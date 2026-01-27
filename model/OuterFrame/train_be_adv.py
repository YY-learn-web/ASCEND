from collections import defaultdict
from OuterFrame.BiEncoder import BiEncoder
import torch
from itertools import chain
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from utilis.earlystop import EarlyStopping
import torch.autograd as autograd
from OuterFrame.Decoder import Decoder
from OuterFrame.MLP import MLP
from InternalFrame.TF_model import TF
from torch import nn
from InternalFrame.metrics import *
from itertools import cycle
from scipy.stats import zscore
import pandas as pd

def correlation(label_test, label_predict, correlation_type):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(corr(lb_test, lb_predict)[0])
    return np.mean(score), score


def eval_be_epoch(model, batch, gene,limit_flag=False):
    if limit_flag:
        limit_size = min(len(batch[0]['drug']), len(batch[0]['drug']))
        x = batch[0]['drug'][:limit_size]
        pert_time = batch[0]['pert_time'][:limit_size]
        cell_id = batch[0]['cell_id'][:limit_size]
        pert_idose = batch[0]['pert_idose'][:limit_size]
    else:
        x = batch[0]['drug']
        pert_time = batch[0]['pert_time']
        cell_id = batch[0]['cell_id']
        pert_idose = batch[0]['pert_idose']
    loss_dict = model.loss_function(*model(x, gene, pert_time, cell_id, pert_idose))
    return loss_dict


def BiEncoder_train_step(s_be, t_be, s_batch, t_batch, data, optimizer,limit_flag=False):
    s_be.train()
    t_be.train()
    
    s_be.zero_grad()
    t_be.zero_grad()
    if limit_flag:
        limit_size = min(len(s_batch[0]['drug']), len(t_batch[0]['drug']))
        source_loss_dict = s_be.loss_function(*s_be(s_batch[0]['drug'][:limit_size], data.gene, s_batch[0]['pert_time'][:limit_size], s_batch[0]['cell_id'][:limit_size], s_batch[0]['pert_idose'][:limit_size]))
        target_loss_dict = t_be.loss_function(*t_be(t_batch[0]['drug'][:limit_size], data.gene, t_batch[0]['pert_time'][:limit_size], t_batch[0]['cell_id'][:limit_size], t_batch[0]['pert_idose'][:limit_size]))
    else:
        source_loss_dict = s_be.loss_function(*s_be(s_batch[0]['drug'], data.gene, s_batch[0]['pert_time'], s_batch[0]['cell_id'], s_batch[0]['pert_idose']))
        target_loss_dict = t_be.loss_function(*t_be(t_batch[0]['drug'], data.gene, t_batch[0]['pert_time'], t_batch[0]['cell_id'], t_batch[0]['pert_idose']))

    optimizer.zero_grad()
    loss = source_loss_dict['loss'] + target_loss_dict['loss']

    loss.backward()
    optimizer.step()

    return loss.cpu().detach().item()


def compute_gradient_penalty(critic, real_samples, fake_samples, cc_output_dim, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 978, real_samples.shape[2])).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 978, cc_output_dim)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), 978, -1)
    gradient_penalty = ((gradients.norm(2, dim=2) - 1) ** 2).mean()
    return gradient_penalty


def critic_be_train_step(data, critic, s_be, t_be,s_batch, t_batch,  device, optimizer,
                        history, cc_output_dim, scheduler=None,clip=None, gp=None, limit_flag=False):
    critic.zero_grad()
    s_be.zero_grad()
    t_be.zero_grad()
    s_be.eval()
    t_be.eval()
    critic.train()
    if limit_flag:
        limit_size = min(len(s_batch[0]['drug']), len(t_batch[0]['drug']))
        s_x = s_batch[0]['drug'][:limit_size]
        t_x = t_batch[0]['drug'][:limit_size]
        s_pert_time = s_batch[0]['pert_time'][:limit_size]
        s_cell_id = s_batch[0]['cell_id'][:limit_size]
        s_pert_idose = s_batch[0]['pert_idose'][:limit_size]
        t_pert_time = t_batch[0]['pert_time'][:limit_size]
        t_cell_id = t_batch[0]['cell_id'][:limit_size]
        t_pert_idose = t_batch[0]['pert_idose'][:limit_size]
    else:
        s_x = s_batch[0]['drug']
        t_x = t_batch[0]['drug']
        s_pert_time = s_batch[0]['pert_time']
        s_cell_id = s_batch[0]['cell_id']
        s_pert_idose = s_batch[0]['pert_idose']
        t_pert_time = t_batch[0]['pert_time']
        t_cell_id = t_batch[0]['cell_id']
        t_pert_idose = t_batch[0]['pert_idose']

    s_latent_code, _ = s_be.encode(s_x, data.gene, s_pert_time, s_cell_id, s_pert_idose)
    t_latent_code, _ = t_be.encode(t_x, data.gene, t_pert_time, t_cell_id, t_pert_idose)

    loss = torch.mean(critic(t_latent_code)) - torch.mean(critic(s_latent_code))

    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=s_latent_code,
                                                    fake_samples=t_latent_code,
                                                    cc_output_dim=cc_output_dim,
                                                    device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()
    history['critic_loss'].append(loss.cpu().detach().item())
    return history


def gan_be_gen_train_step(data, critic, s_be, t_be, s_batch, t_batch, optimizer, belta, history,
                            scheduler=None,limit_flag=False):
    critic.zero_grad()
    s_be.zero_grad()
    t_be.zero_grad()
    critic.eval()
    s_be.train()
    t_be.train()

    if limit_flag:
        limit_size = min(len(s_batch[0]['drug']), len(t_batch[0]['drug']))
        s_x = s_batch[0]['drug'][:limit_size]
        t_x = t_batch[0]['drug'][:limit_size]
        s_pert_time = s_batch[0]['pert_time'][:limit_size]
        s_cell_id = s_batch[0]['cell_id'][:limit_size]
        s_pert_idose = s_batch[0]['pert_idose'][:limit_size]
        t_pert_time = t_batch[0]['pert_time'][:limit_size]
        t_cell_id = t_batch[0]['cell_id'][:limit_size]
        t_pert_idose = t_batch[0]['pert_idose'][:limit_size]
    else:
        s_x = s_batch[0]['drug']
        t_x = t_batch[0]['drug']
        s_pert_time = s_batch[0]['pert_time']
        s_cell_id = s_batch[0]['cell_id']
        s_pert_idose = s_batch[0]['pert_idose']
        t_pert_time = t_batch[0]['pert_time']
        t_cell_id = t_batch[0]['cell_id']
        t_pert_idose = t_batch[0]['pert_idose']

    t_code, _ = t_be.encode(t_x, data.gene, t_pert_time, t_cell_id, t_pert_idose)

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))

    s_loss_dict = s_be.loss_function(*s_be(s_x, data.gene, s_pert_time, s_cell_id, s_pert_idose))
    t_loss_dict = t_be.loss_function(*t_be(t_x, data.gene, t_pert_time, t_cell_id, t_pert_idose))
    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss = recons_loss + belta * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step(loss.item())

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(gen_loss.cpu().detach().item())
    history['gan_loss'].append(loss.cpu().detach().item())

    return history



def train_be_code(data, result_path, **kwargs):

    device = kwargs['device']

    pert_TF_model = TF(drug_input_dim=kwargs['drug_input_dim'], gene_input_dim=kwargs['gene_input_dim'], drug_gene_embed_dim=kwargs['drug_gene_emb_dim'], 
                device=kwargs['device'], hid_dim=kwargs['hid_dim'], num_gene=kwargs['gene_num'], TFlatent_dim=kwargs['TF_latent_dim'],drop=kwargs['TF_drop'],TF_output_dim=kwargs['TF_output_dim'],
                n_layers=kwargs['TF_n_layers'], n_heads=kwargs['TF_n_heads'],
                cell_id_input_dim=kwargs['cell_id_input_dim'], cell_id_emb_dim=kwargs['cell_id_emb_dim'],initializer=kwargs['initializer'])

    ctrl_TF_model = TF(drug_input_dim=kwargs['drug_input_dim'], gene_input_dim=kwargs['gene_input_dim'], drug_gene_embed_dim=kwargs['drug_gene_emb_dim'], 
                device=kwargs['device'], hid_dim=kwargs['hid_dim'], num_gene=kwargs['gene_num'], TFlatent_dim=kwargs['TF_latent_dim'],drop=kwargs['TF_drop'],TF_output_dim=kwargs['TF_output_dim'],
                n_layers=kwargs['TF_n_layers'], n_heads=kwargs['TF_n_heads'],
                cell_id_input_dim=kwargs['cell_id_input_dim'], cell_id_emb_dim=kwargs['cell_id_emb_dim'],initializer=kwargs['initializer'])

    share_encoder = TF(drug_input_dim=kwargs['drug_input_dim'], gene_input_dim=kwargs['gene_input_dim'], drug_gene_embed_dim=kwargs['drug_gene_emb_dim'], 
                device=kwargs['device'], hid_dim=kwargs['hid_dim'], num_gene=kwargs['gene_num'], TFlatent_dim=kwargs['TF_latent_dim'],drop=kwargs['TF_drop'],TF_output_dim=kwargs['TF_output_dim'],
                n_layers=kwargs['TF_n_layers'], n_heads=kwargs['TF_n_heads'],
                cell_id_input_dim=kwargs['cell_id_input_dim'], cell_id_emb_dim=kwargs['cell_id_emb_dim'],initializer=kwargs['initializer'])
    share_decoder = Decoder(input_dim=kwargs['TF_output_dim']*2, hid_dim_list=kwargs['hid_dim_list'], latent_dim=share_encoder.linear_dim, drop=kwargs['ende_drop'],device=kwargs['device'])

    source_be = BiEncoder(private_encoder=pert_TF_model,share_encoder=share_encoder,decoder=share_decoder,**kwargs)
    target_be = BiEncoder(private_encoder=ctrl_TF_model,share_encoder=share_encoder,decoder=share_decoder,**kwargs)

    confounding_classifier = MLP(input_dim=kwargs['TF_output_dim'] * 2,
                                hid_dim1=512,
                                hid_dim2=512,
                                hid_dim3=256,
                                hid_dim4=256,
                                output_dim=kwargs['cc_output_dim'],
                                drop=kwargs['conf_drop'],
                                device=device)

    be_params = [source_be.private_encoder.parameters(),
                target_be.private_encoder.parameters(),
                    share_decoder.parameters(),
                    share_encoder.parameters(),
                    ]
    
    t_be_params = [target_be.private_encoder.parameters(),
                source_be.private_encoder.parameters(),
                share_decoder.parameters(),
                share_encoder.parameters(),
                ]
    
    be_optimizer = torch.optim.AdamW(chain(*be_params), lr=kwargs['be_lr'])
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['cc_lr'])
    t_be_optimizer = torch.optim.RMSprop(chain(*t_be_params), lr=kwargs['tbe_lr'])
    loss_fn = nn.MSELoss()
    
    # os.makedirs(os.path.join(result_path, 'finetune'), exist_ok=True)
    os.chdir(os.path.join(result_path))
    writer = SummaryWriter(os.path.join('En_De_log'))

    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join('target_be.pt'))


    print("Start en-de trainning")
    epochs = int(kwargs['ende_epochs'])
    for epoch in range(epochs):
        data.init_data(batch_size=kwargs['ende_batch'],dataset='adv_train', group='ctrl')
        target_dataloader = data.get_batch_data(batch_size=kwargs['ende_batch'], shuffle=True)
        data.init_data(batch_size=kwargs['ende_batch'],dataset='adv_train', group='pert')
        batch_num = data.batch_num
        source_dataloader = data.get_batch_data(batch_size=kwargs['ende_batch'], shuffle=True)
        loop = tqdm(enumerate(source_dataloader), total=batch_num)
        for step, source_batch in loop:
            target_batch = next(cycle(target_dataloader))
            limit_flag = (len(source_batch[0]['drug']) < kwargs['ende_batch']) or (len(target_batch[0]['drug']) < kwargs['ende_batch'])
            loss = BiEncoder_train_step(s_be=source_be,
                                    t_be=target_be,
                                    s_batch=source_batch,
                                    t_batch=target_batch,
                                    limit_flag=limit_flag,
                                    data=data,
                                    optimizer=be_optimizer
                                    )
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss)
        loop.close()
        writer.add_scalar('train_loss', loss, epoch)
        target_avg_loss_dict = {'loss':0, 'recons_loss':0, 'ortho_loss':0} 
        source_avg_loss_dict= {'loss':0, 'recons_loss':0, 'ortho_loss':0}
        min_loss = float('inf')

        data.init_data(batch_size=kwargs['ende_batch'], dataset='adv_test', group='ctrl')
        target_dataloader = data.get_batch_data(batch_size=kwargs['ende_batch'], shuffle=True)
        data.init_data(batch_size=kwargs['ende_batch'], dataset='adv_test', group='pert')
        batch_num = data.batch_num
        source_dataloader = data.get_batch_data(batch_size=kwargs['ende_batch'], shuffle=True)
        loop = tqdm(enumerate(source_dataloader), total=batch_num)
        with torch.no_grad():   
            for step, source_batch in loop:
                target_batch = next(cycle(target_dataloader))
                limit_flag = (len(source_batch[0]['drug']) < kwargs['ende_batch']) or (len(target_batch[0]['drug']) < kwargs['ende_batch'])
                target_be.eval()
                source_be.eval()
                target_loss_dict = eval_be_epoch(model=target_be,
                                                    batch=target_batch,
                                                    limit_flag=limit_flag,
                                                    gene = data.gene
                                                    )
                source_avg_loss_dict = eval_be_epoch(model=source_be,
                                                        batch=source_batch,
                                                        limit_flag=limit_flag,
                                                        gene=data.gene
                                                        )
                for k ,v in target_loss_dict.items():
                    target_avg_loss_dict[k] += v.cpu().detach().item() / (step + 1)
                for k, v in source_avg_loss_dict.items():
                    source_avg_loss_dict[k] += v.cpu().detach().item() / (step + 1)
            loop.close()
            target_avg_loss = target_avg_loss_dict['loss']
            source_avg_loss = source_avg_loss_dict['loss']
            writer.add_scalar('target_test_loss', target_avg_loss, epoch)
            writer.add_scalar('source_test_loss', source_avg_loss, epoch)
            
            if min_loss > target_avg_loss:
                min_loss = target_avg_loss
                torch.save(target_be.state_dict(), os.path.join('target_be.pt'))
                torch.save(source_be.state_dict(), os.path.join('source_be.pt'))
            
            early_stopping(target_avg_loss, target_be)
            if early_stopping.early_stop:
                torch.save(source_be.state_dict(), os.path.join('source_be.pt'))
                print('Early stopping')
                break


    # Adversarial Learning
    source_be.load_state_dict(torch.load(os.path.join('source_be.pt')))
    target_be.load_state_dict(torch.load(os.path.join('target_be.pt')))
    precision_degree = [10, 20, 50, 100]
    best_dev_pearson = float("-inf")
    Adv_writer = SummaryWriter(os.path.join('Adv_log'))
    early_stopping = EarlyStopping(patience=15, verbose=True, path=os.path.join('target_be.pt'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    t_be_optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True,
                    min_lr=1e-6
                )

    print("Start adversarial training")
    epochs = int(kwargs['critic_epochs'])
    for epoch in range(epochs):
        critic_train_history = defaultdict(list)
        gen_train_history = defaultdict(list)
        adv_min_loss = float('inf')
        data.init_data(batch_size=kwargs['critic_batch'], dataset='critic', group='ctrl')
        target_dataloader = data.get_batch_data(batch_size=kwargs['critic_batch'], shuffle=True)
        data.init_data(batch_size=kwargs['critic_batch'], dataset='critic', group='pert')
        batch_num = data.batch_num
        source_dataloader = data.get_batch_data(batch_size=kwargs['critic_batch'], shuffle=True)
        loop = tqdm(enumerate(source_dataloader), total=batch_num)
        for step, source_batch in loop:
            target_batch = next(cycle(target_dataloader))
            limit_flag = (len(source_batch[0]['drug']) < kwargs['critic_batch']) or (len(target_batch[0]['drug']) < kwargs['critic_batch'])
            critic_train_history = critic_be_train_step(data, critic=confounding_classifier,
                                            s_be=source_be,
                                            t_be=target_be,
                                            s_batch=source_batch,
                                            t_batch=target_batch,
                                            limit_flag=limit_flag,
                                            device=kwargs['device'],
                                            optimizer=classifier_optimizer,
                                            history=critic_train_history,
                                            cc_output_dim=kwargs['cc_output_dim'],
                                            gp=kwargs['gp'])
                                            # clip=0.01,

            
            if step % 5 == 0:
                gen_train_history = gan_be_gen_train_step(data, critic=confounding_classifier,
                                                            s_be=source_be,
                                                            t_be=target_be,
                                                            s_batch=source_batch,
                                                            t_batch=target_batch,
                                                            limit_flag=limit_flag,
                                                            optimizer=t_be_optimizer,
                                                            scheduler=scheduler,
                                                            belta=kwargs['belta'],
                                                            history=gen_train_history)
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(critic_train_loss = critic_train_history['critic_loss'][-1],gen_train_loss = gen_train_history['gan_loss'][-1])
        loop.close()
        Adv_writer.add_scalar('critic_loss', critic_train_history['critic_loss'][-1], epoch)
        Adv_writer.add_scalar('gan_loss', gen_train_history['gan_loss'][-1], epoch)


        if gen_train_history['gan_loss'][-1] < adv_min_loss:
            adv_min_loss = gen_train_history['gan_loss'][-1]
            torch.save(source_be.state_dict(), os.path.join('source_be.pt'))
            torch.save(target_be.state_dict(), os.path.join('target_be.pt'))
        if (epoch+1) % 50 == 0:
            print(f'{epoch}Critic Loss:', critic_train_history['critic_loss'][-1])
            print(f'{epoch}Gan Loss:', gen_train_history['gan_loss'][-1])

        early_stopping(gen_train_history['gan_loss'][-1], target_be)
        if early_stopping.early_stop:
            print(f'{epoch}Critic Loss:', critic_train_history['critic_loss'][-1])
            print(f'{epoch}Gan Loss:', gen_train_history['gan_loss'][-1])
            print('Early stopping')
            break



    target_be.load_state_dict(torch.load(os.path.join('target_be.pt')))
    
    finetune_writer = SummaryWriter(os.path.join('Finetune_log'))
    predictor = MLP(kwargs['TF_output_dim']*2, hid_dim1=512, hid_dim2=512, hid_dim3=256, hid_dim4=256, output_dim=1, drop=kwargs['pred_drop'], device=kwargs['device'], ispredict=True)
    params_predictor = predictor.parameters()
    params_target_be = target_be.parameters()
    optimizer_predictor = torch.optim.AdamW(params_predictor, lr=kwargs['ft_predlr'])
    optimizer_target_be = torch.optim.AdamW(params_target_be, lr=kwargs['ft_belr'])
    os.makedirs('finetuned', exist_ok=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join('finetuned','target_be.pt'))
    best_dev_pearson = float("-inf")

    print("Start finetuning")
    epochs = int(kwargs['ft_epochs'])
    for epoch in range(epochs):
        train_loss_sum = 0
        vali_loss_sum = 0
        lb_np = np.empty([0, kwargs['gene_num']])
        predict_np = np.empty([0, kwargs['gene_num']])
        data.init_data(dataset='ft_train', batch_size=kwargs['ft_batch'])
        batch_num = data.batch_num
        dataloader = data.get_batch_data(batch_size=kwargs['ft_batch'], shuffle=True)
        loop = tqdm(enumerate(dataloader), total=batch_num)
        for step, target_batch in loop:
            target_be.train()
            predictor.train()
            target_be.zero_grad()
            predictor.zero_grad()
            
            output, _ = target_be.encode(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
            if (epoch + 1) % 10 == 0 and (step+1) % 10 == 0:
                # Joint training of target_be and predictor
                predict = predictor(output)
                train_loss = loss_fn(predict, target_batch[1])
                optimizer_predictor.zero_grad()
                optimizer_target_be.zero_grad()
                train_loss.backward()
                optimizer_predictor.step()
                optimizer_target_be.step()
            else:
                output_detached = output.detach()
                predict = predictor(output_detached)
                train_loss_predictor = loss_fn(predict, target_batch[1])
                optimizer_predictor.zero_grad()
                train_loss_predictor.backward()
                optimizer_predictor.step()
                train_loss = train_loss_predictor

            train_loss_sum += train_loss.item()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(train_loss = train_loss.item())
        loop.close()
        train_loss_average = train_loss_sum / (step + 1)
        finetune_writer.add_scalar('train_loss', train_loss_average, epoch)

        rmse_vali_list = []
        pearson_vali_list = []
        spearman_vali_list = []
        precisionk_vali_list = []
        data.init_data(dataset='ft_vali', batch_size=kwargs['ft_batch'])
        batch_num = data.batch_num
        dataloader = data.get_batch_data(batch_size=kwargs['ft_batch'], shuffle=True)
        loop = tqdm(enumerate(dataloader), total=batch_num)
        for step, target_batch in loop:
            target_be.eval()
            predictor.eval()
            output, _ = target_be.encode(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
            predict = predictor(output)
            vali_loss = loss_fn(predict, target_batch[1])
            vali_loss_sum += vali_loss.item()
            lb_np = np.concatenate((lb_np, target_batch[1].cpu().detach().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().detach().numpy()), axis=0)
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(vali_loss = vali_loss.item())
        loop.close()
        
        vali_loss_average = vali_loss_sum / (step + 1)
        finetune_writer.add_scalar('vali_loss', vali_loss_average, epoch)
        rmse_score = RMSE(lb_np, predict_np)
        rmse_vali_list.append(rmse_score)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_vali_list.append(pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_vali_list.append(spearman)

        vali_pearson = pearson
        vali_rmse_score = rmse_score
        
        early_stopping(-pearson, target_be)
        if early_stopping.early_stop:
            torch.save(target_be.state_dict(), os.path.join('finetuned','target_be.pt'))
            torch.save(predictor.state_dict(), os.path.join('finetuned','predictor.pt'))
            print("Early stopping")
            break

        precision = []
        for k in precision_degree:
            precision_neg, precision_pos = precision_k(lb_np, predict_np, k)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            precision.append([precision_pos, precision_neg])
        precisionk_vali_list.append(precision)
        
        if epoch % 10 == 0:
            print(f'{epoch}Validation Loss:', vali_loss_average)
            print(f'{epoch}Validation RMSE:', rmse_score)
            print(f'{epoch}Validation Pearson:', pearson)
            print(f'{epoch}Validation Spearman:', spearman)
            print(f'{epoch}Validation Precision:', precision)
            print('\n')
        if pearson > best_dev_pearson:
            best_dev_pearson = pearson
            torch.save(target_be.state_dict(), 'finetuned/target_be.pt')
            torch.save(predictor.state_dict(), 'finetuned/predictor.pt')

            test_label_np = np.empty([0, kwargs['gene_num']])
            test_predict_np = np.empty([0, kwargs['gene_num']])
            ctrl_profile_np = np.empty([0, kwargs['gene_num']])
            test_loss_sum = 0
            data.init_data(dataset='ft_test', batch_size=kwargs['ft_batch'])
            batch_num = data.batch_num
            dataloader = data.get_batch_data(batch_size=kwargs['ft_batch'], shuffle=True)
            loop = tqdm(enumerate(dataloader), total=batch_num)
            for step, target_batch in loop:
                target_be.eval()
                predictor.eval()
                output, _ = target_be.encode(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
                predict = predictor(output)
                test_loss = loss_fn(predict, target_batch[1])
                test_loss_sum += test_loss.item()
                test_label_np = np.concatenate((test_label_np, target_batch[1].cpu().detach().numpy()), axis=0)
                test_predict_np = np.concatenate((test_predict_np, predict.cpu().detach().numpy()), axis=0)
                ctrl_profile_np = np.concatenate((ctrl_profile_np, target_batch[0]['cell_id'].cpu().detach().numpy()), axis=0)
            loop.close()
            rmse_score = RMSE(test_label_np, test_predict_np)
            test_pearson, _ = correlation(test_label_np, test_predict_np, 'pearson')
            spearman, _ = correlation(test_label_np, test_predict_np, 'spearman')
            test_loss_average = test_loss_sum / (step + 1)
            precision = []
            for k in precision_degree:
                precision_neg, precision_pos = precision_k(test_label_np, test_predict_np, k)
                print("Precision@%d Positive: %.4f" % (k, precision_pos))
                print("Precision@%d Negative: %.4f" % (k, precision_neg))
                precision.append([precision_pos, precision_neg])
            print(f'{epoch}Test Loss:', test_loss_average)
            print(f'{epoch}Test RMSE:', rmse_score)
            print(f'{epoch}Test Pearson:', test_pearson)
            print(f'{epoch}Test Spearman:', spearman)
            print(f'{epoch}Test Precision:', precision)
            # print('\n')
            
            test_label = pd.DataFrame(test_label_np).apply(zscore, axis=1)
            test_predict = pd.DataFrame(test_predict_np).apply(zscore, axis=1)
            ctrl_profile = pd.DataFrame(ctrl_profile_np).apply(zscore, axis=1)
            DEGs_pred = np.array(test_predict) - np.array(ctrl_profile)
            DEGs_label = np.array(test_label) - np.array(ctrl_profile)
            DEGspearson, _ = correlation(DEGs_label, DEGs_pred, 'pearson')
            DEGsspearman, _ = correlation(DEGs_label, DEGs_pred, 'spearman')
            print(f'{epoch}DEGs Pearson:', DEGspearson)
            print(f'{epoch}DEGs Spearman:', DEGsspearman)
            print('\n')