from xCUDO_model.InternalFrame.TF_model import CE
from OuterFrame.MLP import MLP
from tqdm import tqdm
import torch
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from InternalFrame.metrics import *
from itertools import chain
from utilis.earlystop import EarlyStopping

def pretrain_code(data, result_path, **kwargs):

    pert_CE_model = CE(drug_input_dim=kwargs['drug_input_dim'], gene_input_dim=kwargs['gene_input_dim'], drug_gene_embed_dim=kwargs['drug_gene_emb_dim'], 
                device=kwargs['device'], hid_dim=kwargs['hid_dim'], num_gene=kwargs['gene_num'], CElatent_dim=kwargs['CE_latent_dim'],drop=kwargs['CE_drop'],CE_output_dim=kwargs['CE_output_dim'],
                n_layers=kwargs['CE_n_layers'], n_heads=kwargs['CE_n_heads'],
                cell_id_input_dim=kwargs['cell_id_input_dim'], cell_id_emb_dim=kwargs['cell_id_emb_dim'],initializer=kwargs['initializer'])
    # ctrl_CE_model = CE(drug_input_dim=kwargs['drug_input_dim'], gene_input_dim=kwargs['gene_input_dim'], drug_gene_embed_dim=kwargs['drug_gene_emb_dim'], 
    #             device=kwargs['device'], hid_dim=kwargs['hid_dim'], num_gene=kwargs['gene_num'], CElatent_dim=kwargs['CE_latent_dim'],drop=kwargs['CE_drop'],CE_output_dim=kwargs['CE_output_dim'],
    #             n_layers=kwargs['CE_n_layers'], n_heads=kwargs['CE_n_heads'],
    #             cell_id_input_dim=kwargs['cell_id_input_dim'], cell_id_emb_dim=kwargs['cell_id_emb_dim'],initializer=kwargs['initializer'])
    pert_predictor = MLP(kwargs['CE_output_dim'], hid_dim1=512, hid_dim2=512, hid_dim3=256, hid_dim4=256, output_dim=1, drop=kwargs['pred_drop'], device=kwargs['device'], ispredict=True)
    # ctrl_predictor = MLP(kwargs['CE_output_dim'], hid_dim1=512, hid_dim2=512, hid_dim3=256, hid_dim4=256, output_dim=1, drop=kwargs['pred_drop'], device=kwargs['device'], ispredict=True)

    pert_params = [pert_CE_model.parameters(), pert_predictor.parameters()]
    # ctrl_params = [ctrl_CE_model.parameters(), ctrl_predictor.parameters()]
    pert_optimizer = torch.optim.Adam(chain(*pert_params), lr=kwargs['pt_lr'])
    # ctrl_optimizer = torch.optim.Adam(chain(*ctrl_params), lr=kwargs['pt_lr'])
    os.chdir(os.path.join(result_path, 'pretrain'))
    train_writer = SummaryWriter('log')
    vali_writer = SummaryWriter('log')
    loss_fn = nn.MSELoss()
    print("Start source encoder training")
    epochs = int(kwargs['pt_pert_epochs'])
    for epoch in range(epochs):
        patience = 5
        train_loss_sum = 0
        vali_loss_sum = 0
        lb_np = np.empty([0, kwargs['gene_num']])
        predict_np = np.empty([0, kwargs['gene_num']])
        rmse_vali_list = []
        pearson_vali_list = []
        spearman_vali_list = []
        best_dev_pearson = float("-inf")
        data.init_data(batch_size=kwargs['pt_batch'],dataset='pt_train', group='pert')
        batch_num = data.batch_num
        dataloader = data.get_batch_data(batch_size=kwargs['pt_batch'], shuffle=True)
        loop = tqdm(enumerate(dataloader), total=batch_num)
        for step, target_batch in loop:
            pert_CE_model.train()
            pert_predictor.train()
            pert_CE_model.zero_grad()
            pert_predictor.zero_grad()

            output, _ = pert_CE_model(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
            predict = pert_predictor(output)
            
            train_loss = loss_fn(predict, target_batch[1])
            train_loss_sum += train_loss.item()

            pert_optimizer.zero_grad()
            train_loss.backward()
            pert_optimizer.step()
            loop.set_description(f'Epoch {epoch}/{epochs}')
            loop.set_postfix(loss = train_loss.item())
        loop.close()
        train_loss_average = train_loss_sum / (step + 1)
        train_writer.add_scalar('train_loss', train_loss_average, epoch)
        

        data.init_data(batch_size=kwargs['pt_batch'],dataset='pt_vali', group='pert')
        batch_num = data.batch_num
        dataloader = data.get_batch_data(batch_size=kwargs['pt_batch'], shuffle=True)
        loop = tqdm(enumerate(dataloader), total=batch_num)
        for step, target_batch in loop:
            pert_CE_model.eval()
            pert_predictor.eval()
            output, _ = pert_CE_model(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
            predict = pert_predictor(output)
            vali_loss = loss_fn(predict, target_batch[1])
            vali_loss_sum += vali_loss.item()
            lb_np = np.concatenate((lb_np, target_batch[1].cpu().detach().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().detach().numpy()), axis=0)
        loop.close()
        vali_loss_average = vali_loss_sum / (step + 1)
        vali_writer.add_scalar('vali_loss', vali_loss_average, epoch)
        rmse_score = RMSE(lb_np, predict_np)
        rmse_vali_list.append(rmse_score)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_vali_list.append(pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_vali_list.append(spearman)

        if epoch % 10 == 0:
            print(f'{epoch}Validation Loss:', vali_loss_average)
            print(f'{epoch}Validation RMSE:', rmse_score)
            print(f'{epoch}Validation Pearson:', pearson)
            print(f'{epoch}Validation Spearman:', spearman)
            print('\n')
        if pearson > best_dev_pearson:
            patience = 5
            best_dev_pearson = pearson
            torch.save(pert_CE_model.state_dict(), 'pert_CE_model.pt')
            # torch.save(pert_predictor.state_dict(), 'pert_predictor.pt')
            test_label_np = np.empty([0, kwargs['gene_num']])
            test_predict_np = np.empty([0, kwargs['gene_num']])
            test_loss_sum = 0
            data.init_data(batch_size=kwargs['pt_batch'],dataset='pt_test', group='pert')
            for step, target_batch in enumerate(data.get_batch_data(batch_size=kwargs['pt_batch'], shuffle=True)):
                pert_CE_model.eval()
                pert_predictor.eval()
                output, _ = pert_CE_model(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
                predict = pert_predictor(output)
                test_loss = loss_fn(predict, target_batch[1])
                test_loss_sum += test_loss.item()
                test_label_np = np.concatenate((test_label_np, target_batch[1].cpu().detach().numpy()), axis=0)
                test_predict_np = np.concatenate((test_predict_np, predict.cpu().detach().numpy()), axis=0)
            rmse_score = RMSE(test_label_np, test_predict_np)
            pearson, _ = correlation(test_label_np, test_predict_np, 'pearson')
            spearman, _ = correlation(test_label_np, test_predict_np, 'spearman')
            test_loss_average = test_loss_sum / (step + 1)
            print(f'{epoch}Test Loss:', test_loss_average)
            print(f'{epoch}Test RMSE:', rmse_score)
            print(f'{epoch}Test Pearson:', pearson)
            print(f'{epoch}Test Spearman:', spearman)
            print('\n')
        else:
            patience -= 1
            if patience == 0:
                break




    # print("Start target encoder training")
    # epochs = int(kwargs['pt_ctrl_epochs'])
    # for epoch in range(epochs):
    #     train_loss_sum = 0
    #     vali_loss_sum = 0
    #     patience = 5
    #     lb_np = np.empty([0, kwargs['gene_num']])
    #     predict_np = np.empty([0, kwargs['gene_num']])
    #     rmse_vali_list = []
    #     pearson_vali_list = []
    #     spearman_vali_list = []
    #     best_dev_pearson = float("-inf")
    #     data.init_data(dataset='pt_train', batch_size=kwargs['pt_batch'], group='ctrl')
    #     batch_num = data.batch_num
    #     dataloader = data.get_batch_data(batch_size=kwargs['pt_batch'], shuffle=True)
    #     loop = tqdm(enumerate(dataloader), total=batch_num)
    #     for step, target_batch in loop:
    #         ctrl_CE_model.train()
    #         ctrl_predictor.train()
    #         ctrl_CE_model.zero_grad()
    #         ctrl_predictor.zero_grad()

    #         output, _ = ctrl_CE_model(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
    #         predict = ctrl_predictor(output)
            
    #         train_loss = loss_fn(predict, target_batch[1])
    #         train_loss_sum += train_loss.item()

    #         ctrl_optimizer.zero_grad()
    #         train_loss.backward()
    #         ctrl_optimizer.step()
    #         loop.set_description(f'Epoch [{epoch}/{epochs}]')
    #         loop.set_postfix(loss = train_loss.item())
    #     train_loss_average = train_loss_sum / (step + 1)
    #     train_writer.add_scalar('train_loss', train_loss_average, epoch)
    #     loop.close()

    #     data.init_data(batch_size=kwargs['pt_batch'],dataset='pt_vali', group='ctrl')
    #     batch_num = data.batch_num
    #     dataloader = data.get_batch_data(batch_size=kwargs['pt_batch'], shuffle=True)
    #     loop = tqdm(enumerate(dataloader), total=batch_num)
    #     for step, target_batch in loop:
    #         ctrl_CE_model.eval()
    #         ctrl_predictor.eval()
    #         output, _ = ctrl_CE_model(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
    #         predict = ctrl_predictor(output)
    #         vali_loss = loss_fn(predict, target_batch[1])
    #         vali_loss_sum += vali_loss.item()
    #         lb_np = np.concatenate((lb_np, target_batch[1].cpu().detach().numpy()), axis=0)
    #         predict_np = np.concatenate((predict_np, predict.cpu().detach().numpy()), axis=0)
    #     loop.close()
    #     vali_loss_average = vali_loss_sum / (step + 1)
    #     vali_writer.add_scalar('vali_loss', vali_loss_average, epoch)
    #     rmse_score = RMSE(lb_np, predict_np)
    #     rmse_vali_list.append(rmse_score)
    #     pearson, _ = correlation(lb_np, predict_np, 'pearson')
    #     pearson_vali_list.append(pearson)
    #     spearman, _ = correlation(lb_np, predict_np, 'spearman')
    #     spearman_vali_list.append(spearman)
    #     if epoch % 10 == 0:
    #         print(f'{epoch}Validation Loss:', vali_loss_average)
    #         print(f'{epoch}Validation RMSE:', rmse_score)
    #         print(f'{epoch}Validation Pearson:', pearson)
    #         print(f'{epoch}Validation Spearman:', spearman)
    #         print('\n')
    #     if pearson > best_dev_pearson:
    #         best_dev_pearson = pearson
    #         torch.save(ctrl_CE_model.state_dict(), 'ctrl_CE_model.pt')
    #         test_label_np = np.empty([0, kwargs['gene_num']])
    #         test_predict_np = np.empty([0, kwargs['gene_num']])
    #         test_loss_sum = 0
    #         data.init_data(batch_size=kwargs['pt_batch'],dataset='pt_test', group='ctrl')
    #         for step, target_batch in enumerate(data.get_batch_data(batch_size=kwargs['pt_batch'], shuffle=True)):
    #             ctrl_CE_model.eval()
    #             ctrl_predictor.eval()
    #             output, _ = ctrl_CE_model(target_batch[0]['drug'], data.gene, target_batch[0]['pert_time'], target_batch[0]['cell_id'], target_batch[0]['pert_idose'])
    #             predict = ctrl_predictor(output)
    #             test_loss = loss_fn(predict, target_batch[1])
    #             test_loss_sum += test_loss.item()
    #             test_label_np = np.concatenate((test_label_np, target_batch[1].cpu().detach().numpy()), axis=0)
    #             test_predict_np = np.concatenate((test_predict_np, predict.cpu().detach().numpy()), axis=0)
    #         rmse_score = RMSE(test_label_np, test_predict_np)
    #         test_pearson, _ = correlation(test_label_np, test_predict_np, 'pearson')
    #         spearman, _ = correlation(test_label_np, test_predict_np, 'spearman')
    #         test_loss_average = test_loss_sum / (step + 1)
    #         print(f'{epoch}Test Loss:', test_loss_average)
    #         print(f'{epoch}Test RMSE:', rmse_score)
    #         print(f'{epoch}Test Pearson:', test_pearson)
    #         print(f'{epoch}Test Spearman:', spearman)
    #     else:
    #         patience -= 1
    #         if patience == 0:
    #             break

