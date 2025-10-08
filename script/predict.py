import numpy as np
from xCUDO_model.utilis.data_utilis import *
import joblib
from xCUDO_model.OuterFrame.BiEncoder import *
from xCUDO_model.OuterFrame.MLP import *
from xCUDO_model.OuterFrame.Decoder import *
from xCUDO_model.InternalFrame.TF_model import *
from tqdm import tqdm




class Datareader4p(object):
    def __init__(self, drug_file, gene_file, data_path, device, result_path, batch_size=32):
        self.device = device
        self.drug, self.drug_vec = read_drug_string(drug_file)
        self.gene = read_gene(gene_file, device)
        drug, cell, time_dose = self.read_data(data_path)
        feature_dict = self.trans_to_tensor(drug, cell, time_dose, self.drug_vec, result_path) #self.lb
        self.ft = np.concatenate(
            (feature_dict['pert_time'], feature_dict['cell_id'], feature_dict['pert_idose'], feature_dict['drug']), axis=1)
        self.batch_num = len(self.ft) // batch_size

    def read_data(self, adata_path):
        # adata = ad.read_h5ad(adata_path)
        data = pd.read_csv(adata_path)

        drug = data.loc[:,'pert_id']
        cell = data.iloc[:,4:]
        time_dose = data.loc[:,['pert_itime', 'pert_idose']]
        # lb = adata.X
        return np.asarray(drug), np.asarray(cell, dtype=np.float64), np.asarray(time_dose, dtype=np.float64)
    
    def trans_to_tensor(self, ft_drug, ft_cell, ft_time_dose, drug, result_path):
    
        drug_feature = []

        use_pert_time = True
        use_cell_id = True
        use_pert_dose = True

        for i, ft in enumerate(ft_drug):
            drug_fp = drug[ft]
            drug_feature.append(drug_fp)
        cell_id_feature_list = []
        for i, ft in enumerate(ft_cell):
            if use_cell_id:
                cell_id_feature_list.append(np.array(ft, dtype=np.float64))

        feature_dict = dict()

        feature_dict['drug'] = np.asarray(drug_feature)

        ft_time = ft_time_dose[:, 0]
        ft_dose = ft_time_dose[:, 1]

        time_scaler = joblib.load(f'{result_path}/time_scaler.pkl')
        dose_scaler = joblib.load(f'{result_path}/dose_scaler.pkl')
        ft_time = time_scaler.transform(ft_time.reshape(-1, 1))
        ft_dose = np.log(ft_dose + 1e-6)
        ft_dose = dose_scaler.transform(ft_dose.reshape(-1, 1))
        
        if use_pert_time:
            feature_dict['pert_time'] = ft_time
        if use_cell_id:
            feature_dict['cell_id'] = np.asarray(cell_id_feature_list, dtype=np.float64)
        if use_pert_dose:
            feature_dict['pert_idose'] = ft_dose

        return feature_dict

    def get_batch_data(self, batch_size):
        feature = torch.tensor(self.ft, dtype=torch.float64, device=self.device)

        for start_idx in range(0, feature.shape[0], batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            output['drug'] = feature[excerpt, 980:].clone().detach()
            output['pert_time'] = feature[excerpt, 0].clone().detach()
            output['cell_id'] = feature[excerpt, 1:979].clone().detach()
            output['pert_idose'] = feature[excerpt, 979].clone().detach()
            yield output



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())


drug_file = "../data/all_cp_smiles_129vec.csv"
gene_file = "../data/geneid2vec128_978.csv"



def Predicting_process(drug_file, gene_file, **kwargs):
    CE_model = TF(drug_input_dim=kwargs['drug_input_dim'], gene_input_dim=kwargs['gene_input_dim'], drug_gene_embed_dim=kwargs['drug_gene_emb_dim'], 
                device=kwargs['device'], hid_dim=kwargs['hid_dim'], num_gene=kwargs['gene_num'], CElatent_dim=kwargs['CE_latent_dim'],drop=kwargs['CE_drop'],CE_output_dim=kwargs['CE_output_dim'],
                n_layers=kwargs['CE_n_layers'], n_heads=kwargs['CE_n_heads'],
                cell_id_input_dim=kwargs['cell_id_input_dim'], cell_id_emb_dim=kwargs['cell_id_emb_dim'],initializer=kwargs['initializer'])
    
    share_encoder = TF(drug_input_dim=kwargs['drug_input_dim'], gene_input_dim=kwargs['gene_input_dim'], drug_gene_embed_dim=kwargs['drug_gene_emb_dim'], 
                device=kwargs['device'], hid_dim=kwargs['hid_dim'], num_gene=kwargs['gene_num'], CElatent_dim=kwargs['CE_latent_dim'],drop=kwargs['CE_drop'],CE_output_dim=kwargs['CE_output_dim'],
                n_layers=kwargs['CE_n_layers'], n_heads=kwargs['CE_n_heads'],
                cell_id_input_dim=kwargs['cell_id_input_dim'], cell_id_emb_dim=kwargs['cell_id_emb_dim'],initializer=kwargs['initializer'])
    share_decoder = Decoder(input_dim=kwargs['CE_output_dim']*2, hid_dim_list=kwargs['hid_dim_list'], latent_dim=share_encoder.linear_dim, drop=kwargs['ende_drop'],device=kwargs['device'])

    target_dsnae = BiEncoder(private_encoder=CE_model,share_encoder=share_encoder,decoder=share_decoder,**kwargs)

    predictor = MLP(kwargs['CE_output_dim']*2, hid_dim1=512, hid_dim2=512, hid_dim3=256, hid_dim4=256, output_dim=1, drop=kwargs['pred_drop'], device=kwargs['device'], ispredict=True)

    target_dsnae_path = "../saved_model/finetune/finetuned/target_be.pt"
    target_dsnae.load_state_dict(torch.load(target_dsnae_path))
    predictor_path = "../saved_model/finetune/finetuned/predictor.pt"
    predictor.load_state_dict(torch.load(predictor_path))
    result_path = "../results/Model"
    batch_size = 128

    data_path = "" # the path of the data to be predicted
    data = Datareader4p(drug_file, gene_file, data_path, device, result_path,batch_size)
    loop = tqdm(data.get_batch_data(batch_size), total=data.batch_num, desc='Predicting', unit='batch')
    pred_list = []
    with torch.no_grad():
        for target_batch in loop:
            target_dsnae.eval()
            predictor.eval()

            x = target_batch['drug']
            pert_time = target_batch['pert_time']
            cell_id = target_batch['cell_id']
            pert_idose = target_batch['pert_idose']

            code, _ = target_dsnae.encode(x, data.gene, pert_time, cell_id, pert_idose)
            pred = predictor(code)
            pred = pred.cpu().numpy().tolist()
            pred_list += pred
        loop.close()


        pred_data = pd.DataFrame(pred_list)
        gene_vec = pd.read_csv(gene_file,header=None,index_col=0)
        print(pred_data.head())
        print(pred_data.shape)
        pred_data.to_csv('',header=gene_vec.index.tolist(),index=False) # the path of the prediction result
        print(f' predicting finished')



params = {
'drug_input_dim': 128,
'gene_input_dim': 128,
'drug_gene_emb_dim': 128,
'CE_latent_dim': 256,
'CE_output_dim': 128,
'CE_drop': 0.2,
'CE_n_layers':1,
'CE_n_heads':2,
'initializer' : torch.nn.init.kaiming_uniform_,
'device': device,
'hid_dim': 1024,
'gene_num': 978,
'cell_id_input_dim': 978,
'pert_time_emb_dim': 4,
'cell_id_emb_dim': 128,
'pert_dose_emb_dim': 4,
'pt_lr': 0.00037,
'pt_batch': 32,
'pt_pert_epochs': 100,
'pt_ctrl_epochs': 100,
'ende_drop': 0.7,
'ft_batch': 16,
'hid_dim_list': [512, 512, 256, 256, 128],
'latent_dim': 256,
'ft_lr': 7e-05,
'ft_epochs': 300,
'gp': 10,#
'alpha': 1.0,
'belta': 1.0,
'conf_drop': 0.5,
'ae_lr': 1.6e-05,
'cc_lr': 0.0024,
'tae_lr': 0.00017,
'cc_output_dim': 128,
'pred_drop': 0.5,
'ende_epochs': 500,
'ende_batch': 16,
'critic_epochs': 500,
'critic_batch': 16
}

Predicting_process(drug_file, gene_file, **params)
