import numpy as np
import random
import torch
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import anndata as ad


def read_drug_number(input_file, num_feature):
    drug = []
    drug_vec = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            assert len(line) == num_feature + 1, "Wrong format"
            bin_vec = [float(i) for i in line[1:]]
            drug.append(line[0])
            drug_vec.append(bin_vec)
    drug_vec = np.asarray(drug_vec, dtype=np.float64)
    index = []
    for i in range(np.shape(drug_vec)[1]):
        if len(set(drug_vec[:, i])) > 1:
            index.append(i)
    drug_vec = drug_vec[:, index]
    drug = dict(zip(drug, drug_vec))
    return drug, len(index)

def read_drug_string(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        drug = dict()
        drug_vec = dict()
        for line in lines:
            line = line.strip().split(',')
            assert len(line) == 130, "Wrong format"
            drug[line[0]] = line[1]
            drug_vec[line[0]] = [float(i) for i in line[2:]]
    return drug, drug_vec

def read_gene(input_file, device):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        gene = []
        for line in lines:
            line = line.strip().split(',')
            assert len(line) == 129, "Wrong format"
            gene.append([float(i) for i in line[1:]])
    return torch.from_numpy(np.asarray(gene, dtype=np.float64)).to(device)


def create_mask_feature(data, device):
    batch_idx = data['molecules'].get_neighbor_idx_by_batch('atom')
    molecule_length = [len(idx) for idx in batch_idx]
    mask = torch.zeros(len(batch_idx), max(molecule_length)).to(device).double()
    for idx, length in enumerate(molecule_length):
        mask[idx][:length] = 1
    return mask

def choose_mean_example(examples):
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    return examples[index]

def split_data_by_pert_id(pert_id):
    random.shuffle(pert_id)
    num_pert_id = len(pert_id)
    fold_size = int(num_pert_id/10)
    train_pert_id = pert_id[:fold_size*6]
    dev_pert_id = pert_id[fold_size*6: fold_size*8]
    test_pert_id = pert_id[fold_size*8:]
    return train_pert_id, dev_pert_id, test_pert_id

# def read_data(input_file, filter):
#     feature = []
#     label = []
#     data = dict()
#     pert_id = []
#     with open(input_file, 'r') as f:
#         lines = f.readlines()  # skip header
#         for line in lines[1:]:
#             line = line.strip().split(',')
#             assert len(line) == 982, "Wrong format"
#             if line[0] in filter["times"] and line[2] in filter['cells'] and line[3] in filter['doses']:
#                 ft = ','.join([line[1],line[2],line[3],line[0]])
#                 lb = [float(i) for i in line[4:]]
#                 if ft in data.keys():
#                     data[ft].append(lb)
#                 else:
#                     data[ft] = [lb]
#     for ft, lb in sorted(data.items()):
#         ft = ft.split(',')
#         feature.append(ft)
#         pert_id.append(ft[0])
#         if len(lb) == 1:
#             label.append(lb[0])
#         else:
#             lb = choose_mean_example(lb)
#             label.append(lb)
#     return np.asarray(feature), np.asarray(label, dtype=np.float64)

def read_data(adata_path, ST_k, ST_v=None):
    """
    Part of data 
    24h 6h data too much, so we choose 5% of 24h and 10% of 6h as source data
    用于训练模型
    """

    adata = ad.read_h5ad(adata_path)
    # adata.var排序
    adata.var.index = adata.var.index.astype(str)
    gene_col_sorted = sorted(adata.var.index.tolist())
    # adata = adata[:, gene_col_sorted]

    if ST_v == 1 or ST_v == 0:
        drug = adata[adata.obs[f'{ST_k}'] == ST_v].obs.loc[:,'pert_id']
        cell = adata[adata.obs[f'{ST_k}'] == ST_v].obs.loc[:,gene_col_sorted]
        # gene_col_sorted = sorted(cell.columns.tolist())
        cell = cell[gene_col_sorted]
        time_dose = adata[adata.obs[f'{ST_k}'] == ST_v].obs.loc[:,['pert_itime', 'pert_idose']]
        lb = adata[adata.obs[f'{ST_k}'] == ST_v].X
        return np.asarray(drug), np.asarray(cell, dtype=np.float64), np.asarray(time_dose, dtype=np.float64), np.asarray(lb, dtype=np.float64)
    else:
        drug = ['DMSO'] * adata.obs.shape[0]
        cell = adata.X
        time_dose = adata.obs.loc[:,['pert_itime', 'pert_idose']]
        lb = adata.X
        return np.asarray(drug), np.asarray(cell, dtype=np.float64),np.asarray(time_dose, dtype=np.float64), np.asarray(lb, dtype=np.float64)

def read_data_(adata_path, ST_k, ST_v=None):
    """
    Part of data 
    24h 6h data too much, so we choose 5% of 24h and 10% of 6h as source data
    用于vstime
    """

    adata = ad.read_h5ad(adata_path)

    if ST_v == '1' or ST_v == '0':
        drug_cell = adata[adata.obs[f'{ST_k}'] == ST_v].obs.loc[:,['pert_id', 'cell_iname']]
        time_dose = adata[adata.obs[f'{ST_k}'] == ST_v].obs.loc[:,['pert_itime', 'pert_idose']]
        lb = adata[adata.obs[f'{ST_k}'] == ST_v].X
        return np.asarray(drug_cell), np.asarray(time_dose, dtype=np.float64), np.asarray(lb, dtype=np.float64)
    else:
        drug_cell = adata.obs.loc[:,['pert_id', 'cell_iname']]
        time_dose = adata.obs.loc[:,['pert_itime', 'pert_idose']]
        lb = adata.X
        return np.asarray(drug_cell), np.asarray(time_dose, dtype=np.float64), np.asarray(lb, dtype=np.float64)




def transfrom_to_numpy(drug, cell, time_dose, label, drug_vec, result_path, ST):
    
    drug_feature = []

    use_pert_time = True
    use_cell_id = True
    use_pert_dose = True

    for i, ft in enumerate(drug):
        drug_fp = drug_vec[ft]
        drug_feature.append(drug_fp)
    
    cell_id_feature_list = []
    for i, ft in enumerate(cell):
        cell_id_feature_list.append(ft)


    feature_dict = dict()
    feature_dict['drug'] = np.asarray(drug_feature)

    # 将train_ft_time_dose中两列分开,将train,vali,test合并后进行标准化
    time = time_dose[:, 0]
    dose = time_dose[:, 1]

    # if ST == 1:
    #     scaler = MinMaxScaler()
    #     time = scaler.fit_transform(time.reshape(-1, 1))
    #     # 保存scaler
    #     scaler_path = f'{result_path}/time_scaler.pkl' #之前那个version1的被这个覆盖了注意一下
    #     joblib.dump(scaler, scaler_path)
    #     dose = np.log(dose + 1e-6)
    #     dose = scaler.fit_transform(dose.reshape(-1, 1))
    #     # 保存scaler
    #     scaler_path = f'{result_path}/dose_scaler.pkl'
    #     joblib.dump(scaler, scaler_path)
    # else:
    #     time_scaler = joblib.load(f'{result_path}/time_scaler.pkl')
    #     dose_scaler = joblib.load(f'{result_path}/dose_scaler.pkl')
    #     time = time_scaler.transform(time.reshape(-1, 1))
    #     dose = np.log(dose + 1e-6)
    #     dose = dose_scaler.transform(dose.reshape(-1, 1))
    

    if ST == 1:
        time_scaler = MinMaxScaler()
        time = time_scaler.fit_transform(time.reshape(-1, 1))
        time_scaler_path = f'{result_path}/time_scaler.pkl'
        joblib.dump(time_scaler, time_scaler_path)
        dose_scaler = StandardScaler()
        dose = np.log(dose + 1e-6) # 先进行对数变换
        dose = dose_scaler.fit_transform(dose.reshape(-1, 1)) # 再进行 StandardScaler 归一化
        dose_scaler_path = f'{result_path}/dose_scaler.pkl'
        joblib.dump(dose_scaler, dose_scaler_path)
    else:
        time_scaler_path = f'{result_path}/time_scaler.pkl'
        dose_scaler_path = f'{result_path}/dose_scaler.pkl'
        time_scaler = joblib.load(time_scaler_path)
        dose_scaler = joblib.load(dose_scaler_path)
        time = time_scaler.transform(time.reshape(-1, 1))
        dose = np.log(dose + 1e-6) # 推断/测试阶段也先进行对数变换
        dose = dose_scaler.transform(dose.reshape(-1, 1)) # 再进行 StandardScaler 归一化


    feature_dict['pert_time'] = time

    feature_dict['cell_id'] = np.asarray(cell_id_feature_list, dtype=np.float64)

    feature_dict['pert_idose'] = dose

    label_regression = label

    return feature_dict, label_regression
