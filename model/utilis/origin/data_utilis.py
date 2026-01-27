import numpy as np
import random
import torch
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
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
            assert len(line) == 2178, "Wrong format"
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

def read_data(input_file, filter, gene_file):
    data = pd.read_csv(input_file)
    cols = ['pert_id', 'cell_iname', 'pert_itime', 'pert_idose'] + data.columns.to_list()[4:]
    data = data[cols]
    gene_vec = pd.read_csv(gene_file,index_col=0,names=range(128))
    # 判断data的列是否与gene_vec的index一致，不一致就终止任务
    if data.columns.to_list()[4:] != gene_vec.index.to_list():
        print('{input_file} cols is not match gene_vec index')
        exit()

    data = data[data['cell_iname'].isin(filter['cells'])]
    
    # 计算每个组的样本数
    # group_counts = data.groupby(['pert_id', 'cell_iname', 'pert_itime', 'pert_idose']).size()
    
    # 分别处理单样本组和多样本组
    single_sample = data[data.groupby(['pert_id', 'cell_iname', 'pert_itime', 'pert_idose']).transform('size') == 1]
    multi_sample = data[data.groupby(['pert_id', 'cell_iname', 'pert_itime', 'pert_idose']).transform('size') > 1]
    
    # 只对多样本组计算均值
    if not multi_sample.empty:
        multi_sample = multi_sample.groupby(['pert_id', 'cell_iname', 'pert_itime', 'pert_idose']).agg({
            col: 'mean' for col in multi_sample.columns[4:]
        }).reset_index()
    
    # 合并单样本组和处理后的多样本组
    data = pd.concat([single_sample, multi_sample])
    
    ft_drug_cell = data.iloc[:,:2]
    ft_time_dose = data.iloc[:,2:4]
    lb = data.iloc[:,4:]
    # 将dataframe转换为numpy数组
    return np.asarray(ft_drug_cell), np.asarray(ft_time_dose, dtype=np.float64), np.asarray(lb, dtype=np.float64)

def read_data_(adata_path, ST):
    """
    Part of data 
    24h 6h data too much, so we choose 5% of 24h and 10% of 6h as source data
    """

    adata = ad.read_h5ad(adata_path)

    if ST == 1 or ST == 0:
        drug_cell = adata[adata.obs['ST'] == ST].obs.loc[:,['pert_id', 'cell_iname']]
        time_dose = adata[adata.obs['ST'] == ST].obs.loc[:,['pert_itime', 'pert_idose']]
        lb = adata[adata.obs['ST'] == ST].X
        return np.asarray(drug_cell), np.asarray(time_dose, dtype=np.float64), np.asarray(lb, dtype=np.float64)
    else:
        drug_cell = adata.obs.loc[:,['pert_id', 'cell_iname']]
        time_dose = adata.obs.loc[:,['pert_itime', 'pert_idose']]
        lb = adata.X
        return np.asarray(drug_cell), np.asarray(time_dose, dtype=np.float64), np.asarray(lb, dtype=np.float64)


def transfrom_to_tensor(train_ft_drug_cell, train_ft_time_dose, train_label, val_ft_drug_cell, val_ft_time_dose, val_label, test_ft_drug_cell, test_time_dose, test_label, drug, device, filter):
    
    train_drug_feature = []
    val_drug_feature = []
    test_drug_feature = []
    cell_id_set = sorted(filter['cells'])
    # pert_idose_set = sorted(filter['doses'])
    # pert_time_set = sorted(filter['times'])
    use_pert_time = True
    use_cell_id = False
    use_pert_dose = True
    # if len(pert_time_set) >= 1:
    #     pert_time_dict = dict(zip(pert_time_set, list(range(len(pert_time_set)))))
    #     train_pert_time_feature = []
    #     val_pert_time_feature = []
    #     test_pert_time_feature = []
    #     use_pert_time = True
    if len(cell_id_set) >= 1:
        cell_id_dict = dict(zip(cell_id_set, list(range(len(cell_id_set)))))
        train_cell_id_feature = []
        val_cell_id_feature = []
        test_cell_id_feature = []
        use_cell_id = True
    # if len(pert_idose_set) >= 1:
    #     pert_idose_dict = dict(zip(pert_idose_set, list(range(len(pert_idose_set)))))
    #     train_pert_idose_feature = []
    #     val_pert_idose_feature = []
    #     test_pert_idose_feature = []
    #     use_pert_dose = True
    # print('Feature Summary:')
    # print(pert_time_set)
    # print(cell_id_set)
    # print(pert_idose_set)

    for i, ft in enumerate(train_ft_drug_cell):
        drug_fp = drug[ft[0]]
        train_drug_feature.append(drug_fp)
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[1]]] = 1
            train_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float64))

    for i, ft in enumerate(val_ft_drug_cell):
        drug_fp = drug[ft[0]]
        val_drug_feature.append(drug_fp)
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[1]]] = 1
            val_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float64))

    for i, ft in enumerate(test_ft_drug_cell):
        drug_fp = drug[ft[0]]
        test_drug_feature.append(drug_fp)
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[1]]] = 1
            test_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float64))

    train_feature_dict = dict()
    val_feature_dict = dict()
    test_feature_dict = dict()

    train_feature_dict['drug'] = np.asarray(train_drug_feature)
    val_feature_dict['drug'] = np.asarray(val_drug_feature)
    test_feature_dict['drug'] = np.asarray(test_drug_feature)

    # 将train_ft_time_dose中两列分开,将train,vali,test合并后进行标准化
    train_ft_time = train_ft_time_dose[:, 0]
    train_ft_dose = train_ft_time_dose[:, 1]
    val_ft_time = val_ft_time_dose[:, 0]
    val_ft_dose = val_ft_time_dose[:, 1]
    test_ft_time = test_time_dose[:, 0]
    test_ft_dose = test_time_dose[:, 1]

    ft_time = np.concatenate([train_ft_time, val_ft_time, test_ft_time], axis=0)
    ft_dose = np.concatenate([train_ft_dose, val_ft_dose, test_ft_dose], axis=0)
    time_scaler = joblib.load('/dataStor/home/hyzhou/Programs/Gene4/data/version1/time_scaler.pkl')
    dose_scaler = joblib.load('/dataStor/home/hyzhou/Programs/Gene4/data/version1/dose_scaler.pkl')
    ft_time = time_scaler.transform(ft_time.reshape(-1, 1))
    ft_dose = np.log(ft_dose + 1e-6)
    ft_dose = dose_scaler.transform(ft_dose.reshape(-1, 1))

    train_ft_time = ft_time[:len(train_ft_time)]
    val_ft_time = ft_time[len(train_ft_time):len(train_ft_time)+len(val_ft_time)]
    test_ft_time = ft_time[len(train_ft_time)+len(val_ft_time):]
    train_ft_dose = ft_dose[:len(train_ft_dose)]
    val_ft_dose = ft_dose[len(train_ft_dose):len(train_ft_dose)+len(val_ft_dose)]
    test_ft_dose = ft_dose[len(train_ft_dose)+len(val_ft_dose):]

    if use_pert_time:
        train_feature_dict['pert_time'] = torch.from_numpy(train_ft_time).to(device)
        val_feature_dict['pert_time'] = torch.from_numpy(val_ft_time).to(device)
        test_feature_dict['pert_time'] = torch.from_numpy(test_ft_time).to(device)
    if use_cell_id:
        train_feature_dict['cell_id'] = torch.from_numpy(np.asarray(train_cell_id_feature, dtype=np.float64)).to(device)
        val_feature_dict['cell_id'] = torch.from_numpy(np.asarray(val_cell_id_feature, dtype=np.float64)).to(device)
        test_feature_dict['cell_id'] = torch.from_numpy(np.asarray(test_cell_id_feature, dtype=np.float64)).to(device)
    if use_pert_dose:
        train_feature_dict['pert_idose'] = torch.from_numpy(train_ft_dose).to(device)
        val_feature_dict['pert_idose'] = torch.from_numpy(val_ft_dose).to(device)
        test_feature_dict['pert_idose'] = torch.from_numpy(test_ft_dose).to(device)

    train_label_regression = torch.from_numpy(train_label).to(device)
    val_label_regression = torch.from_numpy(val_label).to(device)
    test_label_regression = torch.from_numpy(test_label).to(device)

    return train_feature_dict, val_feature_dict, test_feature_dict, \
            train_label_regression, val_label_regression, test_label_regression, \
            use_pert_time, use_cell_id, use_pert_dose

def transfrom_to_numpy(drug_cell, time_dose, label, drug, device, recoder, ST):
    
    drug_feature = []
    cell_id_set = sorted(recoder['cells'])

    use_pert_time = True
    use_cell_id = True
    use_pert_dose = True

    if len(cell_id_set) >= 1:
        cell_id_dict = dict(zip(cell_id_set, list(range(len(cell_id_set)))))
        cell_id_feature_list = []
    else: 
        print('cell_id_set is empty')
        exit()

    for i, ft in enumerate(drug_cell):
        drug_fp = drug[ft[0]]
        drug_feature.append(drug_fp)
        if use_cell_id:
            cell_id_feature = np.zeros(len(cell_id_set))
            cell_id_feature[cell_id_dict[ft[1]]] = 1
            cell_id_feature_list.append(np.array(cell_id_feature, dtype=np.float64))

    feature_dict = dict()
    feature_dict['drug'] = np.asarray(drug_feature)

    # 将train_ft_time_dose中两列分开,将train,vali,test合并后进行标准化
    time = time_dose[:, 0]
    dose = time_dose[:, 1]

    if ST == 1:
        scaler = MinMaxScaler()
        time = scaler.fit_transform(time.reshape(-1, 1))
        # 保存scaler
        scaler_path = '../results/time_scaler.pkl' #之前那个version1的被这个覆盖了注意一下
        joblib.dump(scaler, scaler_path)
        dose = np.log(dose + 1e-6)
        dose = scaler.fit_transform(dose.reshape(-1, 1))
        # 保存scaler
        scaler_path = '../results/dose_scaler.pkl'
        joblib.dump(scaler, scaler_path)
    else:
        time_scaler = joblib.load('../results/time_scaler.pkl')
        dose_scaler = joblib.load('../results/dose_scaler.pkl')
        time = time_scaler.transform(time.reshape(-1, 1))
        dose = np.log(dose + 1e-6)
        dose = dose_scaler.transform(dose.reshape(-1, 1))

    if use_pert_time:
        feature_dict['pert_time'] = time
    if use_cell_id:
        feature_dict['cell_id'] = np.asarray(cell_id_feature_list, dtype=np.float64)
    if use_pert_dose:
        feature_dict['pert_idose'] = dose

    label_regression = label

    return feature_dict, label_regression
