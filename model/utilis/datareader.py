from utilis.data_utilis import *
from sklearn.model_selection import train_test_split


seed = 42
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class DataReader(object):
    def __init__(self, drug_file, gene_file, device, result_path,
                source_adata_path, target_adata_path,ST_k):
        self.device = device
        self.drug, self.drug_vec = read_drug_string(drug_file)
        self.gene = read_gene(gene_file, device)

        drug, cell, time_dose, label = read_data(source_adata_path, ST_k, 1)
        source_feature_dict,source_label_regression = transfrom_to_numpy(drug, cell, time_dose, label, self.drug_vec, result_path, 1)
        source_feature = np.concatenate(
            (source_feature_dict['pert_time'], source_feature_dict['cell_id'], source_feature_dict['pert_idose'],source_feature_dict['drug']), axis=1)

        # Column 0 is pert_time, columns 1-978 are cell_id, column 979 is pert_idose, and after column 980 is drug
        source_feature_Phase1, source_feature_Phase2, source_label_Phase1, source_label_Phase2 = \
            train_test_split(source_feature, source_label_regression, test_size=0.3, random_state=seed)
        self.source_feature_pttrain, self.source_feature_ptvalitest, self.source_label_pttrain, self.source_label_ptvalitest = \
            train_test_split(source_feature_Phase1, source_label_Phase1, test_size=0.3, random_state=seed)
        self.source_feature_ptvali, self.source_feature_pttest, self.source_label_ptvali, self.source_label_pttest = \
            train_test_split(self.source_feature_ptvalitest, self.source_label_ptvalitest, test_size=0.5, random_state=seed)
        
        self.source_feature_fttrain, self.source_feature_ftvali, self.source_label_fttrain, self.source_label_ftvali = \
            train_test_split(source_feature_Phase2, source_label_Phase2, test_size=0.3, random_state=seed)
        
        drug, cell, time_dose, label = read_data(source_adata_path, ST_k, 0)
        target_feature_test_dict,self.target_label_test = transfrom_to_numpy(drug, cell, time_dose, label, self.drug_vec, result_path, 0)
        self.target_feature_test = np.concatenate(
            (target_feature_test_dict['pert_time'], target_feature_test_dict['cell_id'], target_feature_test_dict['pert_idose'],target_feature_test_dict['drug']), axis=1)
        

        drug, cell, time_dose, label = read_data(target_adata_path, ST_k, None)
        target_feature_dict,target_label_regression = transfrom_to_numpy(drug, cell, time_dose, label, self.drug_vec, result_path, None)
        target_feature = np.concatenate(
            (target_feature_dict['pert_time'], target_feature_dict['cell_id'], target_feature_dict['pert_idose'],target_feature_dict['drug']), axis=1)
        self.target_feature_train, self.target_feature_valitest, self.target_label_train, self.target_label_valitest = \
            train_test_split(target_feature, target_label_regression, test_size=0.3, random_state=seed)
        self.target_feature_vali, self.target_feature_pttest, self.target_label_vali, self.target_label_pttest = \
            train_test_split(self.target_feature_valitest, self.target_label_valitest, test_size=0.5, random_state=seed)
        

    def init_data(self, batch_size, dataset, group=None):
        
        if dataset == 'pt_train' and group == 'pert':
            self.feature = self.source_feature_pttrain # ndarray(num，2179) 0：pert_time 1-1830：cell_id 1831：pert_idose 1832-：drug
            self.label = self.source_label_pttrain
        elif dataset == 'pt_vali' and group == 'pert':
            self.feature = self.source_feature_ptvali
            self.label = self.source_label_ptvali
        elif dataset == 'pt_test' and group == 'pert':
            self.feature = self.target_feature_test
            self.label = self.target_label_test
        elif dataset == 'pt_train' and group == 'ctrl':
            self.feature = self.target_feature_train
            self.label = self.target_label_train
        elif dataset == 'pt_vali' and group == 'ctrl':
            self.feature = self.target_feature_vali
            self.label = self.target_label_vali
        elif dataset == 'pt_test' and group == 'ctrl':
            self.feature = self.target_feature_pttest
            self.label = self.target_label_pttest
        elif dataset == 'adv_train' and group == 'pert':
            self.feature = np.concatenate((self.source_feature_pttrain , self.source_feature_ptvali), axis=0)
            self.label = np.concatenate((self.source_label_pttrain ,self.source_label_ptvali), axis=0)
        elif dataset == 'adv_test' and group == 'pert':
            self.feature = self.source_feature_pttest
            self.label = self.source_label_pttest
        elif dataset == 'adv_train' and group == 'ctrl':
            self.feature = np.concatenate((self.target_feature_train ,self.target_feature_vali), axis=0)
            self.label = np.concatenate((self.target_label_train ,self.target_label_vali), axis=0)
        elif dataset == 'adv_test' and group == 'ctrl':
            self.feature = self.target_feature_pttest
            self.label = self.target_label_pttest
        elif dataset == 'critic' and group == 'pert':
            self.feature = np.concatenate((self.source_feature_pttrain ,self.source_feature_ptvali), axis=0)
            self.label = np.concatenate((self.source_label_pttrain ,self.source_label_ptvali), axis=0)
        elif dataset == 'critic' and group == 'ctrl':
            self.feature = np.concatenate((self.target_feature_train ,self.target_feature_vali), axis=0)
            self.label = np.concatenate((self.target_label_train ,self.target_label_vali), axis=0)
        elif dataset == 'ft_train':
            self.feature = self.source_feature_fttrain
            self.label = self.source_label_fttrain
        elif dataset == 'ft_vali':
            self.feature = self.source_feature_ftvali
            self.label = self.source_label_ftvali
        elif dataset == 'ft_test':
            self.feature = self.target_feature_test
            self.label = self.target_label_test
        
        self.batch_num = self.feature.shape[0] // batch_size + 1


    def get_batch_data(self, batch_size, shuffle=True):
        feature = torch.tensor(self.feature, dtype=torch.float64, device=self.device)
        label = torch.tensor(self.label, dtype=torch.float64, device=self.device)
        if shuffle:
            index = torch.randperm(feature.shape[0]).long()
            index = index.numpy()
        for start_idx in range(0, feature.shape[0], batch_size):
            if shuffle:
                excerpt = index[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            output['drug'] = feature[excerpt, 980:].clone().detach()
            output['pert_time'] = feature[excerpt, 0].clone().detach()
            output['cell_id'] = feature[excerpt, 1:979].clone().detach()
            output['pert_idose'] = feature[excerpt, 979].clone().detach()
            yield output, label[excerpt]
    
