from OuterFrame.train_be_adv import train_be_code
from utilis.datareader import DataReader
import torch
import argparse
import json

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--result_path')
parser.add_argument('--source_adata_path')
parser.add_argument('--target_adata_path')
args = parser.parse_args()

# Laod data
drug_file = args.drug_file
gene_file = args.gene_file
source_adata_path = args.source_adata_path
target_adata_path = args.target_adata_path
result_path = args.result_path

params = json.load(open('../data/params.json'))
params.update({'device': device, 
            'initializer' : torch.nn.init.kaiming_uniform_})


adata = DataReader(drug_file=drug_file, gene_file=gene_file, device=device, result_path=result_path,
                        source_adata_path=source_adata_path,target_adata_path=target_adata_path,ST_k='ST')

train_be_code(adata, result_path, **params)

