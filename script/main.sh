#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python "../model/main.py" \
--drug_file "../data/all_cp_smiles_129vec.csv" \
--gene_file "../data/geneid2vec128_978.csv" \
--source_adata_path "../data/source_data.h5ad" \
--target_adata_path "../data/target_data_example.h5ad" \
--result_path "../results/Model"