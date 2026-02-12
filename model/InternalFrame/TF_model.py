import torch
import torch.nn as nn
import numpy as np
from model.InternalFrame.druggene_attention import DrugGeneAttention


class TF(nn.Module):
    def __init__(self, drug_input_dim, gene_input_dim, drug_gene_embed_dim, device, hid_dim, num_gene, TFlatent_dim, TF_output_dim, drop,
                use_pert_time=True, use_cell_id=True, use_pert_dose=True,initializer=None,n_layers=1, n_heads=1,
                cell_id_input_dim=None,cell_id_emb_dim=None):
        super(TF, self).__init__()
        # assert drug_emb_dim == gene_emb_dim, 'Embedding size mismatch'
        self.num_gene = num_gene
        self.drug_emb_dim = drug_gene_embed_dim
        self.gene_emb_dim = drug_gene_embed_dim
        self.use_pert_time = use_pert_time
        self.use_cell_id = use_cell_id
        self.use_pert_dose = use_pert_dose
        self.device = device
        self.drug_embed = nn.Linear(drug_input_dim, drug_gene_embed_dim, dtype=torch.float64, device=device)
        self.gene_embed = nn.Linear(gene_input_dim, drug_gene_embed_dim, dtype=torch.float64, device=device)
        self.drug_gene_attn = DrugGeneAttention(drug_gene_embed_dim, drug_gene_embed_dim, n_layers=n_layers, n_heads=n_heads, pf_dim=TFlatent_dim,
                                                dropout=drop, device=device)

        self.linear_dim = self.drug_emb_dim + self.gene_emb_dim

        if self.use_pert_time:

            self.linear_dim += 1
        if self.use_cell_id:
            self.cell_id_embed = nn.Linear(cell_id_input_dim, cell_id_emb_dim, dtype=torch.float64, device=device)
            self.linear_dim += cell_id_emb_dim
        if self.use_pert_dose:
            self.linear_dim += 1
        self.linear_1 = nn.Linear(self.linear_dim, hid_dim, dtype=torch.float64, device=device)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hid_dim, TF_output_dim, dtype=torch.float64, device=device)
        self.initializer = initializer
        self.init_weights()

    def init_weights(self):
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if 'drug_gene_attn' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 0.)
                else:
                    self.initializer(parameter)

    def forward(self, input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose):
        # input_drug = [batch * drug_input_dim]
        # input_gene = [num_gene * gene_input_dim]
        num_batch = input_drug.shape[0]
        # input_drug = [batch * drug_input_dim]
        drug_embed = input_drug.unsqueeze(1)
        # drug_embed = [batch * 1 * drug_input_dim]
        drug_embed = drug_embed.repeat(1, self.num_gene, 1)
        # drug_embed = [batch * num_gene * drug_input_dim]
        drug_embed = self.drug_embed(drug_embed)
        # drug_embed = [batch * num_gene * drug_embed_dim]
        gene_embed = input_gene.unsqueeze(0)
        # gene_embed = [1 * num_gene * gene_input_dim]
        gene_embed = gene_embed.repeat(num_batch, 1, 1)
        # gene_embed = [batch * num_gene * gene_input_dim]
        drug_gene_embed = torch.cat((gene_embed, drug_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * gene_emb_dim + drug_emb_dim]
        drug_gene_embed_recon = self.drug_gene_attn(gene_embed, drug_embed, None, None)
        # drug_gene_embed_recon = [batch * num_gene * drug_gene_embed_dim]
        drug_gene_embed_recon = torch.cat((drug_gene_embed_recon, drug_embed), dim=2)
        # drug_gene_embed_recon = [batch * num_gene * drug_gene_embed_dim + drug_emb_dim]
        if self.use_pert_time:
            pert_time_embed = input_pert_time.unsqueeze(1).unsqueeze(1)
            # pert_time_embed = [batch * 1 * pert_time_emb_dim]
            pert_time_embed = pert_time_embed.repeat(1, self.num_gene, 1)
            # pert_time_embed = [batch * num_gene * pert_time_emb_dim]
            drug_gene_embed_recon = torch.cat((drug_gene_embed_recon, pert_time_embed), dim=2)
            # drug_gene_embed_recon = [batch * num_gene * gene_emb_dim + drug_emb_dim + pert_time_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_time_embed), dim=2)
            # drug_gene_embed = [batch * num_gene * gene_emb_dim + drug_emb_dim + pert_time_emb_dim]
        if self.use_cell_id:
            cell_id_embed = self.cell_id_embed(input_cell_id)
            # cell_id_embed = [batch * cell_id_emb_dim]
            cell_id_embed = cell_id_embed.unsqueeze(1)
            # cell_id_embed = [batch * 1 * cell_id_emb_dim]
            cell_id_embed = cell_id_embed.repeat(1, self.num_gene, 1)
            # cell_id_embed = [batch * num_gene * cell_id_emb_dim]
            drug_gene_embed_recon = torch.cat((drug_gene_embed_recon, cell_id_embed), dim=2)
            # drug_gene_embed_recon = [batch * num_gene * gene_emb_dim + drug_emb_dim + pert_time_emb_dim + cell_id_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, cell_id_embed), dim=2)
            # drug_gene_embed = [batch * num_gene * gene_emb_dim + drug_emb_dim + pert_time_emb_dim + cell_id_emb_dim]
        if self.use_pert_dose:
            pert_dose_embed = input_pert_dose.unsqueeze(1).unsqueeze(1)
            # pert_idose_embed = [batch * 1 * pert_idose_emb_dim]
            pert_dose_embed = pert_dose_embed.repeat(1, self.num_gene, 1)
            # pert_idose_embed = [batch * num_gene * pert_idose_emb_dim]
            drug_gene_embed_recon = torch.cat((drug_gene_embed_recon, pert_dose_embed), dim=2)
            # drug_gene_embed_recon = [batch * num_gene * gene_emb_dim + drug_emb_dim + pert_time_emb_dim + cell_id_emb_dim + pert_idose_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_dose_embed), dim=2)
            # drug_gene_embed = [batch * num_gene * gene_emb_dim + drug_emb_dim + pert_time_emb_dim + cell_id_emb_dim + pert_idose_emb_dim]
        drug_gene_embed_recon = self.relu(drug_gene_embed_recon)
        out = self.linear_1(drug_gene_embed_recon.clone().detach().to(torch.float64))
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # # out = [batch * num_gene * hid_dim]
        out = self.linear_2(out)
        # # out = [batch * num_gene * hid_dim]
        return out, drug_gene_embed