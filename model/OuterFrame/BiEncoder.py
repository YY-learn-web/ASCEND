import torch
from torch import nn
from torch.nn import functional as F



class BiEncoder(nn.Module):
    def __init__(self, share_encoder, decoder, private_encoder = None,
                 norm_flag: bool = True, **kwargs):
        super(BiEncoder, self).__init__()
        self.alpha = kwargs['alpha']
        self.norm_flag = norm_flag
        self.share_encoder = share_encoder
        self.decoder = decoder
        self.private_encoder = private_encoder
        # self.TF_output_dim = kwargs['TF_output_dim']

    def p_encode(self, input_drug=None, input_gene=None, input_pert_time=None, input_cell_id=None, input_pert_dose=None):# target_batch=None
        latent_code, _ = self.private_encoder(input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose)
        latent_code = latent_code + torch.randn_like(latent_code, requires_grad=False) * 0.1

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=2)
        else:
            return latent_code
        
    def s_encode(self, input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose):
        latent_code, input_embed = self.share_encoder(input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose)
        latent_code = latent_code + torch.randn_like(latent_code, requires_grad=False) * 0.1

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=2), input_embed
        else:
            return latent_code, input_embed
        
    def encode(self, input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose):
        if self.private_encoder is None:
            p_latent_code = self.p_encode(input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose)
        else:
            p_latent_code = self.p_encode(input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose)
        s_latent_code, input_embed = self.s_encode(input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose)

        return torch.cat((p_latent_code, s_latent_code), dim=2), input_embed
    
    def decode(self, z):
        outputs = self.decoder(z)
        return outputs
    
    def forward(self, input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose):
        if self.private_encoder is None:
            z, input_embed = self.encode(input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose)
        else:
            z, input_embed = self.encode(input_drug, input_gene, input_pert_time, input_cell_id, input_pert_dose)
        return [input_embed, self.decode(z), z]
    
    def loss_function(self, *args) -> dict:
        input = args[0]
        recons = args[1]
        z = args[2]

        p_z = z[:, :, :z.shape[2] // 2]
        s_z = z[:, :, z.shape[2] // 2:]

        recons_loss = F.mse_loss(input, recons)

        s_l2_norm = torch.norm(s_z, p=2, dim=2, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=2, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        s_l2 = s_l2.reshape(-1, s_l2.shape[2])
        p_l2 = p_l2.reshape(-1, p_l2.shape[2])

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
        loss = recons_loss + self.alpha * ortho_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss}