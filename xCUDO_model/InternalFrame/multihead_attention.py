import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.device = device
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim, dtype=torch.float64, device=device)
        self.fc_k = nn.Linear(hid_dim, hid_dim, dtype=torch.float64, device=device)
        self.fc_v = nn.Linear(hid_dim, hid_dim, dtype=torch.float64, device=device)
        self.fc_o = nn.Linear(hid_dim, hid_dim, dtype=torch.float64, device=device)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(self.device).double()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        Q = self.fc_q(query.clone().detach().to(torch.float64).to(self.device))
        K = self.fc_k(key.clone().detach().to(torch.float64).to(self.device))
        V = self.fc_v(value.clone().detach().to(torch.float64).to(self.device))
        # K = self.fc_k(torch.tensor(key,dtype=torch.float64,device=self.device))
        # V = self.fc_v(torch.tensor(value,dtype=torch.float64,device=self.device))
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]
        Q = Q.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).to(self.device)
        K = K.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).to(self.device)
        V = V.reshape(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).to(self.device)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)).to(self.device) / self.scale
        # energy = [batch size, n heads, seq len, seq len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]
        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, seq len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, seq len, n heads, head dim]
        x = x.reshape(batch_size, -1, self.hid_dim)
        # x = [batch size, seq len, hid dim]
        x = self.fc_o(x)
        # x = [batch size, seq len, hid dim]
        return x #, attention