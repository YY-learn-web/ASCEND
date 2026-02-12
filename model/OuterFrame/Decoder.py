import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim_list, latent_dim, drop, device):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.Linear1 = nn.Linear(input_dim, hid_dim_list[0], dtype=torch.float64, device=device)
        self.Linear2 = nn.Linear(hid_dim_list[0], hid_dim_list[1], dtype=torch.float64, device=device)
        self.Linear3 = nn.Linear(hid_dim_list[1], hid_dim_list[2], dtype=torch.float64, device=device)
        self.Linear4 = nn.Linear(hid_dim_list[2], hid_dim_list[3], dtype=torch.float64, device=device)
        self.Linear5 = nn.Linear(hid_dim_list[3], hid_dim_list[4], dtype=torch.float64, device=device)
        self.Linear6 = nn.Linear(hid_dim_list[4], latent_dim, dtype=torch.float64, device=device)

    def forward(self, input):
        output = self.Linear1(input)
        output = self.relu(output)
        output = self.drop(output)

        output = self.Linear2(output)
        output = self.relu(output)
        output = self.drop(output)

        output = self.Linear3(output)
        output = self.relu(output)
        output = self.drop(output)

        output = self.Linear4(output)
        output = self.relu(output)
        output = self.drop(output)

        output = self.Linear5(output)
        output = self.relu(output)
        output = self.drop(output)
        
        output = self.Linear6(output)

        return output
    