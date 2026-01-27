import torch
from torch import nn


class NoiseLayer(nn.Module):
    def __init__(self, noise_std=0.1):
        super(NoiseLayer, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, hid_dim3, hid_dim4, output_dim, drop, device, noise_std=0.1, ispredict=False):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.ispredict = ispredict
        self.Linear1 = nn.Linear(input_dim, hid_dim1, dtype=torch.float64, device=device)
        self.Linear2 = nn.Linear(hid_dim1, hid_dim2, dtype=torch.float64, device=device)
        self.Linear3 = nn.Linear(hid_dim2, hid_dim3, dtype=torch.float64, device=device)
        self.Linear4 = nn.Linear(hid_dim3, hid_dim4, dtype=torch.float64, device=device)
        self.Linear5 = nn.Linear(hid_dim4, output_dim, dtype=torch.float64, device=device)
        self.Noise_layer1 = NoiseLayer(noise_std)
        self.Noise_layer2 = NoiseLayer(noise_std)

        self.drop = nn.Dropout(drop)


    def forward(self, input):
        output = self.Linear1(input)
        output = self.relu(output)
        if self.ispredict:
            output = self.Noise_layer1(output)
        output = self.drop(output)

        output = self.Linear2(output)
        output = self.relu(output)
        output = self.drop(output)

        output = self.Linear3(output)
        output = self.relu(output)
        output = self.drop(output)
        
        output = self.Linear4(output)
        output = self.relu(output)
        if self.ispredict:
            output = self.Noise_layer2(output)
        output = self.drop(output)

        output = self.Linear5(output)
        if self.ispredict:
            output = output.squeeze()

        return output