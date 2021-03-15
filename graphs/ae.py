import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard implementation of Encoder in AE
class Encoder(nn.Module):
    def __init__(self, len_z=512):
        super().__init__()
        self.len_z = len_z

        # Encoder conv layers
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=16,kernel_size=3,padding=1), # padding=1 to keep dims same
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )

        # self.ReLU = torch.nn.ReLU()
        # Encoder linear layer
        self.linear = nn.Linear(in_features=1024, out_features=self.len_z)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # out = torch.tanh(out) # tanh caused many problems!!! made z unstable to +-1
        # out = self.ReLU(out)
        return out
        
# Standard implementation of Decoder in AE
class Decoder(nn.Module):
    def __init__(self, len_z=512):
        super().__init__()
        self.len_z = len_z
        # Decoder linear layer
        self.linear = nn.Linear(in_features=self.len_z, out_features=1024)

        # Encoder conv layers
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.recon_layer = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(out.size(0), 16, 4, 4, 4)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.recon_layer(out)
        return out

class AE(nn.Module):
    def __init__(self, len_z=512):
        super().__init__()
        self.len_z = len_z
        self.encoder = Encoder(self.len_z)
        self.decoder = Decoder(self.len_z)
    
    def forward(self, x1):
        z1 = self.encoder(x1)
        r1 = self.decoder(z1)
        return z1, r1