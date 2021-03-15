import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from graphs.ae import Encoder, Decoder

"""
Uses two paired volumes to jointly train an autoencoder and vector along (x1 - x2)

Input:
    x1: volume 1 at t1
    x2: volume 2 at t2
"""
class LSSLNet(nn.Module):
    def __init__(self, len_z=512):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.tau = Tau(len_z)
    
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        r1 = self.decoder(z1)
        r2 = self.decoder(z2)
        return z1, z2, r1, r2

# Separate Tau to optimize separately
class Tau(nn.Module):
    def __init__(self, len_z):
        super().__init__()
        tau = torch.randn((1,len_z), dtype=torch.float) # initialize tau
        tau = tau / (tau.norm() + 1e-10) # normalize tau as unit vector
        tau.requires_grad = True
        self.vec = torch.nn.Parameter(tau)

class LoCANet(nn.Module):
    def __init__(self, len_z=512, len_k=5):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.len_z = len_z
        self.len_k = len_k
        self._init_tau()
        
    def _init_tau(self):
        self.taus = nn.ModuleList()
        for i in range(self.len_k):
            tau = nn.Linear(in_features=self.len_z, out_features=1, bias=False)
            self.taus.append(tau)

    def get_taus(self):
        return [tau.weight.squeeze() for tau in self.taus]

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        r1 = self.decoder(z1)
        r2 = self.decoder(z2)
        return z1, z2, r1, r2
