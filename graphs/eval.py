import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class DxClassifier(nn.Module):
    def __init__(self, len_z=512):
        super().__init__()
        self.len_z = len_z
        self.classifier = nn.Linear(in_features=self.len_z, out_features=1, bias=True)
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=self.len_z, out_features=int(self.len_z/2), bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=int(self.len_z/2), out_features=1, bias=True)
        # )
    
    def forward(self, z):
        out = self.classifier(z)
        return out


