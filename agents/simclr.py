import pdb
import torch
import torch.nn as nn
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import torchio as tio
import torch.nn.functional as F

from . import base
import graphs
import utils

class SimCLRAgent(base.BaseAgent):
    def __init__(self, config, loaders, k):
        super().__init__(config, loaders, k)
        self.nt_xent_criterion = graphs.simclr.NTXentLoss(self.device, self.config.batch_size, temperature=0.5, use_cosine_similarity=True)
              
    def _get_model(self):
        return graphs.simclr.SimCLRNet(self.config.len_z).to(self.device)

    def _set_loaders(self, loaders):
        super()._set_loaders(loaders)
        self.len_tr_loader = len(self.single_tr_loader)
        self.len_val_loader = len(self.single_val_loader)
        
    def _get_enumerator(self, is_train=True):
        tr_loader = self.single_tr_loader
        val_loader = self.single_val_loader
        if is_train:
            return enumerate(tr_loader)
        else:
            return enumerate(val_loader)

    def _ml_logic_per_pair(self, x1, x2):
        loss_dict = dict()
        loss_dict["loss"] = 0
        
        X1 = x1
        X2 = x2
        batch_size = X1.shape[0]       

        zis, his = self.model(X1)
        zjs, hjs = self.model(X2)
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)        

        # get loss
        loss = self.nt_xent_criterion(zis, zjs)
        loss_dict["loss"] = loss
        return loss_dict
    
    def _ml_logic(self, X, train=True):      
        # transformations - shift intensity, rotate
        x = X[0].squeeze()
        batch_size = x.size(0)
        tr = tio.RandomAffine(scales=(0.95, 1.05), translation=(-5,5), degrees=(-5,5), isotropic=True, image_interpolation='nearest')
        x1 = tr(x)
        x2 = tr(x)

        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        
        loss_dict = self._ml_logic_per_pair(x1, x2)        
        return loss_dict