import pdb
import torch
import torch.nn as nn
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

from . import base
import graphs
import utils


class BetaVAEAgent(base.BaseAgent):
    def __init__(self, config, loaders, k):
        super().__init__(config, loaders, k)
            
    def _get_model(self):
        return graphs.vae.VAE(self.config.len_z, device=self.device, beta=self.config.beta).to(self.device)

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

    def _ml_logic_per_pair(self, x1):
        loss_dict = dict()
        loss_dict["loss"] = 0
        
        
        X1 = x1
        batch_size = X1.shape[0]       

        vae_results = self.model.forward(X1)
        r1 = vae_results[0]
        mu = vae_results[2]
        log_var = vae_results[3]

        rec_loss = nn.MSELoss()
        rec_loss = rec_loss(X1, r1)

        # Calculate KL Loss
        # M_N = batch_size / len(self.tr_loader.dataset)
        M_N = 1
        kl_loss = self.model.kl_loss_function(mu, log_var, M_N)

        loss = rec_loss + kl_loss
        loss_dict["rec_loss"] = rec_loss
        loss_dict["kl_loss"] = kl_loss


        loss_dict["loss"] = loss
        return loss_dict

    def _ml_logic(self, X, train=True):      
        x1 = X[0]
        x1 = x1.to(self.device)
        loss_dict = self._ml_logic_per_pair(x1)        
        return loss_dict