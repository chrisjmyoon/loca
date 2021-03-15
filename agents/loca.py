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


class LoCAAgent(base.BaseAgent):
    def __init__(self, config, loaders, k):
        super().__init__(config, loaders, k)
   
    def _get_model(self):
        return graphs.lssl_net.LoCANet(len_z=512, len_k=self.config.len_k).to(self.device)

    def _get_enumerator(self, is_train=True):
        if is_train:
            return enumerate(self.paired_tr_loader)
        else:
            return enumerate(self.paired_val_loader)
    
    def _ml_logic_per_pair(self, X1, X2):
        loss_dict = dict()
        loss_dict["loss"] = 0

        batch_size = X1.shape[0]      

        z1, z2, r1, r2 = self.model(X1, X2)

        """ Reconstruction Loss """
        # Standard MSE Reconstruction Loss
        rec_loss = nn.MSELoss()
        lambda_rec_loss = 1.0
        r1_loss = lambda_rec_loss*rec_loss(X1, r1) # first pair
        r2_loss = lambda_rec_loss*rec_loss(X2, r2)

        loss_dict["r1_loss"] = r1_loss
        loss_dict["r2_loss"] = r2_loss
        loss = r1_loss + r2_loss

        taus = self.model.get_taus()
        ntaus = len(taus)
        """ END """

        """ Residual Loss """
        # projections of delta_z along tau should sum to delta_z
        # delta_z is normalized to prevent delta_z from degenerating to trivial solution of 0
        # the loss is the norm of the residual
        delta_z = z2 - z1
        alpha_delta_taus = [] # scalar components, alpha, of delta_z along each tau
        delta_z_norm = delta_z.norm(dim=1)
        normalized_delta_z = delta_z / delta_z_norm.unsqueeze(1) 
        
        proj_sum = None
        for tau in taus:
            # sum projections
            tau_dot = torch.dot(tau, tau)
            delta_z_tau = torch.matmul(normalized_delta_z, tau)
            proj_delta_z_tau = tau*(delta_z_tau / tau_dot.unsqueeze(0)).unsqueeze(1)
            if proj_sum is None:
                proj_sum = proj_delta_z_tau 
            else:
                proj_sum = proj_sum + proj_delta_z_tau
            # save alphas for correlation loss
            alpha_delta_z_tau = (delta_z_tau / tau_dot.sqrt())
            alpha_delta_taus.append(alpha_delta_z_tau)

        residual = normalized_delta_z - proj_sum
        residual_loss = residual.norm(dim=1).mean()

        loss_dict["residual_loss"] = residual_loss
        loss = loss + residual_loss
        """ END """

        """ Orthogonality Loss 
        Calculate pairwise disjoint loss (cosine similarity between tau's)
        """
        is_disjoint_loss = True
        if is_disjoint_loss:
            def calc_disjoint_loss(taus):
                ntaus = len(taus)
                num_pairs = ntaus*(ntaus-1)/2
                s_taus = torch.stack(taus) # stacked taus
                cos_sim = nn.CosineSimilarity()
                # compute sum of disjoint loss per pair
                sim_loss = None
                for i in range(ntaus-1):
                    sim = cos_sim(s_taus[i+1:], s_taus[i].unsqueeze(0)).abs()
                    if sim_loss is None:
                        sim_loss = sim.sum()
                    else:
                        sim_loss += sim.sum()
                # sim_loss /= num_pairs 
                return sim_loss
            disjoint_loss = calc_disjoint_loss(taus)
            loss_dict["disjoint_loss"] = disjoint_loss
            loss = loss + disjoint_loss
        """ END """

        """ Correlation Loss """
        # Calculate correlation loss between scalar components of delta_z along each tau
        is_correlation_loss = True
        if is_correlation_loss:
            if batch_size == 1:
                corr_loss = torch.tensor(0).to(self.device)
            else:
                # Learn k by k correlation matrix
                stacked_alphas = torch.stack(alpha_delta_taus)
                tau_alpha_means = stacked_alphas.mean(dim=1)
                x_xmean = (stacked_alphas.T - tau_alpha_means).T
                N = stacked_alphas.size(1)
                alpha_std = stacked_alphas.std(dim=1)
    
                R = []
                for i in range(ntaus-1):
                    numerator = (x_xmean[i+1:]*x_xmean[i]).sum(dim=1) / (N-1)
                    denominator = alpha_std[i+1:]*alpha_std[i]
                    r = numerator / denominator
                    R.extend(r.abs())                
                corr_loss = sum(R) / len(R)
            loss_dict["corr_loss"] = corr_loss
            loss = loss + corr_loss           
        """ END """

        loss_dict["loss"] = loss
        return loss_dict
    
    def _ml_logic(self, X, train=True):      
        X_1 = X

        x1_1 = X_1[0].to(self.device) # first pair
        x2_1 = X_1[1].to(self.device)
        
        meta_1 = X_1[2]      

        # intrasubject
        loss_dict = self._ml_logic_per_pair(x1_1, x2_1)        
        return loss_dict