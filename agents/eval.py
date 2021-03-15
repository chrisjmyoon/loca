"""
Runs evals on trained agent
"""

import pdb
import os
from pathlib import Path
import pandas as pd
import datasets
import torchvision.transforms as transforms
import torch
from torch.nn import functional as F
import utils
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import graphs
from . import eval_finetune
from . import eval_plot

class EvalAgent:
    def __init__(self, agent, k):
        self.agent = agent   
        self.k = k
        # Load checkpoint from appropriate fold
        artifact_dir = os.path.expanduser("~/loca/artifacts/")
        checkpoint_dir = os.path.join(artifact_dir, str(self.agent.config.rand_seed), self.agent.config.exp_id, str(self.k))
        self.checkpoint_fp = os.path.join(checkpoint_dir, "best_checkpoint.pth.tar")

    def _load_fold_cp(self): 
        self.agent.load_checkpoint(self.checkpoint_fp)

    def get_loaders(self, single=True):
        if single:
            trainloader = self.agent.single_tr_loader
            valloader = self.agent.single_val_loader
        else:
            trainloader = self.agent.paired_tr_loader
            valloader = self.agent.paired_tr_loader
        return trainloader, valloader

    def class_separability(self):
        self._load_fold_cp() 
        trainloader, valloader = self.get_loaders(single=True)
        if hasattr(self.agent.model, "len_k"):
            eval_plotter = eval_plot.EvalPlotter(self.agent, trainloader, valloader)
        else:
            eval_plotter = eval_plot.BaselineEvalPlotter(self.agent, trainloader, valloader)
        
        eval_plotter.plot_top_dx(is_train=True)
        eval_plotter.plot_top_dx(is_train=False)

    def eval_dcorr(self):
        self._load_fold_cp() 
        trainloader, valloader = self.get_loaders(single=True)
        if hasattr(self.agent.model, "len_k"):
            eval_plotter = eval_plot.EvalPlotter(self.agent, trainloader, valloader)
        else:
            eval_plotter = eval_plot.BaselineEvalPlotter(self.agent, trainloader, valloader)
        eval_plotter.get_dcorr(is_train=True)
        eval_plotter.get_dcorr(is_train=False)
    
    def plot_tsne_dims(self):
        self._load_fold_cp() 
        trainloader, valloader = self.get_loaders(get_score=True)
        if hasattr(self.agent.model, "len_k"):
            eval_plotter = eval_plot.EvalPlotter(self.agent, trainloader, valloader)
        else:
            eval_plotter = eval_plot.BaselineEvalPlotter(self.agent, trainloader, valloader)
        eval_plotter.plot_tsne_dims(is_train=True)
        eval_plotter.plot_tsne_dims(is_train=False)
    
    def eval_pred_dx(self, mode="frozen"):
        single = True
        self._load_fold_cp() 
        trainloader, valloader = self.get_loaders(single=single)

        if mode == "frozen":
            eval_pred_dx = eval_finetune.EvalFrozenDx(self.agent, trainloader, valloader, single=single)
        elif mode == "finetune":
            eval_pred_dx = eval_finetune.EvalFinetuneDx(self.agent, trainloader, valloader, single=single)
        elif mode == "tau":
            eval_pred_dx = eval_finetune.EvalFrozenDxTaus(self.agent, trainloader, valloader, single=single)
        else:
            raise NotImplementedError
        eval_pred_dx.run()

        ### Save scores
        fig_dir = os.path.expanduser("~/loca/results/")
        fig_dir = os.path.join(fig_dir, str(self.agent.config.rand_seed), self.agent.config.exp_id)
        utils.fs.create_dir(fig_dir)
        self.fig_dir = fig_dir
        scores_dir = os.path.join(self.fig_dir, "scores", "eval_pred_dx", mode)
        utils.fs.create_dir(scores_dir)
        score_fp = os.path.join(scores_dir, "{}_{}_eval_pred_dx".format(self.agent.config.dataset, self.agent.k))
        best_loss_wacc = eval_pred_dx.best_loss_scores["wacc"].item()
        best_wacc = eval_pred_dx.best_wacc_scores["wacc"].item()
        score = np.array([best_loss_wacc, best_wacc])
        np.savetxt(score_fp, score, delimiter=",")

        if self.agent.k == 4:
            best_loss_waccs = []
            best_waccs = []
            for i in range(5):
                fold_score_fp = os.path.join(scores_dir, "{}_{}_eval_pred_dx".format(self.agent.config.dataset, i))
                scores = np.loadtxt(fold_score_fp,delimiter=",")
                best_loss_waccs.append(scores[0])
                best_waccs.append(scores[1])
            best_loss_waccs = np.array(best_loss_waccs)
            best_waccs = np.array(best_waccs)

            mean_loss_wacc = best_loss_waccs.mean()
            std_loss_wacc = best_loss_waccs.std()
            mean_wacc = best_waccs.mean()
            std_wacc = best_waccs.std()
            best_loss_waccs_scores = np.append(best_loss_waccs, [mean_loss_wacc, std_loss_wacc])
            best_waccs_scores = np.append(best_waccs, [mean_wacc, std_wacc])
            stacked_scores = np.stack((best_loss_waccs_scores, best_waccs_scores))
            summary_score_fp = os.path.join(scores_dir, "{}_{}_eval_pred_dx".format(self.agent.config.dataset, "summary"))
            np.savetxt(summary_score_fp, stacked_scores, delimiter=",")
    
