"""
Contains helper functions for eval
"""
import pdb
import graphs
import utils
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import datasets
import torch.nn as nn
from itertools import cycle
"""
Contains helpers to get downstream classification scores
- EvalFrozenDx precomputes all z's for computational efficiency for frozen encoder training
- EvalFrozenDxTaus precomputes all projections of z's along tau's
- EvalFinetuneDx trains the entire model including the encoder if train_encoder is set to True
"""
class EvalFrozenDx:
    def __init__(self, agent, trainloader, valloader, single=True, lr=1e-2):
        self.trainloader = trainloader
        self.valloader = valloader
        self.agent = agent
        self.single = single
        
        # init best scores
        best_scores = dict(
            loss = float('inf'),
            acc = 0,
            wacc = 0,
        )
        self.best_acc_scores = best_scores
        self.best_loss_scores = best_scores
        self.best_wacc_scores = best_scores

        # init dataloader using z
        self.tr_loader = self._init_loader(is_train=True)
        self.val_loader = self._init_loader(is_train=False)
        self._set_len_tr()

        # init model
        self.classifier = self._get_model()
        
        # init criterion
        tr_class_counts = self.trainloader.dataset.class_counts
        N_0 = tr_class_counts[0]
        N_1 = tr_class_counts[1]
        if self.agent.config.dataset == "adni":
            N_D = N_1 # only one class
        else:
            N_D = N_1 + tr_class_counts[2] + tr_class_counts[3]
        pos_weight = torch.tensor(N_0 / N_D)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # init optimizer
        # lr = self.agent.config.eval.lr
        wd = self.agent.config.eval.weight_decay
        
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), 
            lr = lr,
            weight_decay=wd)

        # init scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, verbose=True, patience=10,threshold=1e-3, min_lr=1e-8)

        # initialize meters
        self._init_meters()
        self._init_plotters()    

        # set state dir
        exp_id = self.agent.config.exp_id
        fold = str(self.agent.k)
        rand_seed = str(self.agent.config.rand_seed)
        self.state_dir = "artifacts/eval/{rand_seed}/{exp_id}/findtune_dx/{fold}".format(
            rand_seed = rand_seed,
            exp_id = exp_id,
            fold = fold)

    def _get_model(self):
        return graphs.eval.DxClassifier(len_z=self._len_z()).to(self.agent.device)

    def _len_z(self):
        return self.agent.config.len_z

    def _get_z(self, z):
        return z.cpu()

    def _set_len_tr(self):
        self.len_tr_loader = len(self.tr_loader)

    def _init_loader(self, is_train=True):
        if is_train:
            dataloader = self.trainloader
            shuffle = True
        else:
            dataloader = self.valloader
            shuffle = False

        encoder = self.agent.model.encoder.to(self.agent.device)
        N = len(dataloader.dataset)
        len_z = self._len_z()
        agg_z = np.zeros((N, len_z))
        agg_dx = np.zeros(N)
        agg_age = np.zeros(N)
        batch_count = 0

        # Saving z
        with torch.no_grad():
            for i, X in enumerate(dataloader):      
                if self.single:
                    dxes = X[1]["dxes"]
                    ages = X[1]["ages"]
                    data = X[0]
                    data = data.to(self.agent.device)
                    batch_size = data.size(0)
                    z = encoder(data)
                    z = self._get_z(z)                           
                else:
                    dxes = X[2]["dxes"][0]
                    ages = X[2]["ages"][1] - X[2]["ages"][0] # difference in ages
                    data1 = X[0]
                    data2 = X[1]
                    data1 = data1.to(self.agent.device)
                    data2 = data2.to(self.agent.device)
                    batch_size = data1.size(0)
                    z1 = encoder(data1)
                    z2 = encoder(data2)
                    delta_z = z2 - z1
                    z = self._get_z(delta_z)   
                agg_z[batch_count:batch_count+batch_size] = z
                agg_dx[batch_count:batch_count+batch_size] = dxes[0].cpu()
                agg_age[batch_count:batch_count+batch_size] = ages[0].cpu()
                batch_count += batch_size

        # normalize agg_z
        # normalize per subject
        # agg_z = agg_z - agg_z.mean(axis=1).reshape(-1,1)
        # agg_z = agg_z / agg_z.std(axis=1).reshape(-1,1)
        # normalize per feature
        agg_z = (agg_z - agg_z.mean(axis=0)) / agg_z.std(axis=0)
        agg_z = agg_z / agg_z.std(axis=0)


        z_dataset = datasets.z_dataset.ZDataset(agg_z, agg_dx, agg_age)
        z_loader = torch.utils.data.DataLoader(
            z_dataset, 
            batch_size=self.agent.config.eval.batch_size,
            shuffle=shuffle, 
            num_workers=self.agent.config.num_workers, 
            pin_memory=True)
        return z_loader

    def _init_meters(self):      
        self.meters = dict(
            train_loss_meter =utils.meters.AverageMeter(),
            val_loss_meter = utils.meters.AverageMeter(),
            train_accuracy_meter = utils.meters.AverageMeter(),
            val_accuracy_meter = utils.meters.AverageMeter(),
            val_accuracy_0_meter = utils.meters.AverageMeter(),
            val_accuracy_1_meter = utils.meters.AverageMeter()
        )
    def _reset_meters(self):
        for meter_name, meter in self.meters.items():
            meter.reset()

    def _update_meters(self, results, batch_size, is_train=True):
        loss = results["loss"]
        accuracy = results["accuracy"]
        output  = results["output"]
        dxes = results["dxes"]
        if is_train:
            self.meters["train_loss_meter"].update(loss.item(), batch_size) 
            self.loss_plotter.increment_train_epoch()
            self.meters["train_accuracy_meter"].update(accuracy.item(), batch_size)
        else:
            self.meters["val_loss_meter"].update(loss.item(), batch_size)
            self.meters["val_accuracy_meter"].update(accuracy.item(), batch_size)
            num_0 = (dxes==0).sum()
            num_1 = (dxes==1).sum()
            num_0_correct = num_0 - output[dxes==0].sum()
            num_1_correct = output[dxes==1].sum()
            if num_0:
                self.meters["val_accuracy_0_meter"].update(num_0_correct/num_0, num_0)
            if num_1:
                self.meters["val_accuracy_1_meter"].update(num_1_correct/num_1, num_1)

    def _init_plotters(self):
        self.loss_plotter = utils.viz.LossPlotter("vanilla_loss", title="Dx Classification Loss", len_loader=self.len_tr_loader, env="eval", epoch=0)
    
    def _print_progress(self, epoch, i, is_train=True):
        if is_train:
            print(  "Epoch: [{0}][{1}/{2}]\t  Loss: {loss_val:.4f}\t Accuracy: {accuracy:.4f}".format(
                        epoch, i, self.len_tr_loader,
                        loss_val=self.meters["train_loss_meter"].avg,
                        accuracy=self.meters["train_accuracy_meter"].avg))
            
            self.loss_plotter.plot_train(self.meters["train_loss_meter"].avg)
            self.meters["train_loss_meter"].reset()
            self.meters["train_accuracy_meter"].reset()
        else:
            print(  "Val Epoch: [{0}][{1}/{2}]\t  Loss: {loss_val:.4f}\t Accuracy: {accuracy:.4f}\t"
                    "Acc0: {acc0:.4f}\t Acc1: {acc1:.4f}".format(
                                epoch, i, self.len_tr_loader,
                                loss_val=self.meters["val_loss_meter"].avg,
                                accuracy=self.meters["val_accuracy_meter"].avg,
                                acc0=self.meters["val_accuracy_0_meter"].avg,
                                acc1=self.meters["val_accuracy_1_meter"].avg,))

    def _update_best(self, epoch):
        # Update best and save state
        val_loss_meter = self.meters["val_loss_meter"]
        val_accuracy_meter = self.meters["val_accuracy_meter"]
        wacc = (self.meters["val_accuracy_0_meter"].avg + self.meters["val_accuracy_1_meter"].avg)/2

        best_loss = self.best_loss_scores["loss"]
        best_acc = self.best_acc_scores["acc"]
        best_wacc = self.best_wacc_scores["wacc"]

        is_best_loss = val_loss_meter.avg < best_loss
        is_best_acc = val_accuracy_meter.avg > self.best_acc_scores["acc"]
        is_best_wacc = wacc > self.best_wacc_scores["wacc"]

        best_scores = dict(
                loss = val_loss_meter.avg,
                acc = val_accuracy_meter.avg,
                wacc = wacc
            )
        if is_best_loss:
            self.best_loss_scores = best_scores
        if is_best_acc:
            self.best_acc_scores = best_scores
        if is_best_wacc:
            self.best_wacc_scores = best_scores
        
        state = {
            'epoch': epoch,
            'model': self.classifier,
            'optimizer': self.optimizer,                   
            'loss' : val_loss_meter.avg,
            'acc': val_accuracy_meter.avg,
            'wacc': wacc,                   
            'best_loss': best_loss,
            'best_acc': best_acc,
            'best_wacc': best_wacc
        }
        utils.fs.save_state(self.state_dir, state, is_best_loss)
        utils.fs.save_best(self.state_dir, is_best_acc, best_type="acc")
        utils.fs.save_best(self.state_dir, is_best_wacc, best_type="wacc")

        print("Best loss: {}\t Best acc: {}\t Best wacc: {}\tCurr wacc: {}".format(best_loss, best_acc, best_wacc, wacc))
                       
    def run(self):
        for epoch in range(self.agent.config.eval.num_epochs):
            # Train
            for i, X in enumerate(self.tr_loader): 
                batch_size = X[0].size(0)
                results = self._ml_logic(X)
                loss = results["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()              
                           
                # Update meters
                self._update_meters(results, batch_size, is_train=True)

                if i % self.agent.config.print_freq == 0:
                    self._print_progress(epoch, i, is_train=True)
            # Val
            with torch.no_grad():
                for i, X in enumerate(self.val_loader):          
                    batch_size = X[0].size(0)

                    results = self._ml_logic(X)
                    loss = results["loss"]
                    

                    # Update meters
                    self._update_meters(results, batch_size, is_train=False)

                    if i % self.agent.config.print_freq == 0:
                        self._print_progress(epoch, i, is_train=False)

                self.loss_plotter.increment_val_epoch()        
                self.loss_plotter.plot_val(self.meters["val_loss_meter"].avg)
            self.scheduler.step(self.meters["val_loss_meter"].avg)
            self._update_best(epoch)
            self._reset_meters()

    # returns the eval specific loss and predictions
    def _ml_logic(self, X):
        agg_z = X[0]
        agg_age = X[1]
        agg_dx = X[2]
       
        data = agg_z.to(self.agent.device)
        dxes = agg_dx.to(self.agent.device).squeeze()
        dxes[dxes != 0] = 1

        # Handle vae decoding
        if hasattr(self.agent.model, "kl_loss_function"):
            mu, log_var = self.model.encoder(data)
            ### Use sampling
            z = self.agent.model.reparameterize(mu, log_var)
            dx_pred = self.model.classifier(z)
            ### Use mean
            # dx_pred = self.model.classifier(mu)
        else:        
            dx_pred = self.classifier(data.float())   

        loss = self.criterion(dx_pred.squeeze(), dxes.float())

        # get accuracy
        sigmoid_dx_pred = torch.sigmoid(dx_pred)   
        output = (sigmoid_dx_pred>0.5).float().squeeze()
        accuracy = (output == dxes).float().mean()

        results = dict(
            loss=loss,
            dx_pred=dx_pred,
            accuracy=accuracy,
            output=output,
            dxes=dxes
        )
        return results

class EvalFrozenDxTaus(EvalFrozenDx):
    def __init__(self, agent, trainloader, valloader, single=True, lr=1e-2):
        super().__init__(agent, trainloader, valloader, single, lr)

    def _len_z(self):
        return self.agent.config.len_k

    def _get_z(self, z):
        taus = self.agent.model.get_taus()
        alphas = []
        with torch.no_grad():
            for tau in taus:
                alpha_tau = (torch.matmul(z, tau) / tau.norm()).cpu()
                alphas.append(alpha_tau)
        alphas = torch.stack(alphas).T 
        return alphas

    def __init__(self, agent, trainloader, valloader, single=True):
        super().__init__(agent, trainloader, valloader, single)

    def _len_z(self):
        return self.agent.config.len_k

    def _get_z(self, z):
        taus = self.agent.model.get_taus()
        alphas = []
        with torch.no_grad():
            for tau in taus:
                alpha_tau = (torch.matmul(z, tau) / tau.norm()).cpu()
                alphas.append(alpha_tau)
        alphas = torch.stack(alphas).T 
        return alphas


class EvalFinetuneDx(EvalFrozenDx):
    def __init__(self, agent, trainloader, valloader, single=True, lr=1e-4):
        super().__init__(agent, trainloader, valloader, single, lr)
    
    def _init_loader(self, is_train=True):
        if is_train:
            dataloader = self.trainloader
        else:
            dataloader = self.valloader
        return dataloader
    
    def _get_model(self, train_encoder=True):
        classifier = graphs.eval.DxClassifier(len_z=self._len_z())
        encoder = self.agent.model.encoder

        # Toggle freezing weights
        for param in encoder.parameters():
            param.requires_grad = train_encoder

        model = nn.Sequential(encoder, classifier)
        model = model.to(self.agent.device)
        model = model.train()
        return model
    
    # returns the eval specific loss and predictions
    def _ml_logic(self, X):
        agg_X = X[0]
        agg_dx = X[1]["dxes"][0]
        data = agg_X.to(self.agent.device)
        dxes = agg_dx.to(self.agent.device).squeeze()
        dxes[dxes != 0] = 1

        dx_pred = self.classifier(data.float())   

        loss = self.criterion(dx_pred.squeeze(), dxes.float())

        # get accuracy
        sigmoid_dx_pred = torch.sigmoid(dx_pred)   
        output = (sigmoid_dx_pred>0.5).float().squeeze()
        accuracy = (output == dxes).float().mean()

        results = dict(
            loss=loss,
            dx_pred=dx_pred,
            accuracy=accuracy,
            output=output,
            dxes=dxes
        )
        return results