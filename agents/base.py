import pdb
import shutil
import os
import torch
import utils
from utils.viz import *
class BaseAgent:
    def __init__(self, config, loaders, k):
        self.config = config
        self.k = k

        self.artifact_dir = os.path.join(self.config.save_path, 
            str(self.config.rand_seed),
            self.config.exp_id,
            str(self.k))
        

        # sets device
        if self.config.gpu == "cpu" or not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        self._set_loaders(loaders)

        # initialize model
        self.model = self._get_model()
        self.meters = dict()
        self.plotters = dict()
        # specify loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )        
        if self.config.scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, verbose=True, patience=10,threshold=1e-7)
        
        # initialize state
        self.epoch = 0
        self.best_loss = float('Inf')        

        # initialize visdom loggers
        if self.config.visdom:
            SingleVisdom.reset_window()

        # load from exising state
        if self.config.resume:
            self.load_checkpoint(file_path=self.config.resume)
        if self.config.retrain:
            self.load_model(file_path=self.config.retrain)
    
    def _set_loaders(self, loaders):
        self.paired_tr_loader = loaders["paired_tr_loader"]
        self.paired_val_loader = loaders["paired_val_loader"]
        self.single_tr_loader = loaders["single_tr_loader"]
        self.single_val_loader = loaders["single_val_loader"]
        self.len_tr_loader = len(self.paired_tr_loader)
        self.len_val_loader = len(self.paired_val_loader)
        
    
    def load_checkpoint(self, file_path=None):
        if os.path.isfile(file_path):
            state = torch.load(file_path)
            print("Loaded {}".format(file_path))
        else:
            print("{} does not exist.".format(file_path))
            raise FileNotFoundError
        self.model = state['model']
        self.optimizer = state['optimizer']
        self.epoch = state['epoch']
        # Robustly loading state["meters"] even if state doesn't contain newer keys
        for meter_name, meter in state["meters"].items():
            self.meters[meter_name] = meter
       
        self.best_loss = state['best_loss']
        visdom_state = state["visdom_state"]
        self._load_visdom_state(visdom_state)
        print("Loaded checkpoint to ADNI Agent")
    
    def load_model(self, file_path=None):
        state = super().load_checkpoint(file_path)
        self.model = state['model']

    # Saves model state to checkpoint_path
    def save_checkpoint(self, epoch, is_best=False):
        visdom_state = self._get_visdom_state()
        state = {
            'epoch': epoch,
            'model': self.model,
            'optimizer': self.optimizer,
            'meters': self.meters,
            'best_loss': self.best_loss,
            'visdom_state': visdom_state,
            'notes': self.config.notes
        }

        utils.fs.create_dir(self.artifact_dir)

        checkpoint_path = os.path.join(self.artifact_dir, "checkpoint.pth.tar")
        best_path = os.path.join(os.path.split(checkpoint_path)[0], "best_checkpoint.pth.tar")
        torch.save(state, checkpoint_path)

        if is_best:
            shutil.copyfile(checkpoint_path, best_path)
        
        epoch = state["epoch"]
        epoch_path = os.path.join(os.path.split(checkpoint_path)[0], "{}_checkpoint.pth.tar".format(epoch))
        if epoch % 50 == 0:
            shutil.copyfile(checkpoint_path, epoch_path)
        print("Saved checkpoint")

    def train(self):
        raise NotImplementedError

    def _train_iter(self):
        raise NotImplementedError
    
    def _train_epoch(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
    
    def _print(self, i, max_length, meters, is_train=True):        
        print(  "Epoch: [{0}][{1}/{2}]\t  Loss: {loss_val:.4f}\t".format(
                            self.epoch, i, max_length,
                            loss_val= meters["loss"].avg))

    def _plot(self, meters, is_train=True):
        for loss_name, loss_meter in meters.items():
            if loss_name not in self.plotters:
                self.plotters[loss_name] = utils.viz.LossPlotter(loss_name, title="{} Loss".format(loss_name), len_loader=self.len_tr_loader, epoch=self.epoch)
            
            if is_train:
                self.plotters[loss_name].plot_train(loss_meter.avg)
            else:
                self.plotters[loss_name].plot_val(loss_meter.avg)

    def _get_batch_size(self, X):
        return X[0].size(0)           

    def _update_meters(self, X, ml_results, meters):
        batch_size = self._get_batch_size(X)
        for loss_name, loss_val in ml_results.items():
            if loss_name not in meters:
                meters[loss_name] = utils.meters.AverageMeter()
            meters[loss_name].update(loss_val.item(), batch_size)   

    def _get_visdom_state(self):
        visdom_state = {plotter_name: plotter.export_state() for plotter_name, plotter in self.plotters.items()}
        return visdom_state

    def _load_visdom_state(self, visdom_state):
        for plotter_name, plotter_state in visdom_state.items():
            if plotter_name not in self.plotters:
                self.plotters[plotter_name] = utils.viz.LossPlotter(plotter_name, title="{} Loss".format(plotter_name), len_loader=self.len_tr_loader, epoch=self.epoch)
            self.plotters[plotter_name].load_from_state(plotter_state)

    def train(self):
        for epoch in range(self.epoch, self.config.epochs):
            self._train_epoch()
            loss = self.validate()
            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss
            if self.config.scheduler:
                self.scheduler.step(loss)
            self.save_checkpoint(epoch + 1, is_best=is_best)
            self.epoch += 1

    def _train_epoch(self):      
        self.model.train()

        for meter in self.meters.values():
            meter.reset()
            
        self.optimizer.zero_grad()
        # Note - use cycle() if we want to iterate through larger dataset
        for i, X in self._get_enumerator(is_train=True):
            # Update plotter
            for plotter_name, plotter in self.plotters.items():
                plotter.increment_train_epoch()

            if self._get_batch_size(X) < self.config.batch_size:
                continue

            if self.config.test_mode and i == self.config.test_mode:
                print("Stopping training early because test_mode is True")
                break    

            ml_results = self._ml_logic(X, train=True)
            loss = ml_results["loss"]
            loss.backward()
            if (i *self.config.batch_size) % self.config.desired_bs == 0:
                self.optimizer.step() 
                self.optimizer.zero_grad()

            # Update losses
            self._update_meters(X, ml_results, self.meters)

            # print every config.print_freq
            if i % self.config.print_freq == 0:
                self._print(i, self.len_tr_loader, self.meters, is_train=True)

                # Plot to visdom
                if self.config.visdom:
                    self._plot(self.meters, is_train=True)

                # Reset meters to prevent skewed losses
                for meter in self.meters.values():
                    meter.save_reset() 
                # loss_meter.save_reset()
        print("Finished training epoch: {}".format(self.epoch))  

    # Validates using paired images
    def validate(self):
        print("starting to validate")

        val_meters = dict()
        
        self.model.eval()
        with torch.no_grad():
            for i, X in self._get_enumerator(is_train=False):
                if self._get_batch_size(X) < self.config.batch_size:
                    continue
                if self.config.test_mode and i == self.config.test_mode:
                    print("Stopping validation early because test_mode is True")
                    break

                ml_results = self._ml_logic(X, train=False)              
                
                # Update losses
                self._update_meters(X, ml_results, val_meters)

                # print every config.print_freq
                if i % self.config.print_freq == 0:
                    self._print(i, self.len_val_loader, val_meters, is_train=False)                    

        # Plot to visdom
        if self.config.visdom:
            for plotter in self.plotters.values():
                plotter.increment_val_epoch()
            self._plot(val_meters, is_train=False)
        return val_meters["loss"].avg