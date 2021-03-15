import pdb
import os
import utils
import torchvision.transforms as transforms
import agents
import pandas as pd
import datasets
import torch
import numpy as np 
"""
Basic interface for experiments
"""
class BaseExp:
    def __init__(self, config, k):
        self.config = config
        self.k = k
        """ creates cv splits folder if not exists """
        # create dir
        cv_split_dir_path = os.path.join(self.config.split_dir, "cv", str(self.config.rand_seed), self.config.dataset)
        is_create_splits = not os.path.isdir(cv_split_dir_path)
        # is_create_splits = True
        if is_create_splits:
            self.create_splits()
        else:
            print("{} exists".format(cv_split_dir_path))


        artifact_dir = os.path.join(self.config.save_path, 
            str(self.config.rand_seed),
            self.config.exp_id,
            str(self.k))

        # ensures artifacts folder is created
        utils.fs.create_dir(artifact_dir)
        self.loaders = self._get_loaders()
        self.agent = self.get_agent()

    """
    Returns default transformations
    """
    def get_transforms(self):
        tr = transforms.Compose([
            transforms.ToTensor()
        ])
        return tr
    """
    Get single (for eval and baselines) and paired loaders (for longitudinal methods)
    """
    def _get_loaders(self):
        paired_tr_loader, paired_val_loader = self.get_paired_loaders()
        single_tr_loader, single_val_loader = self.get_single_loaders(paired_tr_loader, paired_val_loader)

        loaders = dict(
            paired_tr_loader = paired_tr_loader,
            paired_val_loader = paired_val_loader,
            single_tr_loader = single_tr_loader,
            single_val_loader = single_val_loader
        )
        return loaders

    def create_splits(self):
        """ Helper to group list of fps by subject to ensure a subject is unique to each split"""
        def group_by_subject(file_indices):     
            # Construct list of fp per subject name
            subject_fps = dict()  
            for fps in file_indices:
                # sub_idx = re.search(r'(?<=S_)\d*', fps[0]).group(0) # match by subject idx
                sub_idx = fps[0].split('-')[0]
                if sub_idx not in subject_fps:
                    subject_fps[sub_idx] = fps
                else:
                    subject_fps[sub_idx] = np.vstack([subject_fps[sub_idx], fps])
            
            # randomly shuffle subjects
            random.seed(self.config.rand_seed)
            subject_fps_items = list(subject_fps.items())
            random.shuffle(subject_fps_items)
            subject_fps = dict(subject_fps_items)
            return subject_fps   
        """ Helper to split list of fps to two groups based on ratio """ 
        def split(subject_fps, ratio=0.2):
            num_subjects = len(subject_fps)
            num_sub_per_split = int(num_subjects*ratio)
            agg_split_fps = []
            split_fps = None

            count = 0
            for i, (subject, fps) in enumerate(subject_fps.items()):
                # add first i subjects to train_subjects
                if count >= num_sub_per_split:
                    agg_split_fps.append(split_fps)
                    split_fps = None
                    count = 0

                if count < num_sub_per_split:
                    if split_fps is None:
                        split_fps = fps
                    else:
                        split_fps = np.vstack([split_fps, fps])
                    count = count + 1
            if split_fps is not None:
                agg_split_fps.append(split_fps)
            return agg_split_fps
        
        """ Create dir if not exists """
        cv_split_dir_path = os.path.join(self.config.split_dir, "cv", str(self.config.rand_seed))
        utils.fs.create_dir(cv_split_dir_path)

        # set seed
        import random
        random.seed(self.config.rand_seed)

        # load img_class1.npy and img_class2.npy containing pairs
        dataset = self.config.dataset
        if dataset == "adni":
            save_dir = os.path.join(cv_split_dir_path, "adni")
            label1_fp = os.path.join(self.config.split_dir, "img_Normal.npy")
            label2_fp = os.path.join(self.config.split_dir, "img_AD.npy")
        elif dataset == "aud":
            save_dir = os.path.join(cv_split_dir_path, "aud")
            label1_fp = os.path.join(self.config.split_dir, "aud_img_C.npy")
            label2_fp = os.path.join(self.config.split_dir, "aud_img_D.npy")
        fps_label1 = np.load(label1_fp)
        fps_label2 = np.load(label2_fp)

        # create save dir if not exists
        utils.fs.create_dir(save_dir)
        """ split and save k folds """
        k = 5
        ratio = 1.0 / k

        # Combine to single list of pairs
        fps = np.concatenate([fps_label1, fps_label2])
        dataset_column = np.array([dataset]*len(fps)).reshape(-1, 1)
        fps = np.hstack([fps, dataset_column])

        # Group by subject 
        subject_fps = group_by_subject(fps)

        # ### Dataset stats
        # label1_subject_fps = group_by_subject(fps_label1)
        # label2_subject_fps = group_by_subject(fps_label2)
        # # num scans by fps
        # num_scans_label1 = len(set(np.concatenate((fps_label1[:,0], fps_label1[:,1])).tolist()))
        # num_scans_label2 = len(set(np.concatenate((fps_label2[:,0], fps_label2[:,1])).tolist()))
        # # num subjects by fps
        # num_subs_label1 = len(label1_subject_fps)
        # num_subs_label2 = len(label2_subject_fps)
        
        # # max scans per subject
        # max_scans_by_subject = 0
        # min_scans_by_subject = float('inf')
        # for sub_name, pairs in subject_fps.items():
        #     scans_by_subject = []
        #     for pair in pairs:
        #         scans_by_subject.append(pair[0])
        #         scans_by_subject.append(pair[1])
        #     num_scans_by_subject = len(set(scans_by_subject))
            
        #     if num_scans_by_subject > max_scans_by_subject:
        #         max_scans_by_subject = num_scans_by_subject
        #     if num_scans_by_subject < min_scans_by_subject:
        #         min_scans_by_subject = num_scans_by_subject
        # ###

        # Split into k folds
        agg_split_fps = split(subject_fps, ratio=ratio)
        # Save folds
        for i in range(k):
            val_fold = agg_split_fps[i]
            train_folds = []
            for j in range(k):
                if i != j:
                    train_folds.append(agg_split_fps[j])
            train_fold = np.concatenate(train_folds)
            # Save fold
            train_fold_fp = "{}_{}_train".format(dataset, i)
            val_fold_fp = "{}_{}_val".format(dataset, i)
            train_fold_fp = os.path.join(save_dir, train_fold_fp)
            val_fold_fp = os.path.join(save_dir, val_fold_fp)
            np.save(train_fold_fp, train_fold)
            np.save(val_fold_fp, val_fold)
            print("Saved fold: {}".format(i))
        pdb.set_trace()

        
    """
    Returns train and val loader using fps
    """
    def _get_loaders_from_fps(self, train_fps, val_fps):
        tr = self.get_transforms()
        # get trainset
        trainset = datasets.paired_dataset.PairedDataset(self.config, train_fps, transform=tr)
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory)
        
        # get valset
        valset = datasets.paired_dataset.PairedDataset(self.config, val_fps, transform=tr)
        val_loader = torch.utils.data.DataLoader(
            valset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory)

        return train_loader, val_loader

    """ get paired loader """ 
    def get_paired_loaders(self):
        # get splits based on k and dataset
        cv_dir = os.path.join(self.config.split_dir, "cv", str(self.config.rand_seed), self.config.dataset)
        train_fold = "{}_{}_train.npy".format(self.config.dataset, self.k)
        val_fold = "{}_{}_val.npy".format(self.config.dataset, self.k)
        train_fold_fp = os.path.join(cv_dir, train_fold)
        val_fold_fp = os.path.join(cv_dir, val_fold)

        train_fold_fps = np.load(train_fold_fp)
        val_fold_fps = np.load(val_fold_fp)
        tr_loader, val_loader = self._get_loaders_from_fps(train_fold_fps, val_fold_fps)
        return tr_loader, val_loader

    """ Collapse paired dataset to single dataset """
    def get_single_loaders(self, tr_loader, val_loader):
        tr_fps = tr_loader.dataset.fps
        val_fps = val_loader.dataset.fps

        tr_fps_list = tr_fps[:,:2].reshape(-1).tolist()
        val_fps_list = val_fps[:,:2].reshape(-1).tolist()

        tr_fp_set = set(tr_fps_list)
        val_fp_set = set(val_fps_list)

        # unique fps in training set
        tr_fps = list(tr_fp_set)
        val_fps = list(val_fp_set)        

        tr = self.get_transforms()

        # get trainset
        trainset = datasets.single_dataset.SingleDataset(self.config, tr_fps, transform=tr)
        train_loader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory)
        
        # get valset
        valset = datasets.single_dataset.SingleDataset(self.config, val_fps, transform=tr)
        val_loader = torch.utils.data.DataLoader(
            valset, 
            batch_size=self.config.batch_size,
            shuffle=False, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory)

        return train_loader, val_loader

    """
    Run agent with different endpoints
    """
    def run(self):
        if self.config.mode == 'train':
            self.agent.train()     
        elif self.config.mode == 'visualize':
            self.agent.visualize()
        elif self.config.mode == "eval":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k)
            eval_agent.eval()       
        elif self.config.mode == "box":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k)
            eval_agent.eval_box()      
        elif self.config.mode == "plot":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k) 
            eval_agent.plot()   
        elif self.config.mode == "eval_class_separability":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k) 
            eval_agent.class_separability() 
        elif self.config.mode == "eval_dcorr":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k)
            eval_agent.eval_dcorr()
        elif self.config.mode == "eval_pred_age":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k) 
        elif self.config.mode == "eval_frozen_dx":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k) 
            eval_agent.eval_pred_dx(mode="frozen")
        elif self.config.mode == "eval_finetune_dx":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k) 
            eval_agent.eval_pred_dx(mode="finetune")
        elif self.config.mode == "eval_frozen_tau":
            eval_agent = agents.eval.EvalAgent(self.agent, self.k) 
            eval_agent.eval_pred_dx(mode="tau")
        else:
            raise Exception("conf['mode'] invalid!")
        print("Finished running experiment")

