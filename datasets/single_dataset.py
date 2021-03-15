import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pdb
import os
import random
from pathlib import Path
import pandas as pd

"""
Loads the entire datasaet to memory as a singleton
"""
class SingleDataDict:
    __instance = None
    @staticmethod
    def getInstance():
        if SingleDataDict.__instance is None:
            SingleDataDict()
        return SingleDataDict.__instance
    
    def __init__(self):
        if SingleDataDict.__instance is None:
            SingleDataDict.__instance = dict()
        else:
            raise Exception("Singleton constructor called twice!")

"""
Returns a single, unpaired, scan
"""
class SingleDataset(Dataset):
    def __init__(self, config, fps, transform=None):
        self.config = config
        self.transform = transform
        self.fps = fps
        self.class_counts = dict()     
        self.fp_to_data = SingleDataDict.getInstance()
        self._get_labels()
        self._load_data_to_memory()

    def __len__(self):
        return len(self.fps)        
    
    def _load_data_to_memory(self):
        # load to memory        
        for fp in self.fps:
            if fp not in self.fp_to_data:
                full_fp = os.path.join(self.config.data_path, fp)                
                data = nib.load(full_fp).get_fdata().astype('float32')                
                if self.transform:
                    data = self.transform(data).unsqueeze(0)
                self.fp_to_data[fp] = data
            # Get num classes
            dx = self.fp_to_dx[fp]
            if dx not in self.class_counts:
                self.class_counts[dx] = 1
            else:
                self.class_counts[dx] += 1

    """
    Get fp_to_dx and fp_to_age mappings
    """
    def _get_labels(self):
        data_dir = str(Path(self.config.data_path).parent)
        fp_dx = pd.read_csv(os.path.join(data_dir, "fp_dx.csv"))
        if self.config.dataset == "adni":
            dxes_to_int = {"CN": 0, "AD": 1}
        elif self.config.dataset == "aud":
            label1 = self.config.datasets["aud"]["label1"]
            label2 = self.config.datasets["aud"]["label2"]
            dxes_to_int = {label1: 0, label2: 1}
            if label1 == "D" or label2 == "D":
                dxes_to_int = {"C": 0, "E": 1, "H": 2, "HE": 3}

        data_dir = data_dir
        fp_dx = fp_dx
        dxes_to_int = dxes_to_int  
        fp_age = pd.read_csv(os.path.join(data_dir, "fp_age.csv"))

        fp_to_dx = dict()
        fp_to_age = dict()
        for index, rows in fp_dx.iterrows():
            row_fp = rows["fp"]
            row_dx = rows["dx"]
            # class we're interested in and not in dict already
            if row_dx in dxes_to_int and row_fp not in fp_to_dx: 
                fp_to_dx[row_fp] = dxes_to_int[row_dx]
        for index, rows in fp_age.iterrows():
            row_fp = rows["fp"]
            row_age = rows["age"]
            # class we're interested in and not in dict already
            if row_fp in fp_to_dx and row_fp not in fp_to_age: 
                fp_to_age[row_fp] = row_age
        self.fp_to_dx = fp_to_dx
        self.fp_to_age = fp_to_age        

    def __getitem__(self, idx):
        fp = self.fps[idx]
        data = self.fp_to_data[fp]

        ages = [self.fp_to_age[fp]]
        dxes = [self.fp_to_dx[fp]]
        meta = dict(
            fps = [fp],
            ages = ages,
            dxes = dxes
        )
        return data, meta