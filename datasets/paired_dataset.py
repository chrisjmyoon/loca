import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pdb
import os
import random
from pathlib import Path
import pandas as pd
from . import single_dataset

"""
Returns paired data organized in fps
fps: [fp1, fp2, dataset]
"""
class PairedDataset(single_dataset.SingleDataset):
    def __init__(self, config, fps, transform=None):
        super().__init__(config, fps, transform)      

    def _load_data_to_memory(self):
        # load to memory
        for fp1, fp2, dataset in self.fps:
            if fp1 not in self.fp_to_data:
                full_fp1 = os.path.join(self.config.data_path, fp1)                
                data1 = nib.load(full_fp1).get_fdata().astype('float32')                
                if self.transform:
                    data1 = self.transform(data1).unsqueeze(0)
                self.fp_to_data[fp1] = data1
                    
            if fp2 not in self.fp_to_data:
                full_fp2 = os.path.join(self.config.data_path, fp2)
                data2 = nib.load(full_fp2).get_fdata().astype('float32')
                if self.transform:
                    data2 = self.transform(data2).unsqueeze(0)
                self.fp_to_data[fp2] = data2

            # Get num classes
            dx = self.fp_to_dx[fp2]
            if dx not in self.class_counts:
                self.class_counts[dx] = 1
            else:
                self.class_counts[dx] += 1    

    def __getitem__(self, idx):
        fp1 = self.fps[idx][0]
        fp2 = self.fps[idx][1]
        data1 = self.fp_to_data[fp1]
        data2 = self.fp_to_data[fp2]

        ages = [self.fp_to_age[fp1], self.fp_to_age[fp2]]
        dxes = [self.fp_to_dx[fp1], self.fp_to_dx[fp2]]
        meta = dict(
            fps = self.fps[idx].tolist(),
            ages = ages,
            dxes = dxes
        )
        return data1, data2, meta
