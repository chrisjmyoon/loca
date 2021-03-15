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
Stores the latent representations z's as the data and returns z
"""
class ZDataset(Dataset):
    def __init__(self, agg_z, agg_dx, agg_age):
        self.agg_z = agg_z
        self.agg_dx = agg_dx
        self.agg_age = agg_age

    def __len__(self):
        return len(self.agg_z)        
    
    def __getitem__(self, idx):
        z = self.agg_z[idx]
        age = self.agg_age[idx]
        dx = self.agg_dx[idx]

        return z, age, dx