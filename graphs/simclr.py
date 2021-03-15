import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np


# Standard implementation of Encoder in AE
class SimEncoder(nn.Module):
    def __init__(self, len_z):
        super().__init__()

        # Encoder conv layers
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=16,kernel_size=3,padding=1), # padding=1 to keep dims same
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
        )

        # Encoder linear layer
        self.linear = nn.Linear(in_features=1024, out_features=len_z)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.tanh(out)
        return out

class SimCLRNet(nn.Module):
    def __init__(self, len_z):
        super().__init__()
        self.len_z = len_z
        self.encoder = SimEncoder(len_z)
        half_len_z = int(len_z/2)
        quarter_len_z = int(len_z/4)
        if self.len_z < 64:
            self.g = nn.Sequential(
                nn.Linear(len_z, half_len_z)
            )
        self.g = nn.Sequential(
            nn.Linear(len_z, half_len_z),
            nn.ReLU(),
            nn.Linear(half_len_z, quarter_len_z)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.g(h)
        return z, h

""" https://github.com/sthalles/SimCLR/blob/master/simclr.py """ 
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)


        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)