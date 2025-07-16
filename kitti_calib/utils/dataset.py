import numpy as np
import torch
from torch.utils.data import Dataset

class KITTIRotationDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)   # shape: (N, 128, 128, 4)
        self.y = np.load(y_path)   # shape: (N, 3, 3)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).permute(2, 0, 1)  # (C, H, W)
        y = torch.from_numpy(self.y[idx])  # (3, 3)
        return x, y
