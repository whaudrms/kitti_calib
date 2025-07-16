import torch
import torch.nn as nn

class CNN4CH(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 9)  # → R9 for rotation matrix
        )

    def forward(self, x):
        r = self.model(x)           # shape (B, 9)
        R = r.view(-1, 3, 3)
        U, _, Vh = torch.linalg.svd(R)
        R_hat = U @ Vh
        # Fix reflection
        det = torch.det(R_hat)
        U[:, :, -1] *= torch.sign(det).unsqueeze(1)
        R_hat = U @ Vh
        return R_hat

