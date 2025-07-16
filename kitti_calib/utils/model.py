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

        # Fix reflection safely (no in-place ops)
        det = torch.det(R_hat)
        sign = torch.sign(det).unsqueeze(1)         # shape (B, 1)
        last_col = U[:, :, -1] * sign               # shape (B, 3)
        U_fixed = torch.cat([U[:, :, :2], last_col.unsqueeze(2)], dim=2)

        R_hat = U_fixed @ Vh
        return R_hat