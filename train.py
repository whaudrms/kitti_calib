from kitti_calib.utils.dataset import KITTIRotationDataset
from kitti_calib.utils.model import CNN4CH
from kitti_calib.utils.metrics import geodesic_distance
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

ds = KITTIRotationDataset("kitti_calib/output/x_inputs.npy", "kitti_calib/output/x_inputs.npy")
loader = DataLoader(ds, batch_size=32, shuffle=True)

model = CNN4CH().cuda()
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    total_loss = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = geodesic_distance(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.6f}")

