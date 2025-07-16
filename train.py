from kitti_calib.utils.dataset import KITTIRotationDataset
from kitti_calib.utils.model import CNN4CH
from kitti_calib.utils.metrics import geodesic_distance
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm  # ✅ tqdm 추가

ds = KITTIRotationDataset("kitti_calib/output/x_inputs.npy", "kitti_calib/output/y_rotations.npy")
loader = DataLoader(ds, batch_size=32, shuffle=True)

model = CNN4CH().cuda()
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(300):
    model.train()
    total_loss = 0

    # ✅ tqdm 진행 바 감싸기
    pbar = tqdm(loader, desc=f"[Epoch {epoch+1}/300]", ncols=100)
    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = geodesic_distance(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ✅ tqdm 진행 바 옆에 실시간 loss 출력
        pbar.set_postfix({"loss": loss.item()})

    # ✅ 에폭 종료 후 평균 loss 출력
    avg_loss = total_loss / len(loader)
    print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")
