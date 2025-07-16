import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

from kitti_calib.utils.dataset import KITTIRotationDataset
from kitti_calib.utils.model import CNN4CH
from kitti_calib.utils.metrics import geodesic_distance

# 하이퍼파라미터
DATA_X = "kitti_calib/output/x_inputs.npy"
DATA_Y = "kitti_calib/output/y_rotations.npy"
BATCH_SIZE = 32
LR = 1e-3
NUM_EPOCHS = 300
PATIENCE = 10
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # 나머지(0.1)는 test

# 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 전체 데이터셋 로드 & 분할
full_ds = KITTIRotationDataset(DATA_X, DATA_Y)
n = len(full_ds)
n_train = int(TRAIN_RATIO * n)
n_val   = int(VAL_RATIO * n)
n_test  = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test],
                                         generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# 모델/옵티마이저
model     = CNN4CH().to(device)
optimizer = Adam(model.parameters(), lr=LR)

best_val_loss   = float("inf")
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ===== 1) Train =====
    model.train()
    train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}] Train", ncols=100)
    train_loss = 0.0

    for x, y in train_pbar:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = geodesic_distance(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

    avg_train_loss = train_loss / len(train_loader)

    # ===== 2) Validation =====
    model.eval()
    val_pbar = tqdm(val_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}] Val  ", ncols=100)
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_pbar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = geodesic_distance(pred, y)
            val_loss += loss.item()
            val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch:3d} ▶ Train: {avg_train_loss:.4f}   Val: {avg_val_loss:.4f}")

    # ===== 3) Early Stopping & Save Best =====
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best.pt")
        print("--> New best model saved (best.pt)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered (no improvement in {PATIENCE} epochs).")
            break

# ===== 4) Save last.pt =====
torch.save(model.state_dict(), "last.pt")
print("Final model saved (last.pt)")

# ===== 5) Test =====
print("\n=== Testing Best Model on Hold-out Set ===")
model.load_state_dict(torch.load("best.pt"))
model.eval()
test_pbar = tqdm(test_loader, desc="Test", ncols=100)
test_loss = 0.0

with torch.no_grad():
    for x, y in test_pbar:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = geodesic_distance(pred, y)
        test_loss += loss.item()
        test_pbar.set_postfix({"test_loss": f"{loss.item():.4f}"})

avg_test_loss = test_loss / len(test_loader)
print(f"\nTest Loss: {avg_test_loss:.4f}")
