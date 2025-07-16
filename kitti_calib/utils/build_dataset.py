import os
import numpy as np
import cv2
from tqdm import tqdm
from kitti_loader import load_point_cloud, load_image, load_velo_calib, load_cam_calib
from projection_utils import project_lidar_to_image, visualize_projection
from augment_utils import random_rotation_matrix

# 경로 설정
ROOT = f'/2011_09_26_drive_0086_sync'  
calib_velo_path = os.path.join(ROOT, 'calib_velo_to_cam.txt')
calib_cam_path = os.path.join(ROOT, 'calib_cam_to_cam.txt')
velodyne_dir = os.path.join(ROOT, 'velodyne_points/data')
image_dir = os.path.join(ROOT, 'image_02/data')

# 캘리브레이션 로드
R_gt, t_gt = load_velo_calib(calib_velo_path)
P_rect, _ = load_cam_calib(calib_cam_path)
# print(R_gt)
# print(t_gt)
# print(P_rect)

# 저장 버퍼
x_list, y_list = [], []

# 프레임 수 결정
frame_ids = sorted([f.split('.')[0] for f in os.listdir(velodyne_dir) if f.endswith('.bin')])

for frame_id in tqdm(frame_ids[:]):  # 예: 앞 200개만 사용
    # 파일 경로
    pc_path = os.path.join(velodyne_dir, f'{frame_id}.bin')
    img_path = os.path.join(image_dir, f'{frame_id}.png')

    # 로딩
    pc = load_point_cloud(pc_path)
    img = load_image(img_path)
    if pc.shape[0] < 100: continue  # poor point clouds

    # 랜덤 회전 노이즈
    delta_R = random_rotation_matrix(max_deg=2)
    noisy_R = delta_R @ R_gt

    # 투영
    uv, pc_cam = project_lidar_to_image(pc, noisy_R, t_gt, P_rect, img.shape)
    if uv.shape[0] < 100: continue  # too few valid points

    if int(frame_id) % 50 == 0:  # 50 프레임마다 시각화
        print(f"[DEBUG] Visualizing frame {frame_id}")
        visualize_projection(img, uv)

    # 깊이 맵 만들기
    H, W = img.shape[:2]
    depth_map = np.zeros((H, W), dtype=np.float32)
    for (u, v), pt in zip(uv, pc_cam):
        u, v = int(u), int(v)
        depth_map[v, u] = pt[2]  # depth = Z_cam

    # 정규화 + resize
    depth_norm = np.clip(depth_map / 80.0, 0, 1)  # normalize
    depth_img = cv2.resize(depth_norm, (128, 128))
    img_resized = cv2.resize(img, (128, 128)) / 255.0  # normalize RGB

    # 입력: RGB + depth → (H, W, 4)
    x = np.concatenate([img_resized, depth_img[..., None]], axis=2).astype(np.float32)
    y = R_gt.astype(np.float32)

    x_list.append(x)
    y_list.append(y)

# numpy 저장
x_np = np.stack(x_list)       # shape: (N, 128, 128, 4)
y_np = np.stack(y_list)       # shape: (N, 3, 3)

np.save('output/x_inputs.npy', x_np)
np.save('output/y_rotations.npy', y_np)

print(f"Saved: {x_np.shape=}, {y_np.shape=}")




