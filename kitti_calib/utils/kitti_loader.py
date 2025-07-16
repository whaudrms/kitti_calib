import numpy as np
import cv2

def load_point_cloud(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def load_image(img_path):
    return cv2.imread(img_path)  # shape: (H, W, 3)

def load_velo_calib(calib_path):
    R = None
    t = None
    with open(calib_path) as f:
        for line in f:
            if line.startswith('R:'):
                vals = [float(x) for x in line.strip().split()[1:]]
                R = np.array(vals).reshape(3, 3)
            elif line.startswith('T:'):
                t = np.array([float(x) for x in line.strip().split()[1:]])
    if R is None or t is None:
        raise ValueError("R 또는 T 정보를 찾을 수 없습니다.")
    return R, t

def load_cam_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    # Parse P_rect_02
    for line in lines:
        if line.startswith('P_rect_02:'):
            P_rect = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)

        if line.startswith('R_rect_02:'):
            R_rect = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)

    return P_rect, R_rect

