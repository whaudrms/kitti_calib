import numpy as np
import matplotlib.pyplot as plt
import cv2

def project_lidar_to_image(points, R_l2c, t_l2c, P_rect, img_shape):
    pc_cam = (R_l2c @ points.T + t_l2c[:, np.newaxis]).T
    pc_cam = pc_cam[pc_cam[:, 2] > 0]

    pts_hom = np.hstack((pc_cam, np.ones((pc_cam.shape[0], 1))))
    uv = (P_rect @ pts_hom.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    H, W = img_shape[:2]
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv[mask], pc_cam[mask]

def visualize_projection(image, uv_points, color='r', point_size=1):
    """
    이미지 위에 투영된 LiDAR 포인트를 시각화합니다.

    Parameters:
    - image: (H, W, 3) RGB 이미지
    - uv_points: (N, 2) 투영된 이미지 좌표
    - color: 점 색상
    - point_size: 점 크기
    """
    img_vis = image.copy()
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.scatter(uv_points[:, 0], uv_points[:, 1], s=point_size, c=color, alpha=0.5)
    plt.title("Projected LiDAR points on Image")
    plt.axis('off')
    plt.show()
