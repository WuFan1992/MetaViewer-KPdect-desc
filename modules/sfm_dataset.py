

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .utils import *
# -----------------------------
# Dataset
# -----------------------------
class SfMDataset(Dataset):
    def __init__(self, points_dict, images, V=5, total_epoch=15000):
        """
        points_dict: dict {point_id: {"id_3d": int, "xyz": np.array, "rgb": np.array,
                                     "error": float, "image_ids": list, "point2d_ids": list}}
        images: dict {image_id: Image}  # 读取的 SfM image 信息
        V: 每个 3D 点采样的视角数
        """
        self.points_dict = points_dict
        self.images = images
        self.V = V
        self.total_epoch = total_epoch
        self.num_points = len(points_dict)
        self.epoch = 0  # 默认初始 epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        point_data = self.points_dict[idx]
        X = point_data["xyz"]
        visible_images = point_data["image_ids"]

        if len(visible_images) == 0:
            # 如果没有可见图像，返回 padding
            return torch.zeros(self.V, dtype=torch.long), torch.zeros(self.V, 2)

        # reference view
        ref_idx = visible_images[0]
        ref_image = self.images[int(ref_idx)]
        ref_C = compute_camera_center(ref_image.qvec, ref_image.tvec)

        # 计算每个可见图像的 viewing angle
        angles = []
        for img_id in visible_images:
            img = self.images[int(img_id)]
            C_i = compute_camera_center(img.qvec, img.tvec)
            theta = compute_viewing_angle(X, C_i, ref_C)
            angles.append(theta)

        # curriculum learning: 最大允许角度
        p_t = self.epoch / self.total_epoch
        theta_max = p_t * (np.pi/2)  # 0~90度逐渐开放

        # candidate views
        candidate_views = [img_id for img_id, angle in zip(visible_images, angles) if angle <= theta_max]
        if len(candidate_views) < self.V:
            candidate_views = visible_images  # fallback

        # 随机采样 V 个视角
        selected_views = random.sample(candidate_views, min(self.V, len(candidate_views)))

        # 对应 2D 点坐标
        selected_2D_points = []
        for sv in selected_views:
            img = self.images[int(sv)]
            # 找到这个 3D 点在该图像中的 2D 坐标
            point_indices_in_img = np.where(img.point3D_ids == int(idx))[0]
            if len(point_indices_in_img) == 0:
                selected_2D_points.append(np.array([0.0,0.0]))
            else:
                pt_idx_in_img = point_indices_in_img[0]
                xy = img.xys[pt_idx_in_img]
                selected_2D_points.append(xy)

        # padding: 如果不足 V 个视角，用 0 填充
        while len(selected_views) < self.V:
            selected_views.append(0)
            selected_2D_points.append(np.array([0.0,0.0]))

        selected_2D_points = np.stack(selected_2D_points, axis=0)  # Vx2

        return torch.tensor(selected_views, dtype=torch.long), torch.tensor(selected_2D_points, dtype=torch.float)

# -----------------------------
# 使用示例
# -----------------------------
# 假设你已经读取

"""
xyzs, _, _, _, ids_img, ids_2dpts = read_points3D_binary('points3D.bin')
images = read_extrinsics_binary('images.bin')

V = 4
total_epoch = 50
dataset = SfMDataset(xyzs, ids_img, ids_2dpts, images, V=V, total_epoch=total_epoch)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# 在训练循环中：
for epoch in range(total_epoch):
    dataset.set_epoch(epoch)
    for batch_image_ids, batch_2D_points in loader:
        # batch_image_ids: list of lists, 16 x V
        # batch_2D_points: list of arrays, 16 x V x 2
        # 这里可以根据 image id 去读取 patch / crop / feature embedding
        pass


"""