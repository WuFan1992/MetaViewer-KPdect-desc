

import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
import os
from .utils import *


class SfMDataset(Dataset):
    def __init__(self, points_dict, images, camera_infos, data_path,
                 total_epoch=15000,
                 theta_limit=np.pi / 2, mode="train"):
        self.points_dict = points_dict
        self.images = images
        self.camera_infos = camera_infos
        self.total_epoch = total_epoch
        self.theta_limit = theta_limit
        self.num_points = len(points_dict)
        self.V = 2
        self.epoch = 0
        self.data_path = os.path.join(data_path, "images/")
        self.mode = mode
        self.train_samples_idx = camera_infos["train_idx_list"]
        self.test_samples_idx = camera_infos["test_idx_list"]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_points

    def get_pair(self, idx, warmup_epoch=500):
        point_data = self.points_dict[idx]
        X = point_data["xyz"]
        visible_images = point_data["image_ids"]

        if len(visible_images) < 2:
            return None

        camera_centers = []
        valid_img_ids = []

        for img_id in visible_images:
            img = self.images[int(img_id)]
            C_i = compute_camera_center(img.qvec, img.tvec)
            camera_centers.append(C_i)
            valid_img_ids.append(int(img_id))

        if len(camera_centers) < 2:
            return None

        pairs = []
        # ------------------ TRAIN ------------------
        if self.mode == "train":
            for (i, C_i), (j, C_j) in itertools.combinations(enumerate(camera_centers), 2):
                img_i = valid_img_ids[i]
                img_j = valid_img_ids[j]
                if not (img_i in self.train_samples_idx and img_j in self.train_samples_idx):
                    continue
                theta = compute_viewing_angle(X, C_i, C_j)
                pairs.append((i, j, theta))

            if len(pairs) == 0:
                return None

            # curriculum: gradually increase allowed max angle
            progress = np.clip(self.epoch / self.total_epoch, 0.0, 1.0)
            allowed_max_angle = progress * self.theta_limit

            # --- warmup阶段: 强制采样中等角度差 theta >= min_theta
            min_theta = np.pi / 6
            if self.epoch < warmup_epoch:
                valid_pairs = [p for p in pairs if p[2] >= min_theta]
            else:
                valid_pairs = [p for p in pairs if p[2] <= allowed_max_angle]

            # fallback: 没有合法 pair 时取最大/最小
            if len(valid_pairs) == 0:
                pairs.sort(key=lambda x: x[2])
                i, j, _ = pairs[0]
            else:
                # 优先选择角度最大的 pair 以保证 variance supervision 有梯度
                valid_pairs.sort(key=lambda x: -x[2])
                i, j, _ = valid_pairs[0]

        # ------------------ TEST ------------------
        elif self.mode == "test":
            for (i, C_i), (j, C_j) in itertools.combinations(enumerate(camera_centers), 2):
                img_i = valid_img_ids[i]
                img_j = valid_img_ids[j]
                if img_i in self.train_samples_idx and img_j in self.test_samples_idx:
                    theta = compute_viewing_angle(X, C_i, C_j)
                    pairs.append((i, j, theta))
                elif img_j in self.train_samples_idx and img_i in self.test_samples_idx:
                    theta = compute_viewing_angle(X, C_j, C_i)
                    pairs.append((j, i, theta))

            if len(pairs) == 0:
                return None
            valid_pairs = [p for p in pairs if p[2] <= self.theta_limit]
            if len(valid_pairs) == 0:
                pairs.sort(key=lambda x: -x[2])
                i, j, _ = pairs[0]
            else:
                valid_pairs.sort(key=lambda x: -x[2])
                i, j, _ = valid_pairs[0]

        else:
            raise ValueError("mode must be 'train' or 'test'")

        selected_views = [valid_img_ids[i], valid_img_ids[j]]

        # 提取 2D 点
        selected_2D_points = []
        for sv in selected_views:
            img = self.images[int(sv)]
            point_indices = np.where(img.point3D_ids == int(idx))[0]
            if len(point_indices) == 0:
                selected_2D_points.append(np.array([0.0, 0.0]))
            else:
                selected_2D_points.append(img.xys[point_indices[0]])
        selected_2D_points = np.stack(selected_2D_points, axis=0)

        return torch.tensor(selected_views, dtype=torch.long), \
                torch.tensor(selected_2D_points, dtype=torch.float)

    def __getitem__(self, idx):
        result = self.get_pair(idx)
        if result is None:
            # 这个点没有合法 pair，直接跳过
            return None
        pair_idx, pts_2d = result

        data0 = load_data(self.data_path, self.camera_infos, pair_idx[0])
        data1 = load_data(self.data_path, self.camera_infos, pair_idx[1])
        
        return data0, data1
        
        
        
        
        
        