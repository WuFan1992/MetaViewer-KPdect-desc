

import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
import os
import random
from .utils import *
from .sfm_loader import build_images_pairs


class SfMDataset(Dataset):
    def __init__(self, points_dict, images, camera_infos, data_path,
                 total_epoch=15000,
                    mode="train"):
      
        self.camera_infos = camera_infos
        self.total_epoch = total_epoch
        self.num_points = len(points_dict)
        self.V = 2
        self.epoch = 0
        self.data_path = os.path.join(data_path, "images/")
        self.mode = mode
        
        self.min_frame_dist = 10
        self.min_matches = 30
      
        frame_index = build_frame_index(images)
        train_data_pairs, test_data_pairs = build_images_pairs(points_dict, images, 
                                                     camera_infos["train_idx_list"], 
                                                     camera_infos["test_idx_list"], frame_index, self.min_frame_dist, self.min_matches)
            
        if mode == "train":
            self.pairs = list(train_data_pairs.items())
        else:
            self.pairs = list(test_data_pairs.items())
        

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_points



    def __getitem__(self, idx):
        
        (img0_id, img1_id), matches = self.pairs[idx]
        
        data0 = load_data(self.data_path, self.camera_infos, img0_id)
        data1 = load_data(self.data_path, self.camera_infos, img1_id)
        
        return data0, data1, matches
        
        
        
        
        
        