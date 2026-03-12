

from torch.utils.data import Dataset
import os

from .utils import *
from .sfm_loader import build_multiview_groups


class SfMDataset(Dataset):
    def __init__(self, points_dict, images, camera_infos, data_path,num_sample=5,
                 total_epoch=15000,mode="train"):
      
        self.camera_infos = camera_infos
        self.total_epoch = total_epoch
        self.V = 2
        self.epoch = 0
        self.data_path = os.path.join(data_path, "images/")
        self.mode = mode
        
        self.min_frame_dist = 10
        
        pkl_path = os.path.join(data_path, "train_test_data.pkl")
        if os.path.exists(pkl_path):
            print("pkl file exists ! ")
            train_data_group, test_data_group = load_pairs(pkl_path)
        else:
            print("create pkl file in: ", pkl_path)
            frame_index = build_frame_index(images)
            train_data_group, test_data_group = build_multiview_groups(points_dict, images, 
                                                     camera_infos["train_idx_list"], 
                                                     camera_infos["test_idx_list"], frame_index, self.min_frame_dist)
            save_pairs(train_data_group, test_data_group, pkl_path)
        
        self.groups = [g for g in train_data_group if len(g['image_ids']) >= num_sample]
        self.num_sample = num_sample
            

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.groups)
    
    def get_img(self, idx):
        
        file_name = self.camera_infos["image_name_list"][idx]        
        img_path = osp.join(self.data_path, file_name)
        
        img = imread(img_path)
        return torch.from_numpy(img).permute(2,0,1).float()/255.0

    def get_pose(self, idx):
        
        pose = self.camera_infos["pose_list"][idx]
        return torch.tensor(pose, dtype=torch.float)        

    def __getitem__(self, idx):
        g = self.groups[idx]
        N = len(g['image_ids'])
        sampled_idx = np.random.choice(N, self.num_sample, replace=False)
        sampled_img_ids = [g['image_ids'][i] for i in sampled_idx]
        sampled_coords = g['coords'][sampled_idx]
        

        return {
            'image_ids': torch.tensor(sampled_img_ids, dtype=torch.long),
            'coords': torch.tensor(sampled_coords, dtype=torch.float)
        }
        
        
        
        
        
def load_raw_data(
    base_dir, scene_info, idx, read_img=True):
    
    
    pose = scene_info["pose_list"][idx]
    K = scene_info["intrinsics_list"][idx]
    
    file_name = scene_info["image_name_list"][idx]
    depth_file_name = scene_info["depth_name_list"][idx]

    img = None
    img_path = osp.join(base_dir, file_name)
    if not osp.isfile(img_path):
        print("ERROR: img_path does not exist:", img_path)

    depth_file_name = osp.join(base_dir, depth_file_name)
    if read_img:
        img = imread(img_path)
    depth = load_depth_from_png(depth_file_name)

    depth=depth.astype(np.float32)/1000
    depth[depth < 1e-5] = 0
    return img, depth, pose, K   