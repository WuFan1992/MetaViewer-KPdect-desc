import torch
import torch.utils.data as data

from modules.sfm_dataset import SfMDataset
from modules.sfm_loader import *
from modules.utils import *

from methods.EmbPose.variance_kpnet import VarianceKPNet
from methods.EmbPose.warper import spvs_coarse

from methods.Xfeat.xfeat import XFeat


def collate_skip_none(batch):
    # batch 是 list，每个元素是 dataset[idx] 的返回值
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data._utils.collate.default_collate(batch)

class Trainer(object):
    def __init__(self, kpnet, data_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kpnet = kpnet.to(self.device)
        self.xfeat = XFeat(top_k=4096).to(self.device)
        
        points,images, cameras, test_images = loadSFM(data_path)
        camera_infos = readColmapCameras(images,cameras, test_images)

        self.dataset = SfMDataset(points, images, camera_infos, data_path)
        self.data_loader = data.DataLoader(self.dataset, batch_size=8, shuffle=True,collate_fn=collate_skip_none)
        self.data_loader_iter = iter(self.data_loader)  # 把数据变成迭代器，方便使用next 一个一个获取
        
    def train_iters(self, iter_num):
        
        for i in range(iter_num):
            try:
                batch = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                batch = next(self.data_loader_iter)
            
            if batch is None:  # 整个 batch 都是 None
                continue
            
            data0, data1 = batch
            
            # XFeat forward 
            xfeat_map0 = self.xfeat.getFeatDesc(data0["img"]).to(self.device) 
            xfeat_map1 = self.xfeat.getFeatDesc(data1["img"]).to(self.device) 
            
            pose0_7d = pose_matrix_to_7d(data0["pose"]).to(self.device)
            pose1_7d = pose_matrix_to_7d(data1["pose"]).to(self.device) 
                        
            positives_md_coarse = spvs_coarse(data0, data1, scale=4)
            
            print("positive_md_coarse = ", positives_md_coarse)

            
            #Check if batch is corrupted with too few correspondences
            is_corrupted = False
            for p in positives_md_coarse:
                if len(p) < 30:
                    is_corrupted = True

            if is_corrupted:
                continue
                        
            xfeat_pred_list = []
            sampled_desc_list = []
            sampled_rel_list = []
            sampled_var_list = []
            
            for i in range(len(positives_md_coarse)):
                pts = positives_md_coarse[i]  # [S_i,4]
                if pts.shape[0] == 0:
                    continue

                coords0 = pts[:, :2].unsqueeze(0).to(self.device)  # [1, S_i, 2]
                coords1 = pts[:, 2:].unsqueeze(0).to(self.device)

                pose0 = pose0_7d[i].unsqueeze(0)  # [1,7]
                pose1 = pose1_7d[i].unsqueeze(0)


                xfeat_pred0, sampled_desc0, sampled_rel0, sampled_var0 = self.kpnet(
                    xfeat_map0[i].unsqueeze(0), pose0, coords0
                )
                xfeat_pred1, sampled_desc1, sampled_rel1, sampled_var1 = self.kpnet(
                    xfeat_map1[i].unsqueeze(0), pose1, coords1
                )

                xfeat_pred_list.append((xfeat_pred0, xfeat_pred1))
                sampled_desc_list.append((sampled_desc0, sampled_desc1))
                sampled_rel_list.append((sampled_rel0, sampled_rel1))
                sampled_var_list.append((sampled_var0, sampled_var1))
            
            print(f"Batch {i}: processed {len(xfeat_pred_list)} elements")
            
            
            
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    variance_kpnet = VarianceKPNet(in_channels=64, pose_dim=7, feature_dim=64, pose_embed=64)
    trainer = Trainer(variance_kpnet, data_path)
    trainer.train_iters(1)


        
            