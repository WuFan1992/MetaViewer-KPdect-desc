import os
import torch.utils.data as data

from modules.sfm_dataset import SfMDataset
from modules.sfm_loader import *


def loadSFM(data_path):
    
    sfm_images_path = os.path.join(data_path, "sparse/0", "images.bin")
    sfm_point3d_path = os.path.join(data_path, "sparse/0/points3D.bin")
    
    points = read_points3D_binary(sfm_point3d_path)
    images = read_extrinsics_binary(sfm_images_path)
    
    return points, images


class Trainer(object):
    def __init__(self, metaviewer, kpnet, data_path):
        
        self.metaviewer = metaviewer
        self.kpnet = kpnet
        
        points,images = loadSFM(data_path)

        self.dataset = SfMDataset(points, images)
        self.data_loader = data.DataLoader(self.dataset, batch_size=8, shuffle=True)
        self.data_loader_iter = iter(self.data_loader)  # 把数据变成迭代器，方便使用next 一个一个获取
        
    def train_iters(self, iter_num):
        
        for i in range(iter_num):
            try:
                view_ids, view_xys = next(self.data_loader_iter)
            except StopIteration:
                # If StopIteration is raised, create a new iterator.
                self.data_loader_iter = iter(self.data_loader)
                view_ids, view_xys = next(self.data_loader_iter)
                
        print("view_ids shape = ", view_ids.shape)
        print("view_xys shape = ", view_xys.shape)
        

if __name__ == "__main__":
    data_path = "datasets/head"
    trainer = Trainer(None, None, data_path)
    trainer.train_iters(1)


        
            