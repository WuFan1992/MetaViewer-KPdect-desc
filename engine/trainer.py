import torch
import torch.utils.data as data

from modules.sfm_dataset import SfMDataset
from modules.sfm_loader import *
from modules.utils import *


def collate_skip_none(batch):
    # batch 是 list，每个元素是 dataset[idx] 的返回值
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data._utils.collate.default_collate(batch)

class Trainer(object):
    def __init__(self, metaviewer, kpnet, data_path):
        
        self.metaviewer = metaviewer
        self.kpnet = kpnet
        
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
            
            data0, data1 = batch  # 或者根据你的 dataset 返回拆分
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    trainer = Trainer(None, None, data_path)
    trainer.train_iters(1)


        
            