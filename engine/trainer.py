import torch
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

from modules.sfm_dataset import SfMDataset
from modules.sfm_loader import *
from modules.utils import *

from methods.EmbPose.variance_kpnet import VarianceKPNet
from methods.EmbPose.warper import spvs_coarse
from methods.EmbPose.loss import *

from methods.Xfeat.xfeat import XFeat




def collate_skip_none(batch):
    # batch 是 list，每个元素是 dataset[idx] 的返回值
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data._utils.collate.default_collate(batch)

class Trainer(object):
    def __init__(self, kpnet, data_path, cpkt_save_path, num_iters):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kpnet = kpnet.to(self.device)
        self.xfeat = XFeat(top_k=4096).to(self.device)
        
        points,images, cameras, test_images = loadSFM(data_path)
        camera_infos = readColmapCameras(images,cameras, test_images)

        self.dataset = SfMDataset(points, images, camera_infos, data_path)
        self.data_loader = data.DataLoader(self.dataset, batch_size=8, shuffle=True,collate_fn=collate_skip_none)
        self.data_loader_iter = iter(self.data_loader)  # 把数据变成迭代器，方便使用next 一个一个获取
        
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.kpnet.parameters()) , lr = 3e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30_000, gamma=0.5)
        
        self.num_iters = num_iters
        self.progress_bar = tqdm(range(0, self.num_iters), desc="Training progress")
        self.writer = SummaryWriter(cpkt_save_path + f'/logdir/scr_kpdect_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.save_ckpt_every = 500
        self.cpkt_save_path = cpkt_save_path
        
    def train_iters(self):
        
        w_rec = 0.2
        w_ds  = 0.05
        w_kp  = 0.5
        w_var = 100
    
        
        for iter in range(self.num_iters):
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
        

            
            #Check if batch is corrupted with too few correspondences
            is_corrupted = False
            for p in positives_md_coarse:
                if len(p) < 30:
                    is_corrupted = True

            if is_corrupted:
                continue
                        


            loss_items = []
            
            for b in range(len(positives_md_coarse)):
                pts = positives_md_coarse[b]  # [S_i,4]
                if pts.shape[0] == 0:
                    continue

                coords0 = pts[:, :2].unsqueeze(0).to(self.device)  # [1, S_i, 2]
                coords1 = pts[:, 2:].unsqueeze(0).to(self.device)

                pose0 = data0["pose"][b].to(self.device)
                pose1 = data1["pose"][b].to(self.device)
                

                xfeat_pred0, sampled_input0, sampled_desc0, sampled_rel0, sampled_var0 = self.kpnet(
                    xfeat_map0[b].unsqueeze(0), pose0_7d[b].unsqueeze(0), coords0
                )
                xfeat_pred1, sampled_input1, sampled_desc1, sampled_rel1, sampled_var1 = self.kpnet(
                    xfeat_map1[b].unsqueeze(0), pose1_7d[b].unsqueeze(0), coords1
                )
                
                loss_rec = reconstr_loss(xfeat_pred0, sampled_input0) + reconstr_loss(xfeat_pred1, sampled_input1)
                
                loss_ds, conf = dual_softmax_loss(sampled_desc0.squeeze(0),sampled_desc1.squeeze(0))
                loss_kp = reliability_loss(sampled_rel0, conf) + reliability_loss(sampled_rel1, conf)
                loss_var = variance_loss_pose_aware_single(sampled_desc0.squeeze(0), sampled_desc1.squeeze(0), sampled_var0, sampled_var1, pose0, pose1, sampled_rel0, sampled_rel1)
                
                loss_rec_w = w_rec * loss_rec
                loss_ds_w  = w_ds  * loss_ds
                loss_kp_w  = w_kp  * loss_kp
                loss_var_w = w_var * loss_var
        
                
                loss_batch = loss_rec_w + loss_ds_w + loss_kp_w + loss_var_w
                loss_items.append(loss_batch)
                if b == 0:
                    acc_coarse_0 = check_accuracy(sampled_desc0.squeeze(0), sampled_desc1.squeeze(0))
                acc_coarse = check_accuracy(sampled_desc0.squeeze(0), sampled_desc1.squeeze(0))
                
            
            loss = torch.stack(loss_items).mean()
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kpnet.parameters(), 1.)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            if (iter+1) % self.save_ckpt_every == 0:
                print('saving iter ', iter+1)
                torch.save(self.kpnet.state_dict(), self.cpkt_save_path + f'variancekpnet_{iter+1}.pth')
                
            self.progress_bar.set_description( 'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} loss_desc: {:.3f} loss_recon: {:.3f} loss_kp: {:.3f}  loss_var: {:.3f}'.format(
                                                                        loss.item(), acc_coarse_0, acc_coarse, loss_ds_w.item(), loss_rec_w.item(), loss_kp_w.item(), loss_var_w.item()) )
            self.progress_bar.update(1)
            
            # Log metrics
            self.writer.add_scalar('Loss/total', loss.item(), iter)
            self.writer.add_scalar('Accuracy/coarse_matching accuracy', acc_coarse, iter)
            self.writer.add_scalar('Loss/description', loss_ds_w, iter)
            self.writer.add_scalar('Loss/recontruction', loss_rec_w, iter)
            self.writer.add_scalar('Loss/reliability', loss_rec_w, iter)
            self.writer.add_scalar('Loss/variance', loss_var, iter)
            
            
            
            


            
            
            
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    cpkt_save_path = "checkpoints/"
    num_iters = 1000
    variance_kpnet = VarianceKPNet(in_channels=64, pose_dim=7, feature_dim=64, pose_embed=64)
    trainer = Trainer(variance_kpnet, data_path, cpkt_save_path, num_iters)
    trainer.train_iters()


        
            