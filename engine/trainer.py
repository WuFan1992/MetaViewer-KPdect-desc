import torch
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

from modules.sfm_dataset import SfMDataset
from modules.sfm_loader import *
from modules.utils import *

from methods.EmbPose.varkpnetmodel import VarianceKPNetModel
from methods.EmbPose.warper import spvs_coarse_orig_res,sample_fixed_points
from methods.EmbPose.loss import *

def plot_matched_keypoints(image1, keypoints1, image2, keypoints2,
                           point_color='r', line_color='b', point_size=40, show_axis=False):
    """
    在两幅图片上绘制匹配关键点，并用线连接匹配点。

    参数：
    - image1, image2: numpy array 或 tensor, 形状 HxW 或 HxWxC
    - keypoints1, keypoints2: torch.Tensor 或 numpy array, 形状 N x 2
    - point_color: str, keypoints 的颜色 (默认红色 'r')
    - line_color: str, 匹配线的颜色 (默认蓝色 'b')
    - point_size: int, keypoints 的大小
    - show_axis: bool, 是否显示坐标轴
    """

    
    # 转成 numpy
    if isinstance(keypoints1, torch.Tensor):
        keypoints1 = keypoints1.cpu().numpy()
    if isinstance(keypoints2, torch.Tensor):
        keypoints2 = keypoints2.cpu().numpy()

    # 如果是灰度图，保证是 HxW
    if len(image1.shape) == 2:
        image1 = np.stack([image1]*3, axis=-1)
    if len(image2.shape) == 2:
        image2 = np.stack([image2]*3, axis=-1)

    # 并排显示两幅图
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    new_h = max(h1, h2)
    new_w = w1 + w2
    new_image = np.zeros((new_h, new_w, 3), dtype=image1.dtype)
    new_image[:h1, :w1, :] = image1
    new_image[:h2, w1:w1+w2, :] = image2

    plt.figure(figsize=(12, 6))
    plt.imshow(new_image)

    # 绘制 keypoints
    plt.scatter(keypoints1[:, 0], keypoints1[:, 1], c=point_color, s=point_size)
    plt.scatter(keypoints2[:, 0] + w1, keypoints2[:, 1], c=point_color, s=point_size)  # x 偏移

    # 绘制匹配线
    for (x1, y1), (x2, y2) in zip(keypoints1, keypoints2):
        plt.plot([x1, x2 + w1], [y1, y2], c=line_color, linewidth=1)

    if not show_axis:
        plt.axis('off')
    plt.show()
    
def sfm_collate_fn(batch):
    """
    batch: list of (data0, data1, matches)
    """
    batch_data0 = {}
    batch_data1 = {}
    matches_list = []

    # 所有 key
    for key in batch[0][0]:  # data0 keys
        batch_data0[key] = torch.stack([torch.tensor(b[0][key]) for b in batch])

    for key in batch[0][1]:  # data1 keys
        batch_data1[key] = torch.stack([torch.tensor(b[1][key]) for b in batch])

    for b in batch:
        matches_list.append(torch.tensor(b[2]))  # variable-length, keep list

    return batch_data0, batch_data1, matches_list



class Trainer(object):
    def __init__(self, kpnet, data_path, cpkt_save_path, num_iters):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kpnet = kpnet.to(self.device)
        #self.xfeat = XFeat(top_k=4096).to(self.device)
        
        points,images, cameras, test_images = loadSFM(data_path)
        camera_infos = readColmapCameras(images,cameras, test_images)

        self.dataset = SfMDataset(points, images, camera_infos, data_path)
        self.data_loader = data.DataLoader(self.dataset, batch_size=8, shuffle=True,collate_fn=sfm_collate_fn)
        self.data_loader_iter = iter(self.data_loader)  # 把数据变成迭代器，方便使用next 一个一个获取
        
        self.optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.kpnet.parameters()) , lr = 3e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30_000, gamma=0.5)
        
        self.num_iters = num_iters
        self.progress_bar = tqdm(range(0, self.num_iters), desc="Training progress")
        self.writer = SummaryWriter(cpkt_save_path + f'/logdir/scr_kpdect_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.save_ckpt_every = 1000
        self.cpkt_save_path = cpkt_save_path
        
    def train_iters(self):
        
        # -------------------------------
        # 初始 loss 权重
        # -------------------------------
        w_rec_init = 0.05
        w_ds_init  = 0.05
        w_kp_init  = 0.5   # 早期 reliability 低，防止梯度塌
        w_var_init = 1    # early stage 保持 small weight

        w_rec_max = 0.2
        w_ds_max  = 0.05
        w_kp_max  = 50
        w_var_max = 500
    
        
        for iter in range(self.num_iters):
            try:
                data0, data1, matches_list = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                data0, data1, matches_list = next(self.data_loader_iter)
            
            img0, img1 = data0["img"].to(self.device), data1["img"].to(self.device)               
            pose0_7d = pose_matrix_to_7d(data0["pose"]).to(self.device)
            pose1_7d = pose_matrix_to_7d(data1["pose"]).to(self.device) 
                        

            
            # -------------------------------
            # 动态 loss 权重 warmup
            # -------------------------------
            progress = np.clip(iter / 2000, 0.0, 1.0)  # 前2000 iteration warmup
            w_rec  = w_rec_init  + (w_rec_max - w_rec_init) * progress
            w_ds   = w_ds_init   + (w_ds_max  - w_ds_init)  * progress
            w_kp   = w_kp_init   + (w_kp_max  - w_kp_init)  * progress

            
            
            if progress < 0.2:
                w_var = 0.5
            elif progress < 0.6:
                w_var = 0.5+ 15 * (progress - 0.2)/0.3
            else:
                w_var = 50 + (100 - 50) * (progress - 0.5)/0.5

            loss_items = []
            
            for b in range(len(matches_list)):
                pts = matches_list[b].to(self.device)  # [S_i,4]
                if pts.shape[0] == 0:
                    continue

                coords0 = pts[:, :2].unsqueeze(0).to(self.device)  # [1, S_i, 2]
                coords1 = pts[:, 2:].unsqueeze(0).to(self.device)
                
                

                pose0 = data0["pose"][b].to(self.device)
                pose1 = data1["pose"][b].to(self.device)
                
                shared_featmap0, variance_map0, desc_map0, reliability_map0 = self.kpnet(img0[b].unsqueeze(0))
                shared_featmap1, variance_map1, desc_map1, reliability_map1 = self.kpnet(img1[b].unsqueeze(0))

                img0_norm = (img0[b]/255).cpu().numpy().transpose(1,2,0) 
                img1_norm = (img1[b]/255).cpu().numpy().transpose(1,2,0) 
                plot_matched_keypoints(img0_norm,pts[:, :2], img1_norm, pts[:, 2:] )
                #print("coords0 = ", pts[:, :2].shape)
                #print("coords1 = ", pts[:,2:].shape)

                sampled_desc0 = sample_map_at_coords(desc_map0, coords0)
                sampled_desc1 = sample_map_at_coords(desc_map1, coords1)
                
                # 防止网络直接用 backbone 绕过 descriptor learning
                shared_featmap0_detach = shared_featmap0.detach() if progress < 0.3 else shared_featmap0
                shared_featmap1_detach = shared_featmap1.detach() if progress < 0.3 else shared_featmap1

                
                backbone_pred0 = self.kpnet.reconstruction(pose0_7d[b].unsqueeze(0), coords0, sampled_desc0, shared_featmap0_detach)
                backbone_pred1 = self.kpnet.reconstruction(pose1_7d[b].unsqueeze(0), coords1, sampled_desc1, shared_featmap1_detach)
                
                # Sample the map to get the point-wise value 
                sampled_backbone0 = sample_map_at_coords(shared_featmap0, coords0)
                sampled_backbone1 = sample_map_at_coords(shared_featmap1, coords1)
                sampled_rel0 = sample_map_at_coords(reliability_map0, coords0)
                sampled_rel1 = sample_map_at_coords(reliability_map1, coords1)
                sampled_var0 = sample_map_at_coords(variance_map0, coords0)
                sampled_var1 = sample_map_at_coords(variance_map1, coords1)
                
                
                loss_rec = cross_view_recon_loss(backbone_pred0, sampled_backbone1) + cross_view_recon_loss(backbone_pred1, sampled_backbone0)
                
                #################
                sim = sampled_desc0 @ sampled_desc1.transpose(1, 2)
                sim = sim.squeeze(0)
                #print("diag", sim.diag().mean(),"mean",  sim.mean())
                ##################
                
                loss_ds, conf = dual_softmax_loss(sampled_desc0.squeeze(0),sampled_desc1.squeeze(0),temp=1.0, hard_negative_k=5)
                #print("loss_ds = ", loss_ds)
                
                
                # detach + normalize conf 避免 early stage reliability loss 梯度太小
                conf_detach = conf.detach()
                conf_detach = conf_detach / (conf_detach.max() + 1e-8)
                loss_kp = reliability_loss(sampled_rel0, conf_detach) + reliability_loss(sampled_rel1, conf_detach)
                
                loss_var = variance_loss_pose_aware_single(sampled_desc0.squeeze(0), sampled_desc1.squeeze(0), sampled_var0, sampled_var1, pose0, pose1, sampled_rel0, sampled_rel1, detach_desc_diff=True,min_weight=0.1)
                
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
            self.writer.add_scalar('Loss/reliability', loss_kp_w, iter)
            self.writer.add_scalar('Loss/variance', loss_var, iter)
            
            
            
            


            
            
            
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    cpkt_save_path = "checkpoints/"
    num_iters = 2000
    variance_kpnet = VarianceKPNetModel(in_channels=64, pose_dim=7, feature_dim=64, pose_embed=64)
    trainer = Trainer(variance_kpnet, data_path, cpkt_save_path, num_iters)
    trainer.train_iters()


        
            