import torch
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import glob
import tqdm

from modules.megadepth.megadepth import MegaDepthDataset
from modules.megadepth.megadepth_wraper import *
from modules.sfm_loader import *
from modules.utils import *

from methods.EmbPose.varkpnetmodel import VUDNet

from methods.EmbPose.loss import *

import numpy as np


def to_numpy_image(img):
    """
    统一把输入转成 (H, W, 3)
    支持:
    [B,C,H,W], [C,H,W], [H,W]
    """

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # ===== 1. 去 batch =====
    if img.ndim == 4:  # [B,C,H,W]
        img = img[0]

    # ===== 2. CHW -> HWC =====
    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img[0]  # -> (H,W)
        elif img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # -> (H,W,3)

    # ===== 3. 灰度 -> RGB =====
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    return img


def plot_multi_view_matches(images, multi_corrs, vis=None, max_points=50, save_path=None):
    """
    可视化多视图匹配点轨迹（严格可见性过滤）
    
    Args:
        images: list of images [C,H,W] 或 [H,W] 或 torch.Tensor
        multi_corrs: [N, V, 2] tensor 或 numpy array
        vis: [N, V] bool，可见性掩码，如果 None，则默认全部可见
        max_points: 最大显示点数
        save_path: 保存路径，如果 None 则 plt.show()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # ===== 转 numpy image =====
    def to_numpy_image(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 4:  # [B,C,H,W]
            img = img[0]
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
            elif img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        return img

    imgs = [to_numpy_image(img) for img in images]
    V = len(imgs)
    H_list = [img.shape[0] for img in imgs]
    W_list = [img.shape[1] for img in imgs]

    H = max(H_list)
    W = sum(W_list)

    canvas = np.zeros((H, W, 3), dtype=imgs[0].dtype)
    offsets = []
    cur_w = 0
    for img in imgs:
        h, w = img.shape[:2]
        canvas[:h, cur_w:cur_w+w] = img
        offsets.append(cur_w)
        cur_w += w

    # ===== 转 numpy 数据 =====
    if isinstance(multi_corrs, torch.Tensor):
        multi_corrs = multi_corrs.detach().cpu().numpy()
    if vis is None:
        vis = np.ones((multi_corrs.shape[0], V), dtype=bool)
    elif isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()

    N = multi_corrs.shape[0]
    if N == 0:
        print("No correspondences")
        return

    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        multi_corrs = multi_corrs[idx]
        vis = vis[idx]
        N = max_points

    # ===== 绘制 =====
    plt.figure(figsize=(15, 5))
    plt.imshow(canvas)
    colors = plt.cm.jet(np.linspace(0, 1, N))

    for i in range(N):
        # 检查该点在所有视图中是否都可见且在图像内
        keep_track = True
        for j in range(V):
            x, y = multi_corrs[i, j]
            h, w = imgs[j].shape[:2]
            if not vis[i, j] or x < 0 or x >= w or y < 0 or y >= h:
                keep_track = False
                break

        if not keep_track:
            continue  # 只保留完全可见的轨迹

        # 画点和连线
        prev_pt = None
        for j in range(V):
            x, y = multi_corrs[i, j]
            plt.scatter(x + offsets[j], y, c=[colors[i]], s=20)
            if prev_pt is not None:
                x0, y0, j0 = prev_pt
                plt.plot([x0 + offsets[j0], x + offsets[j]], [y0, y], c=colors[i], linewidth=1)
            prev_pt = (x, y, j)

    plt.axis('off')
    plt.title(f"Multi-view Correspondences ({V} views)")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=True)
    
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



class TrainerMultiView:
    def __init__(self, kpnet, datapath, cpkt_save_path, num_iters, top_k=4096, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kpnet = kpnet.to(self.device)
        

        #self.dataset = SfMDataset(points, images, camera_infos, datapath)
        megadepth_datapath = "datasets/MegaDepth_v1"
        npz_root = "datasets/scene_info_0.1_0.7"
        npzpaths = glob.glob(npz_root+ '/*.npz')[:]
        npzpaths = ['datasets/scene_info_0.1_0.7/0022_0.1_0.3.npz','datasets/scene_info_0.1_0.7/0022_0.3_0.5.npz', 'datasets/scene_info_0.1_0.7/0022_0.5_0.7.npz' ]
        self.dataset = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = megadepth_datapath,
                            npz_path = path) for path in tqdm.tqdm(npzpaths, desc="[MegaDepth] Loading metadata")] )
        
        
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=True
        )
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.kpnet.parameters()), lr=3e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30000, gamma=0.5)
        
        
        
        self.num_iters = num_iters
        self.cpkt_save_path = cpkt_save_path
        self.top_k = top_k
        
        self.progress_bar = tqdm.tqdm(range(0, self.num_iters), desc="Training progress")
        self.writer = SummaryWriter(cpkt_save_path + f'/logdir/scr_kpdect_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        
        # warm-up params
        self.geo_warmup_steps = 2000
        self.sigma_warmup_steps = 4000


    def train_iters(self):
        self.kpnet.train()
        data_iter = iter(self.data_loader)

        for iter_idx in range(self.num_iters):
            try:
                batch_data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch_data = next(data_iter)
            
            images = batch_data['images']
            H_orig, W_orig = images[0].shape[2:]
            
            # ===== 1. 生成 multi-view 对应点 =====
            multi_corrs = spvs_coarse_multi_cycle(batch_data, scale=4) # 这里的坐标已经返回原图尺寸下了
            V = len(images)  # V 是每一个 batch 里面有多少个view, 默认是5
            B = len(multi_corrs)  # B 是batch 数
            
            
            # ===== 2. 生成不同view 下的样本 =====
            batch_points_dict = split_points_by_visibility(multi_corrs, min_views=(2,3,4,5))
            # 权重
            var_weights = {5:1.0, 4:0.7, 3:0.4, 2:0.1}  # variance loss
            desc_weights = {5:1.0, 4:1.0, 3:1.0, 2:1.0}  # descriptor loss

            
            # ===== 3. 可视化=====
            # ===== 关键：正确取 batch 第一个样本 =====
            # ===== 可视化 =====
            images_vis = []
            for img in batch_data['images']:
                if isinstance(img, torch.Tensor) and img.dim() == 4:
                    images_vis.append(img[0])
                else:
                    images_vis.append(img)

            # ===== 对每种view分别画 =====
            for k in [5,4,3,2]:

                if len(batch_points_dict[k]) == 0:
                    print(f"[VIS] No {k}-view points")
                    continue

                # 只取 batch=0 的点用于可视化
                corrs_k, vis_k = batch_points_dict[k][0]

                # 从 images 中取前 k 个 view
                imgs_k = images[:k]
                pts_k = corrs_k[:, :k, :]    # 对应 k 张图的点
                vis_k = vis_k[:, :k]

                plot_multi_view_matches(
                    imgs_k,
                    pts_k,
                    vis_k,
                    max_points=50,
                    save_path=None
                )
            
            
               
            
             # ===== forward each view =====             
            f_inv_maps, f_geo_maps, sigma_maps = [], [], []
            for v in range(V):
                out = self.kpnet(images[v].to(self.device), return_teacher= True)
                f_inv_maps.append(out["f_inv"])
                f_geo_maps.append(out["f_geo"])
                sigma_maps.append(out["sigma"])
            # ===== sample multi-view features =====
            loss_desc_total = 0.0
            loss_sigma_total = 0.0
            valid_batch = 0
            
            for k in [2,3,4,5]:
                w_var = var_weights[k]
                w_desc = desc_weights[k]
                if len(batch_points_dict[k]) == 0: # 相当于是batch
                    # 这个 batch 没有 k-view 点，直接跳过
                    continue
  
                for b in range(len(batch_points_dict[k])):
                    corrs, vis = batch_points_dict[k][b]


                    if corrs is None or corrs.shape[0] < 20:
                        continue

                    # ===== 限制最大点数 =====
                    max_points = 5000
                    N = corrs.shape[0]
                    if N > max_points:
                        idx = torch.randperm(N)[:max_points]
                        corrs = corrs[idx]
                        vis = vis[idx]
                    N = corrs.shape[0]
                    
                    print(" k = ", k, "N = ", N)

                    # 找出至少有 2 个 view 都有效的点
                    valid_mask_pts = vis.sum(dim=1) >= 2  # [N], True 表示至少 2 个 view 可用
                    if valid_mask_pts.sum() == 0:
                        continue

                    corrs_valid = corrs[valid_mask_pts]  # [N_valid, V, 2]
                    vis_valid = vis[valid_mask_pts]      # [N_valid, V]

                    f_inv_per_point, sigma_per_point, feat_teacher_per_point = [], [], []

                    for v in range(V):
                        valid_mask_view = vis_valid[:, v]  # [N_valid]
                        if valid_mask_view.sum() == 0:
                            # 该 view 没有有效点，填充零
                            C = f_inv_maps[v].shape[1]
                            f_inv_per_point.append(torch.zeros((valid_mask_pts.sum(), C), device=self.device))
                            sigma_per_point.append(torch.zeros((valid_mask_pts.sum(), 1), device=self.device))
                            feat_teacher_per_point.append(torch.zeros((valid_mask_pts.sum(), C), device=self.device))
                            continue

                        coords = corrs_valid[valid_mask_view, v, :].to(self.device)
                        f_inv_sample = sample_map_at_coords(f_inv_maps[v][b:b+1], coords, H_orig, W_orig)
                        sigma_sample = sample_map_at_coords(sigma_maps[v][b:b+1], coords, H_orig, W_orig)
                        feat_teacher_map = self.kpnet.backbone(images[v].to(self.device)).detach()
                        feat_teacher_sample = sample_map_at_coords(feat_teacher_map[b:b+1], coords, H_orig, W_orig)

                        # scatter 回完整 N_valid
                        C = f_inv_sample.shape[-1]
                        full_f_inv = torch.zeros((valid_mask_pts.sum(), C), device=self.device)
                        full_sigma = torch.zeros((valid_mask_pts.sum(), 1), device=self.device)
                        full_teacher = torch.zeros((valid_mask_pts.sum(), C), device=self.device)

                        full_f_inv[valid_mask_view] = f_inv_sample
                        full_sigma[valid_mask_view] = sigma_sample
                        full_teacher[valid_mask_view] = feat_teacher_sample

                        f_inv_per_point.append(full_f_inv)
                        sigma_per_point.append(full_sigma)
                        feat_teacher_per_point.append(full_teacher)

                    # stack → [N_valid, V, C]
                    f_inv = torch.stack(f_inv_per_point, dim=1)
                    sigma = torch.stack(sigma_per_point, dim=1)
                    feat_teacher = torch.stack(feat_teacher_per_point, dim=1)

                    # 计算 multi-view loss
                    loss_desc_b = mv_infonce_masked(f_inv, vis_valid)
                    loss_sigma_b = sigma_loss_teacher_multi_view_masked(feat_teacher, sigma, vis_valid)

                    loss_desc_total += w_desc * loss_desc_b
                    loss_sigma_total += w_var * loss_sigma_b
                    valid_batch += 1
                    
            # =========================
            # normalize losses
            # =========================
            if valid_batch > 0:
                loss_desc = loss_desc_total / valid_batch
                loss_sigma = loss_sigma_total / valid_batch
            else:
                dummy = 0.0 * next(self.kpnet.parameters()).sum()
                loss_desc = dummy
                loss_sigma = dummy

            # =========================
            # loss weights
            # =========================
            w_sigma = 1.0   # 可以后面做schedule

            # =========================
            # final loss
            # =========================
            loss = loss_desc + w_sigma * loss_sigma
            
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kpnet.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            self.progress_bar.set_description(
                f"[Iter {iter_idx}] "
                f"loss:{loss.item():.4f} "
                f"desc:{loss_desc.item():.4f} "
                f"sigma:{loss_sigma.item():.4f} "       # dual-softmax probability loss
                #f"geo:{loss_geo.item():.4f} "       # geometric / variant descriptor reconstruction 
            )       
            
            # 10. 定期保存 checkpoint
            if (iter_idx+1) % 1000 == 0 and self.cpkt_save_path is not None:
                torch.save(self.kpnet.state_dict(), f"{self.cpkt_save_path}/kpnet_iter_{iter_idx}.pth")
            
            self.progress_bar.update(1)
            
            # -------------------------
            # Log metrics to TensorBoard
            # -------------------------
            self.writer.add_scalar('Loss/total', loss.item(), iter_idx)
            self.writer.add_scalar('Loss/description', loss_desc.item(), iter_idx)   # descriptor Mahalanobis
            self.writer.add_scalar('Loss/sigma', loss_sigma.item(), iter_idx)     # dual-softmax probability loss
            #self.writer.add_scalar('Loss/geo', loss_geo.item(), iter_idx)     # geometric / variant descriptor reconstruction



                
            
            
            


            
            
            
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    cpkt_save_path = "checkpoints/"
    num_iters = 10000
    variance_kpnet = VUDNet(feature_dim=64, dim_geo=32,
                 dim_app=16,
                 pose_dim=16,
                 pose_embed=64)
    trainer = TrainerMultiView(variance_kpnet, data_path, cpkt_save_path, num_iters)
    trainer.train_iters()
"""
feature_dim=128,
                 dim_geo=32,
                 dim_app=16,
                 pose_dim=9,
                 pose_embed=128
"""

        
            