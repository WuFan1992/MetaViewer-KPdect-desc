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


def plot_multi_view_matches(images, multi_corrs, max_points=50, save_path=None):
    """
    images: list of 5 images
    multi_corrs: [N, 5, 2]
    """

    # ===== 1. 转 numpy + 标准格式 =====
    imgs = []
    for img in images:
        img = to_numpy_image(img)
        imgs.append(img)

    # ===== debug（非常建议保留）=====
    for i, img in enumerate(imgs):
        print(f"[DEBUG] img[{i}] shape:", img.shape)

    # ===== 2. 处理匹配点 =====
    if isinstance(multi_corrs, torch.Tensor):
        multi_corrs = multi_corrs.detach().cpu().numpy()

    if multi_corrs is None or len(multi_corrs) == 0:
        print("No correspondences")
        return

    N = multi_corrs.shape[0]

    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        multi_corrs = multi_corrs[idx]

    # ===== 3. 拼接图像 =====
    heights = [img.shape[0] for img in imgs]
    widths = [img.shape[1] for img in imgs]

    H = max(heights)
    W = sum(widths)

    canvas = np.zeros((H, W, 3), dtype=imgs[0].dtype)

    offsets = []
    cur_w = 0

    for img in imgs:
        h, w = img.shape[:2]
        canvas[:h, cur_w:cur_w+w] = img
        offsets.append(cur_w)
        cur_w += w

    # ===== 4. 绘制 =====
    plt.figure(figsize=(15, 5))
    plt.imshow(canvas)

    colors = plt.cm.jet(np.linspace(0, 1, len(multi_corrs)))

    for i, pts in enumerate(multi_corrs):

        color = colors[i]

        # 画点
        for j in range(5):
            x, y = pts[j]
            plt.scatter(x + offsets[j], y, c=[color], s=20)

        # 画轨迹线
        for j in range(4):
            x1, y1 = pts[j]
            x2, y2 = pts[j+1]

            plt.plot(
                [x1 + offsets[j], x2 + offsets[j+1]],
                [y1, y2],
                c=color,
                linewidth=1
            )

    plt.axis('off')
    plt.title("Multi-view Correspondences (Tracks)")

    # ===== 5. 显示 or 保存 =====
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


def sfm_collate_fn(batch):
    return batch  # batch 是 list，每个元素就是 dict

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

            # ===== 2. 可视化=====
            # ===== 关键：正确取 batch 第一个样本 =====
            """
            images = []
            for img in batch_data['images']:
                if isinstance(img, torch.Tensor) and img.dim() == 4:
                    images.append(img[0])   # [B,C,H,W] -> [C,H,W]
                else:
                    images.append(img)
            # multi_corrs 也取 batch=0
            multi_corrs_b = multi_corrs[0]  # [num_points, 5, 2]
            # ===== 调用可视化 =====
            plot_multi_view_matches(
                images,
                multi_corrs_b,
                max_points=50,
                save_path=None  # 或 f"debug_{iter_idx}.png"
            )
            """
            
               
            
             # ===== forward each view =====
            shared_feats, f_inv_maps, f_geo_maps, f_app_maps, sigma_maps, reliability_maps = [], [], [], [], [], []
            for v in range(V):
                out = self.kpnet(images[v].to(self.device))
                shared_feats.append(out["shared"])
                f_inv_maps.append(out["f_inv"])
                f_geo_maps.append(out["f_geo"])
                f_app_maps.append(out["f_app"])
                sigma_maps.append(out["sigma"])
                reliability_maps.append(out["reliability"])

            # ===== sample multi-view features =====
            f_inv_list, f_geo_list, f_app_list, rel_list, sf_list, sigma_list = [], [], [], [], [],[]

            for b in range(B):
                corrs = multi_corrs[b]  # [N,V,2]
                if corrs.shape[0] < 30:
                    continue

                # 限制最大点数
                max_points = 5000
                N = corrs.shape[0]
                if N > max_points:
                    idx = torch.randperm(N)[:max_points]
                    corrs = corrs[idx]  # 对 coords 采样
                else:
                    idx = torch.arange(N)


                f_inv_per_point, f_geo_per_point, rel_per_point, sf_per_point, f_app_per_point, sigma_per_point = [], [], [], [], [],[]

                for v in range(V):
                    coords = corrs[:, v, :].to(self.device)

                    f_inv_sample = sample_map_at_coords(f_inv_maps[v][b:b+1], coords, H_orig, W_orig)
                    f_geo_sample = sample_map_at_coords(f_geo_maps[v][b:b+1], coords, H_orig, W_orig)
                    f_app_sample = sample_map_at_coords(f_app_maps[v][b:b+1], coords, H_orig, W_orig)
                    sigma_sample = sample_map_at_coords(sigma_maps[v][b:b+1], coords, H_orig, W_orig)
                    rel_sample      = sample_map_at_coords(reliability_maps[v][b:b+1], coords, H_orig, W_orig)
                    sf_sample       = sample_map_at_coords(shared_feats[v][b:b+1], coords, H_orig, W_orig)

                    f_inv_per_point.append(f_inv_sample)
                    f_geo_per_point.append(f_geo_sample)
                    f_app_per_point.append(f_app_sample)
                    sigma_per_point.append(sigma_sample)
                    sf_per_point.append(sf_sample)
                    rel_per_point.append(rel_sample)
                    


                # stack
                f_inv = torch.stack(f_inv_per_point, dim=1)   # [N,V,C]
                f_geo = torch.stack(f_geo_per_point, dim=1)
                f_app = torch.stack(f_app_per_point, dim=1)
                sigma = torch.stack(sigma_per_point, dim=1)
                shared = torch.stack(sf_per_point, dim=1)
                rel_per_point  = torch.stack(rel_per_point, dim=1)
                
                f_inv_list.append(f_inv)
                f_geo_list.append(f_geo)
                f_app_list.append(f_app)
                rel_list.append(rel_per_point)
                sf_list.append(shared)
                sigma_list.append(sigma)


            if len(f_inv_list) == 0:
                print("⚠️ Skip iteration due to too few correspondences")
                continue

            # =========================
            # compute losses
            # =========================
            loss_desc_total = 0.0
            loss_prob_total = 0.0
            loss_ortho_total = 0.0
            loss_geo_total = 0.0
            loss_recon_total = 0.0
            loss_cross_total = 0.0
            loss_rel_total = 0.0
            valid_batch = 0

            for b in range(len(f_inv_list)):
                f_inv = f_inv_list[b]
                f_geo = f_geo_list[b]
                f_app = f_app_list[b]
                sigma = sigma_list[b]
                rel_pred   = rel_list[b]          # [N,V,1]
                shared       = sf_list[b]       # [N,V,C]

    
                if f_inv.shape[0] == 0:
                    continue
                
                # descriptor loss
                loss_desc_b = dual_softmax_loss(f_inv)
                

                # probabilistic loss
                loss_prob_b = probabilistic_loss(f_inv, sigma)

                # orthogonality
                loss_ortho_geo = orthogonality_loss(f_inv, f_geo)
                loss_ortho_app = orthogonality_loss(f_inv, f_app)
                loss_ortho_total_b = 0.5 * (loss_ortho_geo + loss_ortho_app)
                

                # geometry loss
                T_list = [T.to(self.device, dtype=torch.float) for T in batch_data['T']]
                loss_geo_b = geo_loss(self.kpnet, f_geo, T_list, batch_idx=b)

                # reconstruction loss
                loss_recon_b = recon_loss(self.kpnet, f_inv, f_geo, f_app, shared)

                # cross-view
                loss_cross_b = cross_view_loss(
                        self.kpnet, f_inv, f_geo, f_app, shared, T_list, batch_idx=b
                )
                
                # reliability loss
                loss_rel_b = reliability_loss(rel_pred, f_inv)

                # accumulate
                loss_desc_total += loss_desc_b
                loss_prob_total += loss_prob_b
                loss_ortho_total += loss_ortho_total_b
                loss_geo_total += loss_geo_b
                loss_recon_total += loss_recon_b
                loss_cross_total += loss_cross_b
                loss_rel_total += loss_rel_b
                valid_batch += 1

            # normalize
            if valid_batch > 0:
                loss_desc = loss_desc_total / valid_batch
                loss_prob = loss_prob_total / valid_batch
                loss_ortho = loss_ortho_total / valid_batch
                loss_geo = loss_geo_total / valid_batch
                loss_recon = loss_recon_total / valid_batch
                loss_cross = loss_cross_total / valid_batch
                loss_rel  = loss_rel_total / valid_batch
                
            else:
                dummy = 0.0 * next(self.kpnet.parameters()).sum()
                loss_desc = loss_prob = loss_ortho = loss_geo = loss_recon = loss_cross = loss_rel = dummy

            # -------------------------
            # final weighted loss (smooth schedule)
            # -------------------------
            w_geo = torch.sigmoid(torch.tensor((iter_idx - 2000)/500, device=self.device))
            w_rec = torch.sigmoid(torch.tensor((iter_idx - 3000)/500, device=self.device))
            w_cross = torch.sigmoid(torch.tensor((iter_idx - 3500)/500, device=self.device))


            loss = (
                loss_desc + 
                0.5 * loss_prob +
                50.0 * w_geo * loss_geo +
                1.0 * w_rec * loss_recon +
                1.0 * w_cross * loss_cross +
                20.0 * loss_ortho +
                10.0 * loss_rel
            )

            ######################
            sim = f_inv[:,0,:] @ f_inv[:,1,:].t()
            gt_sim = sim.diag().mean()
            neg_sim = sim.mean()
            print("pos:", gt_sim.item(), "neg:", neg_sim.item())
            #####################
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
                f"prob:{loss_prob.item():.4f} "       # dual-softmax probability loss
                f"geo:{loss_geo.item():.4f} "       # geometric / variant descriptor reconstruction
                f"rec:{loss_recon.item():.4f} "             # reconstruction loss
                f"cross:{loss_cross.item():.4f} "             # feature variance / cross-view
                f"ortho:{loss_ortho.item():.4f} "           # orthogonality loss
                f"rel:{loss_rel.item():.4f}"                # reliability BCE
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
            self.writer.add_scalar('Loss/reconstruction', loss_recon.item(), iter_idx) # reconstruction loss (variant->shared)
            self.writer.add_scalar('Loss/reliability', loss_rel.item(), iter_idx)    # reliability BCE
            self.writer.add_scalar('Loss/variance',loss_cross.item(), iter_idx)       # feature variance loss
            self.writer.add_scalar('Loss/orthogonality', loss_ortho.item(), iter_idx) # orthogonality between inv/var
            self.writer.add_scalar('Loss/prob', loss_desc_total.item(), iter_idx)     # dual-softmax probability loss
            self.writer.add_scalar('Loss/geo', loss_recon_total.item(), iter_idx)     # geometric / variant descriptor reconstruction




                
            
            
            


            
            
            
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    cpkt_save_path = "checkpoints/"
    num_iters = 30000
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

        
            