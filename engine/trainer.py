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

from methods.EmbPose.varkpnetmodel import VarianceKPNetModel

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
        
        points,images, cameras, test_images = loadSFM(datapath)
        camera_infos = readColmapCameras(images,cameras, test_images)
        
        self.camera_infos = camera_infos

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
        
        
        # Loss
        self.desc_loss_fn = MultiViewDualSoftmaxLoss()
        self.var_loss_fn = MultiViewVarianceLoss()
        self.recon_loss_fn = MultiViewReconstructionLoss(self.device)
        
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
               
            
            shared_feats, desc_maps, variance_maps, reliability_maps = [], [], [], []
            # 先遍历每一个view ,每一个view 通过一次网络，上面每一个容器都有V 个元素
            for v in range(V):
                sf, var, desc, rel = self.kpnet(images[v].to(self.device))  # images[v] shape = [B,3,608,800]  desc shape = [B,64,152,200]
                shared_feats.append(sf)
                desc_maps.append(desc)
                variance_maps.append(var)
                reliability_maps.append(rel)
                        

            # 接着遍历Batch，每一个batch 计算一次损失函数 以下的容器每个容器B 个元素
            desc_list, var_list, rel_list, sf_list = [], [], [], []

            for b in range(B):
                corrs = multi_corrs[b]   # [N, 5, 2]
                if corrs.shape[0] < 30:
                    continue
                
                ######### 限制点数避免在我的PC 机上爆显存，但是实际中应该要删去 ##########
                # 🔥 限制最大点数
                max_points = 9000
                N = corrs.shape[0]

                if N > max_points:
                    idx = torch.randperm(N)[:max_points]   # 随机采样
                    corrs = corrs[idx]
                
                #############################################################

                desc_per_point = []
                var_per_point = []
                rel_per_point = []
                sf_per_point = []

                # ===== 遍历5个view =====
                for v in range(V):

                    coords = corrs[:, v, :].to(self.device)   # [N,2]

                    # 取第 b 个 batch 的 feature map
                    desc_map_b = desc_maps[v][b:b+1]   # [1,C,H,W]
                    var_map_b  = variance_maps[v][b:b+1]
                    rel_map_b  = reliability_maps[v][b:b+1]
                    sf_map_b   = shared_feats[v][b:b+1]

                    # 采样 → [N,C]
                    desc_sample = sample_map_at_coords(desc_map_b, coords, H_orig, W_orig)
                    var_sample  = sample_map_at_coords(var_map_b, coords, H_orig, W_orig)
                    rel_sample  = sample_map_at_coords(rel_map_b, coords, H_orig, W_orig)
                    sf_sample   = sample_map_at_coords(sf_map_b, coords, H_orig, W_orig)

                    desc_per_point.append(desc_sample)
                    var_per_point.append(var_sample)
                    rel_per_point.append(rel_sample)
                    sf_per_point.append(sf_sample)

                # ===== stack 成 multi-view =====
                # [N, 5, C]
                desc_per_point = torch.stack(desc_per_point, dim=1)
                var_per_point  = torch.stack(var_per_point, dim=1)
                rel_per_point  = torch.stack(rel_per_point, dim=1)
                sf_per_point   = torch.stack(sf_per_point, dim=1)
                

                desc_list.append(desc_per_point)
                var_list.append(var_per_point)
                rel_list.append(rel_per_point)
                sf_list.append(sf_per_point)

            if len(desc_list) == 0:
                print("⚠️ Skip iteration due to too few correspondences")
                continue                  
            # =========================
            # 3. 统一计算 loss
            # =========================
            loss_desc_total = 0.0
            loss_var_total = 0.0
            loss_rel_total = 0.0
            loss_recon_total = 0.0

            valid_batch = 0

            for b in range(len(desc_list)):

                desc = desc_list[b]   # [N, V, C]
                var  = var_list[b]    # [N, V, 1]
                rel  = rel_list[b]    # [N, V, 1]

                if desc.shape[0] == 0:
                    continue

                # =========================
                # 1. descriptor loss
                # =========================
                loss_desc_b, conf = self.desc_loss_fn(desc, var)

                # =========================
                # 2. variance loss（🔥新加）
                # =========================
                loss_var_b = self.var_loss_fn(desc, var)

                # =========================
                # 3. reliability loss
                # =========================
                rel_pred = rel.squeeze(-1)   # [N,V]

                conf = conf.clamp(0.01, 0.99)

                loss_rel_b = F.binary_cross_entropy(rel_pred, conf.detach())
                

                # =========================
                # accumulate
                # =========================
                loss_desc_total += loss_desc_b
                loss_var_total  += loss_var_b
                loss_rel_total  += loss_rel_b

                valid_batch += 1

            # =========================
            # 4. normalize
            # =========================
            if valid_batch > 0:
                loss_desc = loss_desc_total / valid_batch
                loss_var  = loss_var_total  / valid_batch
                loss_rel  = loss_rel_total  / valid_batch
            else:
                loss_desc = 0.0 * next(self.kpnet.parameters()).sum()
                loss_var  = 0.0 * next(self.kpnet.parameters()).sum()
                loss_rel  = 0.0 * next(self.kpnet.parameters()).sum()

            # Reconstruction Loss
            T_list = [T.to(self.device, dtype=torch.float) for T in batch_data['T']]
            loss_recon = self.recon_loss_fn(self.kpnet, desc_list, sf_list, T_list)
           
            # =========================
            # 5. final loss（🔥建议权重）
            # =========================
            if iter_idx < 2000:
                loss = loss_desc   # 先只训 descriptor
            else:
                loss = loss_desc + 0.2 * loss_var + 0.5 * loss_rel + 0.7 * loss_recon
            
            # 🔥 保底（关键）
            loss = loss + 0.0 * next(self.kpnet.parameters()).sum() 
            


            
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kpnet.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            self.progress_bar.set_description(f"[Iter {iter_idx}] loss:{loss.item():.4f} var:{loss_var.item():.4f} ds:{loss_desc.item():.4f} kp:{loss_rel.item():.4f} rec:{loss_recon.item():.4f}")

            # 10. 定期保存 checkpoint
            if (iter_idx+1) % 1000 == 0 and self.cpkt_save_path is not None:
                torch.save(self.kpnet.state_dict(), f"{self.cpkt_save_path}/kpnet_iter_{iter_idx}.pth")
            
            self.progress_bar.update(1)
            
            # Log metrics
            self.writer.add_scalar('Loss/total', loss.item(), iter_idx)
            self.writer.add_scalar('Loss/description', loss_desc.item(), iter_idx)
            self.writer.add_scalar('Loss/recontruction', loss_recon.item(), iter_idx)
            self.writer.add_scalar('Loss/reliability', loss_rel.item(), iter_idx)
            self.writer.add_scalar('Loss/variance', loss_var.item(), iter_idx)



                
            
            
            


            
            
            
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    cpkt_save_path = "checkpoints/"
    num_iters = 5000
    variance_kpnet = VarianceKPNetModel(in_channels=64, pose_dim=9, feature_dim=64, pose_embed=64)
    trainer = TrainerMultiView(variance_kpnet, data_path, cpkt_save_path, num_iters)
    trainer.train_iters()


        
            