import torch
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import glob
from matplotlib.patches import Circle
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


import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


def visualize_group_points(imgs, coords, title=""):
    """
    imgs: list of [3,H,W] tensor, 每张图
    coords: list of [2] tensor, 每张图对应的一个点 [y,x]
    """
    V = len(imgs)
    fig, axes = plt.subplots(1, V, figsize=(4*V, 4))
    
    if V == 1:
        axes = [axes]
    
    for i in range(V):
        img = imgs[i].cpu().permute(1,2,0).numpy()  # CHW -> HWC
        # 反归一化（ImageNet）
        img = img * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        
        yx = coords[i].cpu().numpy()
        x, y = yx[0], yx[1]
        axes[i].scatter(x, y, s=80, c='r', marker='x')
        axes[i].set_title(f"Image {i}")
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.show()

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
            
                
            # ===== 1. 生成 multi-view 对应点 =====
            multi_corrs = spvs_coarse_multi_cycle(batch_data, scale=4)

            # ===== 2. 可视化=====
            # ===== 关键：正确取 batch 第一个样本 =====
            images = []
            for img in batch_data['images']:
                if isinstance(img, torch.Tensor) and img.dim() == 4:
                    images.append(img[0])   # [B,C,H,W] -> [C,H,W]
                else:
                    images.append(img)

            # ===== 调用可视化 =====
            plot_multi_view_matches(
                images,
                multi_corrs,
                max_points=50,
                save_path=None  # 或 f"debug_{iter_idx}.png"
            )
            

            # 读取 image_ids 和 coords
            all_image_ids = [b['image_ids'].to(self.device) for b in batch_data]  # [V, num_sample]
            all_coords = [b['coords'].to(self.device) for b in batch_data]        # [V, num_sample,2]
            """
            print("all_imag_ids[0] = ", all_image_ids[0])
            image_ids = all_image_ids[0].cpu().numpy()
            names = [self.camera_infos["image_name_list"][i] for i in image_ids]
            print("all_image_ids 0 names =", names)
            """
            # 读取图像并计算 descriptor
            imgs = []
            poses = []
            raw_imgs = [] 
            for b in range(B):
                img_batch = []
                pose_batch = []
                raw_img_batch=[]
                for img_id in all_image_ids[b]:
                    img = self.dataset.get_img(img_id.item()).to(self.device)
                    pose = self.dataset.get_pose(img_id).to(self.device)
                    raw_img =self.dataset.get_raw_img(img_id.item()).to(self.device) 
                    img_batch.append(img)
                    pose_batch.append(pose)
                    raw_img_batch.append(raw_img)
                imgs.append(torch.stack(img_batch, dim=0))  # [num_sample, C,H,W]
                poses.append(torch.stack(pose_batch, dim=0))  # [num_sample, pose_dim]
                raw_imgs.append(torch.stack(raw_img_batch, dim=0))
            ################################################## 
            """
            from matplotlib.patches import Circle   


            # 创建一个 figure，每张图片一个子图
            fig, axes = plt.subplots(1, B, figsize=(5*B, 5))

            if B == 1:
                axes = [axes]  # 保证是 list

            for i in range(5):
                img = raw_imgs[0][i].permute(1,2,0).cpu().numpy()  # H x W x C
                axes[i].imshow(img)
                axes[i].axis('off')
    
                # 在对应 coords 上画圆圈
                x, y = all_coords[0][i].cpu().numpy()
                circ = Circle((x, y), radius=10, edgecolor='red', facecolor='none', linewidth=2)
                axes[i].add_patch(circ)

            plt.show()
            """
            #############################################################       
            H_orig, W_orig = imgs[0].shape[2:]     
            # 假设 kpnet.forward(imgs) 返回 shared_feats, desc_maps, variance_maps, reliability_maps
            shared_feats, desc_maps, variance_maps, reliability_maps = [], [], [], []
            for b in range(B):
                sf, var, desc, rel = self.kpnet(imgs[b])  # imgs[v]: [num_sample,C,H,W]
                shared_feats.append(sf)
                desc_maps.append(desc)
                variance_maps.append(var)
                reliability_maps.append(rel)


            # 对每个 3D 点采样 descriptor & variance
            # 这里假设每个 3D 点在不同图像上对应 coords
            desc_list, var_list, rel_list, sf_list = [], [], [], []
            for b in range(B):
                # sample_map_at_coords(desc_map, coords) 返回 [num_sample, C]
                desc_list.append(sample_map_at_coords(desc_maps[b], all_coords[b], H_orig, W_orig))
                var_list.append(sample_map_at_coords(variance_maps[b], all_coords[b], H_orig, W_orig))
                rel_list.append(sample_map_at_coords(reliability_maps[b], all_coords[b], H_orig, W_orig))
                sf_list.append(sample_map_at_coords(shared_feats[b], all_coords[b], H_orig, W_orig))

        
            # Multi-view soft patch matching
            loss_ds, conf_all = 0, []
            num_sample = desc_list[0].shape[0]
            desc_tensor = torch.stack(desc_list, dim=0)
            var_tensor = torch.stack(var_list, dim=0)
            rel_tensor = torch.stack(rel_list, dim=0)
            coords_tensor = torch.stack(all_coords, dim=0)
            desc_map_tensor = torch.stack(desc_maps, dim=0)
            poses_tensor = torch.stack(poses, dim=0)
            sf_tensor = torch.stack(sf_list, dim=0)   # [B,V,C]
            
                       
            loss_inv = descriptor_invariance_loss(
                desc_tensor,
                var_tensor,
                var_thresh=0.1
            )
            

            loss_ds, conf_all = viewpoint_guide_ds_loss(desc_tensor, desc_map_tensor, coords_tensor, H_orig, W_orig, patch)
            conf_target = torch.stack(conf_all).mean(0).detach()

            # Reliability supervision
            loss_kp = sum([reliability_loss(rel_tensor[:,n], conf_target) for n in range(num_sample)]) / num_sample
            

            
             # ===== viewpoint-guided variance supervision =====
            desc_stack = desc_tensor.permute(1,0,2)  # [V,B,C]
            var_gt = compute_multi_view_variance(desc_stack)  # [B]
            var_pred = var_tensor.mean(1).squeeze(-1)  # [B]
            # 1. variance loss (只在 warm-up 后开启)
            if iter_idx >0: # warmup
                loss_var = F.smooth_l1_loss(var_pred, var_gt)
            else:
                loss_var = torch.tensor(0.0, device=self.device)


            # Reconstruction loss
            # ===== cross-view reconstruction =====
            loss_rec = 0.0
            _, V, _ = desc_tensor.shape
            
            
            for v in range(V):
                for u in range(V):
                    if u == v:
                        continue

                    # source descriptor: batch 内所有样本，第 u 个 view
                    desc_src = desc_tensor[:, u]  # [B, C]
                    sf_tgt = sf_tensor[:, v].detach()  # [B, C]

                    # 加入一点点噪声 ，防止学习decoder(x) = x
                    desc_src = desc_src + 0.02 * torch.randn_like(desc_src)

                    # cross-view reconstruction
                    pred_desc = self.kpnet.reconstruction(
                        desc_src=desc_src,
                        pose_src=poses_tensor[:, u],  # [B, 4, 4]
                        pose_tgt=poses_tensor[:, v]   # [B, 4, 4]
                    )

                    # cosine reconstruction loss with variance weighting
                    var_weight = torch.exp(-var_tensor[:, v, 0])  # [B]
                    loss_rec += (var_weight * (1 - F.cosine_similarity(pred_desc, sf_tgt, dim=1))).mean()

            # 平均到 V*(V-1) 个 view pair
            loss_rec /= V*(V-1)
            
            # 总 loss
            w_var, w_ds, w_kp, w_rec, w_inv = 0.2, 0.1, 5, 0.5, 1.0

            loss = (
                w_var * loss_var +
                w_ds  * loss_ds  +
                w_kp  * loss_kp  +
                w_rec * loss_rec +
                w_inv * loss_inv
            )
            
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.kpnet.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            self.progress_bar.set_description(f"[Iter {iter_idx}] loss:{loss.item():.4f} inv:{loss_inv.item():.4f} var:{loss_var.item():.4f} ds:{loss_ds.item():.4f} kp:{loss_kp.item():.4f} rec:{loss_rec.item():.4f}")

            # 10. 定期保存 checkpoint
            if (iter_idx+1) % 2000 == 0 and self.cpkt_save_path is not None:
                torch.save(self.kpnet.state_dict(), f"{self.cpkt_save_path}/kpnet_iter_{iter_idx}.pth")
            
            self.progress_bar.update(1)
            
            # Log metrics
            self.writer.add_scalar('Loss/total', loss.item(), iter_idx)
            self.writer.add_scalar('Loss/description', loss_ds.item(), iter_idx)
            self.writer.add_scalar('Loss/recontruction', loss_rec.item(), iter_idx)
            self.writer.add_scalar('Loss/reliability', loss_kp.item(), iter_idx)
            self.writer.add_scalar('Loss/variance', loss_var.item(), iter_idx)



                
            
            
            


            
            
            
                
            

                
        

if __name__ == "__main__":
    data_path = "datasets/head"
    cpkt_save_path = "checkpoints/"
    num_iters = 2000
    variance_kpnet = VarianceKPNetModel(in_channels=64, pose_dim=9, feature_dim=64, pose_embed=64)
    trainer = TrainerMultiView(variance_kpnet, data_path, cpkt_save_path, num_iters)
    trainer.train_iters()


        
            