import torch
import torch.nn.functional as F
from modules.utils import sample_map_at_coords

def pose_difference(T1, T2, alpha=1.0, beta=1.0):
    """
    T1, T2: [4,4]  SE(3) matrices
    Returns: scalar pose difference
    """
    # 确保是 float tensor
    T1 = T1.float()
    T2 = T2.float()
    # 旋转部分
    R1 = T1[:3, :3]  # [3,3]
    t1 = T1[:3, 3]   # [3]
    R2 = T2[:3, :3]
    t2 = T2[:3, 3]

    # 相对旋转
    R_rel = torch.matmul(R1, R2.T)  # [3,3]
    trace = R_rel[0,0] + R_rel[1,1] + R_rel[2,2]  # scalar
    theta = torch.acos(torch.clamp((trace - 1)/2, -1+1e-6, 1-1e-6))  # radians

    # 平移差异
    d_t = torch.norm(t1 - t2)  # scalar

    # 综合位姿差异
    pose_diff = alpha * theta + beta * d_t  # scalar

    return pose_diff



def soft_patch_matching(desc_i, desc_map_j, coords_j, H_ori, W_ori,patch=3):

    B, C = desc_i.shape

    pad = patch // 2

    patches = []
    for dy in range(-pad, pad+1):
        for dx in range(-pad, pad+1):

            offset = coords_j.clone()

            offset[:,0] += dy
            offset[:,1] += dx

            p = sample_map_at_coords(desc_map_j, offset, H_ori, W_ori)
            patches.append(p)
    patch_desc = torch.stack(patches, dim=1)  # [B,K,C]

    sim = torch.einsum("bc,bkc->bk", desc_i, patch_desc)

    prob = F.log_softmax(sim, dim=1)

    center = patch_desc.shape[1] // 2

    loss = -prob[:,center].mean()

    conf = prob.exp().max(-1)[0]
    

    return loss, conf


# 注意这里的输入发生变化，不再时target 而时confidence
def reliability_loss(pred, conf):

    conf = conf.detach()

    conf = (conf > 0.5).float()

    return F.binary_cross_entropy(pred.squeeze(), conf)


def viewpoint_guide_ds_loss(desc_tensor, desc_map_tensor, coords_tensor, H_orig, W_orig, patch):
    """
    desc_tensor: [B, V, C] - descriptors for each view
    desc_map_tensor: [B, V, C, H, W] - descriptor maps
    coords_tensor: [B, V, 2] - keypoint coordinates
    H_orig, W_orig: 原图大小
    patch: patch size
    """
    B, V, C = desc_tensor.shape
    loss_ds = 0.0
    conf_all = []

    # 遍历每个 view pair 在 batch 内
    for v in range(V):
        for u in range(V):
            if u == v:
                continue

            # 取 batch 内 B 个样本
            desc_src = desc_tensor[:, u]        # [B, C]
            desc_tgt_map = desc_map_tensor[:, v]  # [B, C, H, W]
            coords_tgt = coords_tensor[:, v]    # [B, 2]

            # soft patch matching
            l, conf = soft_patch_matching(desc_src, desc_tgt_map, coords_tgt,
                                          H_orig, W_orig, patch=patch)
            loss_ds += l
            conf_all.append(conf)

    # 平均到 V*(V-1) 个 view pair
    loss_ds /= V*(V-1)

    return loss_ds, conf_all

                

def cross_view_recon_loss(backbone_pred, target_feat, eps=1e-6):
    """
    backbone_pred: [B, C]
    target_feat:  [B, C]
    """
    # L2 normalize
    pred_norm = F.normalize(backbone_pred, dim=1)
    target_norm = F.normalize(target_feat, dim=1)
    
    # 余弦相似度
    cos_sim = (pred_norm * target_norm).sum(dim=1)  # [B]
    
    # loss = 1 - mean cosine similarity
    loss = 1 - cos_sim.mean()
    return loss



def variance_loss(desc_list, var_list):
    """
    desc_list: list of [N,C]
    var_list:  list of [N,1]
    """

    V = len(desc_list)

    loss = 0
    count = 0

    for i in range(V):
        for j in range(V):

            if i == j:
                continue

            desc_i = desc_list[i].detach()
            desc_j = desc_list[j].detach()

            var = var_list[i].squeeze()

            dist = ((desc_i - desc_j)**2).sum(-1)

            loss += (dist / (var + 1e-6) + torch.log(var + 1e-6)).mean()

            count += 1

    return loss / count

def variance_smooth_loss(var):

    dx = torch.abs(var[:,:,1:,:] - var[:,:,:-1,:])
    dy = torch.abs(var[:,:,:,1:] - var[:,:,:,:-1])

    return dx.mean() + dy.mean()

def compute_relative_view_variance(poses_src, poses_tgt, scale_angle=1.0):
    """
    poses_src, poses_tgt: [B,4,4] 相机pose矩阵
    return: var_gt_view: [B], 用作variance supervision的目标
    """
    R_src = poses_src[:, :3, :3]   # [B,3,3]
    R_tgt = poses_tgt[:, :3, :3]   # [B,3,3]

    # 相对旋转矩阵
    R_rel = torch.matmul(R_tgt, R_src.transpose(1,2))  # [B,3,3]

    # 计算旋转角度 acos((trace(R)-1)/2)
    trace = R_rel[:,0,0] + R_rel[:,1,1] + R_rel[:,2,2]  # [B]
    angle = torch.acos(torch.clamp((trace - 1)/2, -1.0, 1.0))  # [B]

    # scale成合适范围
    var_gt_view = scale_angle * angle

    return var_gt_view  # [B]

def viewpoint_guided_variance_loss(var_tensor, poses_tensor):
    """
    var_tensor: [B, V, 1]   # variance predictions
    poses_tensor: [B, V, 4, 4]  # camera poses for each view in batch
    """
    B, V, _ = var_tensor.shape
    loss_var = 0.0
    num_pairs = 0.0

    for v in range(V):
        for u in range(V):
            if u == v:
                continue

            # 每个 batch 内 u,v pair 的相对旋转角度
            var_gt_view = compute_relative_view_variance(
                poses_src=poses_tensor[:, u],  # [B,4,4]
                poses_tgt=poses_tensor[:, v],  # [B,4,4]
                scale_angle=1.0
            )  # [B]

            # 取预测的 variance
            var_pred = torch.clamp(var_tensor[:, u, 0], min=1e-3)  # 避免log(0)
            
            # heteroscedastic loss 对 batch 平均
            pair_loss = (var_gt_view / var_pred + torch.log(var_pred)).mean()
            #print("pair_loss = ", pair_loss)
            loss_var += pair_loss
            num_pairs+=1

    # 对 V*(V-1) 个 view pair 做平均
    loss_var = loss_var/num_pairs

    return loss_var
    
def fused_variance_loss(self, desc_list, var_list, poses, V, w_desc=0.1):
    """
    计算融合 viewpoint-guided 和 descriptor-guided 的 variance loss
    
    desc_list: list of [N, C] 每个 view 的 descriptor
    var_list: list of [N, 1] 每个 view 的预测 variance
    poses: list of [B, 4, 4] 每个 view 的相机 pose
    V: view 数量
    w_desc: descriptor-based variance 权重
    """
    loss_view = 0
    loss_desc = 0
    
    for v in range(V):
        for u in range(V):
            if u == v:
                continue
            
            # ----- 1. viewpoint-guided variance -----
            var_gt_view = compute_relative_view_variance(
                poses_src=poses[u],
                poses_tgt=poses[v],
                scale_angle=1.0
            )  # [N]
            var_pred = var_list[u].squeeze()  # [N]
            loss_view += (var_gt_view / (var_pred + 1e-6) + torch.log(var_pred + 1e-6)).mean()
            
            # ----- 2. descriptor-guided variance -----
            desc_gt_diff = ((desc_list[u] - desc_list[v])**2).sum(-1)  # L2 差 [N]
            loss_desc += (desc_gt_diff / (var_pred + 1e-6) + torch.log(var_pred + 1e-6)).mean()
    
    # 归一化
    loss_view /= V*(V-1)
    loss_desc /= V*(V-1)
    
    # 总 loss
    loss_var = loss_view + w_desc * loss_desc
    return loss_var


