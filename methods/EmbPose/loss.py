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

                

def compute_multi_view_variance(desc_stack):
    """
    desc_stack: [V, N, C]
    """

    V, N, C = desc_stack.shape

    dist_sum = 0
    count = 0

    for i in range(V):
        for j in range(i+1, V):

            dist = ((desc_stack[i] - desc_stack[j])**2).sum(-1) / C  # [N]

            dist_sum += dist
            count += 1

    var_gt = dist_sum / count

    return var_gt.detach()

def descriptor_invariance_loss(desc_tensor, var_tensor=None, var_thresh=0.1):
    B, V, C = desc_tensor.shape

    # L2 normalize descriptor + 微小噪声
    desc_tensor = F.normalize(desc_tensor + 1e-3 * torch.randn_like(desc_tensor), dim=2)

    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(i+1, V):
            cos = F.cosine_similarity(desc_tensor[:, i], desc_tensor[:, j], dim=1)  # [B]
            pair_loss = 1 - cos  # [B]

            # 初始阶段去掉 mask
            if var_tensor is not None:
                var = 0.5 * (var_tensor[:, i, 0] + var_tensor[:, j, 0])
                mask = torch.sigmoid(10 * (var_thresh - var))  # [0,1]
                pair_loss = pair_loss * mask

            loss += pair_loss.sum()  # 用 sum 而不是 mean
            count += B

    loss /= count
    return loss
    



