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

def variance_loss_multi_view(var_pred, desc_list):
    """
    var_pred: [V,N]
    desc_list: list of descriptors [V,N,C]
    """

    desc_stack = torch.stack(desc_list, dim=0)   # [V,N,C]

    mean_desc = desc_stack.mean(dim=0)

    var_gt = ((desc_stack - mean_desc)**2).sum(-1).mean(0)

    var_gt = var_gt.detach()

    var_pred = var_pred.mean(0)

    loss = var_gt / (var_pred + 1e-6) + torch.log(var_pred + 1e-6)

    return loss.mean()

def soft_patch_matching(desc_i, desc_map_j, coords_j, patch=3):

    B, C = desc_i.shape

    pad = patch // 2

    patches = []

    for dy in range(-pad, pad+1):
        for dx in range(-pad, pad+1):

            offset = coords_j.clone()

            offset[:,0] += dy
            offset[:,1] += dx

            p = sample_map_at_coords(desc_map_j, offset)

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



def cross_view_recon_loss(pred_feat, target_feat):
    loss = F.l1_loss(pred_feat, target_feat)
    return loss


def reconstr_loss(ori, rec):
    diff = ori - rec
    reconstr_loss = torch.norm(diff, dim=2).mean()
    return reconstr_loss

def variance_smooth_loss(var):

    dx = torch.abs(var[:,:,1:,:] - var[:,:,:-1,:])
    dy = torch.abs(var[:,:,:,1:] - var[:,:,:,:-1])

    return dx.mean() + dy.mean()


