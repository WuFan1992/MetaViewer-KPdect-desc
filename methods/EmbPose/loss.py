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

import torch.nn as nn
class MultiViewDualSoftmaxLoss(nn.Module):
    def __init__(self, init_temp=0.2):
        super().__init__()

        # ✅ learnable temperature（用 log 更稳定）
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, desc, var):
        """
        desc: [N, V, C]  (已 or 未 normalize 都可以，内部会 normalize)
        var:  [N, V, 1]

        return:
            loss
            conf: [N, V]
        """

        N, V, C = desc.shape

        # ✅ normalize（非常重要）
        desc = F.normalize(desc, dim=2)

        temp = torch.exp(self.log_temp).clamp(0.01, 10.0)

        total_loss = 0.0
        total_pairs = 0

        conf_accum = torch.zeros(N, V, device=desc.device)

        for i in range(V):
            for j in range(i + 1, V):

                Xi = desc[:, i, :]   # [N, C]
                Xj = desc[:, j, :]   # [N, C]

                var_i = var[:, i, 0]   # [N]
                var_j = var[:, j, 0]   # [N]
                
                # clamp variance
                var_i = torch.clamp(var_i, 1e-3, 1e2)
                var_j = torch.clamp(var_j, 1e-3, 1e2)

                # =========================
                # similarity
                # =========================
                sim = (Xi @ Xj.t()) * temp   # [N, N]

                log_p_ij = F.log_softmax(sim, dim=1)
                log_p_ji = F.log_softmax(sim.t(), dim=1)

                target = torch.arange(N, device=desc.device)

                # =========================
                # ✅ variance weighting
                # =========================
                # 每个点一个 weight
                #weight = 1.0 / (var_i + var_j + 1e-6)   # [N]
                #weight = weight / (weight.mean().detach() + 1e-6)
                
                weight = 1

                loss_ij = F.nll_loss(log_p_ij, target, reduction='none')  # [N]
                loss_ji = F.nll_loss(log_p_ji, target, reduction='none')  # [N]

                loss_ij = (loss_ij * weight).mean()
                loss_ji = (loss_ji * weight).mean()

                total_loss += (loss_ij + loss_ji)
                total_pairs += 1

                # =========================
                # confidence（给 reliability 用）
                # =========================
                with torch.no_grad():
                    p_ij = torch.exp(log_p_ij)
                    p_ji = torch.exp(log_p_ji)

                    conf_i = p_ij.max(dim=1)[0]   # [N]
                    conf_j = p_ji.max(dim=1)[0]

                    conf_accum[:, i] += conf_i
                    conf_accum[:, j] += conf_j

        loss = total_loss / total_pairs

        conf = conf_accum / (V - 1)

        return loss, conf

class MultiViewVarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, desc, var):
        """
        desc: [N, V, C]
        var:  [N, V, 1]
        """

        desc = F.normalize(desc, dim=2)
        
        desc_detached = desc.detach()

        # ===== multi-view mean =====
        mu = desc_detached.mean(dim=1, keepdim=True)
        error = ((desc_detached - mu) ** 2).sum(dim=2)

        # ===== variance =====
        var = var.squeeze(-1)   # [N,V]
        var = torch.clamp(var, 1e-3, 1e2)

        # ===== loss =====
        loss = error / var + torch.log(var)

        return loss.mean()
    
class MultiViewReconstructionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, kpnet, desc_list, sf_list, T_list):
        """
        desc_list: list of [N, V, C]
        sf_list: list of [N, V, C]
        T_list: list of V torch tensors [4,4] (相对 pose)
        """
        loss_total = 0.0
        count = 0

        for b in range(len(desc_list)):
            desc = desc_list[b]  # [N,V,C]
            sf   = sf_list[b]    # [N,V,C]
            V    = desc.shape[1]

            for i in range(V):
                for j in range(V):
                    if i == j:
                        continue
                    # i -> j
                    # kpnet 重建 j 的 sf
                    sf_recon = kpnet.reconstruction(desc[:,i,:], T_list[i], T_list[j])  # [N,C]
                    sf_target = sf[:,j,:]  # [N,C]

                    loss_total += F.mse_loss(sf_recon, sf_target)
                    count += 1

        if count > 0:
            return loss_total / count
        else:
            return 0.0 * next(kpnet.parameters()).sum()


