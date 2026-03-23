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
class MahalanobisDualSoftmaxLoss(nn.Module):
    def __init__(self, init_temp=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, desc, sigma):
        """
        desc:  [N, V, C]   (f_inv)
        sigma: [N, V, 1]   (variance)
        """

        N, V, C = desc.shape
        desc = F.normalize(desc, dim=2)  # [N,V,C]

        temp = torch.exp(self.log_temp).clamp(0.01, 10.0)

        total_loss = 0.0
        total_pairs = 0
        correct_accum = torch.zeros(N, V, device=desc.device)

        for i in range(V):
            for j in range(i + 1, V):
                fi = desc[:, i, :]   # [N,C]
                fj = desc[:, j, :]   # [N,C]

                si = sigma[:, i, 0]  # [N]
                sj = sigma[:, j, 0]  # [N]

                # -------------------------
                # 🔥 Mahalanobis distance using memory-efficient formula
                # -------------------------
                # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
                fi2 = (fi ** 2).sum(dim=1, keepdim=True)    # [N,1]
                fj2 = (fj ** 2).sum(dim=1, keepdim=True)    # [N,1]

                dist = fi2 + fj2.t() - 2 * (fi @ fj.t())    # [N,N]

                # Mahalanobis normalization
                sigma_ij = si[:, None] + sj[None, :]        # [N,N]
                sigma_ij = sigma_ij.detach().clamp(min=0.05, max=2.0) + 1e-6

                dist = dist / sigma_ij

                # similarity for dual-softmax
                sim = -dist * temp

                # -------------------------
                # Dual Softmax Loss
                # -------------------------
                log_p_ij = F.log_softmax(sim, dim=1)
                log_p_ji = F.log_softmax(sim.t(), dim=1)

                target = torch.arange(N, device=desc.device)

                loss_ij = F.nll_loss(log_p_ij, target)
                loss_ji = F.nll_loss(log_p_ji, target)

                total_loss += (loss_ij + loss_ji)
                total_pairs += 1

                # -------------------------
                # accuracy (optional)
                # -------------------------
                with torch.no_grad():
                    pred_ij = sim.argmax(dim=1)
                    pred_ji = sim.argmax(dim=0)
                    gt = torch.arange(N, device=desc.device)

                    correct_i = (pred_ij == gt).float()
                    correct_j = (pred_ji == gt).float()

                    correct_accum[:, i] += correct_i
                    correct_accum[:, j] += correct_j

        # -------------------------
        # final
        # -------------------------
        loss = total_loss / total_pairs
        conf = correct_accum / (V - 1)
        return loss, conf

class MultiViewReconstructionLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, kpnet, desc, sf, T_list):
        """
        desc: [N,V,C]   descriptor per view
        sf:   [N,V,C]   shared features per view
        T_list: list of V [4,4] transformation matrices
        """
        N, V, C = desc.shape
        loss_total = 0.0
        count = 0

        # 归一化 descriptor
        desc_norm = F.normalize(desc, dim=2)

        for i in range(V):
            for j in range(V):
                if i == j:
                    continue

                desc_i = desc_norm[:, i, :]  # [N,C]
                desc_j = desc_norm[:, j, :]  # [N,C]

                # =========================
                # STEP 1: mutual nearest neighbor matching
                # =========================
                sim = desc_i @ desc_j.t()          # [N,N]
                idx_ij = sim.argmax(dim=1)
                idx_ji = sim.argmax(dim=0)
                mutual_mask = (idx_ji[idx_ij] == torch.arange(N, device=sim.device))

                if mutual_mask.sum() == 0:
                    continue

                # =========================
                # STEP 2: reconstruction
                # =========================
                sf_recon = kpnet.reconstruction(
                    desc_i,
                    T_list[i],
                    T_list[j]
                )  # [N,C]

                # =========================
                # STEP 3: soft matching
                with torch.no_grad():
                    prob = F.softmax(desc_i @ desc_j.t(), dim=1)  # [N,N]
                    sf_j_soft = prob @ sf[:, j, :]                # [N,C]

                # =========================
                # STEP 4: MSE loss on mutual matches
                mask_idx = mutual_mask.nonzero(as_tuple=True)[0]  # [M]
                loss = F.mse_loss(sf_recon[mask_idx], sf_j_soft[mask_idx])
                loss_total += loss
                count += 1

        if count > 0:
            return loss_total / count
        else:
            # 保持梯度，避免 None
            return 0.0 * desc.sum()
        
class ReliabilityLoss(nn.Module):
    """
    Reliability loss: BCE between predicted reliability and matching confidence
    """
    def __init__(self):
        super().__init__()

    def forward(self, rel_pred, conf):
        """
        rel_pred: [N, V, 1] predicted reliability
        conf:     [N, V] matching confidence (from descriptor loss)
        """
        rel_pred = rel_pred.squeeze(-1)
        conf = conf.clamp(0.01, 0.99)
        return F.binary_cross_entropy(rel_pred, conf.detach())

class FeatureVarianceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, desc_var, var_pred):
        """
        desc_var: [N,V,C]
        var_pred: [N,V,1]
        """

        N, V, C = desc_var.shape

        # -------------------------
        # GT from feature variation（🔥核心）
        # -------------------------
        with torch.no_grad():
            desc_norm = F.normalize(desc_var, dim=2)

            var_gt = 0.0
            count = 0
            for i in range(V):
                for j in range(i+1, V):
                    diff = desc_norm[:, i, :] - desc_norm[:, j, :]
                    var_gt += (diff ** 2).sum(dim=1)
                    count += 1

            var_gt = var_gt / count
            var_gt = torch.clamp(var_gt, min=0.05)  # 🔥防 collapse

        # -------------------------
        # prediction
        # -------------------------
        var_pred = var_pred.squeeze(-1).mean(dim=1) + self.eps

        # -------------------------
        # log loss（更稳定）
        # -------------------------
        loss = F.l1_loss(torch.log(var_pred), torch.log(var_gt))

        return loss