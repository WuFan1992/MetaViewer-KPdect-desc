import torch
import torch.nn.functional as F
from modules.utils import sample_map_at_coords

# -------------------------
# Loss functions
# -------------------------
"""
def dual_softmax_loss(f_inv):
    N, V, C = f_inv.shape
    total_loss = 0
    count = 0
    for i in range(V):
        for j in range(i+1, V):
            fi = f_inv[:, i, :]
            fj = f_inv[:, j, :]
            
            #sim = fi @ fj.t()
            temperature = 0.1
            sim = (fi @ fj.t()) / temperature
            
            log_p_ij = F.log_softmax(sim, dim=1)
            log_p_ji = F.log_softmax(sim.t(), dim=1)
            target = torch.arange(N, device=f_inv.device)
            loss = F.nll_loss(log_p_ij, target) + F.nll_loss(log_p_ji, target)
            total_loss += loss
            count += 1
    return total_loss / count
"""
def mv_infonce_masked(f_inv, visibility):
    """
    f_inv: [N, V, C]
    visibility: [N, V] (bool)
    """
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=-1)

    loss = 0
    count = 0

    for i in range(V):
        for j in range(V):
            if i == j:
                continue

            mask = visibility[:, i] & visibility[:, j]  # ✅ 只用共可见点

            if mask.sum() < 10:
                continue

            fi = f[mask, i]
            fj = f[mask, j]

            sim = fi @ fj.t()
            logits = sim / 0.07

            labels = torch.arange(fi.shape[0], device=f.device)

            loss += F.cross_entropy(logits, labels)
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=f.device, requires_grad=True)

    return loss / count

def cosine_variance(feat):
    """
    feat: [N, V, C] (already normalized or not)
    return: [N]
    """

    N, V, C = feat.shape

    feat = F.normalize(feat, dim=-1)

    var = 0.0
    count = 0

    for i in range(V):
        for j in range(i + 1, V):
            sim = (feat[:, i] * feat[:, j]).sum(dim=-1)  # cosine
            var += (1.0 - sim)
            count += 1

    return var / count

def sigma_loss_teacher_multi_view_masked(feat_teacher_list, sigma_list, visibility_list, var_weights=None, device=None):
    """
    feat_teacher_list: list of [N, V, C]
    sigma_list: list of [N, V, 1]
    visibility_list: list of [N, V] (bool)
    var_weights: dict {k: weight}
    device: torch device
    """
    if var_weights is None:
        var_weights = {5:1.0, 4:0.7, 3:0.5, 2:0.3}

    if device is None:
        # 如果列表为空，默认使用 CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss = 0.0
    total_count = 0

    for idx, feat_teacher in enumerate(feat_teacher_list):
        sigma_pred = sigma_list[idx]
        visibility = visibility_list[idx]

        N, V, C = feat_teacher.shape
        f_norm = F.normalize(feat_teacher, dim=-1)
        cos_sim = torch.einsum('nvc,nwc->nvw', f_norm, f_norm)

        k = V
        weight = var_weights.get(k, 1.0)

        loss_subset = 0.0
        count_subset = 0

        for i in range(V):
            for j in range(i+1, V):
                mask = visibility[:, i] & visibility[:, j]
                if mask.sum() < 5:
                    continue

                sim = cos_sim[mask, i, j]
                var_target = 1 - sim

                sigma_i = sigma_pred[mask, i, 0]
                sigma_j = sigma_pred[mask, j, 0]

                loss_subset += F.mse_loss(sigma_i, var_target.detach())
                loss_subset += F.mse_loss(sigma_j, var_target.detach())
                count_subset += 2

        if count_subset > 0:
            total_loss += weight * (loss_subset / count_subset)
            total_count += 1

    if total_count > 0:
        return total_loss / total_count
    else:
        # 列表为空时也返回一个可梯度张量
        return torch.tensor(0.0, device=device, requires_grad=True)

def heatmap_mse_loss(pred, target):
    return F.mse_loss(pred, target)

def heatmap_topk_loss(pred, target, topk=1024):
    """
    pred: [B,1,H,W]
    target: [B,1,H,W]
    """

    B = pred.shape[0]
    loss = 0.0

    for b in range(B):
        p = pred[b].view(-1)
        t = target[b].view(-1)

        if t.sum() < 1e-6:
            continue

        k = min(topk, t.numel())

        idx = torch.topk(t, k=k).indices

        loss_b = -torch.log(p[idx] + 1e-6).mean()
        loss += loss_b

    return loss / B

def heatmap_nms_loss(pred):
    """
    encourage local maxima
    """
    maxpool = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
    return F.l1_loss(pred, maxpool)

def heatmap_loss(pred, target):
    loss_mse = heatmap_mse_loss(pred, target)
    loss_topk = heatmap_topk_loss(pred, target)
    loss_nms = heatmap_nms_loss(pred)

    return loss_mse + 0.5 * loss_topk + 0.1 * loss_nms


def reliability_loss_v2_with_target(
    rel_pred,
    f_inv,
    sigma_sample,
    topk=128,
    eps=1e-6
):
    if sigma_sample.dim() == 3:
        sigma_sample = sigma_sample.squeeze(-1)
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=2)

    conf = torch.zeros(N, V, device=f.device)
    count = 0

    for i in range(V):
        for j in range(i + 1, V):

            sim = f[:, i] @ f[:, j].t()

            k_eff = min(topk, sim.size(1))
            topk_val, topk_idx = torch.topk(sim, k=k_eff, dim=1)

            p_ij = F.softmax(topk_val, dim=1)

            sim_T = sim.t()
            topk_val_T = torch.gather(sim_T, 1, topk_idx.t()).t()
            p_ji = F.softmax(topk_val_T, dim=1)

            conf_ij = (p_ij * p_ji).sum(dim=1)

            conf[:, i] += conf_ij
            conf[:, j] += conf_ij
            count += 1

    conf = conf / (count + eps)
    conf = conf.clamp(eps, 1.0)

    sigma_sample = sigma_sample.squeeze(-1).clamp(0, 1)

    target = conf * torch.exp(-sigma_sample)

    rel_pred = rel_pred.squeeze(-1)

    loss = F.mse_loss(rel_pred, target.detach())

    return loss, target


