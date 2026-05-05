import torch
import torch.nn.functional as F
from modules.utils import sample_map_at_coords

# -------------------------
# Loss functions
# -------------------------

def mv_infonce_masked(f_inv, visibility, tau=0.07):
    """
    f_inv: [N, V, C]
    visibility: [N, V] (bool)
    """
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=2)  # [N,V,C]

    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(i+1, V):
            mask = visibility[:, i] & visibility[:, j]  # [N]
            if mask.sum() < 10:
                continue
            fi = f[mask, i]  # [M, C]
            fj = f[mask, j]  # [M, C]

            # 相似度矩阵 [M, M]
            sim = fi @ fj.t() / tau
            labels = torch.arange(sim.shape[0], device=sim.device)  # 正确匹配在对角线

            loss += F.cross_entropy(sim, labels)
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=f.device, requires_grad=True)
    return loss / count

def compute_p_correct(f_inv, visibility, tau=0.1, eps=1e-6):
    """
    f_inv: [N, V, C]
    visibility: [N, V]
    returns: [N, V] p_correct
    """
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=2)  # [N,V,C]

    p_all = torch.zeros(N, V, device=f.device)
    count = torch.zeros(N, V, device=f.device)

    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]  # [N]
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 10:
                continue

            fi = f[idx, i]  # [M, C]
            fj = f[idx, j]  # [M, C]

            # [M, M] 相似度矩阵
            sim = fi @ fj.t() / tau
            prob = F.softmax(sim, dim=1)
            p = torch.diagonal(prob)  # [M], 正确匹配概率

            # 累加
            p_all[idx, i] += p
            p_all[idx, j] += p
            count[idx, i] += 1
            count[idx, j] += 1

    p_all = p_all / (count + eps)
    return p_all.clamp(eps, 1.0)

def sigma_loss_from_pcorrect(f_inv, sigma_pred, visibility):

    p_correct = compute_p_correct(f_inv, visibility)  # [N,V]

    var_target = (1.0 - p_correct).detach()

    sigma_pred = sigma_pred.squeeze(-1)

    loss = F.mse_loss(sigma_pred, var_target)

    return loss, var_target

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


def reliability_loss_from_confidence(
    rel_pred,
    f_inv,
    visibility,
    topk=128,
    eps=1e-6
):
    """
    Reliability loss based only on matching confidence (without sigma)
    
    Args:
        rel_pred: [N, V, 1] predicted reliability scores
        f_inv: [N, V, C] feature descriptors
        visibility: [N, V] visibility mask
    
    Returns:
        loss: scalar
        target: [N, V] target reliability (matching confidence)
    """
    # Compute matching confidence using the same logic as compute_p_correct
    # but directly compute confidence scores
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=2)

    conf = torch.zeros(N, V, device=f.device)
    count = 0

    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]  # [N]
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 10:
                continue

            fi = f[idx, i]  # [M, C]
            fj = f[idx, j]  # [M, C]

            # [M, M] similarity matrix
            sim = fi @ fj.t()

            k_eff = min(topk, sim.size(1))
            topk_val, topk_idx = torch.topk(sim, k=k_eff, dim=1)

            p_ij = F.softmax(topk_val, dim=1)

            sim_T = sim.t()
            topk_val_T = torch.gather(sim_T, 1, topk_idx.t()).t()
            p_ji = F.softmax(topk_val_T, dim=1)

            conf_ij = (p_ij * p_ji).sum(dim=1)

            conf[idx, i] += conf_ij
            conf[idx, j] += conf_ij
            count += 1

    conf = conf / (count + eps)
    conf = conf.clamp(eps, 1.0)

    # Target is just the confidence score (without sigma)
    target = conf.detach()

    rel_pred = rel_pred.squeeze(-1)

    loss = F.mse_loss(rel_pred, target)

    return loss, target

def compute_matching_confidence(
    f_inv,
    visibility,
    topk=128,
    eps=1e-6
):
    """
    return:
        conf: [N, V]
    """
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=2)

    conf = torch.zeros(N, V, device=f.device)
    count = 0

    for i in range(V):
        for j in range(i + 1, V):
            mask = visibility[:, i] & visibility[:, j]
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 10:
                continue

            fi = f[idx, i]
            fj = f[idx, j]

            sim = fi @ fj.t()

            k_eff = min(topk, sim.size(1))
            topk_val, topk_idx = torch.topk(sim, k=k_eff, dim=1)

            p_ij = F.softmax(topk_val, dim=1)

            sim_T = sim.t()
            topk_val_T = torch.gather(sim_T, 1, topk_idx.t()).t()
            p_ji = F.softmax(topk_val_T, dim=1)

            conf_ij = (p_ij * p_ji).sum(dim=1)

            conf[idx, i] += conf_ij
            conf[idx, j] += conf_ij
            count += 1

    conf = conf / (count + eps)
    conf = conf.clamp(eps, 1.0)

    return conf

def reliability_loss_hybrid(
    rel_pred,
    f_inv,
    visibility,
    alpha=0.4,   # p_correct 权重
    topk=128
):
    """
    Hybrid reliability target:
        rel_target = α * p_correct + (1-α) * confidence
    """

    # ===== 1. strict matching probability =====
    p_correct = compute_p_correct(f_inv, visibility)   # [N,V]

    # ===== 2. soft matching confidence =====
    conf = compute_matching_confidence(
        f_inv,
        visibility,
        topk=topk
    )  # [N,V]

    # ===== 3. hybrid target =====
    rel_target = alpha * p_correct + (1 - alpha) * conf
    rel_target = rel_target.detach()

    # ===== 4. loss =====
    rel_pred = rel_pred.squeeze(-1)
    loss = F.mse_loss(rel_pred, rel_target)

    return loss, rel_target


