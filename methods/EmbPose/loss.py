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

def sigma_loss_teacher_multi_view_masked(feat_teacher, sigma_pred, visibility):
    """
    feat_teacher: [N,V,C]
    sigma_pred: [N,V,1]
    visibility: [N,V]
    """
    f_norm = F.normalize(feat_teacher, dim=-1)

    cos_sim = torch.einsum('nvc,nwc->nvw', f_norm, f_norm)  # [N,V,V]

    loss = 0
    count = 0

    V = feat_teacher.shape[1]

    for i in range(V):
        for j in range(i+1, V):
            mask = visibility[:, i] & visibility[:, j]

            if mask.sum() < 10:
                continue

            sim = cos_sim[mask, i, j]

            var_target = 1 - sim

            sigma_i = sigma_pred[mask, i, 0]
            sigma_j = sigma_pred[mask, j, 0]

            loss += F.mse_loss(sigma_i, var_target.detach())
            loss += F.mse_loss(sigma_j, var_target.detach())

            count += 2

    if count == 0:
        return torch.tensor(0.0, device=feat_teacher.device, requires_grad=True)

    return loss / count

def geo_loss(kpnet, f_geo, T_list, batch_idx):
    """
    f_geo: [N, V, C]
    T_list: list of V elements, each [B,4,4]
    batch_idx: 当前样本 index
    """

    N, V, C = f_geo.shape

    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(V):
            if i == j:
                continue

            # --- 取当前 batch 的 pose ---
            T_i = T_list[i][batch_idx]  # [4,4]
            T_j = T_list[j][batch_idx]  # [4,4]

            # --- feature ---
            f_i = f_geo[:, i, :]  # [N,C]
            f_j = f_geo[:, j, :]  # [N,C]

            # --- transform ---
            pred_j = kpnet.transform_geo(f_i, T_i, T_j)
            
            # --- normalize (optional) ---
            pred_j = F.normalize(pred_j, dim=1)
            f_j = F.normalize(f_j, dim=1)

            # --- loss ---
            loss += F.mse_loss(pred_j, f_j.detach())

            count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=f_geo.device)



def reliability_loss(rel_pred, f_inv):
    """
    rel_pred: [N,V,1] predicted reliability
    f_inv:    [N,V,C] descriptor, 用 dual-softmax 生成匹配 confidence
    """
    N, V, C = f_inv.shape
    with torch.no_grad():
        # 生成匹配概率
        conf = torch.zeros(N, V, device=f_inv.device)
        for i in range(V):
            for j in range(V):
                if i == j: continue
                sim = F.normalize(f_inv[:, i, :], dim=1) @ F.normalize(f_inv[:, j, :], dim=1).t()
                conf[:, i] += F.softmax(sim, dim=1).diag()
        conf = conf / (V - 1)
        conf = conf.clamp(0.01, 0.99)
    rel_pred = rel_pred.squeeze(-1)
    return F.binary_cross_entropy(rel_pred, conf)