import torch
import torch.nn.functional as F
from modules.utils import sample_map_at_coords

# -------------------------
# Loss functions
# -------------------------
def dual_softmax_loss(f_inv):
    N, V, C = f_inv.shape
    total_loss = 0
    count = 0
    for i in range(V):
        for j in range(i+1, V):
            fi = f_inv[:, i, :]
            fj = f_inv[:, j, :]
            sim = fi @ fj.t()
            log_p_ij = F.log_softmax(sim, dim=1)
            log_p_ji = F.log_softmax(sim.t(), dim=1)
            target = torch.arange(N, device=f_inv.device)
            loss = F.nll_loss(log_p_ij, target) + F.nll_loss(log_p_ji, target)
            total_loss += loss
            count += 1
    return total_loss / count

def probabilistic_loss(f_inv, sigma):
    N, V, C = f_inv.shape
    loss = 0
    count = 0
    for i in range(V):
        for j in range(i+1, V):
            fi = f_inv[:, i, :]
            fj = f_inv[:, j, :]
            si = sigma[:, i, 0]
            sj = sigma[:, j, 0]
            dist = ((fi - fj) ** 2).sum(dim=1)
            s = si + sj
            loss += (dist / s + torch.log(s)).mean()
            count += 1
    return loss / count

def orthogonality_loss(f_inv, f_geo):
    """
    计算描述子和几何特征的正交性
    f_inv: [N, V, C_inv]
    f_geo: [N, V, C_geo]
    """
    # 统一通道数
    C = min(f_inv.shape[-1], f_geo.shape[-1])
    f_inv_proj = f_inv[..., :C]
    f_geo_proj = f_geo[..., :C]

    # 归一化
    f_inv_norm = F.normalize(f_inv_proj, dim=-1)
    f_geo_norm = F.normalize(f_geo_proj, dim=-1)

    # 计算正交性
    dot = (f_inv_norm * f_geo_norm).sum(dim=-1)  # [N, V]
    return (dot ** 2).mean()

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

            # --- loss ---
            loss += F.mse_loss(pred_j, f_j)

            count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=f_geo.device)

def recon_loss(kpnet, f_inv, f_geo, f_app, shared):
    N, V, C = f_inv.shape
    loss = 0
    count = 0
    for v in range(V):
        pred = kpnet.reconstruct_feature(f_inv[:, v, :], f_geo[:, v, :], f_app[:, v, :])
        loss += F.mse_loss(pred, shared[:, v, :].detach())
        count += 1
    return loss / count

def cross_view_loss(kpnet, f_inv, f_geo, f_app, shared, T_list, batch_idx):

    N, V, C = f_inv.shape
    loss = 0
    count = 0

    for i in range(V):
        for j in range(V):
            if i == j:
                continue

            # ✅ 只取当前 batch 的 pose
            T_i = T_list[i][batch_idx]  # [4,4]
            T_j = T_list[j][batch_idx]  # [4,4]

            pred,_ = kpnet.predict_view(
                f_inv[:, i, :],
                f_geo[:, i, :],
                f_app[:, i, :],
                T_i,
                T_j
            )

            loss += F.mse_loss(pred, shared[:, j, :].detach())
            count += 1

    return loss / count

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