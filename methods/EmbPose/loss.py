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
def mv_infonce(f_inv):
    N, V, C = f_inv.shape
    f = F.normalize(f_inv, dim=-1)

    loss = 0
    count = 0

    for i in range(V):
        for j in range(V):
            if i == j: continue

            sim = f[:, i] @ f[:, j].t()  # [N,N]
            logits = sim / 0.07

            labels = torch.arange(N, device=f.device)

            loss += F.cross_entropy(logits, labels)
            count += 1

    return loss / count


def sigma_viewpoint_loss(f_geo, sigma):
    
    #f_geo:  [N,V,C]
    #sigma:  [N,V,1]
    

    # --- GT: feature variance across views ---
    var = f_geo.var(dim=1).mean(dim=1)   # [N]

    # --- prediction ---
    sigma_pred = sigma.mean(dim=1).squeeze(-1)  # [N]

    # normalize（很重要，避免scale问题）
    var = (var - var.mean()) / (var.std() + 1e-6)
    sigma_pred = (sigma_pred - sigma_pred.mean()) / (sigma_pred.std() + 1e-6)
    

    return F.mse_loss(sigma_pred, var.detach())


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

def pose_distance(T_i, T_j):

    # --- translation ---
    t_i = T_i[:3, 3]
    t_j = T_j[:3, 3]
    dt = torch.norm(t_i - t_j)

    # --- rotation ---
    R_i = T_i[:3, :3]
    R_j = T_j[:3, :3]

    R_rel = R_j @ R_i.T
    cos_theta = (torch.trace(R_rel) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    dR = torch.acos(cos_theta)

    # 🔥 normalize（关键）
    dt = dt / (dt.detach() + 1e-6)
    dR = dR / (dR.detach() + 1e-6)

    return dt + dR


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
            loss += F.mse_loss(pred_j, f_j.detach())

            count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=f_geo.device)

def geo_cycle_loss(kpnet, f_geo, T_list, batch_idx):
    """
    Cycle-consistent geometry loss

    f_geo: [N, V, C]
    T_list: list of [B,4,4]
    """

    N, V, C = f_geo.shape
    loss = 0.0
    count = 0

    for i in range(V):
        for j in range(V):
            for k in range(V):

                if i == j or j == k or i == k:
                    continue

                # --- poses ---
                T_i = T_list[i][batch_idx]
                T_j = T_list[j][batch_idx]
                T_k = T_list[k][batch_idx]

                # --- features ---
                f_i = f_geo[:, i, :]  # [N,C]

                # i → j
                f_j = kpnet.transform_geo(f_i, T_i, T_j)

                # j → k
                f_k = kpnet.transform_geo(f_j, T_j, T_k)

                # k → i
                f_i_cycle = kpnet.transform_geo(f_k, T_k, T_i)

                # --- cycle loss ---
                loss += F.mse_loss(f_i_cycle, f_i.detach())

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