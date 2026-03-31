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

def sigma_loss_teacher(feat_teacher, sigma):
    """
    feat_teacher: [N, V, C]  (DETACHED)
    sigma:        [N, V, 1]
    """

    # --- 1. teacher variance ---
    with torch.no_grad():
        var = cosine_variance(feat_teacher)  # [N]

        # normalize（很重要）
        var = (var - var.min()) / (var.max() - var.min() + 1e-6)

    # --- 2. student prediction ---
    sigma_point = sigma.squeeze(-1).mean(dim=1)  # [N]

    # --- 3. regression loss ---
    loss = F.mse_loss(sigma_point, var)

    return loss

"""
def sigma_loss(kpnet, f_geo, sigma, T_list, batch_idx):
    
    #f_geo: [N, V, C]
    #sigma: [N, V, 1]
    #ranking loss: all points pairwise, no random sampling
    
    device = f_geo.device
    N, V, C = f_geo.shape

    # 1. compute multi-view error per point
    error_list = []

    for i in range(V):
        for j in range(V):
            if i == j:
                continue

            T_i = T_list[i][batch_idx]
            T_j = T_list[j][batch_idx]

            f_i = f_geo[:, i, :]  # [N,C]
            f_j = f_geo[:, j, :]  # [N,C]

            pred_j = kpnet.transform_geo(f_i, T_i, T_j)

            # detach to avoid gradient through error itself
            err = ((pred_j.detach() - f_j.detach()) ** 2).mean(dim=1)  # [N]

            error_list.append(err)

    # [N, num_pairs]
    error = torch.stack(error_list, dim=1).mean(dim=1)  # [N]

    # 2. aggregate sigma per point (average across views)
    sigma_point = sigma.squeeze(-1).mean(dim=1)  # [N]

    # 3. full pairwise ranking loss
    # 构建 pairwise matrix
    diff_sigma = sigma_point.unsqueeze(0) - sigma_point.unsqueeze(1)  # [N,N]
    diff_error = error.unsqueeze(0) - error.unsqueeze(1)                # [N,N]

    target = (diff_error > 0).float()  # 如果 error_i > error_j, target=1

    # 排除对角线（i==j）
    mask = 1 - torch.eye(N, device=device)
    loss_rank = F.binary_cross_entropy_with_logits(
        diff_sigma * mask,
        target * mask
    )

    # 4. multi-view consistency
    loss_cons = 0.0
    count = 0
    for i in range(V):
        for j in range(V):
            if i == j: continue
            loss_cons += (sigma[:, i] - sigma[:, j]).abs().mean()
            count += 1
    loss_cons = loss_cons / count if count > 0 else 0.0

    # 5. anti-collapse
    loss_reg = sigma_point.mean()

    # 6. final
    loss = loss_rank + 0.1 * loss_cons + 0.05 * loss_reg
    return loss
"""

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