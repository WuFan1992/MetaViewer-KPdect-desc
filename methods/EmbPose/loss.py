import torch
import torch.nn.functional as F


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

def variance_loss_pose_aware_single(
    desc1_pts,         # [N,C]  source descriptors
    desc2_pts,         # [N,C]  target descriptors
    var1_pred,         # [N]    predicted variance for source
    var2_pred,         # [N]    predicted variance for target
    pose1,             # [4,4]  source pose
    pose2,             # [4,4]  target pose
    reliability1=None, # [N] optional reliability for source
    reliability2=None, # [N] optional reliability for target
    alpha=1.0,
    beta=1.0
):
    """
    Compute pose-aware variance loss for a single pair of features.
    All inputs are already extracted for N keypoints/features.
    """
    N, C = desc1_pts.shape

    # 1. compute pose difference
    pose_diff = pose_difference(pose1, pose2, alpha, beta)  # [1]
    #pose_diff_norm = pose_diff / (pose_diff.max() + 1e-8)  # scalar normalized
    pose_diff_norm = pose_diff / (alpha + beta + 1e-6)

    # 2. compute descriptor difference per feature
    desc_diff = ((desc1_pts - desc2_pts)**2).sum(dim=1)  # [N]

    # 3. adaptive weight based on pose difference
    # smaller pose_diff -> weight larger
    weight = torch.exp(-pose_diff_norm) * (desc_diff / (desc_diff.max() + 1e-8) + 1e-6)  # [N]
    weight = torch.clamp(weight, min=1e-3)  # 防止权重消失

    # 4. optional reliability weighting
    if reliability1 is not None:
        weight = weight * reliability1  # [N]
    if reliability2 is not None:
        weight = weight * reliability2  # [N]

    # 5. weighted MSE for source and target variance
    loss1 = (weight * (var1_pred - desc_diff)**2).mean()
    loss2 = (weight * (var2_pred - desc_diff)**2).mean()

    # 6. combined loss
    loss = (loss1 + loss2) / 2.0

    return loss


def variance_loss_pose_aware(
    desc_map1,          # [B,C,H,W]  source image descriptor map
    desc_map2,          # [B,C,H,W]  target image descriptor map
    variance_map,       # [B,1,H,W]  predicted variance map (source)
    coords1,            # [B,N,2]    source pixel coordinates (y,x)
    coords2,            # [B,N,2]    target corresponding pixel coordinates (y,x)
    pose1,              # [B,4,4] source pose
    pose2,              # [B,4,4] target pose
    reliability_map=None # [B,1,H,W] optional
):
    """
    Compute variance loss for a single pair of images with pose-aware weighting
    and optional reliability weighting.
    """

    B, C, H, W = desc_map1.shape
    _, N, _ = coords1.shape
    device = desc_map1.device

    loss = 0.0

    # 1. compute pose difference
    pose_diff =  pose_difference(pose1, pose2)
    pose_diff_norm = pose_diff / (pose_diff.max() + 1e-8) # normalize to [0,1]

    for b in range(B):
        y1 = coords1[b,:,0]
        x1 = coords1[b,:,1]
        y2 = coords2[b,:,0]
        x2 = coords2[b,:,1]

        # 2. sample descriptors at corresponding coordinates
        desc1_pts = desc_map1[b, :, y1, x1].T  # [N,C]
        desc2_pts = desc_map2[b, :, y2, x2].T  # [N,C]

        # 3. compute descriptor difference
        desc_diff = ((desc1_pts - desc2_pts)**2).sum(dim=1)  # [N]

        # 4. adaptive weight based on pose difference
        # smaller pose_diff -> weight larger
        weight = torch.exp(-pose_diff_norm[b]) * (desc_diff / (desc_diff.max()+1e-8) + 1e-6)  # [N]

        # 5. optional reliability weighting
        if reliability_map is not None:
            weight = weight * reliability_map[b,0,y1,x1]  # [N]

        # 6. predicted variance for source pixels
        var_pred = variance_map[b,0,y1,x1]  # [N]

        # 7. weighted MSE
        loss += (weight * (var_pred - desc_diff)**2).mean()

    return loss / B



def dual_softmax_loss(X, Y, temp = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf


def reliability_loss(heatmap, target):
    # Compute L1 loss
    target = target.unsqueeze(0).unsqueeze(-1)
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss 


def reconstr_loss(ori, rec):
    diff = ori - rec
    reconstr_loss = torch.norm(diff, dim=2).mean()
    return reconstr_loss


