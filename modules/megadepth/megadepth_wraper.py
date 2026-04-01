import torch
from kornia.utils import create_meshgrid
import numpy as np


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long().clip(0, 2000-1)

    # 边界深度清0
    depth0[:, 0, :] = 0 ; depth1[:, 0, :] = 0 
    depth0[:, :, 0] = 0 ; depth1[:, :, 0] = 0 

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth > 0

    # Draw cross marks on the image for each keypoint
    # for b in range(len(kpts0)):
    #     fig, ax = plt.subplots(1,2)
    #     depth_np = depth0.numpy()[b]
    #     depth_np_plot = depth_np.copy()
    #     for x, y in kpts0_long[b, nonzero_mask[b], :].numpy():
    #         cv2.drawMarker(depth_np_plot, (x, y), (255), cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #     ax[0].imshow(depth_np)
    #     ax[1].imshow(depth_np_plot)

    # Unproject  将I0 上的点反投影到相机空间(这里只需要用到深度和内参)
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform  有了I0 到 I1 之间在相机空间的转换矩阵 T_0to1， 将I0 在相机空间的坐标住那换到I1 的相机空间下
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    #h, w = depth1.shape[1:3]
    #covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
    #     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0

    # w_kpts0_depth = torch.stack(
    #     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    # )  # (N, L)
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2


    valid_mask = nonzero_mask #* consistent_mask* covisible_mask 

    return valid_mask, w_kpts0
"""
@torch.no_grad()
def spvs_coarse_multi_cycle(data, scale=8, cycle_thresh=1.5):
    device = data['images'][0].device
    B = data['images'][0].shape[0]

    images = data['images']
    depths = data['depths']
    Ks = data['Ks']
    T_0to = data['T_0to']
    scales = data['scales']

    ref_img = images[0]
    ref_depth = depths[0]
    ref_K = Ks[0]

    _, _, H, W = ref_img.shape
    h, w = H // scale, W // scale

    # ===== 1. grid =====
    grid = create_meshgrid(h, w, False, device).reshape(1, h*w, 2)
    grid_i = grid.repeat(B, 1, 1) * scale  # [B, L, 2]

    V = len(images)

    all_points = [grid_i]      # list of [B, L, 2]
    valids_per_view = []       # list of [B, L]

    # ===== 2. per-view warp =====
    for i in range(1, V):
        valid_fw, warped_fw = warp_kpts(
            grid_i, ref_depth, depths[i], T_0to[i], ref_K, Ks[i]
        )

        T_i_to_0 = torch.inverse(T_0to[i])
        valid_bw, warped_bw = warp_kpts(
            warped_fw, depths[i], ref_depth, T_i_to_0, Ks[i], ref_K
        )

        dist = torch.norm(grid_i - warped_bw, dim=-1)
        mask_cycle = dist < cycle_thresh

        valid = valid_fw & valid_bw & mask_cycle

        valids_per_view.append(valid)   # ✅ 不再做AND
        all_points.append(warped_fw)

    # ===== 3. build visibility =====
    # view0默认全True
    visibility = torch.stack(
        [torch.ones_like(valids_per_view[0])] + valids_per_view,
        dim=1
    )  # [B, V, L]

    # ===== 4. stack all points =====
    all_points = torch.stack(all_points, dim=1)  # [B, V, L, 2]

    # ===== 5. 输出统一结构 =====
    multi_corrs_batch = []

    for b in range(B):
        vis = visibility[b]        # [V, L]
        pts = all_points[b]        # [V, L, 2]

        # 转成 [L, V, 2]
        pts = pts.permute(1, 0, 2)
        vis = vis.permute(1, 0)    # [L, V]

        # ===== scale回原图 =====
        for v in range(V):
            scale_v = scales[v][b] if isinstance(scales[v], torch.Tensor) else scales[v]
            pts[:, v] /= scale_v

        # ===== 过滤至少2-view =====
        valid_mask = vis.sum(dim=1) >= 2

        pts = pts[valid_mask]   # [N, V, 2]
        vis = vis[valid_mask]   # [N, V]

        multi_corrs_batch.append({
            'points': pts,          # [N, 5, 2]
            'visibility': vis       # [N, 5] bool
        })

    return multi_corrs_batch
"""
"""
def spvs_coarse_multi_cycle(data, scale=8, cycle_thresh=1.5):
    
    #5-view multi-view + cycle consistency
    #支持 batch >= 1
    
    device = data['images'][0].device
    B = data['images'][0].shape[0]  # batch size

    images = data['images']  # list of 5 tensors [B, C, H, W]
    depths = data['depths']  # list of 5 tensors [B, H, W]
    Ks = data['Ks']          # list of 5 tensors [B, 3, 3]
    T_0to = data['T_0to']    # list of 5 tensors [B, 4, 4]
    scales = data['scales']   # list of 5 tensors [B] 或 float

    ref_img = images[0]
    ref_depth = depths[0]
    ref_K = Ks[0]

    _, _, H, W = ref_img.shape
    h, w = H // scale, W // scale

    # ===== 1. grid =====
    grid = create_meshgrid(h, w, False, device).reshape(1, h*w, 2)  # [1, L, 2]
    grid_i = grid.repeat(B, 1, 1) * scale                            # [B, L, 2]

    valid_all = torch.ones((B, h*w), dtype=torch.bool, device=device)
    all_points = [grid_i]

    # ===== 2. cycle consistency for each view =====
    V = len(images)
    for i in range(1, V):
        valid_fw, warped_fw = warp_kpts(
            grid_i, ref_depth, depths[i], T_0to[i], ref_K, Ks[i]
        )

        T_i_to_0 = torch.inverse(T_0to[i])
        valid_bw, warped_bw = warp_kpts(
            warped_fw, depths[i], ref_depth, T_i_to_0, Ks[i], ref_K
        )

        dist = torch.norm(grid_i - warped_bw, dim=-1)
        mask_cycle = dist < cycle_thresh

        valid = valid_fw & valid_bw & mask_cycle
        valid_all &= valid
        all_points.append(warped_fw)

    # ===== 3. final filtering =====
    multi_corrs_batch = []
    for b in range(B):
        final_mask = valid_all[b]
        if final_mask.sum() == 0:
            multi_corrs_batch.append(torch.empty((0, 5, 2), device=device))
            continue

        pts = []
        for p in all_points:
            pts.append(p[b, final_mask])
        multi_corrs = torch.stack(pts, dim=1)  # [num_points, 5, 2]

        # ===== 4. scale 回原图 =====
        for i in range(5):
            multi_corrs[:, i] /= scales[i][b] if isinstance(scales[i], torch.Tensor) else scales[i]

        multi_corrs_batch.append(multi_corrs)

    return multi_corrs_batch  # list of length B, 每个元素 [num_points, 5, 2]
"""

# ================================
# 1️⃣ 生成完整 5-view 对应点
# ================================
@torch.no_grad()
def generate_multi_corrs_from_data(data, scale=8, cycle_thresh=1.5):
    """
    从 data (5-view) 中生成 multi-view 对应点坐标。

    Args:
        data: dict, 包含 keys 'images', 'depths', 'Ks', 'T_0to', 'scales'
              每个 key 是 list of length 5
        scale: 下采样比例
        cycle_thresh: cycle consistency 阈值

    Returns:
        multi_corrs: torch.Tensor, [num_points, 5, 2] 对应原图坐标
    """
    device = data['images'][0].device
    B = 1  # 单个 scene

    images = [img.unsqueeze(0) if img.dim()==3 else img for img in data['images']]
    depths = [d.unsqueeze(0) if d.dim()==2 else d for d in data['depths']]
    Ks = [K.unsqueeze(0) if K.dim()==2 else K for K in data['Ks']]
    T_0to = [T.unsqueeze(0) if T.dim()==2 else T for T in data['T_0to']]
    scales = [torch.tensor([s], device=device) if not isinstance(s, torch.Tensor) else s for s in data['scales']]

    # ===== 1. reference grid =====
    _, _, H, W = images[0].shape
    h, w = H // scale, W // scale
    grid = create_meshgrid(h, w, False, device).reshape(1, h*w, 2)
    grid_i = grid * scale  # 原图坐标

    valid_all = torch.ones((1, h*w), dtype=torch.bool, device=device)
    all_points = [grid_i]

    # ===== 2. cycle consistency =====
    V = len(images)
    for i in range(1, V):
        valid_fw, warped_fw = warp_kpts(grid_i, depths[0], depths[i], T_0to[i], Ks[0], Ks[i])
        T_i_to_0 = torch.inverse(T_0to[i])
        valid_bw, warped_bw = warp_kpts(warped_fw, depths[i], depths[0], T_i_to_0, Ks[i], Ks[0])
        dist = torch.norm(grid_i - warped_bw, dim=-1)
        mask_cycle = dist < cycle_thresh
        valid = valid_fw & valid_bw & mask_cycle
        valid_all &= valid
        all_points.append(warped_fw)

    # ===== 3. final filtering =====
    final_mask = valid_all[0]
    if final_mask.sum() == 0:
        empty_corrs = torch.empty((0, V, 2), device=device)
        empty_vis = torch.empty((0, V), dtype=torch.bool, device=device)
        return empty_corrs, empty_vis

    pts = [p[0, final_mask] for p in all_points]
    multi_corrs_5view = torch.stack(pts, dim=1)  # [num_points, 5, 2]

    # ===== 4. scale 回原图 =====
    for i in range(V):
        multi_corrs_5view[:, i] /= scales[i][0]
    
    # ===== vis mask（所有点都有效）=====
    vis = torch.ones(multi_corrs_5view.shape[:2], dtype=torch.bool, device=device)

    return multi_corrs_5view, vis

@torch.no_grad()
def select_subset_and_recompute_multi_corrs(data_5view, subset_views=3, scale=4, cycle_thresh=1.5):
    """
    从5-view数据中随机选 subset_views 个 view，并重新生成 cycle-consistent 对应点。
    保证 anchor view（view 0）总在 subset 中，避免空 tensor。
    
    Args:
        data_5view: dict, 包含 'images', 'depths', 'Ks', 'T_0to', 'T', 'scales', 'all_5view_ids'
        subset_views: int, 子集 view 数量
        scale: int, 下采样比例
        cycle_thresh: float, cycle-consistency 阈值
    
    Returns:
        subset_ids: list, 选出的 view id
        multi_corrs_list: list of tuple, [(multi_corrs, vis)]，multi_corrs: [N, V, 2], vis: [N, V]
    """
    import numpy as np
    from copy import deepcopy
    import torch

    # 获取所有 view id
    all_ids = data_5view['all_5view_ids']
    if isinstance(all_ids, torch.Tensor):
        all_ids = all_ids.cpu().numpy().flatten()
    elif isinstance(all_ids, list):
        all_ids = np.array(all_ids).flatten()
    elif isinstance(all_ids, np.ndarray):
        all_ids = all_ids.flatten()
    else:
        raise TypeError(f"Unsupported type for all_ids: {type(all_ids)}")

    if subset_views > len(all_ids):
        raise ValueError(f"subset_views={subset_views} > total views={len(all_ids)}")

    # ===== 保证 anchor view（view 0）在子集中 =====
    anchor_id = all_ids[0]
    remaining_ids = [v for v in all_ids if v != anchor_id]
    if subset_views == 1:
        subset_ids = [anchor_id]
    else:
        subset_ids = [anchor_id] + list(np.random.choice(remaining_ids, subset_views-1, replace=False))

    # 获取对应索引
    subset_indices = [int(np.where(all_ids == v)[0][0]) for v in subset_ids]

    # 构造子集 data
    data_subset = deepcopy(data_5view)
    for key in ['images', 'depths', 'Ks', 'T_0to', 'T', 'scales']:
        data_subset[key] = [data_subset[key][i] for i in subset_indices]
    data_subset['all_5view_ids'] = subset_ids

    # ===== 重新生成 cycle-consistent 对应点 =====
    multi_corrs_subset, vis_tensor = generate_multi_corrs_from_data(data_subset, scale=scale, cycle_thresh=cycle_thresh)  # [N, V, 2]

    # vis 全为 True
    #vis_tensor = torch.ones(multi_corrs_subset.shape[:2], dtype=torch.bool)

    # 返回 list of tuple [(multi_corrs, vis)]
    return subset_ids, (multi_corrs_subset, vis_tensor)

"""
    multi_coors: [N, 5, 2]
    N 是匹配点的数量
    
    multi_corrs[i] = [
    [x0, y0],   # image0 (anchor)
    [x1, y1],   # image1
    [x2, y2],   # image2
    [x3, y3],   # image3
    [x4, y4],   # image4
]

"""
"""
batch_5 = dataset[idx]  # 5-view 原始数据

# 从5-view里随机选4-view，并重新计算对应点
batch_4, multi_corrs_4 = select_subset_and_recompute_multi_corrs(batch_5, subset_views=4)

# 从5-view里随机选3-view，并重新计算对应点
batch_3, multi_corrs_3 = select_subset_and_recompute_multi_corrs(batch_5, subset_views=3)

# 从5-view里随机选2-view，并重新计算对应点
batch_2, multi_corrs_2 = select_subset_and_recompute_multi_corrs(batch_5, subset_views=2)

"""
