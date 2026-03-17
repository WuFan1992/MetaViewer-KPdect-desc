import torch
from kornia.utils import create_meshgrid
import pdb


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
    # h, w = depth1.shape[1:3]
    # covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
    #     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0

    # w_kpts0_depth = torch.stack(
    #     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    # )  # (N, L)
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2


    valid_mask = nonzero_mask #* consistent_mask* covisible_mask 

    return valid_mask, w_kpts0


@torch.no_grad()
def spvs_coarse_multi_cycle(data, scale=8, cycle_thresh=1.5):
    """
    5-view multi-view + cycle consistency
    """

    device = data['images'][0].device

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
    grid_i = scale * grid  # [1, L, 2]

    valid_all = torch.ones((1, h*w), dtype=torch.bool, device=device)
    all_points = [grid_i]

    # ===== 2. 对每个 view 做 cycle consistency =====
    for i in range(1, 5):

        # ---------- forward ----------
        valid_fw, warped_fw = warp_kpts(
            grid_i,
            ref_depth,
            depths[i],
            T_0to[i],
            ref_K,
            Ks[i]
        )

        # ---------- backward ----------
        T_i_to_0 = torch.inverse(T_0to[i])

        valid_bw, warped_bw = warp_kpts(
            warped_fw,
            depths[i],
            ref_depth,
            T_i_to_0,
            Ks[i],
            ref_K
        )

        # ---------- cycle consistency ----------
        dist = torch.norm(grid_i - warped_bw, dim=-1)

        mask_cycle = dist < cycle_thresh

        # ---------- 合并mask ----------
        valid = valid_fw & valid_bw & mask_cycle

        valid_all &= valid

        all_points.append(warped_fw)

    # ===== 3. final filtering =====
    final_mask = valid_all[0]

    if final_mask.sum() == 0:
        return []

    # ===== 4. assemble =====
    pts = []
    for p in all_points:
        pts.append(p[0, final_mask])

    # [num_points, 5, 2]
    multi_corrs = torch.stack(pts, dim=1)

    # ===== 5. scale 回原图 =====
    for i in range(5):
        multi_corrs[:, i] /= scales[i]

    return multi_corrs

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

