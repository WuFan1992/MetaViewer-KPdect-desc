import torch
from kornia.utils import create_meshgrid
import pdb
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
    
    N, H, W = depth0.shape
    
    #kpts0_long = kpts0.round().long().clip(0, 2000-1)
    kpts0_long = kpts0.round().long()
    kpts0_long[..., 0] = kpts0_long[..., 0].clamp(0, W - 1)  # x
    kpts0_long[..., 1] = kpts0_long[..., 1].clamp(0, H - 1)  # y

    # 边界深度清0
    depth0[:, 0, :] = 0 ; depth1[:, 0, :] = 0 
    depth0[:, :, 0] = 0 ; depth1[:, :, 0] = 0 

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth > 0

    # Unproject  将I0 上的点反投影到相机空间(这里只需要用到深度和内参)
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform  有了I0 到 I1 之间在相机空间的转换矩阵 T_0to1， 将I0 在相机空间的坐标住那换到I1 的相机空间下
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth


    valid_mask = nonzero_mask #* consistent_mask* covisible_mask 

    return valid_mask, w_kpts0



@torch.no_grad()
def spvs_coarse(data0, data1, scale = 4):
    """
        Supervise corresp with dense depth & camera poses
    """

    # 1. misc
    device = data0['img'].device
    N, _, H0, W0 = data0['img'].shape
    _, _, H1, W1 = data1['img'].shape
    
    #scale = 4
    scale0 = scale
    scale1 = scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    
    
    # -------- poses (numpy) --------
    T0 = data0['pose']      # (4,4)
    T1 = data1['pose']      # (4,4)

    T_0to1 = T1 @ torch.linalg.inv(T0)
    T_1to0 = torch.linalg.inv(T_0to1)
    

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt1_i = scale1 * grid_pt1_c

    # warp kpts bi-directionally and check reproj error
    nonzero_m1, w_pt1_i  =  warp_kpts(grid_pt1_i, data1['depth'], data0['depth'], T_1to0, data1['K'], data0['K']) 
    nonzero_m2, w_pt1_og =  warp_kpts(w_pt1_i, data0['depth'], data1['depth'], T_0to1, data0['K'], data1['K']) 

    # 先将图像1投影到图像2，然后投影点再从图像2投影回图像1，用于双向一致性校验
    dist = torch.linalg.norm( grid_pt1_i - w_pt1_og, dim=-1)
    mask_mutual = (dist < 1.5) & nonzero_m1 & nonzero_m2

    #_, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    # 构建GT 对应点 
    """
    batched_corrs = [
       [x0,y0,x1,y1], ...
       ]
    """
    batched_corrs = [ torch.cat([w_pt1_i[i, mask_mutual[i]] / scale0,
                       grid_pt1_i[i, mask_mutual[i]] / scale1],dim=-1) for i in range(len(mask_mutual))]
    
    
    # batched_corrs[i] 形状 = (Ni, 4) 每一行: [x0, y0, x1, y1] 图0上的点 (x0,y0) 对应 图1上的点 (x1,y1)


    #Remove repeated correspondences - this is important for network convergence
    # 去重的原因是，很多在相机空间的3D 点，投影到图像平面后，投影点的位置很近，再进行离散的网格处理后，可能会出现不同的3D点投影到
    # 同一位置
    corrs = []
    for pts in batched_corrs:
        lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = torch.float32) * -1  # 每个 coarse 像素位置存一个 [x0,y0,x1,y1] 初始化存-1 表示空
        lut_mat21 = torch.clone(lut_mat12)
        src_pts = pts[:, :2] / scale
        tgt_pts = pts[:, 2:] / scale
        try:
            # 如果有多个3D 点同时映射到同一个像素位置，自动覆盖从而实现去重的目的
            # 这里进行了两次去重，一次是src to tgt ，避免one to many 
            # 第二次是去重是tgt to src 避免many to one
            lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)  
            # 例如有A: (10,20) → (30,40) B: (10,20) → (35,45) 第二次赋值会：覆盖第一次 最终的结果是，一个source 像素只会保留一个匹配
            mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
            points = lut_mat12[mask_valid12]

            #Target-src check 
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            lut_mat21[tgt_pts[:,1].long(), tgt_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            # 例如有A: (10,20) → (30,40) B: (11,20) → (30,40) 第二次赋值会：覆盖第一次 最终的结果是，一个source 像素只会保留一个匹配
            mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
            points = lut_mat21[mask_valid21]

            corrs.append(points)
        except:
            pdb.set_trace()
            print('..')

    #Plot for debug purposes    
    # for i in range(len(corrs)):
    #     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2]*8, corrs[i][:, 2:]*8)

    return corrs