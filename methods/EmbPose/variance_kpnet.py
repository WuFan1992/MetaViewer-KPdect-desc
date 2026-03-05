import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Shared Backbone
# -------------------------

class SharedBackbone(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):

        return self.net(x)


# -------------------------
# Descriptor Encoder
# -------------------------

class DescriptorEncoder(nn.Module):
    def __init__(self, in_dim=128, desc_dim=128):
        super().__init__()

        self.conv = nn.Conv2d(in_dim, desc_dim, 3, padding=1)

    def forward(self, x):

        desc = self.conv(x)

        desc = F.normalize(desc, dim=1)

        return desc


# -------------------------
# Variance Head
# -------------------------

class VarianceHead(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(feat_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, 1),

            nn.Softplus()  # variance must be positive
        )

    def forward(self, x):

        return self.net(x)


# -------------------------
# Reliability Head
# -------------------------

class ReliabilityHead(nn.Module):
    def __init__(self, desc_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(desc_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.net(x)


# -------------------------
# Pose Encoder
# -------------------------

class PoseEncoder(nn.Module):
    def __init__(self, pose_dim=7, pose_embed=64):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(pose_dim, 64),
            nn.ReLU(),

            nn.Linear(64, pose_embed)
        )

    def forward(self, pose):

        return self.mlp(pose)


# -------------------------
# Patch-based Pose Attention Fusion
# -------------------------
class PatchPoseAttentionFusion(nn.Module):
    def __init__(self, desc_dim=64, pose_dim=64, num_heads=4, patch_size=5):
        super().__init__()
        self.desc_dim = desc_dim
        self.pose_dim = pose_dim
        self.num_heads = num_heads
        self.patch_size = patch_size

        # linear projections for multi-head attention
        self.q_proj = nn.Linear(desc_dim, desc_dim)
        self.k_proj = nn.Linear(desc_dim, desc_dim)
        self.v_proj = nn.Linear(desc_dim, desc_dim)

        # pose modulation
        self.pose_fc = nn.Linear(pose_dim, desc_dim)

        self.out_fc = nn.Linear(desc_dim, desc_dim)

    def forward(self, desc_points, shared_feat_map, coords, pose_embed):
        """
        desc_points: [B,N,C] 每个采样点的 descriptor
        shared_feat_map: [B,C,H,W] backbone 特征图
        coords: [B,N,2] integer coordinates (y,x)
        pose_embed: [B,pose_dim]
        """
        B, N, C = desc_points.shape
        H, W = shared_feat_map.shape[2], shared_feat_map.shape[3]
        device = desc_points.device

        # 1. 提取每个点周围 patch 特征
        pad = self.patch_size // 2
        padded_feat = F.pad(shared_feat_map, (pad, pad, pad, pad))  # [B,C,H+2*pad,W+2*pad]

        patch_feats = []
        for b in range(B):
            feats_b = []
            for n in range(N):
                y, x = coords[b, n].long()
                patch = padded_feat[b, :, y:y+self.patch_size, x:x+self.patch_size]  # [C,patch,patch]
                patch_flat = patch.flatten(1)  # [C, patch*patch]
                patch_mean = patch_flat.mean(dim=1)  # [C]
                feats_b.append(patch_mean)
            patch_feats.append(torch.stack(feats_b, dim=0))  # [N,C]
        patch_feats = torch.stack(patch_feats, dim=0)  # [B,N,C]

        # 2. multi-head attention (query=desc, key/value=patch_feats)
        q = self.q_proj(desc_points).view(B, N, self.num_heads, C//self.num_heads).transpose(1,2)  # [B,heads,N,head_dim]
        k = self.k_proj(patch_feats).view(B, N, self.num_heads, C//self.num_heads).transpose(1,2)
        v = self.v_proj(patch_feats).view(B, N, self.num_heads, C//self.num_heads).transpose(1,2)

        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / (C//self.num_heads)**0.5  # [B,heads,N,N]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # [B,heads,N,head_dim]

        attn_out = attn_out.transpose(1,2).contiguous().view(B, N, C)  # [B,N,C]

        # 3. pose modulation (FiLM)
        pose_mod = self.pose_fc(pose_embed)  # [B,C]
        pose_mod = pose_mod.unsqueeze(1)     # [B,1,C]
        fused = attn_out * (1 + pose_mod)    # FiLM-like modulation

        # 4. output projection
        fused = self.out_fc(fused)  # [B,N,C]

        return fused


# -------------------------
# Simple per-point decoder
# -------------------------
class PointDecoder(nn.Module):
    def __init__(self, feat_dim=128, out_dim=64):
        super().__init__()
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        """
        x: [B*N, feat_dim]
        return: [B*N, out_dim]
        """
        return self.fc(x)

# -------------------------
# Spatial Feature Masking
# -------------------------

class SpatialFeatureMasking(nn.Module):

    def __init__(self, mask_ratio=0.2):
        super().__init__()

        self.mask_ratio = mask_ratio

    def forward(self, x):

        if not self.training:
            return x

        B, C, H, W = x.shape

        mask = torch.rand(B,1,H,W,device=x.device) > self.mask_ratio

        return x * mask


# -------------------------
# Dense Feature AutoEncoder
# -------------------------

class VarianceKPNet(nn.Module):

    def __init__(self,
                 in_channels=64,
                 pose_dim=7,
                 feature_dim=64,
                 pose_embed=64):
        super().__init__()

        # backbone
        self.backbone = SharedBackbone(in_channels, feature_dim)

        # descriptor branch
        self.descriptor_encoder = DescriptorEncoder(feature_dim, feature_dim)

        # heads
        self.reliability_head = ReliabilityHead(feature_dim)
        self.variance_head = VarianceHead(feature_dim)

        # masking
        self.feature_mask = SpatialFeatureMasking(0.2)

        # pose modules
        self.pose_encoder = PoseEncoder(pose_dim, pose_embed)
        self.fusion = PatchPoseAttentionFusion(desc_dim=feature_dim, pose_dim=pose_embed)
        self.decoder = PointDecoder(feat_dim=feature_dim, out_dim=in_channels)
        
    def sample_map_at_coords(self, fmap, coords):
        """
        fmap: [B,C,H,W]
        coords: [B,N,2]  (y,x)

        return:
            sampled: [B,N,C]
        """
        B, C, H, W = fmap.shape
        _, N, _ = coords.shape

        idx_y = coords[:,:,0]
        idx_x = coords[:,:,1]
        
        sampled = []
        for b in range(B):
            fmap_b = fmap[b]                 # [C,H,W]
            pts = fmap_b[:, idx_y[b].long(), idx_x[b].long()]  # [C,N]
            sampled.append(pts.T)            # [N,C]

        sampled = torch.stack(sampled, dim=0)  # [B,N,C]

        return sampled

    def forward(self, xfeat, pose, coords):
        """
        xfeat: [B,C,H,W]
        pose:  [B,pose_dim]
        coords: [B,N,2] integer coordinates (y,x)
        """

        B, C, H, W = xfeat.shape
        _, N, _ = coords.shape
        device = xfeat.device
        
        # xfeat is 1/8 of original image, we up sample it to 1/4
        xfeat_up = F.interpolate(
            xfeat,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        # 1. shared feature
        shared_feat = self.backbone(xfeat_up)        # [B, feat_dim, H, W]

        # 2. variance map (full map)
        variance_map = self.variance_head(shared_feat)  # [B,1,H,W]
        sampled_var = self.sample_map_at_coords(variance_map, coords)   # [B,N,1]

        # 3. descriptor map (full map)
        desc_map = self.descriptor_encoder(shared_feat)  # [B,feat_dim,H,W]

        # 4. reliability map from descriptor
        reliability_map = self.reliability_head(desc_map)  # [B,1,H,W]
        sampled_rel = self.sample_map_at_coords(reliability_map, coords) # [B,N,1]

        # 5. sample descriptor features at coords
        sampled_desc = self.sample_map_at_coords(desc_map, coords)  # [B,N,C]

        # 6. sample pose features and expand to N
        f_pose = self.pose_encoder(pose)                  # [B,pose_embed]
        f_pose = f_pose.unsqueeze(1).expand(-1,N,-1)     # [B,N,pose_embed]

        # 7. feature masking (在每个采样点)
        sampled_desc = sampled_desc * (torch.rand_like(sampled_desc) > self.feature_mask.mask_ratio)

        # 8. fusion per point
        latent = self.fusion(sampled_desc, shared_feat, coords, f_pose)  # [B,N,feat_dim]
        latent = F.normalize(latent, dim=2)

        # 9. decoder per point (reshape为1x1小图)
        xfeat_pred = self.decoder(latent.view(B*N, -1))  # [B*N,C]
        xfeat_pred = xfeat_pred.view(B,N,-1)  # [B,N,C]

        return xfeat_pred, sampled_desc, sampled_rel, sampled_var

def correspondence_mean_loss(xfeat, xfeat_pred, coords):
    """
    xfeat:      [B, C, H, W]
    xfeat_pred: [B, C, H, W]
    coords:     [B, N, 2]  (y, x) integer coordinates
    """
    B, C, H, W = xfeat.shape
    _, N, _ = coords.shape

    loss = 0.0

    for b in range(B):
        y = coords[b, :, 0]  # [N]
        x = coords[b, :, 1]  # [N]

        # 取出对应位置特征: [N, C]
        f1 = xfeat[b, :, y, x].permute(1, 0)
        f2 = xfeat_pred[b, :, y, x].permute(1, 0)

        # 计算均值
        mean = (f1 + f2) / 2

        loss += ((f1 - mean)**2).mean()
        loss += ((f2 - mean)**2).mean()

    return loss / B
# -------------------------
# Loss函数
# -------------------------
def intra_object_feature_loss(f):
    """
    方法A: 让同一物体N张图片的特征靠近平均值
    f: [B, N, feature_dim]
    """
    f_inv = f.mean(dim=1, keepdim=True)  # [B,1,D]
    loss = ((f - f_inv) ** 2).sum(dim=2).mean()
    return loss

