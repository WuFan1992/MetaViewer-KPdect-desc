import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -------------------------
# Shared Backbone
# -------------------------


class SharedBackbone(nn.Module):

    def __init__(self, out_dim=128, freeze=True):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(
            resnet.conv1,   # stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # stride 2
            resnet.layer1   # stride 1
        )

        self.proj = nn.Sequential(
            nn.Conv2d(256, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True)
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):

        feat = self.backbone(x)   # [B,256,H/4,W/4]
        feat = self.proj(feat)    # [B,out_dim,H/4,W/4]

        return feat


# -------------------------
# Descriptor Encoder
# -------------------------

class DescriptorEncoder(nn.Module):

    def __init__(self, in_dim=128, desc_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, desc_dim, 3, padding=1),
            nn.GroupNorm(8, desc_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(desc_dim, desc_dim, 3, padding=1),
            nn.GroupNorm(8, desc_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(desc_dim, desc_dim, 1)
        )

    def forward(self, x):
        desc = self.net(x)
        desc = F.normalize(desc, dim=1)
        return desc


# -------------------------
# Variance Head
# -------------------------

class VarianceHead(nn.Module):
    def __init__(self, feat_dim=128, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(

            nn.Conv2d(feat_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        var_pred = F.softplus(self.net(x)) + self.epsilon
        return var_pred


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
        self.pad = patch_size // 2

        # multi-head projections
        self.q_proj = nn.Linear(desc_dim, desc_dim)
        self.k_proj = nn.Linear(desc_dim, desc_dim)
        self.v_proj = nn.Linear(desc_dim, desc_dim)

        # pose modulation
        self.pose_fc = nn.Linear(pose_dim, desc_dim)

        self.out_fc = nn.Linear(desc_dim, desc_dim)

    def forward(self, desc_points, shared_feat_map, coords, pose_embed):
        """
        desc_points: [B, N, C]
        shared_feat_map: [B, C, H, W]
        coords: [B, N, 2]  (y, x)
        pose_embed: [B, pose_dim]
        """
        B, N, C = desc_points.shape
        H, W = shared_feat_map.shape[2], shared_feat_map.shape[3]

        device = desc_points.device

        # --------- 1. 提取 patch 特征 ---------
        # pad fmap
        padded_feat = F.pad(shared_feat_map, (self.pad, self.pad, self.pad, self.pad))  # [B,C,H+2*pad,W+2*pad]

        # 生成 patch 坐标偏移
        offsets = torch.arange(-self.pad, self.pad+1, device=device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing='ij')  # [patch,patch]
        dy = dy.flatten()
        dx = dx.flatten()
        K = self.patch_size ** 2

        # 扩展 coords
        y = coords[..., 0].unsqueeze(-1) + dy.view(1,1,-1)  # [B,N,K]
        x = coords[..., 1].unsqueeze(-1) + dx.view(1,1,-1)  # [B,N,K]

        # clamp 防止越界
        y = y.clamp(0, H + 2*self.pad - 1)
        x = x.clamp(0, W + 2*self.pad - 1)

        # long 类型索引
        y = y.long()
        x = x.long()

        # 生成 batch index
        b_idx = torch.arange(B, device=device).view(B,1,1).expand(-1,N,K)  # [B,N,K]

        # gather
        patch_feats = padded_feat[b_idx, :, y, x]  # [B,N,K,C] ??? 需要 permute
        patch_feats = patch_feats.permute(0,1,3,2)  # [B,N,C,K]
        patch_feats = patch_feats.mean(dim=-1)      # [B,N,C] 取 patch 平均

        # --------- 2. multi-head attention ---------
        q = self.q_proj(desc_points).view(B,N,self.num_heads,C//self.num_heads).transpose(1,2)  # [B,heads,N,head_dim]
        k = self.k_proj(patch_feats).view(B,N,self.num_heads,C//self.num_heads).transpose(1,2)
        v = self.v_proj(patch_feats).view(B,N,self.num_heads,C//self.num_heads).transpose(1,2)

        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / (C//self.num_heads)**0.5  # [B,heads,N,N]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # [B,heads,N,head_dim]
        attn_out = attn_out.transpose(1,2).contiguous().view(B,N,C)  # [B,N,C]

        # --------- 3. pose modulation (FiLM) ---------
        pose_mod = self.pose_fc(pose_embed).unsqueeze(1)  # [B,1,C]
        fused = attn_out * (1 + pose_mod)                  # FiLM-like modulation

        # --------- 4. output projection ---------
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

class VarianceKPNetModel(nn.Module):

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
        
    
    def reconstruction(self, pose, coords, sampled_desc, shared_feat):
        
        B, N, _ = coords.shape
        # 6. sample pose features and expand to N
        f_pose = self.pose_encoder(pose)                  # [B,pose_embed]
        f_pose = f_pose.unsqueeze(1).expand(-1,N,-1)     # [B,N,pose_embed]

        # 7. feature masking (在每个采样点)
        sampled_desc = sampled_desc * (torch.rand_like(sampled_desc) > self.feature_mask.mask_ratio)

        # 8. fusion per point
        latent = self.fusion(sampled_desc, shared_feat, coords, f_pose)  # [B,N,feat_dim]
        latent = F.normalize(latent, dim=2)

        # 9. decoder per point (reshape为1x1小图)
        backbone_pred = self.decoder(latent.view(B*N, -1))  # [B*N,C]
        backbone_pred = backbone_pred.view(B,N,-1)  # [B,N,C]
        
        return backbone_pred
        
    
    def forward(self, img):
        """
        img: [B,3,H,W]
        pose:  [B,pose_dim]
        coords: [B,N,2] integer coordinates (y,x)
        """

        # 1. shared feature
        shared_featmap = self.backbone(img)        # [B, feat_dim, H, W]

        # 2. variance map (full map)
        variance_map = self.variance_head(shared_featmap)  # [B,1,H,W]

        # 3. descriptor map (full map)
        desc_map = self.descriptor_encoder(shared_featmap)  # [B,feat_dim,H,W]
        
        # 5. reliability map from descriptor
        reliability_map = self.reliability_head(desc_map)  # [B,1,H,W]

        return shared_featmap, variance_map, desc_map, reliability_map
        



