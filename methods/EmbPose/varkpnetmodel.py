import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .utils import *
from methods.Xfeat.xfeat import *

# -------------------------
# Shared Backbone
# -------------------------
"""
class SharedBackbone(nn.Module):

    def __init__(self, out_dim=128, freeze=True):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # stem
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # feature stages
        self.layer1 = resnet.layer1   # H/4   C=256
        self.layer2 = resnet.layer2   # H/8   C=512
        self.layer3 = resnet.layer3   # H/16  C=1024

        # FPN fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(256 + 512 + 1024, 512, 3, padding=1),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True)
        )

        if freeze:
            for p in (
                list(self.conv1.parameters()) +
                list(self.bn1.parameters()) +
                list(self.layer1.parameters()) +
                list(self.layer2.parameters()) +
                list(self.layer3.parameters())
            ):
                p.requires_grad = False


    def forward(self, x):

        # stem
        x = self.conv1(x)   # H/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # H/4

        # multi-scale features
        f1 = self.layer1(x)   # [B,256,H/4,W/4]
        f2 = self.layer2(f1)  # [B,512,H/8,W/8]
        f3 = self.layer3(f2)  # [B,1024,H/16,W/16]

        # upsample to H/4
        f2_up = F.interpolate(
            f2, size=f1.shape[-2:],
            mode='bilinear', align_corners=False
        )

        f3_up = F.interpolate(
            f3, size=f1.shape[-2:],
            mode='bilinear', align_corners=False
        )

        # FPN fusion
        feat = torch.cat([f1, f2_up, f3_up], dim=1)

        feat = self.fuse(feat)

        return feat
"""
"""
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
        with torch.no_grad():
            feat = self.backbone(x)   # [B,256,H/4,W/4]
        feat = self.proj(feat)    # [B,out_dim,H/4,W/4]

        return feat
"""

class SharedBackbone_XFeat(nn.Module):
    def __init__(self, out_dim=128, freeze=True):
        super().__init__()

        # 初始化 XFeat
        self.xfeat = XFeat()  # 如果有预训练权重可以加载
        self.out_dim = out_dim

        # XFeat 输出是 [B, 64, H/8, W/8]，proj + 上采样到 H/4, W/4
        self.proj = nn.Sequential(
            nn.Conv2d(64, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True)
        )

        # 是否冻结 XFeat backbone
        if freeze:
            for p in self.xfeat.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: [B,3,H,W]

        feat = self.xfeat.getFeatDesc(x)  # [B,64,H/8,W/8]

        feat = self.proj(feat)               # [B,out_dim,H/8,W/8]
        # 上采样到 H/4, W/4
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)

        return feat


# -------------------------
# Descriptor Encoder
# -------------------------

class DualDescriptorHead(nn.Module):
    def __init__(self, in_dim=128, dim=128, bottleneck=32):
        super().__init__()

        # invariant（高容量）
        self.inv_head = nn.Sequential(
            nn.Conv2d(in_dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

        # 🔥 variant（低容量 bottleneck）
        self.var_head = nn.Sequential(
            nn.Conv2d(in_dim, bottleneck, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(bottleneck, dim, 1)
        )

    def forward(self, x):
        f_inv = F.normalize(self.inv_head(x), dim=1)
        f_var = self.var_head(x)
        return f_inv, f_var


# -------------------------
# Variance Head
# -------------------------

class UncertaintyHead(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, feat):
        log_sigma2 = self.net(feat)
        sigma2 = F.softplus(log_sigma2) + 1e-4
        return sigma2


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
    def __init__(self, pose_dim=9, pose_embed=64):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(pose_dim, 64),
            nn.ReLU(),

            nn.Linear(64, pose_embed)
        )

    def forward(self, pose):

        return self.mlp(pose)
    
class FiLMModulation(nn.Module):

    def __init__(self, pose_dim=128, feat_dim=64):
        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(pose_dim, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, feat_dim * 2)
        )

    def forward(self, feat, pose_embed):

        """
        feat: [B,C]
        pose_embed: [B,pose_dim]
        """

        gamma_beta = self.mlp(pose_embed)

        gamma, beta = gamma_beta.chunk(2, dim=1)

        out = gamma * feat + beta

        return out



# -------------------------
# Simple per-point decoder
# -------------------------
class PointDecoder(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x):
        return self.net(x)



# -------------------------
# Dense Feature AutoEncoder
# -------------------------

class VarianceKPNetModel(nn.Module):

    def __init__(self,
                 in_channels=64,
                 pose_dim=9,
                 feature_dim=64,
                 pose_embed=128):
        super().__init__()

        # -------------------------
        # Backbone
        # -------------------------
        self.backbone = SharedBackbone_XFeat(out_dim=feature_dim)

        # -------------------------
        # Descriptor (解耦)
        # -------------------------
        self.descriptor_encoder = DualDescriptorHead(feature_dim, feature_dim)

        # -------------------------
        # Heads（全部从 shared feature 出发）
        # -------------------------
        self.reliability_head = ReliabilityHead(feature_dim)

        # 🔥 改：variance 从 shared_feat，不是 desc
        self.uncertainty_head = UncertaintyHead(feature_dim)

        # -------------------------
        # Pose modules
        # -------------------------
        self.pose_encoder = PoseEncoder(pose_dim, pose_embed)

        # -------------------------
        # FiLM
        # -------------------------
        self.film = FiLMModulation(pose_embed, feature_dim)

        # -------------------------
        # Decoder（只作用在 var branch）
        # -------------------------
        self.decoder = PointDecoder(feature_dim)

    # =========================================================
    # Reconstruction（只用 desc_var）
    # =========================================================
    def reconstruction(self, desc_var, desc_inv, pose_src, pose_tgt, use_inv=False):
        """
        desc_var: [N,C]
        desc_inv: [N,C]
        """

        device = desc_var.device

        pose_src = pose_src.to(device).float()
        pose_tgt = pose_tgt.to(device).float()

        if pose_src.dim() == 2:
            pose_src = pose_src.unsqueeze(0)
        if pose_tgt.dim() == 2:
            pose_tgt = pose_tgt.unsqueeze(0)

        pose_ij = pose_matrix_to_9d(pose_src, pose_tgt)
        pose_embed = self.pose_encoder(pose_ij)

        if pose_embed.dim() == 1:
            pose_embed = pose_embed.unsqueeze(0)

        if pose_embed.shape[0] > 1:
            pose_embed = pose_embed.mean(dim=0, keepdim=True)

        N = desc_var.shape[0]
        pose_embed = pose_embed.expand(N, -1)

        # -------------------------
        # ✅ 不再 detach！！
        # -------------------------
        latent = desc_var

        # -------------------------
        # FiLM modulation
        # -------------------------
        delta = self.film(latent, pose_embed)
        latent = latent + delta

        # -------------------------
        # ❗阶段控制（避免 shortcut）
        # -------------------------
        if use_inv:
            latent = latent + desc_inv

        # -------------------------
        # decode
        # -------------------------
        pred_desc = self.decoder(latent)
        pred_desc = F.normalize(pred_desc, dim=1)

        return pred_desc
    
    # =========================================================
    # Forward
    # =========================================================
    def forward(self, img):

        # -------------------------
        # 1. shared feature
        # -------------------------
        shared_featmap = self.backbone(img)  # [B,C,H,W]

        # -------------------------
        # 2. descriptor（解耦）
        # -------------------------
        desc_inv, desc_var = self.descriptor_encoder(shared_featmap)

        # -------------------------
        # 3. uncertainty（🔥关键改动）
        # -------------------------
        variance_map = self.uncertainty_head(shared_featmap)

        # -------------------------
        # 4. reliability（🔥建议也用 shared_feat）
        # -------------------------
        reliability_map = self.reliability_head(shared_featmap)

        # -------------------------
        # 输出（全部返回）
        # -------------------------
        return {
            "shared_feat": shared_featmap,
            "desc_inv": desc_inv,
            "desc_var": desc_var,
            "variance": variance_map,
            "reliability": reliability_map
        }



