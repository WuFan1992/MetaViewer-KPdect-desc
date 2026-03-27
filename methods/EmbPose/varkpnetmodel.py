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

class TripleDescriptorHead(nn.Module):
    def __init__(self, in_dim=128, dim_inv=128, dim_geo=32, dim_app=16):
        super().__init__()

        # invariant（matching用）
        self.inv_head = nn.Sequential(
            nn.Conv2d(in_dim, dim_inv, 3, padding=1),
            nn.GroupNorm(8, dim_inv),
            nn.ReLU(),
            nn.Conv2d(dim_inv, dim_inv, 1)
        )

        # geometry（viewpoint）
        self.geo_head = nn.Sequential(
            nn.Conv2d(in_dim, dim_geo, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_geo, dim_geo, 1)
        )

        # appearance（光照/纹理）
        self.app_head = nn.Sequential(
            nn.Conv2d(in_dim, dim_app, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_app, dim_app, 1)
        )

    def forward(self, x):
        f_inv = F.normalize(self.inv_head(x), dim=1)
        f_geo = self.geo_head(x)
        f_app = self.app_head(x)
        return f_inv, f_geo, f_app


# -------------------------
# Variance Head
# -------------------------

class UncertaintyHead(nn.Module):
    def __init__(self, in_dim=32):  # 🔥 改成 geo dim
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, f_geo):
        log_sigma = self.net(f_geo)

        # 🔥 bounded sigma（防爆）
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return sigma


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
    
    
class GeometryTransform(nn.Module):
    def __init__(self, geo_dim=32, pose_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(geo_dim + pose_dim, 128),
            nn.ReLU(),
            nn.Linear(128, geo_dim)
        )

    def forward(self, f_geo, pose_embed):
        """
        f_geo: [N,C]
        pose_embed: [N,P]
        """
        x = torch.cat([f_geo, pose_embed], dim=1)
        delta = self.mlp(x)
        return f_geo + delta
    

class FeatureDecoder(nn.Module):
    def __init__(self, dim_inv=128, dim_geo=32, dim_app=16, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_inv + dim_geo + dim_app, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, f_inv, f_geo, f_app):
        x = torch.cat([f_inv, f_geo, f_app], dim=1)
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

class VUDNet(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 dim_geo=32,
                 dim_app=16,
                 pose_dim=16,
                 pose_embed=128):
        super().__init__()

        # -------------------------
        # Backbone
        # -------------------------
        self.backbone = SharedBackbone_XFeat(out_dim=feature_dim)

        # -------------------------
        # Disentangled encoder
        # -------------------------
        self.encoder = TripleDescriptorHead(
            in_dim=feature_dim,
            dim_inv=feature_dim,
            dim_geo=dim_geo,
            dim_app=dim_app
        )

        # -------------------------
        # Heads
        # -------------------------
        self.uncertainty_head = UncertaintyHead(dim_geo)
        self.reliability_head = ReliabilityHead(feature_dim)

        # -------------------------
        # Geometry modules
        # -------------------------
        self.pose_encoder = PoseEncoder(pose_dim, pose_embed)
        self.pose_proj = nn.Linear(64, dim_geo)  # 将 pose_embed 投影到 f_geo 的维度
        self.geo_transform = GeometryTransform(dim_geo, pose_embed)

        # -------------------------
        # Decoder（reconstruction）
        # -------------------------
        self.decoder = FeatureDecoder(
            dim_inv=feature_dim,
            dim_geo=dim_geo,
            dim_app=dim_app,
            out_dim=feature_dim
        )

    # =========================================================
    # 1️⃣ 基础 forward（只做 feature 提取）
    # =========================================================
    def forward(self, img):

        shared = self.backbone(img)

        f_inv, f_geo, f_app = self.encoder(shared)

        sigma = self.uncertainty_head(f_geo)

        reliability = self.reliability_head(shared)

        return {
            "shared": shared,
            "f_inv": f_inv,
            "f_geo": f_geo,
            "f_app": f_app,
            "sigma": sigma,
            "reliability": reliability
        }

    # =========================================================
    # 2️⃣ 几何变换（真正用到 pose_encoder + geo_transform）
    # =========================================================
    def transform_geo(self, f_geo, T_i, T_j):
        """
        f_geo: [N, C]
        T_i, T_j: [4,4]
        """

        # --- 1. compute relative pose ---
        T_i_inv = torch.inverse(T_i)
        T_ji = T_j @ T_i_inv  # i → j

        # --- 2. flatten ---
        pose = T_ji.reshape(1, -1)  # [1,16]

        # --- 3. encode ---
        pose_embed = self.pose_encoder(pose)  # [1, pose_embed]

        # expand to N points
        N = f_geo.shape[0]
        pose_embed = pose_embed.expand(N, -1)  # [N, pose_embed]

        # --- 4. transform ---
        f_geo_j = self.geo_transform(f_geo, pose_embed)

        return f_geo_j

    # =========================================================
    # 3️⃣ reconstruction（真正用 decoder）
    # =========================================================
    def reconstruct_feature(self, f_inv, f_geo, f_app):
        """
        all: [N,C]
        """
        return self.decoder(f_inv, f_geo.detach(), f_app)

    # =========================================================
    # 4️⃣ joint prediction（论文里可以画图）
    # =========================================================
    def predict_view(self, f_inv_i, f_geo_i, f_app_i, pose_i, pose_j):
        """
        从 view i 预测 view j
        """

        # 1. geometry transform
        f_geo_j = self.transform_geo(f_geo_i, pose_i, pose_j)

        # 2. reconstruction
        shared_j = self.reconstruct_feature(
            f_inv_i,
            f_geo_j,
            f_app_i
        )

        return shared_j, f_geo_j