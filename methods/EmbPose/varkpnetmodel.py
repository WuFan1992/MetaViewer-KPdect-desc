import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .utils import *


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
        #desc = F.normalize(desc, dim=1)
        return desc


# -------------------------
# Variance Head
# -------------------------

class VarianceHead(nn.Module):
    def __init__(self, feat_dim=128, epsilon=0.01):
        super().__init__()
        self.epsilon = epsilon

        self.net = nn.Sequential(
            # 第1层卷积 + GroupNorm + ReLU
            nn.Conv2d(feat_dim, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),

            # 第2层卷积 + GroupNorm + ReLU
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),

            # 第3层卷积 → 输出1通道
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        """
        x: [B, feat_dim, H, W]  (可以使用 normalize 前的 descriptor)
        输出:
            var_pred: [B, 1, H, W], >= epsilon
        """
        var_pred = self.net(x)
        var_pred = F.softplus(var_pred) + self.epsilon  # 保证输出正数
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

        # backbone
        self.backbone = SharedBackbone(in_channels, feature_dim)

        # descriptor branch
        self.descriptor_encoder = DescriptorEncoder(feature_dim, feature_dim)

        # heads
        self.reliability_head = ReliabilityHead(feature_dim)
        self.variance_head = VarianceHead(feature_dim)


        # pose modules
        self.pose_encoder = PoseEncoder(pose_dim, pose_embed)

        # FiLM
        self.film = FiLMModulation(pose_embed, feature_dim)

        # decoder
        self.decoder = PointDecoder(feature_dim)
        
    
    def reconstruction(self,
                   desc_src,
                   pose_src,
                   pose_tgt):
        """
        desc_src: [B,C]
        pose_src: [B,4,4]
        pose_tgt: [B,4,4]
        """

        # relative pose
        pose_ij = pose_matrix_to_9d(pose_src, pose_tgt)

        pose_embed = self.pose_encoder(pose_ij)

        # FiLM modulation
        latent = self.film(desc_src, pose_embed)

        latent = F.normalize(latent, dim=1)

        # decoder
        pred_desc = self.decoder(latent)

        pred_desc = F.normalize(pred_desc, dim=1)

        return pred_desc
        
    
    def forward(self, img):
        """
        img: [B,3,H,W]
        pose:  [B,pose_dim]
        coords: [B,N,2] integer coordinates (y,x)
        """

        # 1. shared feature
        shared_featmap = self.backbone(img)        # [B, feat_dim, H, W]
        
        # 2. descriptor map (full map)
        desc_map = self.descriptor_encoder(shared_featmap)  # [B,feat_dim,H,W]
        
        # 2. variance map (full map)
        variance_map = self.variance_head(desc_map)  # [B,1,H,W]
        #desc_map = F.normalize(desc_map, dim=1)

        
        # 5. reliability map from descriptor
        reliability_map = self.reliability_head(desc_map)  # [B,1,H,W]

        return shared_featmap, variance_map, desc_map, reliability_map
        



