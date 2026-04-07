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
from torchvision.models import mobilenet_v3_small
class SharedBackbone_Light(nn.Module):
    def __init__(self, out_dim=128, freeze=True):
        super().__init__()

        # === 使用轻量 backbone ===
        backbone = mobilenet_v3_small(pretrained=True).features  # output [B,576,H/8,W/8]
        self.student = backbone
        self.out_dim = out_dim

        # 投影到指定维度
        self.proj = nn.Sequential(
            nn.Conv2d(576, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True)
        )

        if freeze:
            for p in self.student.parameters():
                p.requires_grad = False


    def forward(self, x):
        # student 特征（描述子用）
        feat = self.student(x)           # [B,576,H/8,W/8]
        feat = self.proj(feat)           # [B,out_dim,H/8,W/8]
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        return feat

    # teacher 用的纯特征（variance head 输入）
    def forward_mobilenet(self, x):
        with torch.no_grad():
            feat = self.student(x)
            feat = self.proj(feat)
            feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        return feat
"""
from torchvision.models import mobilenet_v2


class SharedBackbone_MobileNet(nn.Module):
    def __init__(self, out_dim=128, freeze=False):
        super().__init__()

        mobilenet = mobilenet_v2(pretrained=True)

        # 取 feature extractor（stride 32 → 改成 stride 16）
        self.features = mobilenet.features[:14]  # 到 stride 16

        self.proj = nn.Sequential(
            nn.Conv2d(96, out_dim, 1),
            nn.GroupNorm(8, out_dim),
            nn.ReLU(inplace=True)
        )

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.features(x)  # [B,96,H/16,W/16]
        feat = self.proj(feat)
        feat = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False)
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
        f_geo = F.normalize(self.geo_head(x))
        f_app = self.app_head(x)
        return f_inv, f_geo, f_app


# -------------------------
# Variance Branch (Teacher-Student)
# -------------------------
class VarianceHead(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()   # output [0,1]
        )

    def forward(self, feat_teacher):
        # feat_teacher: [B,C,H,W]
        sigma_pred = self.net(feat_teacher)
        return sigma_pred


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

        # ✅ 修改初始化：让 Sigmoid 输出稍偏正
        final_conv = self.net[-2]  # 取倒数第二层 Conv2d
        nn.init.kaiming_uniform_(final_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        final_conv.bias.data.fill_(0.2)  # bias 初始为 0.2 → Sigmoid 输出 ~0.55-0.6

    def forward(self, x):
        return self.net(x)
    
        
class HeatmapHead(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)   # [B,1,H,W]
    

# -------------------------
# Dense Feature AutoEncoder
# -------------------------

class VUDNet(nn.Module):
    def __init__(self,
                 feature_dim=128,
                 dim_geo=32,
                 dim_app=16):
        super().__init__()



        # backbone
        self.backbone = SharedBackbone_MobileNet(out_dim=feature_dim)

        # disentangle
        self.encoder = TripleDescriptorHead(
            in_dim=feature_dim,
            dim_inv=feature_dim,
            dim_geo=dim_geo,
            dim_app=dim_app
        )

        
        # independent variance branch (teacher-student)
        self.variance_branch = VarianceHead(feature_dim)
        
        # reliability head
        self.reliability_head = ReliabilityHead(feature_dim)
        
        # heatmap
        self.heatmap_head = HeatmapHead(feature_dim)
        

    def forward(self, img):

        shared_feat = self.backbone(img)

        f_inv, f_geo, _ = self.encoder(shared_feat)
        
        # -------- ✅ reliability（关键新增）--------
        reliability = self.reliability_head(shared_feat)
        
        sigma_pred = self.variance_branch(shared_feat)
        
        # heatmap
        heatmap = self.heatmap_head(shared_feat)
        
        out = {
            "f_inv": f_inv,
            "f_geo": f_geo,
            "reliability": reliability, 
            "sigma": sigma_pred, 
            "heatmap": heatmap
        }


        return out
        
