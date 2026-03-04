import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Encoder: 提取viewpoint invariant feature
# -------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 全局池化得到视角不变特征
        )
    
    def forward(self, x):  # x: [B, C, H, W]
        f = self.conv(x)   # [B, hidden_dim, 1, 1]
        f = f.view(f.size(0), -1)  # [B, hidden_dim]
        return f

# -------------------------
# Pose projector: 将位姿投影到特征空间
# -------------------------
class PoseProjector(nn.Module):
    def __init__(self, pose_dim=7, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, pose):
        return self.mlp(pose)

# -------------------------
# Decoder: 融合 f_inv + pose_feat_i -> xfeat_i_pred
# -------------------------
class Decoder(nn.Module):
    def __init__(self, feature_dim=128, out_channels=64, out_size=(8,8)):
        super().__init__()
        self.out_H, self.out_W = out_size
        self.fc = nn.Linear(feature_dim, feature_dim * self.out_H * self.out_W)
        self.deconv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, out_channels, 3, padding=1)
        )
    
    def forward(self, f):
        f = self.fc(f)  # [B, C*H*W]
        f = f.view(f.size(0), -1, self.out_H, self.out_W)
        x_pred = self.deconv(f)
        return x_pred

# -------------------------
# Multi-view Autoencoder 网络
# -------------------------
class MultiViewAutoEncoder(nn.Module):
    def __init__(self, in_channels=64, pose_dim=7, feature_dim=128, out_size=(8,8)):
        super().__init__()
        self.encoder = Encoder(in_channels, feature_dim)
        self.pose_projector = PoseProjector(pose_dim, feature_dim)
        self.decoder = Decoder(feature_dim, in_channels, out_size)
    
    def forward(self, xfeat, poses):
        """
        xfeat: [B, N, C, H, W]
        poses: [B, N, pose_dim]
        """
        B, N, C, H, W = xfeat.shape
        
        # 1. Encoder提取每张图片特征
        xfeat_flat = xfeat.view(B*N, C, H, W)
        f = self.encoder(xfeat_flat)  # [B*N, feature_dim]
        f = f.view(B, N, -1)          # [B, N, feature_dim]

        # 2. 计算viewpoint-invariant特征: N张特征的平均
        f_inv = f.mean(dim=1)          # [B, feature_dim]

        # 3. Pose投影
        poses_flat = poses.view(B*N, -1)
        pose_feat = self.pose_projector(poses_flat)
        pose_feat = pose_feat.view(B, N, -1)

        # 4. 融合 f_inv + pose_feat_i
        f_inv_expand = f_inv.unsqueeze(1).expand(-1, N, -1)  # [B, N, feature_dim]
        fused = f_inv_expand + pose_feat                      # 可改为concat

        fused_flat = fused.view(B*N, -1)

        # 5. Decoder预测回原始xfeat
        xfeat_pred = self.decoder(fused_flat)
        xfeat_pred = xfeat_pred.view(B, N, C, H, W)

        return xfeat_pred, f

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

# -------------------------
# 训练示例
# -------------------------
if __name__ == "__main__":
    B, N, C, H, W = 2, 5, 64, 8, 8
    pose_dim = 7
    xfeat = torch.randn(B, N, C, H, W)
    poses = torch.randn(B, N, pose_dim)

    model = MultiViewAutoEncoder(C, pose_dim, feature_dim=128, out_size=(H,W))
    xfeat_pred, f_all = model(xfeat, poses)

    # 重建损失
    recon_loss = F.mse_loss(xfeat_pred, xfeat)

    # N张图片特征聚拢loss
    intra_loss = intra_object_feature_loss(f_all)

    # 总损失
    loss = recon_loss + 0.1 * intra_loss

    print("xf_pred:", xfeat_pred.shape)  # [B, N, C, H, W]
    print("f_all:", f_all.shape)        # [B, N, feature_dim]
    print("loss:", loss.item())