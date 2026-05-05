import torch.nn as nn
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from methods.EmbPose.varkpnetmodel import *


def pad_to_same_height(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    
    max_h = max(h1, h2)
    
    def pad(img, target_h):
        h, w, c = img.shape
        pad_h = target_h - h
        return np.pad(img, ((0,pad_h),(0,0),(0,0)), mode='constant')
    
    return pad(img1, max_h), pad(img2, max_h)

def build(checkpoint_path="checkpoints/kpnet_iter_thirtyscene_pcorrect+conf_150000.pth" ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化网络
    net= VUDNet(feature_dim=64, dim_geo=32, dim_app=16)
    net = net.to(device)

    # 加载训练好的权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint)  # 假设你保存的是 model_state_dict
    net.eval()
    return net


class VUDNet_helper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def load_image(self, path, device):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2,0,1).unsqueeze(0).to(device)
        return img_tensor, img
    
    def extract_keypoints(self, heatmap, reliability, sigma, img_shape, num_keypoints=4096):
        hmap = heatmap.squeeze().cpu().numpy()
        rel = reliability.squeeze().cpu().numpy()
        sig = sigma.squeeze().cpu().numpy()
        
        score = hmap * rel * (1 - sig)

        # ✅ ====== 新增：mask 掉边界 ======
        margin = 16   # 👉 可以调：8 / 16 / 32
        score[:margin, :] = 0
        score[-margin:, :] = 0
        score[:, :margin] = 0
        score[:, -margin:] = 0

        # 提取候选点
        coords = np.argwhere(score > 0.0)[:, [1, 0]]  # (x,y)
        scores = score[score > 0.0]

        # 排序
        idx = np.argsort(scores)[::-1]
        coords = coords[idx]
        scores = scores[idx]

        # score阈值过滤
        mask = scores > 0.001
        coords = coords[mask]
        scores = scores[mask]
        #print(f"{len(coords)}")

        # top-K
        coords = coords[:num_keypoints]
        scores = scores[:num_keypoints]

        # 映射回原图
        Hf, Wf = hmap.shape
        Hi, Wi = img_shape[:2]
    
        scale_x = Wi / Wf
        scale_y = Hi / Hf
    
        coords = coords.astype(np.float32)
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
    
        return coords, scores

    def coords_to_feat(self, feat_map, coords, img_shape):
        # feat_map: [1,C,H,W], coords: [N,2] 原图坐标
        B,C,H,W = feat_map.shape
        x = np.clip((coords[:,0] * W / img_shape[1]).astype(int), 0, W-1)
        y = np.clip((coords[:,1] * H / img_shape[0]).astype(int), 0, H-1)
        idx = y * W + x
        feat = feat_map.squeeze(0).permute(1,2,0).reshape(-1,C)[idx]
        return F.normalize(feat, dim=1).cpu().numpy()
    
    def match_keypoints(self, f1, f2):
        # f1, f2: [N,C]
        sim = f1 @ f2.T
        

        # 正向最近邻
        idx12 = np.argmax(sim, axis=1)
        # 反向最近邻
        idx21 = np.argmax(sim, axis=0)

        # mutual nearest neighbor
        matches = []
        for i, j in enumerate(idx12):
            if idx21[j] == i:
                matches.append([i, j])

        matches = np.array(matches)
        return matches
    
    
    def match(self, img1_path, img2_path, top_k=4096):
        
        img_tensor1, img1 = self.load_image(img1_path, "cuda")
        img_tensor2, img2 = self.load_image(img2_path, "cuda")
        
        
        out1 = self.model(img_tensor1)
        out2 = self.model(img_tensor2)

        coords1, scores1 = self.extract_keypoints(
            out1['heatmap'], out1['reliability'], out1['sigma'], img1.shape, top_k
        )
        coords2, scores2 = self.extract_keypoints(
            out2['heatmap'], out2['reliability'], out2['sigma'], img2.shape, top_k
        )
        
        # ✅ ====== 关键修改（只加这一段）======
        if len(coords1) == 0 or len(coords2) == 0:
            return np.zeros((0,2)), np.zeros((0,2)), np.array([]), np.array([])
        # ====================================== 

        f1 = self.coords_to_feat(out1['f_inv'], coords1, img1.shape)
        f2 = self.coords_to_feat(out2['f_inv'], coords2, img2.shape)

        matches = self.match_keypoints(f1, f2)
        # ✅ 再加一个保险（防止 match 为空）
        if len(matches) == 0:
            return np.zeros((0,2)), np.zeros((0,2)), np.array([]), np.array([])
        
        # Debug
        #self.visualize(img1, img2, coords1, coords2, matches)

        return coords1[matches[:,0]], coords2[matches[:,1]], scores1[matches[:,0]], scores2[matches[:,1]]
    
    
    def visualize(self, img1, img2, coords1, coords2, matches):
    
        # ✅ ====== 新增：padding ======
        img1, img2 = pad_to_same_height(img1, img2)

        concat_img = np.concatenate([img1, img2], axis=1)
    
        plt.figure(figsize=(15,7))
        plt.imshow(concat_img)
    
        for m in matches:
            pt1 = coords1[m[0]]
            pt2 = coords2[m[1]] + np.array([img1.shape[1],0])
        
            plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],'r',linewidth=1)
    
        plt.title("Keypoint Matches")
        plt.axis('off')
        plt.show()

