import torch
import os
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

def plot_keypoints_on_image(image, keypoints, point_color='r', point_size=40, show_axis=False):
    """
    在图片上绘制 keypoints 并显示。

    参数：
    - image: numpy array 或 tensor, 形状 HxW 或 HxWxC
    - keypoints: torch.Tensor 或 numpy array, 形状 N x 2, 每行是 [x, y]
    - point_color: str, keypoints 的颜色 (默认红色 'r')
    - point_size: int, keypoints 的大小 (默认 40)
    - show_axis: bool, 是否显示坐标轴 (默认 False)
    """
    # 如果 keypoints 是 torch tensor，转成 numpy
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()

    plt.imshow(image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c=point_color, s=point_size)
    if not show_axis:
        plt.axis('off')
    plt.show()

def plot_matched_keypoints(image1, keypoints1, image2, keypoints2,
                           point_color='r', line_color='b', point_size=40, show_axis=False):
    """
    在两幅图片上绘制匹配关键点，并用线连接匹配点。

    参数：
    - image1, image2: numpy array 或 tensor, 形状 HxW 或 HxWxC
    - keypoints1, keypoints2: torch.Tensor 或 numpy array, 形状 N x 2
    - point_color: str, keypoints 的颜色 (默认红色 'r')
    - line_color: str, 匹配线的颜色 (默认蓝色 'b')
    - point_size: int, keypoints 的大小
    - show_axis: bool, 是否显示坐标轴
    """
    # 转成 numpy
    if isinstance(keypoints1, torch.Tensor):
        keypoints1 = keypoints1.cpu().numpy()
    if isinstance(keypoints2, torch.Tensor):
        keypoints2 = keypoints2.cpu().numpy()

    # 如果是灰度图，保证是 HxW
    if len(image1.shape) == 2:
        image1 = np.stack([image1]*3, axis=-1)
    if len(image2.shape) == 2:
        image2 = np.stack([image2]*3, axis=-1)

    # 并排显示两幅图
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    new_h = max(h1, h2)
    new_w = w1 + w2
    new_image = np.zeros((new_h, new_w, 3), dtype=image1.dtype)
    new_image[:h1, :w1, :] = image1
    new_image[:h2, w1:w1+w2, :] = image2

    plt.figure(figsize=(12, 6))
    plt.imshow(new_image)

    # 绘制 keypoints
    plt.scatter(keypoints1[:, 0], keypoints1[:, 1], c=point_color, s=point_size)
    plt.scatter(keypoints2[:, 0] + w1, keypoints2[:, 1], c=point_color, s=point_size)  # x 偏移

    # 绘制匹配线
    for (x1, y1), (x2, y2) in zip(keypoints1, keypoints2):
        plt.plot([x1, x2 + w1], [y1, y2], c=line_color, linewidth=1)

    if not show_axis:
        plt.axis('off')
    plt.show()


from methods.EmbPose.varkpnet import VarianceKPNet

var_kpnet = VarianceKPNet(os.path.abspath(os.path.dirname(__file__)) + '/../checkpoints/variancekpnet_8000.pth', 50)
img0_tensor = io.imread("datasets/head/images/seq-02/frame-000147.color.png")
img1_tensor = io.imread("datasets/head/images/seq-01/frame-000463.color.png")
res0 = var_kpnet.detectAndCompute(img0_tensor)
res1 = var_kpnet.detectAndCompute(img1_tensor)

idx0, idx1 = var_kpnet.match(res0[0]["descriptors"], res1[0]["descriptors"])

plot_matched_keypoints(img0_tensor, res0[0]["keypoints"][idx0], img1_tensor, res1[0]["keypoints"][idx1])



