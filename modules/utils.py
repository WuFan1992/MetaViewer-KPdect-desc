
import numpy as np
import math
import os.path as osp
import cv2
from skimage.io import imread
# -----------------------------
# 计算相机中心
# -----------------------------
def compute_camera_center(qvec, tvec):
    """qvec: quaternion, tvec: translation"""
    # 先转为旋转矩阵
    w, x, y, z = qvec
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y- z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x**2 + y**2)]
    ])
    C = -R.T @ tvec
    return C

# -----------------------------
# 计算 viewing angle
# -----------------------------
def compute_viewing_angle(X, C, ref_C):
    v1 = X - C
    v2 = X - ref_C
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_theta)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getIntrinsic(FoVx, width, height):
    K = np.eye(3)
    focal_length = fov2focal(FoVx, width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    return K

def getExtrinsic(R, t):
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3,3] = t
    return pose


def load_depth_from_png(tiff_file_path):
    depth = cv2.imread(tiff_file_path, cv2.IMREAD_ANYDEPTH)
    depth[depth==65535]=0
    return depth



def load_raw_data(
    base_dir, scene_info, idx, read_img=True):
    
    
    pose = scene_info["pose_list"][idx]
    K = scene_info["intrinsics_list"][idx]
    
    file_name = scene_info["image_name_list"][idx]
    depth_file_name = scene_info["depth_name_list"][idx]

    img = None
    img_path = osp.join(base_dir, file_name)
    if not osp.isfile(img_path):
        print("ERROR: img_path does not exist:", img_path)

    depth_file_name = osp.join(base_dir, depth_file_name)
    if read_img:
        img = imread(img_path)
    depth = load_depth_from_png(depth_file_name)

    depth=depth.astype(np.float32)/1000
    depth[depth < 1e-5] = 0
    return img, depth, pose, K


def load_data(base_dir, scene_info, idx):

    img, depth, pose, K = load_raw_data(base_dir, scene_info, idx, read_img=True)

    img = img.astype(np.float32).transpose(2,0,1)

    return {
        "pose": pose.astype(np.float32),
        "K": K.astype(np.float32),
        "depth": depth.astype(np.float32),
        "img": img
    }


