
import numpy as np
import random

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






