import torch
import torch.nn.functional as F


"""
6D rotation 是很多视觉模型使用的稳定表示（来自论文 On the Continuity of Rotation Representations in Neural Networks）。

"""

def compute_relative_pose(pose_i, pose_j):
    """
    pose_i: [B,4,4]
    pose_j: [B,4,4]

    return:
        R_ij [B,3,3]
        t_ij [B,3]
    """

    pose_i_inv = torch.inverse(pose_i)

    T_ij = torch.matmul(pose_j, pose_i_inv)

    R_ij = T_ij[:, :3, :3]
    t_ij = T_ij[:, :3, 3]

    # ⭐ translation normalization（非常重要）
    t_ij = t_ij / (torch.norm(t_ij, dim=1, keepdim=True) + 1e-8)

    return R_ij, t_ij


def rotation_matrix_to_6d(R):
    """
    R: [B,3,3]
    return: [B,6]
    """

    return R[:, :, :2].reshape(R.shape[0], 6)


def pose_matrix_to_9d(pose_i, pose_j):
    """
    pose_i: [B,4,4]
    pose_j: [B,4,4]

    return:
        pose_9d: [B,9]
    """

    R_ij, t_ij = compute_relative_pose(pose_i, pose_j)

    rot6d = rotation_matrix_to_6d(R_ij)

    pose_9d = torch.cat([rot6d, t_ij], dim=1)

    return pose_9d


class DynamicLossWeights:
    def __init__(self, init_weights=None, target_contrib=0.25, momentum=0.9):
        """
        init_weights: 初始权重 dict, e.g. {'var':0.2,'ds':0.1,'kp':5,'rec':0.5}
        target_contrib: 每个任务期望贡献到总 loss 的比例
        momentum: 平滑更新
        """
        self.weights = init_weights if init_weights is not None else {'var':0.2,'ds':0.1,'kp':5,'rec':0.5}
        self.target = target_contrib
        self.momentum = momentum
        # 保存历史平均 loss
        self.loss_ema = {'var': 1.0, 'ds': 1.0, 'kp': 1.0, 'rec': 1.0}

    def update_ema(self, losses_dict):
        """
        losses_dict: 当前 iteration 的各任务 loss, e.g. {'var':1.42,'ds':2.55,...}
        """
        for k in losses_dict:
            self.loss_ema[k] = self.momentum*self.loss_ema[k] + (1-self.momentum)*losses_dict[k]

    def get_weights(self):
        """
        根据 EMA loss 计算权重，使每个任务贡献接近 target
        w_task = target / loss_task_mean
        """
        new_weights = {}
        for k, l in self.loss_ema.items():
            new_weights[k] = self.target / (l + 1e-6)
        self.weights = new_weights
        return self.weights
