"""数据增强模块：对轨迹历史帧进行随机旋转、缩放和加噪。"""

import torch
import numpy as np


class TrajectoryAugmentation:
    """轨迹数据增强，包含随机旋转、随机缩放和高斯噪声。

    所有增强仅作用于历史帧，未来帧同步变换以保证一致性。
    """

    def __init__(
        self,
        enable: bool = True,
        rotation_range: float = 3.14159,  # [-pi, pi]
        scale_range: tuple = (0.9, 1.1),
        noise_std: float = 0.02,
    ):
        self.enable = enable
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std

    def _random_rotation_matrix(self, batch_size: int, device: torch.device) -> tuple:
        """生成随机旋转矩阵 [B, 2, 2] 及对应的旋转角 [B]。"""
        angle = (torch.rand(batch_size, device=device) * 2 - 1) * self.rotation_range
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot = torch.stack([
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1),
        ], dim=-2)
        return rot, angle

    def __call__(
        self,
        history: torch.Tensor,  # [B, T_obs, 11]
        future: torch.Tensor,    # [B, T_pred, 2]
    ) -> tuple:
        """对一批轨迹应用数据增强。

        参数:
            history: 历史轨迹 [B, T_obs, 11]（11 维含航向角/曲率）
            future: 未来轨迹真值 [B, T_pred, 2]

        返回:
            增强后的 (history, future) 元组
        """
        if not self.enable:
            return history, future

        B = history.shape[0]
        device = history.device

        # 1. 随机旋转：对 x,y 坐标和 vx,vy 速度分量同步旋转
        rot, angle = self._random_rotation_matrix(B, device)  # rot [B,2,2], angle [B]

        # 旋转 position (x, y)
        pos = history[:, :, :2]  # [B, T, 2]
        pos_rot = torch.bmm(pos, rot.transpose(-2, -1))  # [B, T, 2]
        history = torch.cat([pos_rot, history[:, :, 2:]], dim=-1)

        # 旋转 velocity (vx, vy)
        vel = history[:, :, 2:4]  # [B, T, 2]
        vel_rot = torch.bmm(vel, rot.transpose(-2, -1))  # [B, T, 2]
        history[:, :, 2:4] = vel_rot

        # 更新航向角（index 9）：坐标旋转 angle 后，航向角同步 + angle，
        # 否则航向角特征与旋转后的 (vx, vy) 方向矛盾。
        # 曲率（index 10）是航向角差分，旋转不变量，无需更新。
        if history.shape[-1] > 9:
            history[:, :, 9] = history[:, :, 9] + angle.unsqueeze(-1)

        # 旋转未来轨迹
        future_rot = torch.bmm(future, rot.transpose(-2, -1))  # [B, T, 2]
        future = future_rot

        # 2. 随机缩放
        scale = (
            torch.rand(B, 1, 1, device=device) * (self.scale_range[1] - self.scale_range[0])
            + self.scale_range[0]
        )
        history[:, :, :2] *= scale
        history[:, :, 2:4] *= scale
        future *= scale

        # 也缩放距离相关特征（修复广播：scale 需与 history 同维度）
        if history.shape[-1] >= 7:
            scale_3d = scale.squeeze(-1).unsqueeze(-1)  # [B, 1] → [B, 1, 1]
            history[:, :, 6:9] *= scale_3d

        # 3. 高斯噪声（仅对位置坐标）
        noise = torch.randn_like(history[:, :, :2]) * self.noise_std
        history[:, :, :2] += noise

        return history, future
