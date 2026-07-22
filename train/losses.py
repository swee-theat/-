"""损失函数模块：Winner-Takes-All 轨迹损失 + 模态分类损失 + 联合损失。

核心策略（Winner-Takes-All, WTA）:
1. 计算 K 个预测轨迹与真值之间的 ADE
2. 选择 ADE 最小的模态作为"胜者"
3. 仅胜者模态的轨迹损失反向传播梯度
4. 模态选择器始终接收交叉熵损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TrajectoryLoss(nn.Module):
    """Winner-Takes-All 轨迹预测损失。

    对 K 个模态分别计算 smooth L1 损失，
    仅最佳模态（最小 ADE）贡献梯度。
    """

    def __init__(self, loss_type: str = "smooth_l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        trajectories: torch.Tensor,  # [B, K, T_pred, 2]
        ground_truth: torch.Tensor,   # [B, T_pred, 2]
    ) -> tuple:
        """计算 WTA 轨迹损失。

        参数:
            trajectories: K 个模态的预测轨迹
            ground_truth: 真实未来轨迹

        返回:
            (traj_loss, best_mode_idx)
            - traj_loss: scalar，最佳模态的 smooth L1 损失
            - best_mode_idx: [B]，每个样本的最佳模态索引
        """
        B, K, T_pred, _ = trajectories.shape
        gt = ground_truth.unsqueeze(1)  # [B, 1, T_pred, 2]

        # 计算每个模态与真值的逐点 L2 距离
        diff = trajectories - gt  # [B, K, T_pred, 2]

        if self.loss_type == "smooth_l1":
            # Smooth L1 (Huber) loss
            per_point_loss = F.smooth_l1_loss(
                trajectories,
                gt.expand(-1, K, -1, -1),
                reduction="none",
                beta=1.0,
            )  # [B, K, T_pred, 2]
            # 每个模态的总损失（对 T_pred 和 2 维求和 / 平均）
            per_mode_loss = per_point_loss.sum(dim=(-1, -2))  # [B, K]
        elif self.loss_type == "mse":
            per_mode_loss = (diff ** 2).sum(dim=(-1, -2))  # [B, K]
        elif self.loss_type == "l1":
            per_mode_loss = diff.abs().sum(dim=(-1, -2))  # [B, K]
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

        # WTA: 选择损失最小的模态
        best_mode_loss, best_mode_idx = per_mode_loss.min(dim=-1)  # [B], [B]

        # 平均所有样本的胜者损失
        traj_loss = best_mode_loss.mean()

        return traj_loss, best_mode_idx


class ModeLoss(nn.Module):
    """模态选择交叉熵损失。

    对模态选择器使用硬标签（one-hot at best_mode）。
    硬标签鼓励模态专业化，而非分摊概率。
    """

    def forward(
        self,
        mode_logits: torch.Tensor,  # [B, K]
        best_mode_idx: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """计算模态分类交叉熵损失。

        参数:
            mode_logits: 模态 logits（未经 softmax）
            best_mode_idx: 最佳模态索引（来自 WTA）

        返回:
            mode_loss: scalar 交叉熵损失
        """
        return F.cross_entropy(mode_logits, best_mode_idx)


class CombinedLoss(nn.Module):
    """联合损失: L = λ_traj * L_traj + λ_mode * L_mode。

    默认权重: λ_traj=0.7, λ_mode=0.3
    """

    def __init__(
        self,
        lambda_traj: float = 0.7,
        lambda_mode: float = 0.3,
        traj_loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.lambda_traj = lambda_traj
        self.lambda_mode = lambda_mode
        self.traj_loss_fn = TrajectoryLoss(loss_type=traj_loss_type)
        self.mode_loss_fn = ModeLoss()

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        ground_truth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算联合损失。

        参数:
            model_output: 模型输出字典，包含 trajectories, mode_logits
            ground_truth: [B, T_pred, 2] 真实未来轨迹

        返回:
            {
                "loss": 联合总损失,
                "traj_loss": WTA 轨迹损失,
                "mode_loss": 模态分类损失,
                "best_mode_idx": 最佳模态索引 [B],
            }
        """
        trajectories = model_output["trajectories"]  # [B, K, T_pred, 2]
        mode_logits = model_output["mode_logits"]    # [B, K]

        # 计算 WTA 轨迹损失
        traj_loss, best_mode_idx = self.traj_loss_fn(trajectories, ground_truth)

        # 计算模态分类损失
        mode_loss = self.mode_loss_fn(mode_logits, best_mode_idx)

        # 联合损失
        total_loss = self.lambda_traj * traj_loss + self.lambda_mode * mode_loss

        return {
            "loss": total_loss,
            "traj_loss": traj_loss,
            "mode_loss": mode_loss,
            "best_mode_idx": best_mode_idx,
        }
