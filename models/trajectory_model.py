"""顶层轨迹预测模型：组装编码器 + 多模态预测器 + 不确定性估计器。

完整数据流:
    历史轨迹 [B, T_obs, input_dim]
    → TrajectoryEncoder → [B, hidden_dim]
    → MultimodalPredictor → trajectories [B, K, T_pred, 2] + mode_probs [B, K]
    → UncertaintyEstimator → uncertainties [B, K, T_pred, 2]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from .trajectory_encoder import TrajectoryEncoder
from .multimodal_predictor import MultimodalPredictor
from .uncertainty_estimator import UncertaintyEstimator


class TrajectoryModel(nn.Module):
    """轻量化多模态轨迹预测模型。

    目标参数量: ~198K
    目标硬件: RTX 3060 (6GB) 可完整训练
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 128,
        num_mlp_layers: int = 3,
        dropout: float = 0.1,
        num_modes: int = 3,
        pred_len: int = 30,
        pooling_type: str = "attention",
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.pred_len = pred_len
        self.use_uncertainty = use_uncertainty

        # 轨迹编码器
        self.encoder = TrajectoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_mlp_layers=num_mlp_layers,
            dropout=dropout,
            pooling_type=pooling_type,
        )

        # 多模态预测器
        self.predictor = MultimodalPredictor(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            pred_len=pred_len,
            dropout=dropout,
        )

        # 不确定性估计器（可选）
        if use_uncertainty:
            self.uncertainty = UncertaintyEstimator(
                hidden_dim=hidden_dim,
                num_modes=num_modes,
                pred_len=pred_len,
                dropout=dropout,
                share_across_modes=True,
            )
        else:
            self.uncertainty = None

    def forward(
        self,
        history: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        参数:
            history: [B, T_obs, input_dim] 历史轨迹特征
            return_attn_weights: 是否返回注意力权重（用于可视化）

        返回:
            {
                "trajectories":     [B, K, T_pred, 2],
                "mode_probs":       [B, K],
                "mode_logits":      [B, K],
                "uncertainties":    [B, K, T_pred, 2] (if use_uncertainty),
                "encoded_feature":  [B, hidden_dim],
                "attn_weights":     [B, T_obs] (if return_attn_weights),
            }
        """
        # 编码
        encoded, attn_weights = self.encoder(
            history, return_attn_weights=return_attn_weights
        )

        # 多模态预测
        pred_output = self.predictor(encoded)

        # 不确定性估计
        if self.uncertainty is not None:
            uncertainties = self.uncertainty(encoded)
        else:
            uncertainties = None

        output = {
            "trajectories": pred_output["trajectories"],
            "mode_probs": pred_output["mode_probs"],
            "mode_logits": pred_output["mode_logits"],
            "uncertainties": uncertainties,
            "encoded_feature": encoded,
        }

        if return_attn_weights:
            output["attn_weights"] = attn_weights

        return output

    def predict(
        self,
        history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """推理模式预测（no_grad + eval）。

        参数:
            history: [B, T_obs, input_dim]

        返回:
            与 forward() 相同的输出字典
        """
        self.eval()
        with torch.no_grad():
            return self.forward(history)

    def get_best_trajectory(
        self, history: torch.Tensor
    ) -> torch.Tensor:
        """获取最可能的单条预测轨迹。

        参数:
            history: [B, T_obs, input_dim]

        返回:
            best_traj: [B, T_pred, 2] 概率最高的模态轨迹
        """
        output = self.predict(history)
        best_mode_idx = torch.argmax(output["mode_probs"], dim=-1)  # [B]
        B = history.shape[0]
        best_traj = output["trajectories"][
            torch.arange(B, device=history.device), best_mode_idx
        ]  # [B, T_pred, 2]
        return best_traj
