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
from .lane_encoder import LaneEncoder


class TrajectoryModel(nn.Module):
    """轻量化多模态轨迹预测模型。

    目标参数量:
        - 独立不确定性头(share_uncertainty=False): ~244K
        - 共享不确定性头(share_uncertainty=True): ~195K（为车道线+交互扩展模块腾预算）
    目标硬件: RTX 3060 (6GB) 可完整训练
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_mlp_layers: int = 3,
        dropout: float = 0.1,
        num_modes: int = 3,
        pred_len: int = 30,
        pooling_type: str = "attention",
        use_uncertainty: bool = True,
        use_temporal_conv: bool = True,
        share_uncertainty: bool = False,
        use_lane: bool = True,
        use_city: bool = True,
        branch_hidden_dim: int = 96,
        num_cities: int = 7,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.pred_len = pred_len
        self.use_uncertainty = use_uncertainty
        self.use_lane = use_lane
        self.use_city = use_city

        # 轨迹编码器
        self.encoder = TrajectoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_mlp_layers=num_mlp_layers,
            dropout=dropout,
            pooling_type=pooling_type,
            use_temporal_conv=use_temporal_conv,
        )

        # 车道线编码器（可选）
        if use_lane:
            self.lane_encoder = LaneEncoder(
                node_dim=6, hidden_dim=hidden_dim, num_layers=3, dropout=dropout
            )
        else:
            self.lane_encoder = None

        # 城市 embedding（可选，域感知偏置）
        if use_city:
            self.city_embed = nn.Embedding(num_cities, hidden_dim)
        else:
            self.city_embed = None

        # 多模态预测器（branch_hidden_dim 压缩中间层，第二阶段设 96 省约 30K）
        self.predictor = MultimodalPredictor(
            hidden_dim=hidden_dim,
            num_modes=num_modes,
            pred_len=pred_len,
            dropout=dropout,
            branch_hidden_dim=branch_hidden_dim,
        )

        # 不确定性估计器（可选）
        if use_uncertainty:
            self.uncertainty = UncertaintyEstimator(
                hidden_dim=hidden_dim,
                num_modes=num_modes,
                pred_len=pred_len,
                dropout=dropout,
                # 独立头(False)：每模态各自方差；共享头(True)：省约 48K 参数，
                # 用于容纳车道线+交互两个扩展模块、满足总参数 ≤288K。
                share_across_modes=share_uncertainty,
            )
        else:
            self.uncertainty = None

    def forward(
        self,
        history: torch.Tensor,
        lane_nodes: Optional[torch.Tensor] = None,
        lane_adj: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        city_id: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        参数:
            history: [B, T_obs, input_dim] 历史轨迹特征
            lane_nodes: [B, N, 6] 车道线节点特征（可选）
            lane_adj: [B, N, N] 车道线邻接矩阵（可选）
            lane_mask: [B, N] 车道线节点掩码（可选）
            city_id: [B] 城市 ID（可选，域感知偏置）
            return_attn_weights: 是否返回注意力权重（用于可视化）

        返回:
            {
                "trajectories":     [B, K, T_pred, 2],
                "mode_probs":       [B, K],
                "mode_logits":      [B, K],
                "uncertainties":    [B, K, T_pred, 2] (if use_uncertainty),
                "encoded_feature":  [B, hidden_dim],
                "attn_weights":     [B, T+N] 或 [B, T_obs] (if return_attn_weights),
            }
        """
        # 编码（取池化前特征 [B, T, H]）
        enc_feats, _ = self.encoder(history, pool=False)

        # 车道线编码 + 注意力池化阶段融合（轨迹帧 + 车道节点一起池化）
        if self.use_lane and lane_nodes is not None and self.lane_encoder is not None:
            lane_feats = self.lane_encoder(lane_nodes, lane_adj, lane_mask)  # [B, N, H]
            fused = torch.cat([enc_feats, lane_feats], dim=1)  # [B, T+N, H]
        else:
            fused = enc_feats  # [B, T, H]

        # 统一注意力池化 [B, T(+N), H] → [B, H]
        encoded, attn_weights = self.encoder.pooling(
            fused, return_weights=return_attn_weights
        )

        # 城市特征（additive 偏置）
        if self.use_city and city_id is not None and self.city_embed is not None:
            encoded = encoded + self.city_embed(city_id)

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
        lane_nodes: Optional[torch.Tensor] = None,
        lane_adj: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        city_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """推理模式预测（no_grad + eval）。

        参数:
            history: [B, T_obs, input_dim]
            其余为可选车道线/城市输入，见 forward()

        返回:
            与 forward() 相同的输出字典
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                history,
                lane_nodes=lane_nodes,
                lane_adj=lane_adj,
                lane_mask=lane_mask,
                city_id=city_id,
            )

    def get_best_trajectory(
        self,
        history: torch.Tensor,
        lane_nodes: Optional[torch.Tensor] = None,
        lane_adj: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        city_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取最可能的单条预测轨迹。

        参数:
            history: [B, T_obs, input_dim]
            其余为可选车道线/城市输入，见 forward()

        返回:
            best_traj: [B, T_pred, 2] 概率最高的模态轨迹
        """
        output = self.predict(
            history,
            lane_nodes=lane_nodes,
            lane_adj=lane_adj,
            lane_mask=lane_mask,
            city_id=city_id,
        )
        best_mode_idx = torch.argmax(output["mode_probs"], dim=-1)  # [B]
        B = history.shape[0]
        best_traj = output["trajectories"][
            torch.arange(B, device=history.device), best_mode_idx
        ]  # [B, T_pred, 2]
        return best_traj
