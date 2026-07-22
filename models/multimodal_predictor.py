"""多模态预测器：模态选择 + K 个独立轨迹预测分支。

数据流:
    编码特征 [B, hidden_dim]
    ├── 模态选择器 → mode_probs [B, K]
    └── K 个轨迹分支 → trajectories [B, K, T_pred, 2]
"""

import torch
import torch.nn as nn


class MultimodalPredictor(nn.Module):
    """多模态轨迹预测头。

    包含：
    - 模态选择器：输出每个模态的概率
    - K 个独立轨迹分支：每个分支预测完整未来轨迹
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_modes: int = 3,
        pred_len: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.pred_len = pred_len
        self.output_dim = pred_len * 2  # 每个模态输出 T_pred * 2 (x, y)

        # 模态选择器
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modes),
        )

        # K 个独立的轨迹预测分支
        # 每个分支: hidden_dim → hidden_dim → T_pred*2
        self.traj_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )
            for _ in range(num_modes)
        ])

        # 用不同随机种子初始化各分支的最后一层，鼓励模态多样性
        self._init_diverse_branches()

    def _init_diverse_branches(self):
        """用不同的随机种子初始化各分支，防止模态坍缩。"""
        for k, branch in enumerate(self.traj_branches):
            # 最后一层 Linear 用不同种子
            last_linear = branch[-1]
            generator = torch.Generator()
            generator.manual_seed(42 + k * 100)
            nn.init.xavier_uniform_(last_linear.weight, generator=generator)
            nn.init.zeros_(last_linear.bias)

    def forward(
        self, encoded_feature: torch.Tensor
    ) -> dict:
        """前向传播。

        参数:
            encoded_feature: [B, hidden_dim] 编码后的场景表征

        返回:
            {
                "trajectories": [B, K, T_pred, 2],
                "mode_probs": [B, K],
                "mode_logits": [B, K],
            }
        """
        B = encoded_feature.shape[0]

        # 模态概率
        mode_logits = self.mode_selector(encoded_feature)  # [B, K]
        mode_probs = torch.softmax(mode_logits, dim=-1)

        # K 个轨迹分支分别预测
        trajectories = []
        for branch in self.traj_branches:
            traj_flat = branch(encoded_feature)  # [B, T_pred*2]
            traj = traj_flat.view(B, self.pred_len, 2)  # [B, T_pred, 2]
            trajectories.append(traj)

        # 堆叠 K 个模态 [B, K, T_pred, 2]
        trajectories = torch.stack(trajectories, dim=1)

        return {
            "trajectories": trajectories,
            "mode_probs": mode_probs,
            "mode_logits": mode_logits,
        }
