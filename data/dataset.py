"""PyTorch Dataset 类：加载预处理好的轨迹数据。"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict


class ArgoverseTrajectoryDataset(Dataset):
    """Argoverse 轨迹预测数据集。

    每个样本:
        history:  [T_obs, 11] 历史轨迹特征 (动态维度，取决于 preprocessing 输出)
        future:   [T_pred, 2]  未来轨迹真值 (x, y)
        category: int          场景类型 (0-4)
        seq_id:   str          序列标识（用于调试）
    """

    CATEGORY_NAMES = {
        0: "直行",
        1: "左转",
        2: "右转",
        3: "掉头",
        4: "路口",
    }

    def __init__(self, npz_path: str):
        """从预处理好的 .npz 文件加载数据集。

        参数:
            npz_path: .npz 文件路径，包含 histories, futures, categories, seq_ids
        """
        data = np.load(npz_path, allow_pickle=True)

        self.histories = data["histories"]      # [N, T_obs, input_dim]
        self.futures = data["futures"]            # [N, T_pred, 2]
        self.categories = data["categories"]      # [N]
        self.seq_ids = data["seq_ids"]            # [N]

        # 验证数据形状
        assert self.histories.shape[1] >= 2, f"历史帧数异常: {self.histories.shape}"
        assert self.futures.shape[1] >= 2, f"未来帧数异常: {self.futures.shape}"
        assert len(self.histories) == len(self.futures), "历史和未来样本数不一致"

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        history = torch.from_numpy(self.histories[idx].copy()).float()
        future = torch.from_numpy(self.futures[idx].copy()).float()
        category = int(self.categories[idx])
        seq_id = str(self.seq_ids[idx])

        return {
            "history": history,     # [T_obs, input_dim]
            "future": future,        # [T_pred, 2]
            "category": category,    # int
            "seq_id": seq_id,        # str
        }

    @property
    def obs_len(self) -> int:
        return self.histories.shape[1]

    @property
    def pred_len(self) -> int:
        return self.futures.shape[1]

    @property
    def input_dim(self) -> int:
        return self.histories.shape[2]

    def get_category_stats(self) -> Dict[str, int]:
        """返回每个场景类型的样本数量统计。"""
        stats = {}
        for label, name in self.CATEGORY_NAMES.items():
            count = int(np.sum(self.categories == label))
            stats[name] = count
        return stats
