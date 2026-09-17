"""DataLoader 封装：批量加载、数据增强、划分数据集。"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from .augmentation import TrajectoryAugmentation


def collate_fn(batch: list) -> dict:
    """自定义 batch 拼接函数。

    参数:
        batch: list of dict, 每个 dict 包含 history, future, category, seq_id
               （第二阶段另含 lane_nodes/lane_adj/lane_mask/city_id）

    返回:
        合并后的 batch dict（tensor 在 dim=0 拼接）
    """
    histories = torch.stack([item["history"] for item in batch])
    futures = torch.stack([item["future"] for item in batch])
    categories = torch.tensor([item["category"] for item in batch], dtype=torch.long)
    seq_ids = [item["seq_id"] for item in batch]

    # 第二阶段新增字段（车道线 + 城市；兼容不含这些字段的旧 dataset）
    if all("lane_nodes" in item for item in batch):
        lane_nodes = torch.stack([item["lane_nodes"] for item in batch])
        lane_adj = torch.stack([item["lane_adj"] for item in batch])
        lane_mask = torch.stack([item["lane_mask"] for item in batch])
        city_ids = torch.tensor([item["city_id"] for item in batch], dtype=torch.long)
        return {
            "history": histories,       # [B, T_obs, input_dim]
            "future": futures,          # [B, T_pred, 2]
            "category": categories,     # [B]
            "seq_id": seq_ids,          # list[str]
            "lane_nodes": lane_nodes,   # [B, 32, 6]
            "lane_adj": lane_adj,       # [B, 32, 32]
            "lane_mask": lane_mask,     # [B, 32]
            "city_id": city_ids,        # [B]
        }

    return {
        "history": histories,       # [B, T_obs, input_dim]
        "future": futures,            # [B, T_pred, 2]
        "category": categories,       # [B]
        "seq_id": seq_ids,            # list[str]
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """创建 DataLoader，适配 Windows 多进程 spawn 模式。

    参数:
        dataset: PyTorch Dataset 实例
        batch_size: batch 大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数（Windows 建议不超过 4）
        pin_memory: 是否使用锁页内存（GPU 训练时建议开启）

    返回:
        配置好的 DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # 丢弃最后不够一个 batch 的样本
        persistent_workers=(num_workers > 0),  # 保持 workers 存活
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """确定性划分数据集（基于哈希，跨运行一致）。

    注意：返回的是 torch.utils.data.Subset，不能直接保存为文件。
    实际的 7:1:2 划分在预处理阶段更合理，此函数用于中间验证。

    参数:
        dataset: 完整数据集
        train_ratio: 训练集比例（默认 0.7）
        val_ratio: 验证集比例（默认 0.1）
        test_ratio: 测试集比例（默认 0.2）

    返回:
        (train_dataset, val_dataset, test_dataset) 元组
    """
    from torch.utils.data import Subset, random_split

    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"  数据集划分: 训练 {train_size}, 验证 {val_size}, 测试 {test_size}")
    return train_ds, val_ds, test_ds
