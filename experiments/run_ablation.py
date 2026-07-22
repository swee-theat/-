"""消融实验批量运行器：遍历所有消融配置，训练并收集结果。

对应论文实验第二阶段（表3/表6）：
- 去不确定性、均值池化、LSTM池化
- MLP层数(2/3/4)
- 模态数(1/2/3/5)
- 去上下文特征(6维)、最小特征(4维)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.config import load_config, config_to_dict, save_config
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer
from eval.evaluator import Evaluator
from eval.metrics import compute_min_ade, compute_min_fde, compute_miss_rate


ABLATION_CONFIGS = {
    "去不确定性": "configs/ablation/no_uncertainty.yaml",
    "均值池化": "configs/ablation/mean_pooling.yaml",
    "LSTM池化": "configs/ablation/lstm_pooling.yaml",
    "2层MLP": "configs/ablation/mlp_2layers.yaml",
    "4层MLP": "configs/ablation/mlp_4layers.yaml",
    "K=1单模态": "configs/ablation/k1.yaml",
    "K=2双模态": "configs/ablation/k2.yaml",
    "K=5多模态": "configs/ablation/k5.yaml",
    "无上下文(6维)": "configs/ablation/no_context.yaml",
    "最小特征(4维)": "configs/ablation/dim4.yaml",
}


def run_single_ablation(
    name: str,
    config_path: str,
    base_config_path: str,
    data_dir: str,
    device: str,
) -> dict:
    """运行单个消融实验并返回结果。

    参数:
        name: 实验名称
        config_path: 本次消融的配置路径
        base_config_path: 基础配置路径（用于继承）
        data_dir: 预处理数据目录
        device: 计算设备

    返回:
        包含名称、指标和参数量的字典
    """
    print(f"\n{'='*60}")
    print(f"  消融实验: {name}")
    print(f"{'='*60}")

    # 加载配置（继承默认配置）
    config = load_config(config_path, base_path=base_config_path)
    cfg = config_to_dict(config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    # 为每个消融实验创建独立日志目录
    exp_name = name.replace("(", "").replace(")", "").replace("=", "_").replace(" ", "_")
    log_dir = f"outputs/logs/ablation_{exp_name}"
    ckpt_dir = f"outputs/checkpoints/ablation_{exp_name}"
    cfg["logging"]["log_dir"] = log_dir
    cfg["logging"]["checkpoint_dir"] = ckpt_dir

    set_seed(train_cfg.get("seed", 42))
    logger = setup_logger(log_dir, name=exp_name)

    # 加载数据
    processed_dir = Path(data_dir)
    train_ds = ArgoverseTrajectoryDataset(str(processed_dir / "train.npz"))
    val_ds = ArgoverseTrajectoryDataset(str(processed_dir / "val.npz"))

    train_loader = create_dataloader(
        train_ds, batch_size=data_cfg["batch_size"],
        shuffle=True, num_workers=data_cfg.get("num_workers", 4),
    )
    val_loader = create_dataloader(
        val_ds, batch_size=data_cfg["batch_size"],
        shuffle=False, num_workers=data_cfg.get("num_workers", 2),
    )

    # 构建模型
    model = TrajectoryModel(
        input_dim=model_cfg.get("input_dim", 9),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_mlp_layers=model_cfg.get("num_mlp_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
        num_modes=model_cfg.get("num_modes", 3),
        pred_len=data_cfg.get("pred_len", 30),
        pooling_type=model_cfg.get("pooling", "attention"),
        use_uncertainty=model_cfg.get("use_uncertainty", True),
    )

    total_params, _ = count_parameters(model)

    # 训练
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=cfg, logger=logger, tb_writer=None,
    )
    result = trainer.train()

    # 测试集评估（快速版：仅关键指标）
    logger.info("评估测试集...")
    test_ds = ArgoverseTrajectoryDataset(str(processed_dir / "test.npz"))
    test_loader = create_dataloader(
        test_ds, batch_size=data_cfg["batch_size"],
        shuffle=False, num_workers=data_cfg.get("num_workers", 2),
    )

    # 加载最佳模型
    best_ckpt = Path(ckpt_dir) / "best_model.pt"
    if best_ckpt.exists():
        from utils.checkpoint import load_checkpoint
        load_checkpoint(str(best_ckpt), model, device=device)

    evaluator = Evaluator(model, test_loader, device=device)
    eval_result = evaluator.evaluate()
    overall = eval_result["overall"]

    return {
        "实验名称": name,
        "参数量": total_params,
        "最佳Val_minADE": result["best_val_min_ade"],
        "测试minADE": overall.get("min_ade", float("nan")),
        "测试minFDE": overall.get("min_fde", float("nan")),
        "测试MR(2m)": overall.get("miss_rate_2m", float("nan")),
        "终点命中率(3m)": overall.get("endpoint_hit_3m", float("nan")),
        "早停Epoch": result["best_epoch"] + 1,
    }


def main():
    parser = argparse.ArgumentParser(description="批量运行消融实验")
    parser.add_argument("--base-config", default="configs/default.yaml", help="基础配置")
    parser.add_argument("--data-dir", default="data/processed", help="预处理数据目录")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="指定运行的实验名称（默认全部运行）")
    args = parser.parse_args()

    results = []
    experiments = args.experiments or list(ABLATION_CONFIGS.keys())

    for name in tqdm(experiments, desc="消融实验进度"):
        if name not in ABLATION_CONFIGS:
            print(f"跳过未知实验: {name}")
            continue
        try:
            result = run_single_ablation(
                name, ABLATION_CONFIGS[name], args.base_config,
                args.data_dir, args.device,
            )
            results.append(result)
        except Exception as e:
            print(f"实验 {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    # 导出结果汇总表（对应论文表3/表6）
    if results:
        df = pd.DataFrame(results)
        output_path = Path("outputs/results/ablation_summary.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n消融实验结果已导出到: {output_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
