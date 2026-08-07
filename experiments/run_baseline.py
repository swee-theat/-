"""主训练入口：使用默认配置训练基线模型。"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import setup_logger, create_tb_writer
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer
from eval.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="训练基线轨迹预测模型")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--data-dir", default="data/processed", help="预处理数据目录")
    parser.add_argument("--resume", default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    cfg = config_to_dict(config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    # 固定随机种子
    set_seed(train_cfg.get("seed", 42))

    # 日志
    logger = setup_logger(log_cfg["log_dir"], "baseline")
    tb_writer = create_tb_writer(log_cfg["log_dir"])
    logger.info(f"配置: {args.config}")
    logger.info(f"模型参数: input_dim={model_cfg['input_dim']}, hidden_dim={model_cfg['hidden_dim']}, "
                f"K={model_cfg['num_modes']}, pooling={model_cfg['pooling']}")

    # 加载数据
    processed_dir = Path(args.data_dir)
    train_ds = ArgoverseTrajectoryDataset(str(processed_dir / "train.npz"))
    val_ds = ArgoverseTrajectoryDataset(str(processed_dir / "val.npz"))
    logger.info(f"训练集: {len(train_ds)} 样本, 验证集: {len(val_ds)} 样本")

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
        use_temporal_conv=model_cfg.get("use_temporal_conv", True),
    )

    # 打印参数量
    count_parameters(model)

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        logger=logger,
        tb_writer=tb_writer,
    )

    result = trainer.train(resume_from=args.resume)

    # 测试集评估
    logger.info("开始在测试集上评估...")
    test_ds = ArgoverseTrajectoryDataset(str(processed_dir / "test.npz"))
    test_loader = create_dataloader(
        test_ds, batch_size=data_cfg["batch_size"],
        shuffle=False, num_workers=data_cfg.get("num_workers", 2),
    )

    # 加载最佳模型
    best_ckpt = Path(log_cfg["checkpoint_dir"]) / "best_model.pt"
    if best_ckpt.exists():
        logger.info(f"加载最佳模型: {best_ckpt}")
        from utils.checkpoint import load_checkpoint
        load_checkpoint(str(best_ckpt), model, device=args.device)

    evaluator = Evaluator(model, test_loader, device=args.device)
    eval_result = evaluator.evaluate()
    evaluator.export_csv(eval_result, "./outputs/results/baseline")

    logger.info(f"训练完成！最佳 Val minADE: {result['best_val_min_ade']:.4f}")
    logger.info(f"测试集 minADE: {eval_result['overall'].get('min_ade', 'N/A')}")
    logger.info(f"测试集 minFDE: {eval_result['overall'].get('min_fde', 'N/A')}")
    logger.info(f"测试集 MR: {eval_result['overall'].get('miss_rate_2m', 'N/A')}")

    tb_writer.close()


if __name__ == "__main__":
    main()
