"""效率实验运行器：参数扫描 + 延迟/显存/FPS 基准测试。

对应论文实验第三阶段（表4/表5）：
- hidden_dim 扫描 (32/64/128/256/512)
- 推理延迟 vs Batch Size
- 训练/推理显存
- CPU 模式延迟
- FPS 吞吐量
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer
from eval.evaluator import Evaluator
from eval.latency import benchmark_latency, benchmark_cpu_latency, benchmark_throughput
from eval.memory import measure_training_memory, measure_inference_memory


EFFICIENCY_CONFIGS = {
    "dim32": "configs/efficiency/dim32.yaml",
    "dim64": "configs/efficiency/dim64.yaml",
    "dim128(基线)": "configs/default.yaml",
    "dim256": "configs/efficiency/dim256.yaml",
    "dim512": "configs/efficiency/dim512.yaml",
}


def run_parameter_sweep(
    base_config_path: str,
    data_dir: str,
    device: str,
) -> pd.DataFrame:
    """运行 hidden_dim 参数扫描实验。

    返回:
        DataFrame 包含各 hidden_dim 下的参数量、ADE、FDE、延迟等
    """
    results = []
    print("\n" + "=" * 60)
    print("  参数扫描实验（hidden_dim 扫描）")
    print("=" * 60)

    for name, config_path in EFFICIENCY_CONFIGS.items():
        print(f"\n--- {name} ---")

        config = load_config(config_path, base_path=base_config_path)
        cfg = config_to_dict(config)
        train_cfg = cfg["training"]
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

        hidden_dim = model_cfg["hidden_dim"]
        exp_name = f"efficiency_{name.replace('(', '').replace(')', '')}"
        log_dir = f"outputs/logs/{exp_name}"
        ckpt_dir = f"outputs/checkpoints/{exp_name}"
        cfg["logging"]["log_dir"] = log_dir
        cfg["logging"]["checkpoint_dir"] = ckpt_dir

        set_seed(train_cfg.get("seed", 42))
        logger = setup_logger(log_dir, name=exp_name)

        # 数据
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

        # 模型
        model = TrajectoryModel(
            input_dim=model_cfg.get("input_dim", 9),
            hidden_dim=hidden_dim,
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
        train_result = trainer.train()

        # 测试集评估
        best_ckpt = Path(ckpt_dir) / "best_model.pt"
        if best_ckpt.exists():
            from utils.checkpoint import load_checkpoint
            load_checkpoint(str(best_ckpt), model, device=device)

        test_ds = ArgoverseTrajectoryDataset(str(processed_dir / "test.npz"))
        test_loader = create_dataloader(
            test_ds, batch_size=data_cfg["batch_size"],
            shuffle=False, num_workers=data_cfg.get("num_workers", 2),
        )
        evaluator = Evaluator(model, test_loader, device=device)
        eval_result = evaluator.evaluate()

        # 推理延迟
        latencies = benchmark_latency(
            model, input_dim=model_cfg.get("input_dim", 9),
            obs_len=data_cfg.get("obs_len", 20),
            batch_sizes=[1, 8, 16, 32, 64], device=device,
        )

        # 训练显存
        train_mem = measure_training_memory(
            model, input_dim=model_cfg.get("input_dim", 9),
            obs_len=data_cfg.get("obs_len", 20),
            pred_len=data_cfg.get("pred_len", 30),
            batch_size=data_cfg["batch_size"], device=device,
        )

        results.append({
            "hidden_dim": hidden_dim,
            "参数量": total_params,
            "Val_minADE": train_result["best_val_min_ade"],
            "测试minADE": eval_result["overall"].get("min_ade", float("nan")),
            "测试minFDE": eval_result["overall"].get("min_fde", float("nan")),
            "测试MR": eval_result["overall"].get("miss_rate_2m", float("nan")),
            "延迟_batch1_ms": latencies[1]["mean_ms"],
            "延迟_batch16_ms": latencies[16]["mean_ms"],
            "延迟_batch64_ms": latencies[64]["mean_ms"],
            "训练显存_MB": train_mem.get("peak_allocated_mb", float("nan")),
            "最优Epoch": train_result.get("best_epoch", 0),
        })

    df = pd.DataFrame(results)
    output_path = Path("outputs/results/parameter_sweep.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n参数扫描结果已导出: {output_path}")
    return df


def run_latency_benchmarks(
    model: TrajectoryModel,
    input_dim: int = 9,
    obs_len: int = 20,
    device: str = "cuda",
) -> dict:
    """运行完整的延迟和吞吐量基准测试。"""
    print("\n--- 延迟基准测试 ---")

    # GPU 延迟
    gpu_lat = benchmark_latency(
        model, input_dim=input_dim, obs_len=obs_len,
        batch_sizes=[1, 8, 16, 32, 64],
        num_warmup=50, num_iter=200, device=device,
    )
    print("GPU 延迟 (ms):")
    for bs, info in gpu_lat.items():
        print(f"  batch={bs:3d}: {info['mean_ms']:7.3f} ± {info['std_ms']:.3f} ms | FPS={info['fps']:.1f}")

    # CPU 延迟
    cpu_lat = benchmark_cpu_latency(
        model, input_dim=input_dim, obs_len=obs_len,
        batch_sizes=[1, 8, 16],
    )
    print("CPU 延迟 (ms):")
    for bs, info in cpu_lat.items():
        print(f"  batch={bs:3d}: {info['mean_ms']:7.3f} ± {info['std_ms']:.3f} ms")

    # 吞吐量
    throughput = benchmark_throughput(
        model, input_dim=input_dim, obs_len=obs_len,
        batch_sizes=[8, 16, 32, 64], device=device,
    )
    print(f"最大吞吐量: {throughput['max_fps']:.1f} FPS (batch={throughput['optimal_batch_size']})")

    return {
        "gpu_latency": gpu_lat,
        "cpu_latency": cpu_lat,
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="效率实验")
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--sweep-only", action="store_true", help="仅运行参数扫描")
    parser.add_argument("--benchmark-only", action="store_true", help="仅运行基准测试")
    args = parser.parse_args()

    if args.benchmark_only:
        # 仅基准测试：加载已训练的基线模型
        config = load_config(args.base_config)
        cfg = config_to_dict(config)
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        model = TrajectoryModel(
            input_dim=model_cfg.get("input_dim", 9),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_mlp_layers=model_cfg.get("num_mlp_layers", 3),
            dropout=model_cfg.get("dropout", 0.1),
            num_modes=model_cfg.get("num_modes", 3),
            pred_len=data_cfg.get("pred_len", 30),
        )
        model.to(args.device)
        model.eval()

        run_latency_benchmarks(model, device=args.device)

    elif args.sweep_only:
        run_parameter_sweep(args.base_config, args.data_dir, args.device)
    else:
        # 全量效率实验
        df = run_parameter_sweep(args.base_config, args.data_dir, args.device)
        print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
