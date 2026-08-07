"""可视化实验运行器：生成论文全部图表（图3-10）。

对应论文实验第四阶段（定性分析+可视化）。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import pandas as pd
from pathlib import Path

from utils.config import load_config, config_to_dict
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from eval.latency import benchmark_latency, benchmark_cpu_latency
from eval.memory import measure_training_memory, measure_inference_memory


def main():
    parser = argparse.ArgumentParser(description="生成论文可视化图表")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--figures", nargs="+", default=None,
                        help="指定生成的图表（默认全部）")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config_to_dict(config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

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
    model.to(args.device)

    # 加载训练好的权重
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"加载模型权重: {ckpt_path}")
        from utils.checkpoint import load_checkpoint
        load_checkpoint(str(ckpt_path), model, device=args.device)
    else:
        print(f"警告: checkpoint 不存在 {ckpt_path}，使用随机权重生成图表")

    model.eval()

    # 加载测试数据
    processed_dir = Path(args.data_dir)
    test_ds = ArgoverseTrajectoryDataset(str(processed_dir / "test.npz"))
    test_loader = create_dataloader(
        test_ds, batch_size=data_cfg["batch_size"],
        shuffle=True, num_workers=data_cfg.get("num_workers", 2),
    )

    figures_to_generate = args.figures or ["all"]

    # 图 4: 训练曲线（需要训练历史，如果有的话）
    if "all" in figures_to_generate or "fig4" in figures_to_generate:
        from visualization.plot_training_curves import plot_training_curves
        # 尝试从保存的训练历史加载，否则用示例数据
        try:
            history = torch.load("outputs/checkpoints/training_history.pt", weights_only=True)
            plot_training_curves(
                history.get("train_losses", [1.0] * 20),
                history.get("val_ades", [2.0] * 20),
                best_epoch=history.get("best_epoch", 10),
            )
        except FileNotFoundError:
            print("  图4: 训练历史文件不存在，将在实际训练后生成")

    # 图 5: Pareto 前沿
    if "all" in figures_to_generate or "fig5" in figures_to_generate:
        from visualization.plot_pareto import plot_pareto_frontier
        try:
            sweep_df = pd.read_csv("outputs/results/parameter_sweep.csv")
            plot_pareto_frontier(sweep_df.to_dict("records"))
        except FileNotFoundError:
            print("  图5: 参数扫描结果不存在，将在效率实验后生成")

    # 图 6: 消融柱状图
    if "all" in figures_to_generate or "fig6" in figures_to_generate:
        from visualization.plot_ablation_bars import plot_ablation_bars
        try:
            ablation_df = pd.read_csv("outputs/results/ablation_summary.csv")
            plot_ablation_bars(ablation_df.to_dict("records"))
        except FileNotFoundError:
            print("  图6: 消融结果不存在，将在消融实验后生成")

    # 图 7: 延迟曲线
    if "all" in figures_to_generate or "fig7" in figures_to_generate:
        from visualization.plot_latency import plot_latency_comparison
        latencies = benchmark_latency(
            model, input_dim=model_cfg.get("input_dim", 9),
            obs_len=data_cfg.get("obs_len", 20),
            batch_sizes=[1, 8, 16, 32, 64],
            device=args.device,
        )
        plot_latency_comparison(latencies)

    # 图 8: 轨迹可视化
    if "all" in figures_to_generate or "fig8" in figures_to_generate:
        from visualization.plot_trajectories import plot_trajectory_samples
        plot_trajectory_samples(model, test_loader, device=args.device)

    # 图 9: 不确定性可视化
    if "all" in figures_to_generate or "fig9" in figures_to_generate:
        from visualization.plot_uncertainty import plot_uncertainty_comparison
        if model_cfg.get("use_uncertainty", True):
            plot_uncertainty_comparison(model, test_loader, device=args.device)
        else:
            print("  图9: 模型未启用不确定性，跳过")

    # 图 10: 失败案例
    if "all" in figures_to_generate or "fig10" in figures_to_generate:
        from visualization.plot_failure_cases import plot_failure_cases
        plot_failure_cases(model, test_loader, device=args.device)

    # 图 3: 池化对比（需要消融数据）
    if "fig3" in figures_to_generate:
        from visualization.plot_pooling_comparison import plot_pooling_comparison
        try:
            ablation_df = pd.read_csv("outputs/results/ablation_summary.csv")
            plot_pooling_comparison(ablation_df.to_dict("records"))
        except FileNotFoundError:
            print("  图3: 消融结果不存在，将在消融实验后生成")

    # 显存测量
    print("\n--- 显存占用 ---")
    train_mem = measure_training_memory(
        model, input_dim=model_cfg.get("input_dim", 9),
        obs_len=data_cfg.get("obs_len", 20),
        pred_len=data_cfg.get("pred_len", 30),
        batch_size=data_cfg["batch_size"], device=args.device,
    )
    print(f"  训练峰值显存: {train_mem.get('peak_allocated_mb', 'N/A'):.1f} MB")
    print(f"  模型参数显存: {train_mem.get('model_params_mb', 'N/A'):.2f} MB")

    infer_mem = measure_inference_memory(
        model, input_dim=model_cfg.get("input_dim", 9),
        obs_len=data_cfg.get("obs_len", 20),
        batch_size=data_cfg["batch_size"], device=args.device,
    )
    print(f"  推理峰值显存: {infer_mem.get('peak_allocated_mb', 'N/A'):.1f} MB")

    # CPU 延迟
    print("\n--- CPU 模式延迟 ---")
    cpu_lat = benchmark_cpu_latency(
        model, input_dim=model_cfg.get("input_dim", 9),
        obs_len=data_cfg.get("obs_len", 20),
    )
    for bs, info in cpu_lat.items():
        print(f"  batch={bs}: {info['mean_ms']:.3f} ms ± {info['std_ms']:.3f} ms")

    print(f"\n图表生成完成！输出目录: outputs/figures/")


if __name__ == "__main__":
    main()
