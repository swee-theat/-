#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量化多模态轨迹预测系统 — Web 可视化 Demo

提供三种预测方式：
  1. 鼠标手绘轨迹（Canvas 点击）
  2. 上传 Argoverse CSV 文件
  3. 内置示例轨迹

使用方法:
    cd da_chuang
    python demo/app.py
    浏览器打开 http://localhost:5000
"""

import os, sys, csv, json, io

# Windows 终端编码修正（必须在其他输出之前）
sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trajectory_model import TrajectoryModel

# ── 配置 ──────────────────────────────────────────────
OBS_LEN = 20          # 观测帧数 (2秒)
PRED_LEN = 30         # 预测帧数 (3秒)
INPUT_DIM = 11        # 特征维度 (11=完整+航向+曲率)
HIDDEN_DIM = 128
NUM_MODES = 5         # K=5 多模态
CHECKPOINT_PATH = "outputs/checkpoints/full_train_K5/best_model_indep.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 加载模型 ──────────────────────────────────────────
model = TrajectoryModel(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_mlp_layers=3,
    dropout=0.1,
    num_modes=NUM_MODES,
    pred_len=PRED_LEN,
    pooling_type="attention",
    use_uncertainty=True,
    use_temporal_conv=True,
)

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    # 兼容不同 checkpoint 格式
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        info = checkpoint.get("config", {})
        print(f"[OK] 从 checkpoint 加载模型 (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("[OK] 从 state_dict 加载模型")
    model_loaded = True
except FileNotFoundError:
    print(f"[警告] 未找到模型文件: {CHECKPOINT_PATH}")
    print("       将使用随机初始化模型（仅供界面测试）")
    model_loaded = False
except Exception as e:
    print(f"[警告] 模型加载失败: {e}")
    print("       将使用随机初始化模型（仅供界面测试）")
    model_loaded = False

model.eval()
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")
print(f"运行设备: {DEVICE}")
print(f"K = {NUM_MODES} 模态")


# ── 特征工程（11维，与 preprocessing.py 严格一致）───────
def compute_features(obs_centered: np.ndarray) -> np.ndarray:
    """计算 11 维特征。

    与 data/preprocessing.py 的 extract_agent_track() 保持一致：
    - 坐标以最后观测帧为原点居中
    - 速度用前向差分（dt=0.1），t=0 复制 t=1 的速度
    - 时间步归一化: i / (obs_len - 1)

    11 维: [x, y, vx, vy, t/T, 1, dist/100, rel_x/100, rel_y/100, heading, curvature]

    参数:
        obs_centered: [T, 2] 以最后观测帧为原点的坐标

    返回:
        features: [T, 11]
    """
    T = min(len(obs_centered), OBS_LEN)
    features = np.zeros((T, INPUT_DIM), dtype=np.float32)

    # 速度（前向差分，dt=0.1，与 preprocessing 一致）
    vx = np.zeros(T, dtype=np.float32)
    vy = np.zeros(T, dtype=np.float32)
    if T > 1:
        vx[1:] = (obs_centered[1:, 0] - obs_centered[:-1, 0]) / 0.1
        vy[1:] = (obs_centered[1:, 1] - obs_centered[:-1, 1]) / 0.1
        vx[0] = vx[1]  # 第一帧复制第二帧速度
        vy[0] = vy[1]

    # 航向角和曲率（与 preprocessing 一致，展开 + 裁剪）
    heading = np.unwrap(np.arctan2(vy, vx))
    curvature = np.zeros(T, dtype=np.float32)
    if T > 1:
        curvature[1:] = (heading[1:] - heading[:-1]) / 0.1
        curvature[0] = curvature[1]
    curvature = np.clip(curvature, -3.0, 3.0)

    for i in range(T):
        features[i] = [
            obs_centered[i, 0],           # 0: x（以最后观测帧为原点）
            obs_centered[i, 1],           # 1: y
            vx[i],                         # 2: vx
            vy[i],                         # 3: vy
            i / (OBS_LEN - 1),             # 4: t/T（0 → 1.0）
            1.0,                            # 5: 常值偏置
            0.0,                            # 6: 最近邻距离/100
            0.0,                            # 7: 相对最近邻 x/100
            0.0,                            # 8: 相对最近邻 y/100
            heading[i],                     # 9: 航向角（rad）
            curvature[i],                   # 10: 曲率（rad/s）
        ]

    return features


# ── CSV 文件解析 ──────────────────────────────────────
def parse_csv_trajectory(file_content: str) -> np.ndarray:
    """从 Argoverse CSV 内容中提取 agent 轨迹。

    参数:
        file_content: CSV 文件原始文本

    返回:
        trajectory: [N, 2] numpy 数组
    """
    reader = csv.reader(io.StringIO(file_content))
    header = next(reader)

    data = []
    for row in reader:
        if len(row) < 5:
            continue
        obj_type = row[2] if len(row) > 2 else ""
        if obj_type == "AGENT":
            data.append({
                "timestamp": float(row[0]),
                "x": float(row[3]),
                "y": float(row[4]),
            })

    data.sort(key=lambda d: d["timestamp"])
    return np.array([[d["x"], d["y"]] for d in data], dtype=np.float32)


# ── 推理函数 ──────────────────────────────────────────
def run_inference(obs_traj: np.ndarray) -> dict:
    """对观测轨迹进行多模态预测。

    与训练预处理严格一致：
    - 以最后观测帧为原点居中
    - 预测输出 = 相对最后帧的位移

    参数:
        obs_traj: [T, 2] 原始观测坐标（绝对坐标）

    返回:
        API 响应字典（坐标已还原为绝对坐标）
    """
    # 居中：以最后观测帧为原点（与 preprocessing 一致）
    last_point = obs_traj[-1].copy()
    obs_centered = obs_traj - last_point

    # 计算 11 维特征
    features = compute_features(obs_centered)
    input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    # [1, K, 30, 2] → [K, 30, 2]（相对于最后观测帧的位移）
    trajectories = output["trajectories"][0].cpu().numpy()
    mode_probs = output["mode_probs"][0].cpu().numpy()
    uncertainties = output.get("uncertainties")
    if uncertainties is not None:
        uncertainties = uncertainties[0].cpu().numpy()  # [K, 30, 2]
    else:
        uncertainties = np.zeros((NUM_MODES, PRED_LEN, 2), dtype=np.float32)

    # 还原绝对坐标：预测位移 + 最后观测点
    abs_trajectories = trajectories + last_point  # [K, 30, 2]

    # 各模态统计
    stats = []
    for k in range(NUM_MODES):
        traj = abs_trajectories[k]  # 使用绝对坐标做统计
        final_point = traj[-1]
        total_dist = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1)))
        avg_unc = float(np.mean(uncertainties[k]))
        stats.append({
            "mode": k + 1,
            "probability": float(mode_probs[k]),
            "final_x": float(final_point[0]),
            "final_y": float(final_point[1]),
            "total_distance": float(total_dist),
            "avg_uncertainty": float(avg_unc),
        })

    return {
        "success": True,
        "data": {
            "trajectories": abs_trajectories.tolist(),       # 绝对坐标
            "trajectories_relative": trajectories.tolist(),  # 相对位移（调试用）
            "probabilities": mode_probs.tolist(),
            "uncertainties": [
                uncertainties[k].tolist() for k in range(NUM_MODES)
            ],
            "last_point": last_point.tolist(),  # 最后观测点坐标
            "statistics": stats,
            "summary": {
                "best_mode": int(np.argmax(mode_probs)) + 1,
                "confidence": float(np.max(mode_probs)),
                "observation_points": len(obs_traj),
                "prediction_horizon": PRED_LEN,
                "avg_uncertainty": float(np.mean(uncertainties)),
            },
        },
    }


# ── Flask 应用 ────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# ── API: 模型信息 ─────────────────────────────────────
@app.route("/api/v1/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "success": True,
        "model": {
            "name": "轻量化多模态轨迹预测模型",
            "parameters": total_params,
            "device": str(DEVICE),
            "k_modes": NUM_MODES,
            "observation_length": OBS_LEN,
            "prediction_length": PRED_LEN,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "checkpoint": CHECKPOINT_PATH,
        },
    })


# ── API: 健康检查 ─────────────────────────────────────
@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(DEVICE),
    })


# ── API: 坐标序列预测 ─────────────────────────────────
@app.route("/api/v1/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "trajectory" not in data:
            return jsonify({"success": False, "error": "缺少 trajectory 字段"}), 400

        obs = np.array(data["trajectory"], dtype=np.float32)
        if len(obs) < OBS_LEN:
            return jsonify({
                "success": False,
                "error": f"观测点数量不足，需要 ≥{OBS_LEN} 个点，当前 {len(obs)} 个",
            }), 400

        # 只取前 OBS_LEN 帧
        obs = obs[:OBS_LEN]
        result = run_inference(obs)
        # 返回原始观测轨迹供前端显示
        result["data"]["observation"] = obs.tolist()
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── API: CSV 文件上传预测 ─────────────────────────────
@app.route("/api/v1/predict_csv", methods=["POST"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "未上传文件"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"success": False, "error": "仅支持 CSV 文件"}), 400

        # 读取并解析 CSV
        content = file.read().decode("utf-8")
        traj = parse_csv_trajectory(content)

        if len(traj) < OBS_LEN:
            return jsonify({
                "success": False,
                "error": f"CSV 中 agent 轨迹点不足，需要 >={OBS_LEN}，实际 {len(traj)}",
            }), 400

        obs = traj[:OBS_LEN]
        result = run_inference(obs)

        # 返回观测轨迹供前端绘图
        result["data"]["observation"] = obs.tolist()

        # 如果 CSV 有 ground truth，一并返回
        if len(traj) >= OBS_LEN + PRED_LEN:
            gt = traj[OBS_LEN:OBS_LEN + PRED_LEN]
            result["data"]["ground_truth"] = gt.tolist()

        result["data"]["full_trajectory_length"] = len(traj)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── API: 内置示例 ─────────────────────────────────────
# 注意: 示例速度需匹配训练数据分布（中位数 ~2 m/s, 均值 ~5 m/s）
# x 每帧增量 0.2 = 2 m/s, 与训练数据低速场景一致
@app.route("/api/v1/examples", methods=["GET"])
def examples():
    examples = [
        {
            "name": "straight",
            "description": "匀速直线运动 (~2 m/s)",
            "trajectory": [[float(i * 0.2), float(i * 0.1)] for i in range(OBS_LEN)],
        },
        {
            "name": "curve",
            "description": "恒定曲率右转弯 (~1.9 m/s, 半径5m)",
            # 真实车辆转弯：恒定角速度，圆弧轨迹（非正弦振荡 S 弯）
            # 车速 2m/s，半径 5m，角速度 ω=v/R=0.4 rad/s，20帧=2秒，总转角约 45°
            # 曲率恒定 = 1/R = 0.2/m，与 Argoverse 真实车辆转弯一致
            "trajectory": [
                [float(5.0 * np.sin(i * 0.04)), float(5.0 * (1.0 - np.cos(i * 0.04)))]
                for i in range(OBS_LEN)
            ],
        },
        {
            "name": "lane_change",
            "description": "S 形换道 (~2 m/s)",
            "trajectory": [
                [float(i * 0.2), 0.0 if i < 10 else float((i - 10) * 0.08)]
                for i in range(OBS_LEN)
            ],
        },
        {
            "name": "slow_down",
            "description": "减速右转 (~1-2 m/s)",
            "trajectory": [
                [float(i * 0.2 * (1.0 - i * 0.02)), float(i * 0.03 + i * i * 0.0016)]
                for i in range(OBS_LEN)
            ],
        },
        {
            "name": "u_turn",
            "description": "掉头场景 (~1.9 m/s, 半径1.2m半圆弧)",
            "trajectory": [
                [float(np.sin(i * np.pi / 20) * 1.2),
                 float(1.2 - np.cos(i * np.pi / 20) * 1.2)]
                for i in range(OBS_LEN)
            ],
        },
    ]
    return jsonify({"success": True, "examples": examples})


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 确保 templates 目录存在
    os.makedirs("demo/templates", exist_ok=True)

    print("=" * 58)
    print("  轻量化多模态轨迹预测系统 -- Web Demo")
    print("=" * 58)
    print(f"  模型参数: {total_params:,}")
    print(f"  模态数 K: {NUM_MODES}")
    print(f"  观测帧数: {OBS_LEN} (2s) -> 预测帧数: {PRED_LEN} (3s)")
    print(f"  运行设备: {DEVICE}")
    print(f"  模型状态: {'[已加载]' if model_loaded else '[随机初始化]'}")
    print()
    print(f"  Web 界面: http://localhost:5000")
    print(f"  API 文档: http://localhost:5000/api/v1/health")
    print(f"  示例数据: http://localhost:5000/api/v1/examples")
    print("=" * 58)

    app.run(host="0.0.0.0", port=5000, debug=False)
