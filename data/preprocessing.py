"""数据预处理模块：将 Argoverse CSV 原始数据转换为 9 维特征。

特征设计（9 维）:
    [0] x:               居中的 X 坐标
    [1] y:               居中的 Y 坐标
    [2] vx:              X 方向速度
    [3] vy:              Y 方向速度
    [4] t/T:             归一化时间步
    [5] 1:               常数偏置项
    [6] dist_nearest/100: 到最近智能体的归一化距离
    [7] rel_x/100:        相对最近智能体的 X 位移
    [8] rel_y/100:        相对最近智能体的 Y 位移
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


# 预期列名映射（兼容大小写和不同命名风格）
COLUMN_MAP = {
    "timestamp": ["TIMESTAMP", "timestamp", "time", "TIME"],
    "track_id": ["TRACK_ID", "track_id", "id", "ID", "agent_id"],
    "object_type": ["OBJECT_TYPE", "object_type", "type", "TYPE"],
    "x": ["X", "x", "pos_x", "POS_X"],
    "y": ["Y", "y", "pos_y", "POS_Y"],
    "city": ["CITY_NAME", "city", "CITY"],
}


def _detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """自动检测 CSV 列名映射。"""
    cols = {c.upper(): c for c in df.columns}
    mapping = {}
    for target, candidates in COLUMN_MAP.items():
        for cand in candidates:
            if cand.upper() in cols:
                mapping[target] = cols[cand.upper()]
                break
    return mapping


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 CSV DataFrame 标准化为统一列名。"""
    detected = _detect_columns(df)
    required = ["timestamp", "track_id", "object_type", "x", "y"]
    missing = [k for k in required if k not in detected]
    if missing:
        raise ValueError(
            f"CSV 缺少必要列: {missing}。已检测到列: {list(df.columns)}"
        )

    normalized = pd.DataFrame()
    normalized["timestamp"] = df[detected["timestamp"]]
    normalized["track_id"] = df[detected["track_id"]]
    normalized["object_type"] = df[detected["object_type"]]
    normalized["x"] = df[detected["x"]].astype(np.float32)
    normalized["y"] = df[detected["y"]].astype(np.float32)

    return normalized


def compute_velocities(
    x: np.ndarray, y: np.ndarray, dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """计算前向差分速度。

    参数:
        x: X 坐标序列 [T]
        y: Y 坐标序列 [T]
        dt: 时间步长（秒），Argoverse 默认为 0.1s

    返回:
        (vx, vy) 元组，各为 [T]
    """
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)

    # 前向差分
    vx[1:] = (x[1:] - x[:-1]) / dt
    vy[1:] = (y[1:] - y[:-1]) / dt

    # t=0 使用 t=1 的速度（避免零点问题）
    if len(vx) > 1:
        vx[0] = vx[1]
        vy[0] = vy[1]

    return vx, vy


def compute_nearest_agent_features(
    x: np.ndarray,
    y: np.ndarray,
    all_agents: Dict[int, Tuple[np.ndarray, np.ndarray]],
    current_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算当前智能体相对于场景中最近其他智能体的特征。

    参数:
        x, y: 当前智能体的坐标 [T]
        all_agents: {track_id: (x_arr[T], y_arr[T])} 场景中所有智能体
        current_id: 当前智能体的 track_id

    返回:
        (dist_nearest[T], rel_x[T], rel_y[T])
    """
    T = len(x)
    dist_nearest = np.full(T, 100.0, dtype=np.float32)  # 默认远距离
    rel_x = np.zeros(T, dtype=np.float32)
    rel_y = np.zeros(T, dtype=np.float32)

    if len(all_agents) <= 1:
        return dist_nearest, rel_x, rel_y

    for t in range(T):
        min_dist = float("inf")
        nearest_x, nearest_y = 0.0, 0.0
        for agent_id, (ax, ay) in all_agents.items():
            if agent_id == current_id:
                continue
            if t < len(ax):
                dx = x[t] - ax[t]
                dy = y[t] - ay[t]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_x = dx
                    nearest_y = dy
        if min_dist < float("inf"):
            dist_nearest[t] = min_dist
            rel_x[t] = nearest_x
            rel_y[t] = nearest_y

    return dist_nearest, rel_x, rel_y


def extract_agent_track(
    df: pd.DataFrame,
    track_id: int,
    obs_len: int = 20,
    pred_len: int = 30,
) -> Optional[Dict]:
    """从单个场景 CSV 中提取一个智能体的完整轨迹。

    参数:
        df: 标准化后的单个 CSV DataFrame
        track_id: 目标智能体 ID
        obs_len: 观测帧数（默认 20）
        pred_len: 预测帧数（默认 30）

    返回:
        {
            "history": np.ndarray [obs_len, 9],
            "future": np.ndarray [pred_len, 2],
            "category": int (场景类型标签),
            "seq_id": str
        }
        如果轨迹长度不足则返回 None
    """
    total_len = obs_len + pred_len
    mask = df["track_id"] == track_id
    agent_df = df[mask].sort_values("timestamp")

    if len(agent_df) < total_len:
        return None

    # 取前 total_len 帧
    agent_df = agent_df.iloc[:total_len]
    x = agent_df["x"].values.astype(np.float32)
    y = agent_df["y"].values.astype(np.float32)

    # 居中：以最后观测帧为原点
    x_center = x[obs_len - 1]
    y_center = y[obs_len - 1]
    x_centered = x - x_center
    y_centered = y - y_center

    # 计算速度
    vx, vy = compute_velocities(x_centered, y_centered)

    # 收集场景中所有智能体（用于最近距离计算）
    all_agents = {}
    for tid in df["track_id"].unique():
        tid_df = df[df["track_id"] == tid].sort_values("timestamp")
        if len(tid_df) >= total_len:
            tid_df = tid_df.iloc[:total_len]
            tx = tid_df["x"].values.astype(np.float32) - x_center
            ty = tid_df["y"].values.astype(np.float32) - y_center
            all_agents[tid] = (tx, ty)

    # 最近智能体特征
    dist_nearest, rel_x, rel_y = compute_nearest_agent_features(
        x_centered, y_centered, all_agents, track_id
    )

    # 时间步归一化
    t_normalized = np.arange(total_len, dtype=np.float32) / (obs_len - 1)

    # 航向角（瞬时运动方向），展开消除 ±π 跳变
    heading = np.unwrap(np.arctan2(vy, vx))  # [total_len]

    # 曲率（航向角变化率，rad/s），裁剪到合理车辆动力学范围
    curvature = np.zeros_like(heading)
    if total_len > 1:
        curvature[1:] = (heading[1:] - heading[:-1]) / 0.1
        curvature[0] = curvature[1]  # 第一帧复制第二帧
    curvature = np.clip(curvature, -3.0, 3.0)  # 车辆极限曲率 ±3 rad/s

    # 构建 11 维特征
    history_feats = np.stack([
        x_centered[:obs_len],                    # 0: x
        y_centered[:obs_len],                    # 1: y
        vx[:obs_len],                            # 2: vx
        vy[:obs_len],                            # 3: vy
        t_normalized[:obs_len],                  # 4: t/T
        np.ones(obs_len, dtype=np.float32),      # 5: 常值偏置 1
        dist_nearest[:obs_len] / 100.0,          # 6: dist/100
        rel_x[:obs_len] / 100.0,                 # 7: rel_x/100
        rel_y[:obs_len] / 100.0,                 # 8: rel_y/100
        heading[:obs_len],                       # 9: 航向角（rad）
        curvature[:obs_len],                     # 10: 曲率（rad/s）
    ], axis=-1)  # [obs_len, 11]

    # 未来轨迹（仅 x,y，已经居中）
    future = np.stack([
        x_centered[obs_len:],
        y_centered[obs_len:],
    ], axis=-1)  # [pred_len, 2]

    # 场景类型分类（基于未来轨迹曲率）
    category = classify_scene_type(future)

    # 序列 ID
    seq_id = f"{track_id}"

    return {
        "history": history_feats,
        "future": future,
        "category": category,
        "seq_id": seq_id,
    }


def classify_scene_type(future_traj: np.ndarray) -> int:
    """根据未来轨迹曲率和速度对场景类型进行分类。

    类别:
        0: 直行
        1: 左转
        2: 右转
        3: 掉头
        4: 路口/交叉

    参数:
        future_traj: 未来轨迹 [T_pred, 2]，已居中

    返回:
        场景类型标签 (0-4)
    """
    dx = np.diff(future_traj[:, 0])
    dy = np.diff(future_traj[:, 1])

    # 计算每帧的航向角变化
    headings = np.arctan2(dy, dx)
    heading_changes = np.diff(headings)

    # 处理角度环绕（-pi 到 pi）
    heading_changes = np.arctan2(
        np.sin(heading_changes), np.cos(heading_changes)
    )

    total_curvature = np.sum(heading_changes)

    # 计算平均速度
    velocities = np.sqrt(dx**2 + dy**2)
    avg_speed = np.mean(velocities)

    # 分类阈值
    U_TURN_THRESH = np.deg2rad(150)  # 总曲率 > 150°
    TURN_THRESH = np.deg2rad(30)     # 总曲率 > 30°

    if abs(total_curvature) > U_TURN_THRESH:
        return 3  # 掉头
    elif total_curvature > TURN_THRESH:
        return 1  # 左转
    elif total_curvature < -TURN_THRESH:
        return 2  # 右转

    # 路口判定：速度骤降到峰值速度的 30% 以下
    max_v = np.max(velocities)
    is_slow = max_v > 0 and avg_speed < 0.3 * max_v
    if is_slow:
        return 4  # 路口/交叉

    return 0  # 直行


def process_csv_file(
    csv_path: str,
    obs_len: int = 20,
    pred_len: int = 30,
) -> List[Dict]:
    """处理单个 Argoverse CSV 文件，提取所有有效智能体轨迹。

    参数:
        csv_path: CSV 文件路径
        obs_len: 观测帧数
        pred_len: 预测帧数

    返回:
        样本列表，每个样本为 {"history", "future", "category", "seq_id"} 字典
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        warnings.warn(f"读取 {csv_path} 失败: {e}")
        return []

    try:
        df = _normalize_columns(df)
    except ValueError as e:
        warnings.warn(f"标准化 {csv_path} 失败: {e}")
        return []

    samples = []
    for track_id in df["track_id"].unique():
        sample = extract_agent_track(df, track_id, obs_len, pred_len)
        if sample is not None:
            sample["seq_id"] = f"{Path(csv_path).stem}_{track_id}"
            samples.append(sample)

    return samples


def process_dataset(
    raw_dir: str,
    output_path: str,
    obs_len: int = 20,
    pred_len: int = 30,
    split: str = "train",
) -> int:
    """批量处理数据集目录下所有 CSV 文件，保存为压缩 numpy 格式。

    参数:
        raw_dir: 原始 CSV 目录
        output_path: 输出 .npz 文件路径
        obs_len: 观测帧数
        pred_len: 预测帧数
        split: 数据集划分名称

    返回:
        处理成功的样本总数
    """
    csv_dir = Path(raw_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {raw_dir}")

    csv_files = sorted(csv_dir.rglob("*.csv"))
    print(f"  找到 {len(csv_files)} 个 CSV 文件在 {raw_dir}")

    all_histories = []
    all_futures = []
    all_categories = []
    all_seq_ids = []
    total_samples = 0

    for i, csv_path in enumerate(csv_files):
        samples = process_csv_file(str(csv_path), obs_len, pred_len)
        for s in samples:
            all_histories.append(s["history"])
            all_futures.append(s["future"])
            all_categories.append(s["category"])
            all_seq_ids.append(s["seq_id"])
            total_samples += 1

        if (i + 1) % 1000 == 0:
            print(f"  已处理 {i+1}/{len(csv_files)} 文件, 累计 {total_samples} 样本...")

    # 保存为压缩 numpy 文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        histories=np.array(all_histories, dtype=np.float32),
        futures=np.array(all_futures, dtype=np.float32),
        categories=np.array(all_categories, dtype=np.int32),
        seq_ids=np.array(all_seq_ids),
    )

    print(f"  {split} 集处理完成: {total_samples} 样本 → {output_path}")
    return total_samples
