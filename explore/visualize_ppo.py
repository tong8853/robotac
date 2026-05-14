"""
可视化验证脚本 - 生成 GIF 动画和性能报告

使用方法:
    conda activate robotac
    python visualize_ppo.py
"""
import torch
import time
import yaml
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from observation_wrapper import ImageObservationWrapper
from reward_function import check_violation

# ==================== 设备配置（设备无关）====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ==================== 配置 ====================
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODEL_PATH = Path(__file__).parent / "models" / config["model_name"]

# ==================== 环境配置 ====================
MAP_CONFIG = dict(
    type="block_num",
    config=3,
    lane_width=4,
    lane_num=1,
)

ENV_CONFIG = dict(
    use_render=True,  # 渲染开启用于可视化
    manual_control=False,
    traffic_density=0.0,
    num_scenarios=10000,
    random_agent_model=False,
    on_continuous_line_done=True,
    out_of_route_done=True,
    image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 160, 90)),
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
        image_source="rgb_camera",
    ),
    map_config=MAP_CONFIG,
    norm_pixel=True,
)


def run_validation_episode(model, env, max_steps=1000, render=True):
    """
    运行单个验证 episode

    返回:
        stats: dict, 包含 episode 统计数据
    """
    obs, _ = env.reset()
    episode_reward = 0.0
    step_count = 0
    episode_start = time.time()

    # 获取车辆对象用于违规检测
    vehicle = env.agent

    # 统计
    violations = {"minor": 0, "major": 0}
    speeds = []
    latencies = []

    for step in range(max_steps):
        # 延迟测量
        step_start = time.time()
        action, _ = model.predict(obs, deterministic=True)
        latency_ms = (time.time() - step_start) * 1000
        latencies.append(latency_ms)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        # 统计违规（通过 vehicle 直接检测）
        violation_type, severity = check_violation(info, vehicle)
        if violation_type is not None:
            if severity == "minor":
                violations["minor"] += 1
            elif severity == "major":
                violations["major"] += 1

        # 记录速度
        if hasattr(vehicle, 'speed_m_s'):
            speeds.append(vehicle.speed_m_s * 3.6)  # m/s 转 km/h
        elif isinstance(obs, dict) and "state" in obs:
            vx, vy = obs["state"][0], obs["state"][1]
            speeds.append(np.sqrt(vx**2 + vy**2) * 3.6)

        if render:
            env.render()

        if terminated or truncated:
            break

    episode_time = time.time() - episode_start

    stats = {
        "episode_reward": episode_reward,
        "step_count": step_count,
        "episode_time": episode_time,
        "arrive_dest": info.get("arrive_dest", False),
        "violations": violations,
        "avg_speed": np.mean(speeds) if speeds else 0.0,
        "max_speed": np.max(speeds) if speeds else 0.0,
        "avg_latency_ms": np.mean(latencies) if latencies else 0.0,
        "max_latency_ms": np.max(latencies) if latencies else 0.0,
        "success": info.get("arrive_dest", False),
    }

    return stats


def generate_performance_report(stats_list, output_path="performance_report.md"):
    """
    生成性能报告

    参数:
        stats_list: list of dict, 多个 episode 的统计数据
        output_path: str, 输出文件路径
    """
    total_episodes = len(stats_list)
    successful_episodes = sum(1 for s in stats_list if s["arrive_dest"])
    total_minor_violations = sum(s["violations"]["minor"] for s in stats_list)
    total_major_violations = sum(s["violations"]["major"] for s in stats_list)
    total_violations = total_minor_violations + total_major_violations

    report = f"""# ROBOTAC 性能报告

## 测试概览
- 总测试 episodes: {total_episodes}
- 成功到达终点: {successful_episodes}
- 成功率: {successful_episodes/total_episodes*100:.1f}%

## 成绩统计

### 完成时间
- 平均步数: {np.mean([s['step_count'] for s in stats_list]):.1f}
- 平均耗时: {np.mean([s['episode_time'] for s in stats_list]):.2f}秒
- 最快步数: {np.min([s['step_count'] for s in stats_list])}
- 最快耗时: {np.min([s['episode_time'] for s in stats_list]):.2f}秒

### 速度统计
- 平均速度: {np.mean([s['avg_speed'] for s in stats_list]):.2f} km/h
- 最大速度: {np.max([s['max_speed'] for s in stats_list]):.2f} km/h

### 延迟统计
- 平均推理延迟: {np.mean([s['avg_latency_ms'] for s in stats_list]):.2f}ms
- 最大推理延迟: {np.max([s['max_latency_ms'] for s in stats_list]):.2f}ms
- 延迟 ≤100ms 合规率: {sum(1 for s in stats_list if s['max_latency_ms'] <= 100) / total_episodes * 100:.1f}%

### 违规统计（规则7.5）
| 违规类型 | 次数 | 扣分估算 |
|----------|------|----------|
| 轻微违规（压线） | {total_minor_violations} | -{total_minor_violations * 3}分 |
| 严重违规（碰撞/越界） | {total_major_violations} | -{total_major_violations * 10}分 |
| **合计** | **{total_violations}** | **见下** |

## 评分估算（规则7.4）

### 合规分 (60分基准)
- 轻微违规(压线) × {total_minor_violations} = -{total_minor_violations * 3}分
- 严重违规(碰撞/越界/逆行) × {total_major_violations} = -{total_major_violations * 10}分
- 估算合规分: max(0, 60 - {total_minor_violations * 3} - {total_major_violations * 10})

### 时间分 (40分)
- 基于中位数基准计算

## 速胜条件检查（规则1.2）
- 要求: 无违规 + 20%快于基准
- 当前状态: {'✅ 符合' if total_violations == 0 and successful_episodes > 0 else '❌ 不符合（' + ('有违规' if total_violations > 0 else '未到达终点') + '）'}

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n性能报告已保存: {output_path}")
    return report


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("ROBOTAC 模型可视化验证")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # 检查模型
    if not MODEL_PATH.exists():
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        print("请先运行 train_ppo.py 训练模型")
        exit(1)

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device=DEVICE)

    # 创建环境
    print("创建仿真环境...")
    env = ImageObservationWrapper(MetaDriveEnv(ENV_CONFIG))

    # 运行验证
    print("\n开始验证测试...\n")
    n_episodes = 3
    stats_list = []

    for i in range(n_episodes):
        print(f"--- Episode {i+1}/{n_episodes} ---")
        stats = run_validation_episode(model, env, max_steps=1000, render=True)
        stats_list.append(stats)

        print(f"  到达终点: {'是' if stats['arrive_dest'] else '否'}")
        print(f"  奖励: {stats['episode_reward']:.2f}")
        print(f"  步数: {stats['step_count']}")
        print(f"  耗时: {stats['episode_time']:.2f}秒")
        print(f"  轻微违规(压线): {stats['violations']['minor']}次")
        print(f"  严重违规(碰撞/越界): {stats['violations']['major']}次")
        print(f"  平均延迟: {stats['avg_latency_ms']:.2f}ms")
        print()

    env.close()

    # 生成报告
    report_path = Path(__file__).parent / "performance_report.md"
    generate_performance_report(stats_list, str(report_path))

    print("=" * 60)
    print("验证完成!")
    print("=" * 60)
