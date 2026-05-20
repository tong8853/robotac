"""
可视化验证脚本 - 生成 GIF 动画和性能报告（第一人称视角）

使用方法:
    conda activate robotac
    python visualize_ppo.py
"""
import torch
import time
import yaml
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.utils.doc_utils import generate_gif
from observation_wrapper import ImageObservationWrapper
from reward_function import check_violation

# ==================== 设备配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ==================== 配置 ====================
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODEL_PATH = Path(__file__).parent / "models" / config["model_name"]
GIF_DIR = Path(__file__).parent / "Gif"
GIF_DIR.mkdir(exist_ok=True)


# ==================== 环境配置 ====================
MAP_CONFIG = dict(
    type="block_num",
    config=3,
    lane_width=4,
    lane_num=1,
)

ENV_CONFIG = dict(
    use_render=False,
    manual_control=False,
    num_agents=1,
    traffic_density=0.0,
    num_scenarios=10000,
    random_agent_model=False,
    on_continuous_line_done=True,
    out_of_route_done=True,
    image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 320, 180)),  # 提升分辨率以生成更清晰的GIF
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
        image_source="rgb_camera",
    ),
    map_config=MAP_CONFIG,
    norm_pixel=False,  # 与训练一致
)


def generate_performance_report(stats_list, output_path="performance_report.md"):
    """生成性能报告"""
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


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("ROBOTAC 模型可视化验证 (第一人称视角GIF)")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    if not MODEL_PATH.exists():
        print(f"错误: 模型文件不存在: {MODEL_PATH}")
        exit(1)

    print(f"加载模型: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device=DEVICE)

    # 创建单一环境
    print("创建仿真环境...")
    env = ImageObservationWrapper(MetaDriveEnv(ENV_CONFIG))

    print("\n开始验证测试 (生成第一人称视角GIF)...\n")
    n_episodes = 3
    stats_list = []

    for i in range(n_episodes):
        print(f"--- Episode {i+1}/{n_episodes} ---")

        # 重置环境
        obs, _ = env.reset()

        episode_reward = 0.0
        step_count = 0
        episode_start = time.time()
        vehicle = env.agent

        violations = {"minor": 0, "major": 0}
        speeds = []
        latencies = []
        frames = []  # 存储GIF帧

        max_steps = 1000
        for step in range(max_steps):
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            latency_ms = (time.time() - step_start) * 1000
            latencies.append(latency_ms)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            violation_type, severity = check_violation(info, vehicle)
            if violation_type is not None:
                if severity == "minor":
                    violations["minor"] += 1
                elif severity == "major":
                    violations["major"] += 1

            if hasattr(vehicle, 'speed_m_s'):
                speeds.append(vehicle.speed_m_s * 3.6)

            # 捕获第一人称视角帧
            img_obs = env.env.observations['default_agent'].img_obs
            frame = img_obs.get_image()  # (H, W, 3), uint8 [0, 255] with norm_pixel=False
            frames.append(frame[..., ::-1])  # RGB转BGR

            if terminated or truncated:
                break

        episode_time = time.time() - episode_start

        # 保存GIF（使用MetaDrive内置函数）
        gif_path = GIF_DIR / f"episode_{i+1}.gif"
        generate_gif(frames, gif_name=str(gif_path), duration=100)
        print(f"  GIF 已保存: {gif_path}")

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
        }
        stats_list.append(stats)

        print(f"  到达终点: {'是' if stats['arrive_dest'] else '否'}")
        print(f"  奖励: {stats['episode_reward']:.2f}")
        print(f"  步数: {stats['step_count']}")
        print(f"  捕获帧数: {len(frames)}")
        print(f"  轻微违规: {stats['violations']['minor']}次")
        print(f"  严重违规: {stats['violations']['major']}次")
        print()

    env.close()

    report_path = Path(__file__).parent / "performance_report.md"
    generate_performance_report(stats_list, str(report_path))

    print("=" * 60)
    print(f"验证完成! GIF已保存到: {GIF_DIR}")
    print("=" * 60)
