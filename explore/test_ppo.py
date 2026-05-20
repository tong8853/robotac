"""
测试脚本 - 验证训练好的模型

使用方法:
    conda activate robotac
    python test_ppo.py
"""
import torch
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
    use_render=False,
    manual_control=False,
    num_agents=1,
    traffic_density=0.0,
    num_scenarios=10000,
    random_agent_model=False,
    on_continuous_line_done=True,
    out_of_route_done=True,
    image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 320, 180)),  # 提升分辨率以生成更清的GIF
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
        image_source="rgb_camera",
    ),
    map_config=MAP_CONFIG,
    norm_pixel=False,
)


def test_model(model, env, n_episodes=5, max_steps=1000):
    """测试模型性能"""
    results = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0

        # 获取车辆对象用于违规检测
        vehicle = env.agent

        violations = {"minor": 0, "major": 0}

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # 统计违规
            violation_type, severity = check_violation(info, vehicle)
            if violation_type is not None:
                if severity == "minor":
                    violations["minor"] += 1
                elif severity == "major":
                    violations["major"] += 1

            if terminated or truncated:
                break

        results.append({
            "episode": i + 1,
            "reward": total_reward,
            "steps": step_count,
            "arrive_dest": info.get("arrive_dest", False),
            "violations": violations,
        })

        status = "✅ 到达终点" if info.get("arrive_dest") else "❌ 未到达"
        print(f"Episode {i+1}: {status}, 奖励={total_reward:.2f}, 步数={step_count}")
        print(f"  轻微违规(压线): {violations['minor']}次, 严重违规(碰撞/越界): {violations['major']}次")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("PPO 模型测试")
    print("=" * 50)

    if not MODEL_PATH.exists():
        print(f"错误: 模型不存在: {MODEL_PATH}")
        exit(1)

    print(f"加载模型: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device=DEVICE)

    print("创建环境...")
    env = ImageObservationWrapper(MetaDriveEnv(ENV_CONFIG))

    print("\n开始测试...\n")
    results = test_model(model, env, n_episodes=5)

    # 统计
    success_count = sum(1 for r in results if r["arrive_dest"])
    total_minor = sum(r["violations"]["minor"] for r in results)
    total_major = sum(r["violations"]["major"] for r in results)
    total_violations = total_minor + total_major

    print("\n" + "=" * 50)
    print(f"测试完成: {success_count}/{len(results)} 成功")
    print(f"总违规: 轻微{total_minor}次, 严重{total_major}次")
    print("=" * 50)

    env.close()
