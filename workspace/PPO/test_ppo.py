"""
模型测试脚本 - 查看训练效果

使用方法：
    python test_ppo.py
    python test_ppo.py --model ./models/ppo_metadrive.zip
"""

import argparse
import numpy as np

from stable_baselines3 import PPO

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod


# ==================== 配置 ====================

MAP_CONFIG = {
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
    BaseMap.GENERATE_CONFIG: 3,
    BaseMap.LANE_WIDTH: 4,
    BaseMap.LANE_NUM: 1,
}

ENV_CONFIG = dict(
    use_render=True,  # 测试时开启渲染
    manual_control=False,
    traffic_density=0.0,
    num_scenarios=10000,
    random_agent_model=False,
    on_continuous_line_done=True,
    out_of_route_done=True,
    image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 320, 180)),
    norm_pixel=False,
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
    ),
    map_config=MAP_CONFIG,
)


# ==================== 主程序 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/ppo_metadrive.zip")
    args = parser.parse_args()

    print("=" * 50)
    print("测试模型...")
    print(f"模型路径: {args.model}")
    print("=" * 50)

    # 加载模型
    try:
        model = PPO.load(args.model)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("使用随机策略")
        model = None

    # 创建环境
    env = MetaDriveEnv(ENV_CONFIG)

    episode_count = 0
    total_rewards = []
    total_lengths = []

    try:
        obs, _ = env.reset()

        for step in range(100000):
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()  # 随机策略

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                episode_count += 1
                total_rewards.append(info.get("episode_reward", 0))
                total_lengths.append(info.get("episode_length", 0))

                if info.get("arrive_dest"):
                    print(f"Episode {episode_count}: 到达终点！步数={info['episode_length']}, 奖励={info['episode_reward']:.2f}")
                else:
                    print(f"Episode {episode_count}: 失败 步数={info['episode_length']}, 奖励={info['episode_reward']:.2f}")

                obs, _ = env.reset()

                if episode_count >= 10:
                    break

    finally:
        env.close()

    # 统计
    if total_rewards:
        print("\n" + "=" * 50)
        print(f"测试完成：共 {episode_count} 局")
        print(f"平均奖励：{np.mean(total_rewards):.2f}")
        print(f"平均步数：{np.mean(total_lengths):.2f}")
        print(f"最大奖励：{np.max(total_rewards):.2f}")
        print("=" * 50)