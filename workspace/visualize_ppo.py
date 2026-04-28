"""
可视化训练好的 PPO 模型
运行: python visualize_ppo.py
"""

from stable_baselines3 import PPO
from metadrive import MetaDriveEnv
import os

# 配置（与训练时保持一致）
MAP_CONFIG = {
    "generate_type": "big_block_num",
    "generate_config": 3,
    "lane_width": 4,
    "lane_num": 1,
}

ENV_CONFIG = dict(
    use_render=True,       # 关键：打开渲染！
    manual_control=False,
    traffic_density=0.0,
    num_scenarios=1,
    random_agent_model=False,
    on_continuous_line_done=True,
    out_of_route_done=True,
    image_observation=True,
    sensors=dict(rgb_camera=(160, 90)),
    norm_pixel=True,
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
    ),
    map_config=MAP_CONFIG,
)

# 模型路径
MODEL_PATH = "./models/ppo_metadrive.zip"


def main():
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型 {MODEL_PATH}")
        print("请先运行 train_ppo.py 训练模型")
        return

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    # 创建渲染环境
    print("创建渲染环境...")
    env = MetaDriveEnv(ENV_CONFIG)

    # 开始测试
    print("\n开始渲染测试（按 Ctrl+C 退出）")
    print("-" * 50)

    obs, _ = env.reset()
    episode_count = 0
    total_reward = 0

    while True:
        # 使用模型预测动作
        action, _ = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 如果episode结束
        if terminated or truncated:
            episode_count += 1
            done_type = ""

            if info.get("arrive_dest"):
                done_type = "🚗 到达终点！"
            elif info.get("out_of_route"):
                done_type = "❌ 驶出路线"
            elif info.get("on_lane"):
                done_type = "❌ 违规"
            else:
                done_type = "⚠️ 其他结束"

            print(f"Episode {episode_count}: {done_type} | "
                  f"奖励: {total_reward:.2f} | "
                  f"步数: {info.get('episode_length', 0)}")

            total_reward = 0
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()