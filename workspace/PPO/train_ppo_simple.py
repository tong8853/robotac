"""
SB3 PPO 最简示例 - 向量观测版本
运行: python train_ppo_simple.py
"""

import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO

# 1. 创建环境（使用 gymnasium 接口）
env = gym.make(
    "MetaDrive-validation-v0",
    config={
        "num_scenarios": 1,
        "use_render": False,      # 训练时不渲染
        "manual_control": False,
        "traffic_density": 0.0,   # 无其他车辆
    }
)

# 2. 创建 PPO 模型（使用 MlpPolicy 因为是向量观测）
model = PPO(
    "MlpPolicy",          # 多层感知机策略
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    verbose=1,
)

# 3. 训练
print("开始训练...")
model.learn(total_timesteps=10000)
print("训练完成！")

# 4. 保存模型
model.save("ppo_simple")
print("模型已保存: ppo_simple")

# 5. 测试（渲染）
print("\n开始测试...")
test_env = gym.make(
    "MetaDrive-validation-v0",
    config={
        "num_scenarios": 1,
        "use_render": True,   # 测试时打开渲染
        "manual_control": False,
    }
)

obs, _ = test_env.reset()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)

    if terminated or truncated:
        if info.get("arrive_dest"):
            print(f"到达终点！步数: {step + 1}")
        obs, _ = test_env.reset()

test_env.close()
env.close()