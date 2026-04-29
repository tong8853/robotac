"""
PPO 训练脚本 - 单车路径规划

使用方法：
    python train_ppo.py
"""
import torch
from pathlib import Path

import tqdm
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod


# ==================== 设备配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 创建日志和模型目录
log_dir = Path("./logs/ppo_metadrive")
model_dir = Path("./models")
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# 训练配置
MODEL_PATH = model_dir / "ppo_metadrive.zip"
TOTAL_TIMESTEPS = 50000  # 总训练步数
CONTINUE_TRAINING = MODEL_PATH.exists()  # 自动检测是否已有模型


# ==================== 配置 ====================

MAP_CONFIG = {
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
    BaseMap.GENERATE_CONFIG: 3,  # 3个路段（小地图，快速验证）
    BaseMap.LANE_WIDTH: 4,
    BaseMap.LANE_NUM: 1,
}

ENV_CONFIG = dict(
    use_render=False,  # 训练时关闭渲染加速
    manual_control=False,
    traffic_density=0.0,
    num_scenarios=10000,
    random_agent_model=False,
    on_continuous_line_done=True,
    out_of_route_done=True,
    image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 160, 90)),  # 降低分辨率加速
    norm_pixel=True,  # SB3需要归一化像素
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
    ),
    map_config=MAP_CONFIG,
)


# ==================== 回调 ====================

class TensorboardCallback(BaseCallback):
    """训练过程中记录关键指标到 TensorBoard"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        # 获取 info 字典（terminated=True 时的信息）
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self):
        if self.episode_rewards:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths))
            self.logger.record("rollout/ep_rew_max", np.max(self.episode_rewards))
            self.episode_rewards = []
            self.episode_lengths = []
        return True


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("PPO 训练开始 - 单车路径规划")
    print("=" * 50)

    # 创建向量化环境（减少内存占用）
    env = make_vec_env(
        lambda: MetaDriveEnv(ENV_CONFIG),
        n_envs=1,  # 只用1个环境（减少内存）
        seed=42,
    )

    # 创建或加载 PPO 模型
    if CONTINUE_TRAINING:
        print(f"检测到已有模型，从断点继续训练: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env, device=DEVICE)
        # 继续训练时，目标步数 = 总步数 - 已训练步数
        remaining_steps = TOTAL_TIMESTEPS - model.num_timesteps
        print(f"已训练: {model.num_timesteps}, 剩余: {remaining_steps}")
    else:
        print(f"从头开始训练")
        model = PPO(
            "MultiInputPolicy",  # 使用MultiInputPolicy处理字典输入
            env,
            learning_rate=3e-4,  # 学习率
            n_steps=512,  # 减少步数
            batch_size=32,  # 减小批次
            n_epochs=5,  # 减少轮数
            gamma=0.99,  # 折扣因子
            gae_lambda=0.95,  # GAE参数
            clip_range=0.2,  # PPO裁剪范围
            device=DEVICE,  # 使用GPU或CPU
            tensorboard_log=str(log_dir),  # TensorBoard日志目录
            verbose=1,
        )
        remaining_steps = TOTAL_TIMESTEPS

    # 开始训练
    print(f"目标训练步数: {remaining_steps}")
    model.learn(
        total_timesteps=remaining_steps,
        callback=TensorboardCallback(),
        progress_bar=False,
    )

    # 保存模型
    save_path = model_dir / "ppo_metadrive.zip"
    model.save(save_path)
    print(f"模型已保存: {save_path}")

    # 测试：创建单环境测试（使用gym.make避免引擎重复初始化）
    print("\n" + "=" * 50)
    print("测试模型...")
    print("=" * 50)

    test_env = gym.make("MetaDrive-validation-v0", config=ENV_CONFIG)
    obs, _ = test_env.reset()
    for i in range(5):
        total_reward = 0
        obs, _ = test_env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        if info.get("arrive_dest"):
            print(f"Episode {i}: 到达终点！奖励={total_reward:.2f}")
        else:
            print(f"Episode {i}: 失败，奖励={total_reward:.2f}")

    print("测试完成")
    env.close()
    test_env.close()