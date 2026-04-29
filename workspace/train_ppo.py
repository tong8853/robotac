"""
PPO 训练脚本 - 单车路径规划

使用方法：
    python train_ppo.py
"""
import torch
from pathlib import Path
import yaml

import numpy as np
import tqdm
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import wandb
from wandb.integration import sb3

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod


# ==================== 加载配置 ====================
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ==================== 设备配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# 路径配置
log_dir = Path(config["log_dir"])
model_dir = Path(config["model_dir"])
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# 训练配置
MODEL_PATH = model_dir / config["model_name"]
TOTAL_TIMESTEPS = config["total_timesteps"]
CONTINUE_TRAINING = MODEL_PATH.exists()  # 自动检测是否已有模型


# ==================== 配置 ====================

MAP_CONFIG = {
    BaseMap.GENERATE_TYPE: config["map_config"]["generate_type"],
    BaseMap.GENERATE_CONFIG: config["map_config"]["generate_config"],
    BaseMap.LANE_WIDTH: config["map_config"]["lane_width"],
    BaseMap.LANE_NUM: config["map_config"]["lane_num"],
}

ENV_CONFIG = dict(
    use_render=config["env_config"]["use_render"],
    manual_control=config["env_config"]["manual_control"],
    traffic_density=config["env_config"]["traffic_density"],
    num_scenarios=config["env_config"]["num_scenarios"],
    random_agent_model=config["env_config"]["random_agent_model"],
    on_continuous_line_done=config["env_config"]["on_continuous_line_done"],
    out_of_route_done=config["env_config"]["out_of_route_done"],
    image_observation=config["env_config"]["image_observation"],
    sensors=dict(rgb_camera=(RGBCamera, config["env_config"]["rgb_camera_width"], config["env_config"]["rgb_camera_height"])),
    norm_pixel=True,  # SB3需要归一化像素
    vehicle_config=dict(
        show_lidar=config["env_config"]["show_lidar"],
        show_navi_mark=config["env_config"]["show_navi_mark"],
        show_line_to_navi_mark=config["env_config"]["show_line_to_navi_mark"],
    ),
    map_config=MAP_CONFIG,
)


# ==================== 回调 ====================

class TensorboardCallback(BaseCallback):
    """训练过程中记录关键指标到 TensorBoard 和 wandb"""

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
            mean_rew = np.mean(self.episode_rewards)
            max_rew = np.max(self.episode_rewards)
            mean_len = np.mean(self.episode_lengths)

            # 记录到 TensorBoard
            self.logger.record("rollout/ep_rew_mean", mean_rew)
            self.logger.record("rollout/ep_len_mean", mean_len)
            self.logger.record("rollout/ep_rew_max", max_rew)

            # 本地打印
            print(f"[迭代 {self.num_timesteps // 512}] 奖励均值: {mean_rew:.2f}, 最大: {max_rew:.2f}, 步数均值: {mean_len:.0f}")

            # 记录到 wandb
            wandb.log({
                "iteration": self.num_timesteps // 512,
                "timesteps": self.num_timesteps,
                "ep_rew_mean": mean_rew,
                "ep_rew_max": max_rew,
                "ep_len_mean": mean_len,
            })

            self.episode_rewards = []
            self.episode_lengths = []
        return True


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("PPO 训练开始 - 单车路径规划")
    print("=" * 50)

    # 初始化 wandb
    wandb.init(
        name=config["wandb"]["name"],
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config={
            "learning_rate": config["learning_rate"],
            "n_steps": config["n_steps"],
            "batch_size": config["batch_size"],
            "n_epochs": config["n_epochs"],
            "gamma": config["gamma"],
            "gae_lambda": config["gae_lambda"],
            "clip_range": config["clip_range"],
            "total_timesteps": TOTAL_TIMESTEPS,
        },
        sync_tensorboard=True,
        resume="allow" if CONTINUE_TRAINING else False,
    )

    # 创建向量化环境（减少内存占用）
    env = make_vec_env(
        lambda: MetaDriveEnv(ENV_CONFIG),
        n_envs=config["n_envs"],
        seed=config["seed"],
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
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            device=DEVICE,  # 使用GPU或CPU
            tensorboard_log=str(log_dir),  # TensorBoard日志目录
            verbose=1,
        )
        remaining_steps = TOTAL_TIMESTEPS

    # 添加 wandb 回调
    wandb_callback = sb3.WandbCallback(
        model_save_freq=config["wandb"]["model_save_freq"],
        model_save_path=str(model_dir),
        verbose=1,
    )
    callbacks = [TensorboardCallback(), wandb_callback]

    # 开始训练
    print(f"目标训练步数: {remaining_steps}")
    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
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

    # 关闭 wandb
    wandb.finish()