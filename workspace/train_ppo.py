"""
PPO 训练脚本 - 单车路径规划

使用方法：
    python train_ppo.py
"""
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

    # 创建 PPO 模型（MultiInputPolicy 处理字典观测空间）
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
        tensorboard_log="./logs/ppo_metadrive",  # TensorBoard日志目录
        verbose=1,
    )

    # 开始训练
    total_timesteps = 50000  # 训练50000步
    print(f"目标训练步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardCallback(),
        progress_bar=False,
    )

    # 保存模型
    save_path = "./models/ppo_metadrive.zip"
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