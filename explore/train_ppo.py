"""
PPO 训练脚本 - ROBOTAC 数字仿真挑战赛
符合规则第4节技术要求和第7.5节违规判定标准

基于 MetaDrive-Tutorials 官方示例优化
使用方法:
    conda activate robotac
    python train_ppo.py
"""
import torch
import time
import yaml
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from observation_wrapper import ImageObservationWrapper
from reward_function import compute_reward, check_violation

# ==================== 设备配置（设备无关）====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ==================== 加载配置 ====================
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 路径配置
log_dir = Path(config["log_dir"])
model_dir = Path(config["model_dir"])
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# 训练配置
MODEL_PATH = model_dir / config["model_name"]
TOTAL_TIMESTEPS = config["total_timesteps"]
CONTINUE_TRAINING = MODEL_PATH.exists()

# ==================== 环境配置（合规检查）====================
MAP_CONFIG = dict(
    type="block_num",  # 使用 BigGenerateMethod.BLOCK_NUM
    config=3,
    lane_width=4,
    lane_num=1,
)

# 合规配置：仅使用单一RGB摄像头，无Lidar，无导航标记
# 符合规则4.1：仅接收摄像头实时图像数据作为输入
ENV_CONFIG = dict(
    use_render=False,
    manual_control=False,
    num_agents=1,
    traffic_density=0.0,  # 无其他车辆干扰
    num_scenarios=10000,
    random_agent_model=False,
    on_continuous_line_done=True,  # 压白实线检测
    out_of_route_done=True,  # 偏离路线检测
    image_observation=True,  # 使用图像观测（规则4.1要求）
    sensors=dict(rgb_camera=(RGBCamera, 320, 180)),  # 提升分辨率以生成更清晰的GIF
    vehicle_config=dict(
        show_lidar=False,  # 禁用Lidar - 规则4.1
        show_navi_mark=False,  # 禁用导航标记 - 规则4.1
        show_line_to_navi_mark=False,  # 禁用导航线 - 规则4.1
        image_source="rgb_camera",
    ),
    map_config=MAP_CONFIG,
    norm_pixel=False,  # 保持uint8格式，与赛方样例一致
)


def make_env(seed=0):
    """创建单个环境的工厂函数（用于向量化环境）"""
    def _init():
        env = MetaDriveEnv(dict(
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
            norm_pixel=False,
            start_seed=seed,  # MetaDrive 使用 start_seed
        ))
        env = ImageObservationWrapper(env)
        return env
    return _init


# ==================== 延迟检查回调 ====================
class LatencyGuardCallback(BaseCallback):
    """
    延迟守卫：确保每步推理延迟 ≤100ms
    符合规则4.1：单次指令响应延迟 ≤100ms
    """

    def __init__(self, latency_threshold_ms=100, verbose=0):
        super().__init__(verbose)
        self.latency_threshold_ms = latency_threshold_ms
        self.latency_violations = 0
        self.total_steps = 0

    def _on_step(self):
        return True

    def check_latency(self, predict_time_ms):
        """检查预测延迟是否超标"""
        self.total_steps += 1
        if predict_time_ms > self.latency_threshold_ms:
            self.latency_violations += 1
            if self.verbose > 0:
                print(f"[延迟警告] 步骤{self.total_steps}: {predict_time_ms:.1f}ms > {self.latency_threshold_ms}ms")
        return predict_time_ms <= self.latency_threshold_ms

    def get_stats(self):
        return {
            "total_steps": self.total_steps,
            "violations": self.latency_violations,
            "compliance_rate": (self.total_steps - self.latency_violations) / self.total_steps * 100
            if self.total_steps > 0 else 100.0
        }


# ==================== 训练回调 ====================
class TensorboardCallback(BaseCallback):
    """训练过程中记录关键指标到TensorBoard"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.violation_counts = {
            "minor": 0,  # 压线
            "major": 0,  # 碰撞/越界/逆行
        }

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            # 记录违规次数
            violation_type, severity = check_violation(info)
            if violation_type is not None:
                if severity == "minor":
                    self.violation_counts["minor"] += 1
                elif severity == "major":
                    self.violation_counts["major"] += 1

            # 记录episode信息
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

    def _on_rollout_end(self):
        if self.episode_rewards:
            mean_rew = np.mean(self.episode_rewards)
            max_rew = np.max(self.episode_rewards)
            mean_len = np.mean(self.episode_lengths)

            self.logger.record("rollout/ep_rew_mean", mean_rew)
            self.logger.record("rollout/ep_len_mean", mean_len)
            self.logger.record("rollout/ep_rew_max", max_rew)
            self.logger.record("rollout/violations/minor", self.violation_counts["minor"])
            self.logger.record("rollout/violations/major", self.violation_counts["major"])

            print(f"[迭代 {self.num_timesteps // config['n_steps']}] "
                  f"奖励均值: {mean_rew:.2f}, 最大: {max_rew:.2f}, "
                  f"步数均值: {mean_len:.0f}, "
                  f"轻微违规: {self.violation_counts['minor']}次, "
                  f"严重违规: {self.violation_counts['major']}次")

            # 重置计数
            self.violation_counts = {"minor": 0, "major": 0}
            self.episode_rewards = []
            self.episode_lengths = []
        return True


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("PPO 训练开始 - ROBOTAC 数字仿真挑战赛")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"训练步数: {TOTAL_TIMESTEPS}")
    print(f"模型保存路径: {MODEL_PATH}")
    print("合规配置: 仅使用RGB摄像头, 无Lidar, 无导航标记")
    print("=" * 60)

    # 创建向量化环境
    n_envs = config["n_envs"]
    seed = config["seed"]
    if n_envs == 1:
        # 单环境使用 DummyVecEnv
        env = make_vec_env(make_env(seed), n_envs=1, seed=seed)
    else:
        # 多环境使用 SubprocVecEnv（Linux/Mac 推荐）
        env = SubprocVecEnv([make_env(seed + i) for i in range(n_envs)])

    # 创建或加载PPO模型（设备无关）
    if CONTINUE_TRAINING:
        print(f"检测到已有模型，从断点继续训练: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env, device=DEVICE)
        remaining_steps = TOTAL_TIMESTEPS - model.num_timesteps
        print(f"已训练: {model.num_timesteps}, 剩余: {remaining_steps}")
    else:
        print("从头开始训练")
        # 使用 MultiInputPolicy 处理 Dict 观测空间（图像+状态向量）
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            device=DEVICE,  # 设备无关
            tensorboard_log=str(log_dir),
            verbose=0,
        )
        remaining_steps = TOTAL_TIMESTEPS

    # 延迟守卫
    latency_guard = LatencyGuardCallback(latency_threshold_ms=100, verbose=0)
    callbacks = [TensorboardCallback(), latency_guard]

    # 开始训练
    print(f"\n开始训练，目标步数: {remaining_steps}")
    start_time = time.time()

    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        progress_bar=False,
    )

    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time:.1f}秒")

    # 保存模型
    save_path = model_dir / config["model_name"]
    model.save(save_path)
    print(f"模型已保存: {save_path}")

    # 测试模型
    print("\n" + "=" * 50)
    print("测试模型...")
    print("=" * 50)

    env.close()  # 关闭训练环境
    test_env_config = {**ENV_CONFIG, "start_seed": 42}  # 使用训练时的种子
    test_env = ImageObservationWrapper(MetaDriveEnv(test_env_config))

    for i in range(3):
        total_reward = 0
        obs, _ = test_env.reset()
        episode_start = time.time()
        step_count = 0
        max_steps = 1000

        # 获取车辆对象用于违规检测
        vehicle = test_env.agent

        while step_count < max_steps:
            # 延迟测量
            step_start = time.time()
            action, _ = model.predict(obs, deterministic=True)
            predict_time_ms = (time.time() - step_start) * 1000

            # 延迟检查
            latency_guard.check_latency(predict_time_ms)

            obs, reward, terminated, truncated, info = test_env.step(action)

            # 使用 reward_function 计算奖励
            reward, done_info = compute_reward(obs, reward, terminated, info, vehicle)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        episode_time = time.time() - episode_start
        arrive_dest = info.get("arrive_dest", False)

        print(f"Episode {i+1}: "
              f"{'到达终点' if arrive_dest else '未到达终点'}, "
              f"奖励={total_reward:.2f}, "
              f"步数={step_count}, "
              f"耗时={episode_time:.1f}秒")

    print("\n测试完成")
    env.close()
    test_env.close()

    # 延迟统计
    latency_stats = latency_guard.get_stats()
    print(f"\n延迟统计: 总步骤={latency_stats['total_steps']}, "
          f"超标次数={latency_stats['violations']}, "
          f"合规率={latency_stats['compliance_rate']:.1f}%")
