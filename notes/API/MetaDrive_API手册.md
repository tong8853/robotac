# MetaDrive 强化学习完整 API 手册

**适用场景**：ROBOTAC 复赛 - 基于视觉的强化学习自动驾驶  
**版本**：MetaDrive 0.4.3

---

## 目录

1. [环境创建与配置](#1-环境创建与配置)
2. [核心方法](#2-核心方法)
3. [观测结构（重点）](#3-观测结构重点)
4. [车辆状态属性](#4-车辆状态属性)
5. [Reward 函数（可自定义）](#5-reward-函数可自定义)
6. [Done 终止条件](#6-done-终止条件)
7. [Action 动作空间](#7-action-动作空间)
8. [环境配置参数](#8-环境配置参数)
9. [SB3 集成](#9-sb3-集成)
10. [完整训练示例](#10-完整训练示例)

---

## 1. 环境创建与配置

### 1.1 基本环境创建

```python
from metadrive import MetaDriveEnv

env = MetaDriveEnv(dict(
    use_render=False,           # 是否渲染图形窗口（训练时设为False）
    manual_control=False,       # 是否手动控制（键盘）
    num_scenarios=1000,         # 场景数量（随机生成地图）
    start_seed=0,               # 随机种子
    traffic_density=0.0,        # 交通密度（0 = 无社会车辆）
))
```

### 1.2 图像观测配置（复赛必需）

```python
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera

env = MetaDriveEnv(dict(
    # 启用图像观测
    image_observation=True,     # 返回图像作为观测
    image_on_cuda=False,        # 是否将图像放在GPU上（需要PyTorch）
    
    # 传感器配置
    sensors=dict(
        rgb_camera=(RGBCamera, 400, 300),  # (宽, 高)
        # depth_camera=(DepthCamera, 400, 300),  # 深度摄像头
        # semantic_camera=(SemanticCamera, 400, 300),  # 语义分割
    ),
    
    # 车辆配置
    vehicle_config=dict(
        image_source="rgb_camera",  # 使用哪个摄像头
        show_lidar=False,
        show_navi_mark=False,
    ),
))
```

### 1.3 地图配置

```python
env = MetaDriveEnv(dict(
    # 简单方式
    map=3,  # 3 = 简单场景，4 = 复杂场景
    
    # 详细方式
    map_config={
        "generate_type": "big_block_num",  # 生成方式
        "lane_width": 3.5,                 # 车道宽度
        "lane_num": 3,                     # 车道数量
        "exit_length": 50,                 # 出口长度
    },
    
    random_lane_width=True,   # 随机车道宽度
    random_lane_num=True,     # 随机车道数量
))
```

---

## 2. 核心方法

### 2.1 初始化 / 重置

```python
# 方式1：无参数重置
obs, info = env.reset()

# 方式2：指定种子
obs, info = env.reset(seed=42)

# 方式3：重置到指定场景
obs, info = env.reset(seed=100)
# env.current_seed  # 获取当前种子
```

**返回值：**
- `obs`: 观测数据（dict）
- `info`: 环境信息（dict）

### 2.2 执行动作（核心！）

```python
obs, reward, terminated, truncated, info = env.step(action)
```

**参数：**
- `action`: 动作数组，格式为 `[steering, acceleration]`
  - `steering`: 方向盘转角，范围 `[-1.0, 1.0]`
    - `-1.0` = 左转最大
    - `0.0` = 直行
    - `+1.0` = 右转最大
  - `acceleration`: 加速踏板，范围 `[0.0, 1.0]`
    - `0.0` = 刹车/滑行
    - `0.5` = 中等加速
    - `1.0` = 全速加速

**返回值：**
| 变量 | 类型 | 说明 |
|------|------|------|
| `obs` | dict | 观测数据 |
| `reward` | float | 当前步的奖励值 |
| `terminated` | bool | 是否终止（撞车/到达终点/越界） |
| `truncated` | bool | 是否截断（超时） |
| `info` | dict | 详细信息 |

### 2.3 渲染

```python
# 方式1：显示文本信息
env.render(text={
    "speed": 60.5,
    "step": 100,
    "reward": 1.23,
})

# 方式2：渲染为图像（无窗口）
frame = env.render(mode="top_down")  # 返回 numpy 数组
```

### 2.4 关闭环境

```python
env.close()  # 释放资源
```

---

## 3. 观测结构（重点）

### 3.1 图像观测模式

当 `image_observation=True` 时：

```python
obs, _ = env.reset()

print(obs.keys())
# dict_keys(['image', 'state'])

# ========== 图像数据 ==========
# obs["image"]: 形状 (H, W, C)，范围 [0.0, 1.0]，float32

# 获取图像（处理GPU/CPU）
if hasattr(obs["image"], "get"):
    img = obs["image"].get()  # GPU tensor → numpy
else:
    img = obs["image"]        # 直接是 numpy

# 转换为 uint8 [0, 255] 并保存
import cv2
img_uint8 = (img * 255).astype("uint8")
cv2.imwrite("frame.png", img_uint8)

# ========== 状态数据 ==========
# obs["state"]: 形状 (11,)，包含车辆状态向量
# 索引对应:
# [0-1] 相对位置 (local_x, local_y)
# [2] heading
# [3] velocity
# [4] steering
# [5-6] ...
```

### 3.2 非图像观测模式

```python
# 只有 state 向量
obs, _ = env.reset()
# obs: shape = (11,) 或更大，取决于配置
```

---

## 4. 车辆状态属性

### 4.1 通过 env.agent 访问

```python
o, r, d, _, info = env.step(action)

vehicle = env.agent  # 获取当前车辆对象

# ========== 速度相关 ==========
vehicle.speed_km_h      # 速度 (km/h)，float
vehicle.speed_m_s       # 速度 (m/s)，float
vehicle.max_speed_km_h  # 最大速度 (km/h)
vehicle.velocity        # 速度向量 [vx, vy]

# ========== 位置相关 ==========
vehicle.position        # 位置 [x, y, z]
vehicle.heading_theta   # 航向角 (弧度)

# ========== 车道相关 ==========
vehicle.lane            # 当前所在车道对象
vehicle.lane_index      # 车道索引 (road_id, lane_id, offset)
vehicle.lane.speed_limit  # 车道限速 (km/h)

# ========== 状态标志 ==========
vehicle.crash_vehicle   # 是否撞到车辆 (bool)
vehicle.crash_building  # 是否撞到建筑 (bool)
vehicle.crash_object    # 是否撞到障碍物 (bool)
vehicle.crash_sidewalk  # 是否撞到人行道 (bool)
vehicle.crash_human     # 是否撞到人 (bool)

# ========== 车道线检测 ==========
vehicle.on_yellow_continuous_line  # 是否压黄色实线
vehicle.on_white_continuous_line   # 是否压白色实线
vehicle.on_broken_line             # 是否在虚线上
vehicle.on_crosswalk               # 是否在人行道上

# ========== 导航相关 ==========
vehicle.navigation.route_completion  # 路线完成度 (0.0 ~ 1.0)
vehicle.navigation.current_road      # 当前道路
vehicle.navigation.current_lane      # 当前车道
```

### 4.2 通过 info 访问

```python
o, r, d, t, info = env.step(action)

# info 包含:
info = {
    "velocity": [0.0, 0.0],           # 速度向量
    "speed": 0.0,                     # 速度标量
    "crash_vehicle": False,
    "crash_building": False,
    "crash_object": False,
    "crash_sidewalk": False,
    "out_of_road": False,             # 是否越界
    "arrive_dest": False,             # 是否到达终点
    "route_completion": 0.0,          # 路线完成度
    "step_reward": 0.0,               # 当前步奖励
    "navigation_command": "forward",  # 导航指令
}
```

### 4.3 通过导航模块访问

```python
navi = env.agent.navigation

navi.current_road           # 当前道路对象
navi.current_lane           # 当前车道对象
navi.current_ref_lanes      # 参考车道列表
navi.route_completion       # 路线完成度 (0.0~1.0)
navi.get_current_lane_width()  # 当前车道宽度
```

---

## 5. Reward 函数（可自定义）

### 5.1 默认 Reward 组成

MetaDrive 默认的 reward_function 返回：

```python
reward = driving_reward + speed_reward

# driving_reward: 前进奖励，基于纵向位移
# speed_reward: 速度奖励，基于当前速度/最大速度

# 额外奖励/惩罚：
+ success_reward         # 到达终点 (默认 10.0)
- out_of_road_penalty   # 越界 (默认 5.0)
- crash_vehicle_penalty # 撞车 (默认 5.0)
- crash_object_penalty  # 撞障碍物 (默认 5.0)
- crash_sidewalk_penalty# 撞人行道 (默认 0.0)
```

### 5.2 自定义 Reward 函数

```python
from metadrive import MetaDriveEnv
import numpy as np

class MyMetaDriveEnv(MetaDriveEnv):
    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        
        # ===== 速度奖励 =====
        speed_reward = vehicle.speed_km_h / vehicle.max_speed_km_h
        
        # ===== 中心线奖励 =====
        # 获取当前车道中心线的横向距离
        current_lane = vehicle.navigation.current_lane
        _, lateral = current_lane.local_coordinates(vehicle.position)
        lane_width = vehicle.navigation.get_current_lane_width()
        lateral_reward = 1.0 - (abs(lateral) / (lane_width / 2))
        lateral_reward = np.clip(lateral_reward, 0.0, 1.0)
        
        # ===== 撞车惩罚 =====
        crash_penalty = 0.0
        if vehicle.crash_vehicle:
            crash_penalty = -10.0
        elif vehicle.crash_building:
            crash_penalty = -10.0
        
        # ===== 越界惩罚 =====
        out_of_road_penalty = -5.0 if self._is_out_of_road(vehicle) else 0.0
        
        # ===== 成功奖励 =====
        success_reward = 0.0
        if self._is_arrive_destination(vehicle):
            success_reward = 50.0
        
        # ===== 总奖励 =====
        total_reward = (
            speed_reward * 0.3 +
            lateral_reward * 0.5 +
            crash_penalty +
            out_of_road_penalty +
            success_reward
        )
        
        info = {
            "speed_reward": speed_reward,
            "lateral_reward": lateral_reward,
            "crash_penalty": crash_penalty,
            "route_completion": vehicle.navigation.route_completion,
        }
        
        return total_reward, info

# 使用自定义环境
env = MyMetaDriveEnv(dict(
    use_render=False,
    image_observation=True,
))
```

### 5.3 推荐 Reward 设计

```python
def compute_reward(vehicle, done_info):
    """
    推荐 Reward 设计（用于复赛）
    """
    # 1. 速度奖励：鼓励开快
    speed_reward = vehicle.speed_km_h / 80.0  # 归一化到 80km/h
    
    # 2. 中心线奖励：鼓励居中行驶
    current_lane = vehicle.navigation.current_lane
    _, lateral = current_lane.local_coordinates(vehicle.position)
    lane_width = vehicle.navigation.get_current_lane_width()
    lateral_reward = 1.0 - abs(lateral) / (lane_width / 2)
    lateral_reward = max(0.0, lateral_reward)
    
    # 3. 前进奖励：鼓励沿车道前进
    route_reward = vehicle.navigation.route_completion
    
    # 4. 惩罚项
    crash_penalty = 0.0
    if vehicle.crash_vehicle:
        crash_penalty = -10.0
    if vehicle.crash_building:
        crash_penalty = -10.0
        
    off_road_penalty = -5.0 if done_info.get("out_of_road", False) else 0.0
    
    # 5. 成功奖励
    success_reward = 20.0 if vehicle.navigation.route_completion > 0.99 else 0.0
    
    # 综合
    total = (
        speed_reward * 0.2 +
        lateral_reward * 0.3 +
        route_reward * 0.3 +
        crash_penalty +
        off_road_penalty +
        success_reward
    )
    
    return total
```

---

## 6. Done 终止条件

### 6.1 默认终止条件

```python
# 在 step 返回中，terminated = True 当：

# 1. 撞车
vehicle.crash_vehicle  # 撞到车
vehicle.crash_building # 撞到建筑
vehicle.crash_object   # 撞到障碍物
vehicle.crash_sidewalk # 撞到人行道
vehicle.crash_human    # 撞到人

# 2. 越界（默认启用）
# out_of_road_done = True (默认)

# 3. 压实线（可选）
# on_continuous_line_done = True (默认)

# 4. 到达终点
vehicle.navigation.route_completion >= 0.99

# 5. 超时
episode_step >= horizon (默认1000)
```

### 6.2 配置终止条件

```python
env = MetaDriveEnv(dict(
    # 终止条件配置
    out_of_road_done=True,        # 越界是否终止
    crash_vehicle_done=True,      # 撞车是否终止
    crash_object_done=True,       # 撞障碍物是否终止
    crash_human_done=True,        # 撞人是否终止
    on_continuous_line_done=True, # 压实线是否终止
    on_broken_line_done=False,    # 压虚线是否终止
    
    # 超时设置
    horizon=1000,                 # 最大步数
))
```

### 6.3 自定义 Done 条件

```python
class MyMetaDriveEnv(MetaDriveEnv):
    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        done_info = {}
        
        # 检查默认条件
        done = done or vehicle.crash_vehicle
        done = done or vehicle.crash_building
        done = done or self._is_out_of_road(vehicle)
        
        # 自定义：速度太慢也终止（可选）
        if vehicle.speed_km_h < 1.0 and self.episode_lengths[vehicle_id] > 100:
            done = True
            done_info["too_slow"] = True
        
        # 成功终止
        if self._is_arrive_destination(vehicle):
            done = True
            done_info["success"] = True
        
        # 超时
        if self.episode_lengths[vehicle_id] >= self.config["horizon"]:
            done = True
            done_info["timeout"] = True
        
        return done, done_info
```

---

## 7. Action 动作空间

### 7.1 动作空间定义

```python
# Box 空间
env.action_space
# Box(-1.0, 1.0, (2,), float32)

# 含义：
# action[0] = steering (方向盘): [-1.0, 1.0]
# action[1] = acceleration (加速): [0.0, 1.0]
```

### 7.2 动作示例

```python
# 直行
action = [0.0, 0.3]

# 左转
action = [-0.5, 0.3]

# 右转
action = [0.5, 0.3]

# 刹车
action = [0.0, 0.0]

# 倒车（部分环境支持）
action = [0.0, -0.3]
```

---

## 8. 环境配置参数

### 8.1 完整配置示例

```python
config = dict(
    # ===== 环境基础 =====
    use_render=False,
    manual_control=False,
    num_scenarios=10000,       # 场景池大小
    start_seed=0,
    
    # ===== 地图 =====
    map=3,                     # 地图难度 1-10
    random_lane_width=True,    # 随机车道宽度
    random_lane_num=True,      # 随机车道数量
    
    # ===== 交通 =====
    traffic_density=0.0,       # 交通密度，0=无车
    random_traffic=False,      # 随机交通
    
    # ===== 观测 =====
    image_observation=True,    # 图像观测
    image_on_cuda=False,       # GPU图像
    
    # ===== 传感器 =====
    sensors=dict(
        rgb_camera=(RGBCamera, 400, 300),
    ),
    vehicle_config=dict(
        image_source="rgb_camera",
        show_lidar=False,
        show_navi_mark=False,
    ),
    
    # ===== 奖励配置 =====
    success_reward=10.0,       # 成功奖励
    out_of_road_penalty=5.0,   # 越界惩罚
    crash_vehicle_penalty=5.0, # 撞车惩罚
    crash_object_penalty=5.0,  # 撞障碍物惩罚
    crash_sidewalk_penalty=0.0,# 撞人行道惩罚
    driving_reward=1.0,        # 前进奖励系数
    speed_reward=0.1,          # 速度奖励系数
    use_lateral_reward=False,  # 是否使用横向奖励
    
    # ===== 终止条件 =====
    out_of_road_done=True,     # 越界终止
    crash_vehicle_done=True,   # 撞车终止
    crash_object_done=True,    # 撞障碍物终止
    on_continuous_line_done=True,  # 压实线终止
    horizon=1000,              # 最大步数
)

env = MetaDriveEnv(config)
```

### 8.2 常用参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_render` | bool | False | 是否渲染窗口 |
| `image_observation` | bool | False | 启用图像观测 |
| `traffic_density` | float | 0.1 | 交通密度 |
| `horizon` | int | 1000 | 最大步数 |
| `success_reward` | float | 10.0 | 成功奖励 |
| `crash_vehicle_penalty` | float | 5.0 | 撞车惩罚 |
| `out_of_road_penalty` | float | 5.0 | 越界惩罚 |

---

## 9. SB3 集成

### 9.1 封装为 Gym 环境

```python
import gymnasium as gym
from metadrive import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 方式1：直接使用
env = MetaDriveEnv(dict(
    use_render=False,
    image_observation=True,
    num_scenarios=100,
    start_seed=0,
    traffic_density=0.0,
))

# 方式2：使用 VecEnv（推荐）
def make_env():
    def _init():
        env = MetaDriveEnv(dict(
            use_render=False,
            image_observation=True,
            num_scenarios=1000,
            traffic_density=0.0,
        ))
        return env
    return _init

vec_env = DummyVecEnv([make_env])

# 方式3：Gymnasium 包装器
from metadrive.envs.gym_wrapper import create_gym_wrapper
GymEnv = create_gym_wrapper(MetaDriveEnv)
env = GymEnv(dict(
    use_render=False,
    image_observation=True,
))
```

### 9.2 训练 PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 创建环境
env = MetaDriveEnv(dict(
    use_render=False,
    image_observation=True,
    num_scenarios=1000,
    traffic_density=0.0,
))

# 创建模型
model = PPO(
    "CnnPolicy",           # 使用CNN处理图像
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
)

# 训练
model.learn(total_timesteps=100000)

# 保存模型
model.save("ppo_metadrive")

# 加载模型
# model = PPO.load("ppo_metadrive")
```

### 9.3 评估模型

```python
from metadrive import MetaDriveEnv

env = MetaDriveEnv(dict(
    use_render=False,
    image_observation=True,
    num_scenarios=10,
    start_seed=0,
))

obs, _ = env.reset()
episode_reward = 0
episode_steps = 0

for i in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_reward += reward
    episode_steps += 1
    
    if terminated or truncated:
        print(f"Episode reward: {episode_reward}, steps: {episode_steps}")
        episode_reward = 0
        episode_steps = 0
        obs, _ = env.reset()

env.close()
```

---

## 10. 完整训练示例

```python
#!/usr/bin/env python
"""
ROBOTAC 复赛 - 完整训练示例
"""
import cv2
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.envs.gym_wrapper import create_gym_wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# 1. 创建 Gym 兼容环境
GymEnv = create_gym_wrapper(MetaDriveEnv)

env = GymEnv(dict(
    use_render=False,
    image_observation=True,
    num_scenarios=1000,
    start_seed=0,
    traffic_density=0.0,
    horizon=1000,
    
    # 奖励配置
    success_reward=20.0,
    out_of_road_penalty=10.0,
    crash_vehicle_penalty=10.0,
))

# 2. 创建模型
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    device="cpu",  # 用CPU训练
)

# 3. 设置回调
callbacks = [
    CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="ppo_metadrive"),
]

# 4. 训练
model.learn(
    total_timesteps=500000,
    callback=callbacks,
    progress_bar=True,
)

# 5. 保存最终模型
model.save("ppo_metadrive_final")

print("训练完成！")
```

---

## 附录：A. 常见问题

### Q1: 图像数据获取失败
```python
# 确保启用 image_observation
env = MetaDriveEnv(dict(image_observation=True, ...))

obs = env.reset()
print(obs.keys())  # 应该有 'image' key
```

### Q2: reward 为 NaN
```python
# 检查是否有除零或无效计算
lateral_reward = 1.0 - abs(lateral) / (lane_width / 2 + 1e-6)
```

### Q3: 训练太慢
```python
# 使用多进程
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    return GymEnv(dict(...))

vec_env = SubprocVecEnv([make_env for _ in range(4)])
```

### Q4: 如何只用CPU训练
```python
model = PPO("CnnPolicy", env, device="cpu")
```

---

## 附录：B. 参考资料

1. **MetaDrive 官方文档**: https://metadrive.readthedocs.io/
2. **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
3. **Gymnasium**: https://gymnasium.farama.org/
4. **PPO 论文**: Proximal Policy Optimization Algorithms

---

## 下一步

运行测试脚本验证环境：
```bash
conda activate robotac
python workspace/保存图像数据.py
```