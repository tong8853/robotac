# Stable Baselines3 + MetaDrive 强化学习教程

## 环境要求
```bash
pip install metadrive stable-baselines3 gymnasium
```

---

## 一、核心 API

### 1. 创建环境
```python
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv

# 方式1：使用 gymnasium 的 make
env = gym.make("MetaDrive-validation-v0", config={"num_scenarios": 10})

# 方式2：直接实例化
env = MetaDriveEnv(config={"num_scenarios": 10})
```

### 2. 环境基本操作
```python
# 重置环境，返回观察值
observation, info = env.reset()

# 执行动作，返回 (下一状态, 奖励, 是否终止, 是否截断, 信息)
observation, reward, terminated, truncated, info = env.step(action)

# 关闭环境
env.close()
```

### 3. 常用环境配置
```python
config = {
    "num_scenarios": 1,           # 场景数量
    "start_seed": 0,              # 随机种子
    "vehicle_config": {
        "lidar": {"num_lasers": 3, "distance": 50},  # 激光雷达
        "navigation": True,                             # 导航
    },
    "use_render": True,           # 是否渲染（训练时设为 False）
    "manual_control": False,      # 是否手动控制
}
```

---

## 二、PPO 算法

### 1. 创建 PPO 模型
```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",           # 策略网络类型
    env=env,                      # 环境
    learning_rate=3e-4,           # 学习率
    n_steps=2048,                 # 每次更新收集的步数
    batch_size=64,                # 批次大小
    n_epochs=10,                  # 每次更新的 epoch 数
    gamma=0.99,                   # 折扣因子
    gae_lambda=0.95,              # GAE 参数
    clip_range=0.2,               # PPO 裁剪范围
    ent_coef=0.01,                # 熵系数（鼓励探索）
    verbose=1,                    # 日志级别
)
```

### 2. 训练模型
```python
# 方式1：指定总步数
model.learn(total_timesteps=100000)

# 方式2：指定步数 + 回调 + 保存间隔
model.learn(
    total_timesteps=100000,
    callback=None,                # 回调函数
    log_interval=10,              # 每10次迭代打印日志
    progress_bar=True,            # 显示进度条
)
```

### 3. 保存和加载模型
```python
# 保存模型
model.save("ppo_metadrive")

# 加载模型
model = PPO.load("ppo_metadrive", env=env)
```

### 4. 模型推理
```python
# 单步推理
observation, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        observation, _ = env.reset()

# 渲染模式
env = gym.make("MetaDrive-validation-v0", config={"use_render": True})
```

---

## 三、常用策略网络

| Policy | 适用场景 |
|--------|----------|
| `MlpPolicy` | 状态是向量（数值数组） |
| `CnnPolicy` | 状态是图像 |
| `MultiInputPolicy` | 状态包含向量+图像 |

MetaDrive 默认输出向量状态，用 `MlpPolicy` 即可。

---

## 四、完整训练流程

```python
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO

# 1. 创建环境
env = gym.make("MetaDrive-validation-v0", config={"num_scenarios": 1})

# 2. 创建模型
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

# 3. 训练
model.learn(total_timesteps=50000)

# 4. 保存
model.save("ppo_metadrive")

# 5. 测试
env = gym.make("MetaDrive-validation-v0", config={"use_render": True})
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

---

## 五、Monitor 和评估

```python
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# 包装环境（记录训练数据）
env = Monitor(env)

# 评估模型
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

---

## 六、超参数调优建议

| 参数 | 默认值 | 调整建议 |
|------|--------|----------|
| learning_rate | 3e-4 | 改小有助于收敛，如 1e-4 |
| n_steps | 2048 | 内存允许可增大到 4096 |
| gamma | 0.99 | 任务越长可以适当增大 |
| ent_coef | 0 | 设为 0.01 鼓励探索 |

---

## 七、常见问题

**Q: 训练时内存不足？**
A: 减小 `n_steps` 或 `batch_size`

**Q: 模型不收敛？**
A: 检查奖励函数是否合理，或尝试降低学习率

**Q: 如何使用自定义奖励函数？**
A: 继承 MetaDriveEnv，重写 `reward_function()` 方法