# ROBOTAC 复赛备赛指南

**面向对象**：已有YOLO视觉背景的人工智能专业研究生  
**比赛时间**：2026年6月（复赛）  
**赛题**：基于视觉的强化学习自动驾驶

---

## 一、比赛规则核心要点

### 1.1 输入输出
| 项目 | 要求 |
|------|------|
| **输入** | 前置摄像头图像（1080P/60fps/65°视场角） |
| **输出** | 方向盘转角 + 加速/制动踏板行程 |
| **延迟** | 单次响应 ≤100ms |

### 1.2 禁止事项
- ❌ 预设仿真环境固定坐标或任务节点
- ❌ 用规则硬编码替代AI决策
- ❌ 修改平台接口参数或篡改数据

### 1.3 评分标准
- **合规分（60分）**：轻微违规扣3分，严重违规扣10分，3次以上取消资格
- **时间分（40分）**：以中位数为基准，每快10%加4分

---

## 二、备赛任务清单（4周）

### 第1周：环境熟悉 + 数据采集

| 天数 | 任务 | 详细步骤 |
|------|------|----------|
| Day 1-2 | 配置开发环境 | 1. 激活robotac环境<br>2. 安装metadrive及依赖<br>3. 验证环境可运行 |
| Day 3-4 | 获取图像数据 | 1. 学习MetaDrive API<br>2. 编写脚本获取摄像头图像<br>3. 保存10张测试图片 |
| Day 5-7 | 理解环境结构 | 1. 打印obs/reward/done/info结构<br>2. 理解action_space和observation_space<br>3. 了解车辆状态变量 |

**里程碑**：能从仿真器保存图片

### 第2周：强化学习基础 + SB3

| 天数 | 任务 | 详细步骤 |
|------|------|----------|
| Day 8-10 | 安装SB3 | 1. pip install stable-baselines3<br>2. 阅读SB3官方文档<br>3. 理解PPO算法原理 |
| Day 11-12 | 跑通Demo | 1. 运行SB3的CartPole例子<br>2. 理解train和evaluate流程 |
| Day 13-14 | 集成测试 | 1. 将MetaDrive封装为Gym环境<br>2. 运行官方案例drive_in_single_agent_env.py |

**里程碑**：能用SB3跑通一个简单环境

### 第3周：搭建第一个RL模型

| 天数 | 任务 | 详细步骤 |
|------|------|----------|
| Day 15-17 | 设计Reward函数 | 1. 速度奖励：speed_reward = speed * 0.3<br>2. 中心线奖励：lateral_reward = 1 - abs(lateral_distance)<br>3. 碰撞惩罚：crash_penalty = -100<br>4. 越界惩罚：off_road_penalty = -10 |
| Day 18-20 | CNN特征提取 | 1. 使用ResNet18提取图像特征<br>2. 或使用简单的CNN架构 |
| Day 21 | 训练测试 | 1. 运行PPO训练1000步<br>2. 观察小车行为<br>3. 调整reward权重 |

**里程碑**：小车能自己向前开（有奖励反馈）

### 第4周：调参与测试

| 天数 | 任务 | 详细步骤 |
|------|------|----------|
| Day 22-24 | 调参 | 1. 调整reward权重<br>2. 调整学习率<br>3. 调整网络结构 |
| Day 25-26 | 延迟优化 | 1. 确保推理时间≤100ms<br>2. 优化模型推理速度 |
| Day 27 | 泛化测试 | 1. 用5个不同随机seed测试<br>2. 记录各seed得分 |
| Day 28 | 提交 | 1. 整理代码<br>2. 打包可运行脚本<br>3. 生成GIF演示 |

**里程碑**：能提交一个可运行的RL版本

---

## 三、技术栈

### 3.1 必须掌握
| 技能 | 用途 |
|------|------|
| Python基础 | 编程 |
| Gymnasium/OpenAI Gym | 强化学习环境接口 |
| Stable-Baselines3 | PPO等强化学习算法 |
| NumPy/OpenCV | 数据处理 |
| Matplotlib | 可视化 |

### 3.2 推荐掌握
| 技能 | 用途 |
|------|------|
| PyTorch | 深度学习框架 |
| ResNet/CNN | 图像特征提取 |
| Weights & Biases | 实验记录 |

---

## 四、代码模板

### 4.1 创建环境
```python
from metadrive import MetaDriveEnv

env = MetaDriveEnv(dict(
    image_observation=True,  # 启用摄像头
    sensors={"rgb_camera": (RGBCamera, 400, 300)},
    num_scenarios=1000,
    start_seed=0,
    traffic_density=0.1,
))
```

### 4.2 环境交互
```python
obs, _ = env.reset(seed=1)
# action = [steering, acceleration]
obs, reward, done, info = env.step([0.0, 0.1])
env.close()
```

### 4.3 Reward函数示例
```python
def compute_reward(vehicle):
    speed_reward = vehicle.speed * 0.3
    lateral_reward = 1.0 - abs(vehicle.lateral_distance)
    crash_penalty = -100 if vehicle.crashed else 0
    off_road_penalty = -10 if vehicle.out_of_road else 0
    return speed_reward + lateral_reward + crash_penalty + off_road_penalty
```

---

## 五、常见问题

### Q1: 需要自己构建环境吗？
**不需要**。MetaDrive本身就是完整环境，直接实例化使用即可。

### Q2: 嵌入式Python需要装PyTorch吗？
**不需要**。嵌入式环境只负责运行推理，训练在本地conda环境进行。

### Q3: 图像数据需要自己采集吗？
**是的**。需要编写脚本采集摄像头图像数据用于训练。

### Q4: 延迟要求如何满足？
- 模型推理使用ONNX或轻量模型
- 嵌入式环境运行推理，本地训练

---

## 六、检查清单

- [ ] Week 1：能从仿真器保存图片
- [ ] Week 2：能用SB3跑通一个简单环境
- [ ] Week 3：小车能自己向前开（有奖励反馈）
- [ ] Week 4：能提交一个可运行的RL版本

---

## 七、参考资料

1. MetaDrive官方文档
2. Stable-Baselines3文档：https://stable-baselines3.readthedocs.io/
3. Gymnasium文档：https://gymnasium.farama.org/
4. PPO论文：Proximal Policy Optimization Algorithms