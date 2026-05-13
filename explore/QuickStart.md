# ROBOTAC 快速启动指南

## 环境准备

### 1. Conda 环境
```bash
conda activate robotac
```

### 2. 验证环境
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import metadrive; print(f'MetaDrive: {metadrive.__version__}')"
python -c "from stable_baselines3 import PPO; print('SB3 OK')"
```

## 训练模型

```bash
cd d:/robotac/explore
conda activate robotac
python train_ppo.py
```

训练会自动:
- 使用 CPU 或 GPU（自动检测）
- 保存模型到 `models/ppo_metadrive.zip`
- 记录 TensorBoard 日志到 `logs/`

### 查看训练日志
```bash
tensorboard --logdir=./logs
```

## 测试模型

```bash
python test_ppo.py
```

## 可视化验证

```bash
python visualize_ppo.py
```

这将生成 `performance_report.md` 性能报告。

## 目录结构

```
explore/
├── train_ppo.py        # 主训练脚本
├── test_ppo.py         # 测试脚本
├── visualize_ppo.py    # 可视化验证
├── reward_function.py  # 奖励函数（合规）
├── config.yaml         # 配置文件
├── QuickStart.md       # 本文件
├── models/             # 保存的模型
│   └── ppo_metadrive.zip
└── logs/               # 训练日志
    └── ppo_metadrive/
```

## 规则合规说明

本实现符合 ROBOTAC 规则:

- ✅ 仅使用单一 RGB 摄像头（无 Lidar）
- ✅ 100ms 延迟检查
- ✅ 违规惩罚（压线/碰撞/越界/逆行）
- ✅ 无硬编码路径点
- ✅ 设备无关代码（CPU/GPU 自动检测）

## 参考资料

- MetaDrive-Tutorials: `explore/MetaDrive-Tutorials/`
- 官方示例: `workspace/赛方样例/`
- API 文档: `notes/API/`

## 常见问题

### Q: 训练很慢怎么办？
A: 确保安装了 CUDA 版本的 PyTorch:
```bash
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Q: 如何恢复中断的训练？
A: 模型会自动保存和加载，只需重新运行 `python train_ppo.py`

### Q: 验证时环境无法初始化？
A: 某些环境下需要设置 `USE_GIL=0`，详细参考 `notes/规则与指南/复赛准备计划.md`
