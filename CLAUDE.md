# 项目状态

## 当前进度
- **初赛**：已完成，等待审核
- **复赛**：第1阶段（环境搭建 + 数据采集）

## 已完成任务
- [x] 跑通 1-仿真环境测试.ipynb（到达终点 289 步）
- [x] 本地 conda 环境配置完成（robotac）
- [x] 安装 PyTorch GPU 版（CUDA 11.7）

## 复赛阶段
- 阶段一：环境搭建（进行中）
- 阶段二：数据采集（待开始）
- 阶段三：模型训练（待开始）

## 当前任务
1. 安装 stable-baselines3
2. 确认 Kaggle/Colab GPU 环境可用
3. 编写环境测试脚本
4. 数据采集脚本开发

## 技术环境
- **本地编译器**：robotac conda 环境
- **提交编译器**：./python-3.7.0-embed-amd64/python.exe
- **已安装库**：metadrive, gymnasium, numpy, opencv-python, matplotlib, torch 1.13.1+cu117
- **待安装库**：stable-baselines3（已安装）

## 学习计划
- [ ] 学会使用 stable-baselines3（PPO 算法）
- [ ] 学会数据采集与保存
- [ ] 学会模型训练与保存
- [ ] 学会嵌入式环境适配
- [ ] 学习配置驱动（YAML 管理超参数）

## 文件
- 计划文档：复赛准备计划.md
- 规则文档：机器人大赛仿真规则-V1.0.md