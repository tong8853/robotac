#!/usr/bin/env python
"""
Day 3-4 任务：获取并保存摄像头图像数据
运行后会保存 10 张图片到 workspace/captures 文件夹
"""
import os
import cv2
import numpy as np
from metadrive import MetaDriveEnv

# 创建保存目录
save_dir = "workspace/captures"
os.makedirs(save_dir, exist_ok=True)

# 配置环境
config = dict(
    use_render=False,  # 关闭渲染以提高速度
    manual_control=False,
    traffic_density=0.0,
    num_scenarios=1,
    start_seed=1010,
    image_observation=True,  # 关键：开启图像观测
    vehicle_config=dict(
        image_source="main_camera",  # 使用主摄像头
        show_lidar=False,
        show_navi_mark=False,
    ),
)

env = MetaDriveEnv(config)

try:
    # 重置环境
    o, _ = env.reset(seed=21)
    print("环境已初始化")
    print(f"Observation 包含: {o.keys()}")
    print(f"图像形状: {o['image'].shape}")

    # 保存 10 张图片
    for i in range(10):
        # 获取图像数据
        # image 已归一化为 [0, 1]，需要乘以 255
        img = o["image"].get() if hasattr(o["image"], "get") else o["image"]

        # 取最后一帧（RGB通道）
        img_rgb = img[..., -1]  # shape: (H, W, C)

        # 转换为 uint8
        img_uint8 = (img_rgb * 255).astype(np.uint8)

        # 保存
        filepath = os.path.join(save_dir, f"frame_{i:03d}.png")
        cv2.imwrite(filepath, img_uint8)
        print(f"已保存: {filepath}")

        # 执行一步（不控制让它自己开）
        o, r, d, _, info = env.step([0, 0])

        if d:
            print(" episode done")
            break

    print(f"\n完成！共保存 {i+1} 张图片到 {save_dir}")

finally:
    env.close()