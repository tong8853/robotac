"""
自定义观测变换 - 将 MetaDrive 图像格式转换为 SB3 CnnPolicy 兼容格式

MetaDrive 图像格式: (H, W, C, history) 或 (1, H, W, C, history)
SB3 CnnPolicy 期望: (C, H, W) 或 (batch, C, H, W)

使用方法:
    from observation_wrapper import ImageObservationWrapper
    env = ImageObservationWrapper(meta_drive_env)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ImageObservationWrapper(gym.ObservationWrapper):
    """
    将 MetaDrive 多帧图像观测转换为 SB3 CnnPolicy 兼容格式

    输入 observation: dict with 'image' key
        image shape: (H, W, C, history) 或 (1, H, W, C, history)
        - H: 高度 (如 90)
        - W: 宽度 (如 160)
        - C: 通道数 (3 for RGB)
        - history: 历史帧数 (如 3)

    输出 observation: dict with 'image' and 'state'
        image shape: (C * history, H, W) = (9, 90, 160)
        - 将所有历史帧的通道拼接在一起
        - 转换为 (C, H, W) 格式以适配 CNN
    """

    def __init__(self, env):
        super().__init__(env)
        self._update_observation_space()

    def reset(self, **kwargs):
        """重写 reset 以处理观测转换"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def _update_observation_space(self):
        original_image_shape = self.env.observation_space['image'].shape
        # shape 可能是: (H, W, C, history) 或 (batch, H, W, C, history)
        # 例如: (90, 160, 3, 3) 或 (1, 90, 160, 3, 3)

        if len(original_image_shape) == 4:
            # 格式: (H, W, C, history)
            H, W, C, history = original_image_shape
        elif len(original_image_shape) == 5:
            # 格式: (batch, H, W, C, history)
            _, H, W, C, history = original_image_shape
        else:
            raise ValueError(f"Unexpected image shape: {original_image_shape}")

        # 新格式: (C * history, H, W)
        new_shape = spaces.Box(
            low=0,
            high=255,
            shape=(C * history, H, W),
            dtype=np.uint8
        )

        self.observation_space = spaces.Dict({
            'image': new_shape,
            'state': self.env.observation_space['state']
        })

    def observation(self, observation):
        image = observation['image']
        state = observation['state']

        # 处理不同格式
        if len(image.shape) == 4:
            # 格式: (H, W, C, history)
            H, W, C, history = image.shape
        elif len(image.shape) == 5:
            # 格式: (batch, H, W, C, history)
            image = image[0]
            H, W, C, history = image.shape
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # 对每一帧进行转置: (H, W, C) -> (C, H, W)
        frames = []
        for i in range(history):
            frame = image[:, :, :, i]  # (H, W, C)
            frame_transposed = np.transpose(frame, (2, 0, 1))  # (C, H, W)
            frames.append(frame_transposed)

        # 拼接所有帧: (C*history, H, W)
        new_image = np.concatenate(frames, axis=0)

        # 归一化到 [0, 1] 范围（SB3 CnnPolicy 期望）
        new_image = new_image.astype(np.float32) / 255.0

        return {
            'image': new_image,
            'state': state
        }


class SimplifiedImageWrapper(gym.ObservationWrapper):
    """
    简化版：只取最后一帧，转为 (C, H, W) 格式
    """

    def __init__(self, env):
        super().__init__(env)
        self._update_observation_space()

    def reset(self, **kwargs):
        """重写 reset 以处理观测转换"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def _update_observation_space(self):
        original_image_shape = self.env.observation_space['image'].shape

        if len(original_image_shape) == 4:
            H, W, C, history = original_image_shape
        elif len(original_image_shape) == 5:
            _, H, W, C, history = original_image_shape
        else:
            raise ValueError(f"Unexpected image shape: {original_image_shape}")

        new_shape = spaces.Box(
            low=0,
            high=255,
            shape=(C, H, W),
            dtype=np.uint8
        )

        self.observation_space = spaces.Dict({
            'image': new_shape,
            'state': self.env.observation_space['state']
        })

    def observation(self, observation):
        image = observation['image']
        state = observation['state']

        # 处理不同格式
        if len(image.shape) == 4:
            # (H, W, C, history) -> 取最后一帧
            image = image[:, :, :, -1]
        elif len(image.shape) == 5:
            # (batch, H, W, C, history) -> 取最后一帧
            image = image[0, :, :, :, -1]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # 转置: (H, W, C) -> (C, H, W)
        new_image = np.transpose(image, (2, 0, 1))

        # 归一化
        new_image = new_image.astype(np.float32) / 255.0

        return {
            'image': new_image,
            'state': state
        }


# 测试代码
if __name__ == "__main__":
    from metadrive.envs import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    print("测试 ImageObservationWrapper...")

    env = MetaDriveEnv(dict(
        use_render=False,
        image_observation=True,
        sensors=dict(rgb_camera=(RGBCamera, 160, 90)),
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
            image_source="rgb_camera",
        ),
        map_config=dict(type='block_num', config=3, lane_width=4, lane_num=1),
        norm_pixel=True,
    ))

    print(f"原始观测空间: {env.observation_space}")
    print(f"图像空间: {env.observation_space['image']}")

    wrapped_env = ImageObservationWrapper(env)
    print(f"包装后观测空间: {wrapped_env.observation_space}")

    obs, info = wrapped_env.reset()
    print(f"图像形状: {obs['image'].shape}")
    print(f"state形状: {obs['state'].shape}")

    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"Step {i+1}: image shape={obs['image'].shape}, reward={reward:.2f}")
        if terminated or truncated:
            break

    wrapped_env.close()
    print("测试完成!")
