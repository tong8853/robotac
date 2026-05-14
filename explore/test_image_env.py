"""快速测试图像观测环境"""
from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from observation_wrapper import ImageObservationWrapper

print('测试图像观测配置...')

env = MetaDriveEnv(dict(
    use_render=False,
    manual_control=False,
    traffic_density=0.0,
    num_scenarios=10000,
    random_agent_model=False,
    on_continuous_line_done=True,
    out_of_route_done=True,
    image_observation=True,
    sensors=dict(rgb_camera=(RGBCamera, 160, 90)),
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
        image_source='rgb_camera',
    ),
    map_config=dict(type='block_num', config=3, lane_width=4, lane_num=1),
    norm_pixel=True,
))

print(f'原始观测空间: {env.observation_space}')
wrapped = ImageObservationWrapper(env)
print(f'包装后观测空间: {wrapped.observation_space}')

obs, info = wrapped.reset()
print(f'图像形状: {obs["image"].shape}')
print(f'state形状: {obs["state"].shape}')

# 测试几步
for i in range(3):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped.step(action)
    print(f'Step {i+1}: image shape={obs["image"].shape}, reward={reward:.2f}')
    if terminated or truncated:
        break

wrapped.close()
print('图像观测测试通过!')
