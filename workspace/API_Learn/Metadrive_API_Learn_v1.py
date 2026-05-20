# pynput: 键盘监听库 - 用于捕获用户按键事件（手动控制时需要）
from pynput import keyboard
# time: 时间库 - 用于记录程序运行时间、计算耗时等
import time
# metadrive.component.sensors.rgb_camera: RGB摄像头传感器 - 提供车辆前方摄像头图像
from metadrive.component.sensors.rgb_camera import RGBCamera
# metadrive.utils.doc_utils.generate_gif: GIF生成工具 - 将帧序列合成动画
from metadrive.utils.doc_utils import generate_gif
# metadrive: 核心仿真环境 - 提供MetaDriveEnv等环境类
from metadrive import MetaDriveEnv
# metadrive.component.map.base_map: 地图基类 - 定义地图相关配置
from metadrive.component.map.base_map import BaseMap
# metadrive.component.map.pg_map: 程序生成地图 - 地图生成方法配置
from metadrive.component.map.pg_map import MapGenerateMethod

# 存当前按下的所有键
cur_keys = set()

# 当按下按键
def on_press(key):
    try:
        if hasattr(key, 'char'):  # has attribute
            cur_keys.add(key.char.lower())
    except:
        pass

# 当松开按键
def on_release(key):
    try:
        if hasattr(key, 'char') and key.char.lower() in cur_keys:
            cur_keys.remove(key.char.lower())
    except:
        pass

# 启动监听器
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# 地图配置 - 随机模式
map_config = {
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,  # 随机地图生成方法
    BaseMap.GENERATE_CONFIG: 5,  # 地图包含 5 个随机生成的路段
    BaseMap.LANE_WIDTH: 4,  # 车道宽度：米
    BaseMap.LANE_NUM: 1,  # 车道数量
}

# 只有直接运行此脚本时才执行（不是被导入时）
if __name__ == "__main__":
    # 环境配置字典
    config = dict(
        use_render=True,  # 显示渲染画面
        manual_control=False,  # 关闭内置手动控制（用键盘）
        traffic_density=0.0,  # 无其他车辆
        num_scenarios=10000,  # 最大场景数
        random_agent_model=False,  # 不随机车辆模型
        on_continuous_line_done=True,  # 压线停止后结束
        out_of_route_done=True,  # 出界后结束
        image_observation=True, # 返回图像数据 
        sensors=dict(rgb_camera=(RGBCamera, 320, 180)),  # RGB摄像头，分辨率320x180
        norm_pixel=False,  # 像素不归一化
        vehicle_config=dict(
            show_lidar=True,  # 显示雷达
            show_navi_mark=True,  # 显示导航标记
            show_line_to_navi_mark=True,  # 不显示导航连线
        ),
        map_config=map_config,  # 地图配置
    )
    
    env = MetaDriveEnv(config)  # 创建环境实例
    frames = []  # 用于存储GIF帧
    try:
        # env.reset() 返回 (observation, info) 两个值
        obs, _ = env.reset(seed=21)  # 重置环境，返回初始观察（seed=21固定地图）
        start = time.time()  # 记录开始时间

        for i in range(1, 1000000000):  # 主循环，最多10亿步
            # --- 决策逻辑开始 ---  # 每次循环初始化为0
            throttle = 0
            steering = 0

            if 'i' in cur_keys: throttle += 1.0  # I键按住→加速
            if 'k' in cur_keys: throttle -= 1.0  # K键按住→刹车
            if 'j' in cur_keys: steering += 0.6  # J键按住→左转
            if 'l' in cur_keys: steering -= 0.6  # L键按住→右转
            # --- 决策逻辑结束 ---

            # 执行动作，返回5个值
            obs, reward, terminated, truncated, diagnostics = env.step([steering, throttle])

            ret = obs["image"][..., -1]  # 提取最后一帧图像
            frames.append(ret[..., ::-1])  # 存入GIF帧（翻转色彩）

            text = {"Keyboard Control": "I (Go), K (Brake), J (Left), L (Right)"}
            env.render(text=text)  # 显示控制说明

            if diagnostics["arrive_dest"]:  # 如果到达终点
                print('到达终点', diagnostics["episode_length"], '得分', diagnostics["episode_reward"])
                print(f'花费时间 {time.time()-start:.2f} 秒')
                generate_gif(frames, gif_name="mannual-control.gif")
                break  # 退出循环

            if terminated:  # 如果本局失败（撞车/超时等）
                print('失败，重新开始')
                obs, _ = env.reset(seed=21)  # 重置环境
                frames = []  # 清空帧缓存

    finally:
        env.close()  # 关闭环境
        listener.stop()  # 停止键盘监听