from pynput import keyboard
import time
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.utils.doc_utils import generate_gif
from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

# 存储当前按下的所有键
current_keys = set()

def on_press(key):
    try:
        if hasattr(key, 'char'):
            current_keys.add(key.char.lower()) # 统一转小写，防止大小写干扰
    except AttributeError:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char.lower() in current_keys:
            current_keys.remove(key.char.lower())
    except AttributeError:
        pass

# 启动监听器
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE, 
            BaseMap.GENERATE_CONFIG: "SCS",  # 直路-弯道-直路
            BaseMap.LANE_WIDTH: 4,
            BaseMap.LANE_NUM: 1}

if __name__ == "__main__":
    config = dict(
        use_render=True,
        manual_control=False,
        traffic_density=0.0,
        num_scenarios=10000,
        random_agent_model=False,
        on_continuous_line_done=True,
        out_of_route_done=True,
        image_observation=True,
        sensors=dict(rgb_camera=(RGBCamera, 320, 180)),
        norm_pixel=False,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=False,
            show_line_to_navi_mark=False,
        ),
        map_config=map_config,
    )

    env = MetaDriveEnv(config)
    frames = []
    try:
        o, _ = env.reset(seed=21)
        start = time.time()
        for i in range(1, 1000000000):
            # --- 决策逻辑开始 ---
            throttle_brake = 0
            steer = 0
            
            if 'i' in current_keys: throttle_brake += 1.0 # 前进
            if 'k' in current_keys: throttle_brake -= 1.0 # 刹车
            if 'j' in current_keys: steer += 0.6          # 左转
            if 'l' in current_keys: steer -= 0.6          # 右转
            # --- 决策逻辑结束 ---

            o, r, tm, tc, info = env.step([steer, throttle_brake])
            
            # 获取摄像头画面用于生成 GIF
            ret = o["image"][..., -1]
            frames.append(ret[..., ::-1])
            
            text = {"Keyboard Control": "I (Go), K (Brake), J (Left), L (Right)"} 
            env.render(text=text)

            if info["arrive_dest"]:
                print('到达终点', info["episode_length"], '得分', info["episode_reward"])
                print(f'花费时间 {time.time()-start:.2f} 秒')
                generate_gif(frames, gif_name="mannual-control.gif")
                break
                
            if tm:
                print('失败，重新开始')
                o, _ = env.reset(seed=21) # 失败后重置环境
                frames = [] # 清空帧缓存

    finally:
        env.close()
        listener.stop() # 记得关闭监听器