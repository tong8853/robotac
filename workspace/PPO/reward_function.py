#奖励函数
def compute_reward(obs, reward, terminated, diagnostics):
    """
    计算每一步的奖励。

    参数：
        obs：当前观察
        reward：环境原始奖励
        terminated：是否结束
        diagnostics：诊断信息

    返回：
        total_reward：综合奖励
        done_info：额外信息字典
    """
    total_reward = 0.0
    done_info = {}

    # 1. 原始奖励（环境自带）
    total_reward += reward

    # 2. 到达终点（大奖励）
    if diagnostics.get("arrive_dest", False):
        total_reward += 100.0
        done_info["goal"] = True

    # 3. 撞车（惩罚）
    if terminated:
        total_reward -= 10.0
        done_info["crash"] = True

    # 4. 保持运动（防止不动拿分）
    # 如果车辆速度>0，给小奖励
    speed = obs.get("velocity", [0])[0] if isinstance(obs, dict) else 0
    if speed > 0.5:
        total_reward += 0.01

    return total_reward, done_info