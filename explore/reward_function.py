"""
奖励函数 - 符合ROBOTAC规则第7.5节违规判定标准

违规惩罚（规则7.5）:
- 轻微违规（压白实线）: 等效扣3分
- 严重违规（碰撞、越界、逆行）: 等效扣10分

Goal: 速胜条件 - 无违规 + 20%快于基准时间
"""

import numpy as np


def compute_reward(obs, reward, terminated, info, vehicle=None):
    """
    计算每一步的奖励

    符合规则:
    - 7.5 违规判定与处罚
    - 1.2 速胜条件（无违规 + 20%快于基准）

    参数:
        obs: 观察（包含image和state）
        reward: 环境原始奖励
        terminated: 是否 episode 结束
        info: 环境返回的额外信息
        vehicle: 可选，车辆对象用于直接检测违规

    返回:
        total_reward: 综合奖励
        done_info: 额外信息字典
    """
    total_reward = 0.0
    done_info = {}

    # ==================== 1. 原始环境奖励 ====================
    total_reward += reward

    # ==================== 2. 违规惩罚（规则7.5）====================
    # 通过 vehicle 对象直接检测（更准确）
    if vehicle is not None:
        # 轻微违规：压白实线（规则7.5 - 压白实线）
        if getattr(vehicle, 'on_white_continuous_line', False):
            total_reward -= 10.0
            done_info["violation"] = "on_continuous_line"
            done_info["severity"] = "minor"

        # 轻微违规：压黄实线
        if getattr(vehicle, 'on_yellow_continuous_line', False):
            total_reward -= 10.0
            done_info["violation"] = "on_yellow_line"
            done_info["severity"] = "minor"

        # 严重违规：越界/偏离路线（规则7.5 - 偏离任务路线）
        if getattr(vehicle, 'crash_sidewalk', False):
            total_reward -= 50.0
            done_info["violation"] = "crash_sidewalk"
            done_info["severity"] = "major"

    # 通过 info 检测（后备方案）
    # 严重违规：碰撞（规则7.5 - 碰撞）
    if info.get("crash_vehicle", False) or info.get("crash_building", False) or info.get("crash_object", False):
        total_reward -= 50.0
        done_info["violation"] = "crash"
        done_info["severity"] = "major"

    # 严重违规：越界（规则7.5 - 偏离任务路线）
    if info.get("out_of_road", False):
        total_reward -= 50.0
        done_info["violation"] = "out_of_road"
        done_info["severity"] = "major"

    # ==================== 3. 目标达成奖励 ====================
    if info.get("arrive_dest", False):
        total_reward += 200.0  # 大奖励到达终点
        done_info["goal"] = True

    # ==================== 4. 速度奖励（鼓励快速行驶）====================
    # 符合规则1.2：时间排名加分
    speed = 0.0
    if isinstance(obs, dict) and "state" in obs:
        # 从state中获取速度 [vx, vy, heading, ...]
        vx = obs["state"][0]
        vy = obs["state"][1]
        speed = np.sqrt(vx**2 + vy**2)
    elif isinstance(obs, dict) and "velocity" in obs:
        speed = np.linalg.norm(obs["velocity"])
    elif vehicle is not None:
        # 直接从车辆对象获取速度
        speed = getattr(vehicle, 'speed_m_s', 0.0) * 3.6  # m/s 转 km/h

    # 速度奖励：鼓励保持较高速度
    if speed > 5.0:
        total_reward += 0.05 * speed  # 速度越快，奖励越高

    # ==================== 5. 存活奖励（防止卡死）====================
    if not terminated and speed < 0.5:
        total_reward -= 0.1  # 停滞惩罚

    # ==================== 6. 距离奖励（鼓励前进）====================
    route_completion = info.get("route_completion", 0.0)
    distance_to_dest = info.get("distance_to_dest", None)

    # 路线完成度奖励
    if route_completion > 0:
        total_reward += 0.5 * route_completion  # 提高权重

    # 距离目的地奖励（如果可用）
    if distance_to_dest is not None and distance_to_dest > 0:
        # 距离越近奖励越高
        total_reward += 0.01 * (1.0 / (distance_to_dest + 1.0))

    # ==================== 7. 存活步数奖励（鼓励存活更久）====================
    # 每步存活给予少量正奖励，累积起来鼓励更长行驶
    if not terminated:
        total_reward += 0.01  # 存活奖励

    return total_reward, done_info


def check_violation(info, vehicle=None):
    """
    检查违规类型（供外部调用统计）

    返回:
        violation_type: str, 违规类型或None
        severity: str, "minor" 或 "major"
    """
    # 通过 vehicle 直接检测（更可靠）
    if vehicle is not None:
        # 轻微违规：压线
        if getattr(vehicle, 'on_white_continuous_line', False):
            return "on_white_continuous_line", "minor"
        if getattr(vehicle, 'on_yellow_continuous_line', False):
            return "on_yellow_continuous_line", "minor"
        # 严重违规：越界/偏离路线
        if getattr(vehicle, 'crash_sidewalk', False):
            return "crash_sidewalk", "major"
        if getattr(vehicle, 'out_of_route', False):
            return "out_of_route", "major"

    # 通过 info 检测（后备方案）
    # 严重违规：越界/偏离路线
    if info.get("out_of_road", False) or info.get("out_of_route", False):
        return "out_of_road", "major"
    # 严重违规：碰撞
    if info.get("crash_vehicle", False) or info.get("crash_building", False) or info.get("crash_object", False):
        return "crash", "major"
    # 轻微违规：压线
    if info.get("on_continuous_line", False):
        return "on_continuous_line", "minor"

    return None, None


def compute_violation_penalty(violation_type, severity):
    """
    根据违规类型和严重程度计算惩罚值

    符合规则7.5:
    - 轻微违规: 每次扣3分（等效惩罚值）
    - 严重违规: 每次扣10分（等效惩罚值）
    """
    penalties = {
        "minor": {
            "on_continuous_line": 10.0,  # 压白实线
            "on_white_continuous_line": 10.0,
            "on_yellow_continuous_line": 10.0,
        },
        "major": {
            "crash": 50.0,  # 碰撞
            "crash_vehicle": 50.0,
            "crash_building": 50.0,
            "crash_object": 50.0,
            "out_of_road": 50.0,  # 偏离路线
            "crash_sidewalk": 50.0,
        }
    }

    return penalties.get(severity, {}).get(violation_type, 0.0)
