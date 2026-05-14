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
    计算每一步的奖励 - 优化版

    核心策略：
    1. route_completion 是关键 - 持续给出进度奖励
    2. 违规惩罚确保安全行驶
    3. 到达终点给予大奖赏
    """
    total_reward = 0.0
    done_info = {}

    # ==================== 1. 原始环境奖励 ====================
    total_reward += reward

    # ==================== 2. 违规惩罚（规则7.5）====================
    if vehicle is not None:
        if getattr(vehicle, 'on_white_continuous_line', False):
            total_reward -= 10.0
            done_info["violation"] = "on_continuous_line"
            done_info["severity"] = "minor"

        if getattr(vehicle, 'on_yellow_continuous_line', False):
            total_reward -= 10.0
            done_info["violation"] = "on_yellow_line"
            done_info["severity"] = "minor"

        if getattr(vehicle, 'crash_sidewalk', False):
            total_reward -= 50.0
            done_info["violation"] = "crash_sidewalk"
            done_info["severity"] = "major"

    if info.get("crash_vehicle", False) or info.get("crash_building", False) or info.get("crash_object", False):
        total_reward -= 50.0
        done_info["violation"] = "crash"
        done_info["severity"] = "major"

    if info.get("out_of_road", False):
        total_reward -= 50.0
        done_info["violation"] = "out_of_road"
        done_info["severity"] = "major"

    # ==================== 3. 目标达成奖励（最重要） ====================
    if info.get("arrive_dest", False):
        total_reward += 500.0  # 大幅提高到达终点奖励
        done_info["goal"] = True
        return total_reward, done_info  # 提前返回，避免其他奖励干扰

    # ==================== 4. route_completion 奖励（核心导航奖励）====================
    route_completion = info.get("route_completion", 0.0)
    # route_completion 从 0 到 1，表示完成比例
    # 每增加 0.01 奖励 1.0，这样完成 100% 路程能获得约 100 奖励
    total_reward += 1.0 * route_completion

    # ==================== 5. 速度奖励（鼓励保持行驶）====================
    speed = 0.0
    if vehicle is not None:
        speed = getattr(vehicle, 'speed_m_s', 0.0) * 3.6  # km/h

    # 速度奖励：鼓励保持 10-60 km/h 的舒适速度
    if 10.0 < speed < 60.0:
        total_reward += 0.1
    elif speed >= 60.0:
        total_reward += 0.05  # 速度过快略微惩罚

    # 停滞惩罚
    if not terminated and speed < 0.5:
        total_reward -= 1.0  # 停滞严重惩罚

    # ==================== 6. 存活奖励 ====================
    if not terminated:
        total_reward += 0.1  # 每步存活奖励

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
