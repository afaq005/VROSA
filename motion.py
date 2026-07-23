"""
motion.py — Low-level robot movement helpers.

Uses cmd_pub from ros_node (imported lazily to avoid any init-order issues)
and reads/writes the shared state dict for position + interrupt handling.
"""

import math
import time

import numpy as np
from geometry_msgs.msg import Twist

from state import state, interrupt_flag
from logger import log

SAFE_DIST = 0.75  # metres — stop if obstacle closer than this during movement


# ── Internal: get publisher lazily ───────────────────────────────
def _cmd_pub():
    """Lazily fetch the ROS publisher so import order doesn't matter."""
    import ros_node
    return ros_node.cmd_pub


# ════════════════════════════════════════════════════════════
# PRIMITIVES
# ════════════════════════════════════════════════════════════
def _stop() -> None:
    """Publish zero-velocity Twist to halt the robot."""
    _cmd_pub().publish(Twist())
    time.sleep(0.2)


def _drive(linear: float = 0.0, angular: float = 0.0,
           duration: float | None = None) -> None:
    """Publish a velocity command. If duration is given, stop afterward."""
    t = Twist()
    t.linear.x  = float(linear)
    t.angular.z = float(angular)
    _cmd_pub().publish(t)
    if duration:
        time.sleep(duration)
        _stop()


# ════════════════════════════════════════════════════════════
# OBSTACLE CHECK
# ════════════════════════════════════════════════════════════
def _check_obstacle(direction: int) -> float:
    """Closest obstacle in movement direction.
    Works in camera-only mode (use_lidar=False) using depth map.
    LiDAR z filter: z > -0.10 excludes Nova Carter floor returns.
    """
    # ── LiDAR (if enabled) ───────────────────────────────────
    if state.get("use_lidar", True) and direction == 1:
        lidar = state["lidar"]
        if lidar:
            pts = [p["dist"] for p in lidar
                   if p["x"] > 0.05 and abs(p["y"]) < 1.0
                   and p["z"] > -0.10 and p["z"] < 2.0]
            if pts:
                return min(pts)

    # ── Depth map (camera-only fallback, always works) ────────
    if state.get("use_depth", True) and direction == 1:
        dm = state.get("depth_map")
        if dm is not None:
            H, W = dm.shape
            # Narrow centre strip (30% width), lower 2/3 height — avoids sky/ceiling
            # strip = dm[H // 3:, W * 35 // 100: W * 65 // 100]
            strip = dm[H // 3: H * 55 // 100, W * 35 // 100: W * 65 // 100]

            valid = strip[(strip > 0.15) & (strip < 10.0)]
            if len(valid) > 0:
                return float(np.percentile(valid, 8))

    return 99.0


# ════════════════════════════════════════════════════════════
# HIGH-LEVEL MOVES
# ════════════════════════════════════════════════════════════
# def _move_dist(meters: float, speed: float = 0.4) -> float:
def _move_dist(meters: float, speed: float = 0.4, safe_dist: float = None) -> float:
    """Move by a signed number of metres with live obstacle avoidance.

    Returns the actual distance travelled (may be less if blocked).
    """
    interrupt_flag.clear()
    sx, sy        = state["x"], state["y"]
    target        = abs(meters)
    sign          = 1 if meters > 0 else -1
    timeout       = (target / speed) * 2.5
    t0            = time.time()
    stopped_early = False

    while True:
        if interrupt_flag.is_set():
            log("Movement interrupted!", "WARN")
            break

        # obs = _check_obstacle(sign)
        _sd = safe_dist if safe_dist is not None else SAFE_DIST
        obs = _check_obstacle(sign)
        # if obs < SAFE_DIST:
        if obs < _sd:
            _stop()
            direction_word    = "front" if sign > 0 else "rear"
            log(f"⚠ Obstacle {obs:.2f}m — stopped ({direction_word})", "WARN")
            state["last_vlm"] = f"BLOCKED: obstacle {obs:.2f}m {direction_word}"
            stopped_early     = True
            break

        # Slow-down zone: crawl at 40% speed between SAFE_DIST and 1.5 m
        # drive_speed = speed * 0.4 if obs < 1.5 else speed
        drive_speed = speed * 0.4 if obs < (_sd + 0.5) else speed

        _drive(linear=sign * drive_speed)
        time.sleep(0.05)

        d = math.sqrt((state["x"] - sx) ** 2 + (state["y"] - sy) ** 2)
        if d >= target or time.time() - t0 > timeout:
            break

    _stop()
    actual = math.sqrt((state["x"] - sx) ** 2 + (state["y"] - sy) ** 2)
    if stopped_early:
        log(f"Moved {actual:.2f}m before obstacle stop", "ROBOT")
    return actual


# def _rotate(deg: float, speed: float = 0.4) -> None:
#     """Rotate by a signed number of degrees (+ = left, - = right)."""
#     interrupt_flag.clear()
#     dur  = abs(deg) / math.degrees(speed)
#     sign = 1 if deg > 0 else -1
#     t0   = time.time()
#     while time.time() - t0 < dur:
#         if interrupt_flag.is_set():
#             log("Rotation interrupted!", "WARN")
#             break
#         _drive(angular=sign * speed)
#         time.sleep(0.05)
#     _stop()

def _rotate(deg: float, speed: float = 0.4) -> None:
    """Rotate by a signed number of degrees using odometry feedback.
    Tracks actual yaw change from odometry — not time-based.
    """
    interrupt_flag.clear()
    if abs(deg) < 1.0:
        return

    start_yaw = state["yaw"]
    target    = abs(deg)
    sign      = 1 if deg > 0 else -1
    timeout   = abs(deg) / 10.0 + 5.0   # generous: min 5s, +1s per 10°
    t0        = time.time()

    while time.time() - t0 < timeout:
        if interrupt_flag.is_set():
            log("Rotation interrupted!", "WARN")
            break

        current = state["yaw"]
        diff    = current - start_yaw
        # Unwrap to handle ±180° crossing
        while diff >  180: diff -= 360
        while diff < -180: diff += 360

        travelled = abs(diff)
        if travelled >= target - 2.0:   # 2° tolerance
            break

        # Slow down in last 20°
        rate = speed * 0.4 if (target - travelled) < 20 else speed
        _drive(angular=sign * rate)
        time.sleep(0.05)

    _stop()
    time.sleep(0.1)    