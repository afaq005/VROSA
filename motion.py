"""
motion.py — Low-level translational/rotational motion primitives.

All translational commands pass through safety.check_obstacle().  Navigation can
request a larger clearance but can never reduce the configured hard-stop
threshold.  Missing/stale safety sensing fails closed.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from geometry_msgs.msg import Twist

from config import (
    CONTROL_PERIOD_S,
    ROBOT,
    STOP_ANGULAR_EPS_RPS,
    STOP_STABLE_TIME_S,
    STOP_VELOCITY_EPS_MPS,
    STOP_WAIT_TIMEOUT_S,
)
from experiment_logger import record_event, timed
from logger import log
from safety import check_obstacle, log_stop_measurement, record_safety_unknown
from state import interrupt_flag, state

HARD_STOP_DIST = ROBOT["hard_stop_distance"]
STOP_TRIGGER_DIST = ROBOT["stop_trigger_distance"]
SAFE_DIST = STOP_TRIGGER_DIST  # compatibility for existing imports
DEFAULT_SPEED = ROBOT["max_linear_speed"]
DEFAULT_ROTATE_SPEED = ROBOT["max_angular_speed"]
SLOWDOWN_MARGIN = ROBOT["slowdown_margin"]
ODOM_MAX_AGE_S = float(ROBOT.get("odom_max_age_s", 0.30))
MOVE_TOLERANCE_M = float(ROBOT.get("move_position_tolerance_m", 0.03))
MOVE_SLOWDOWN_M = float(ROBOT.get("move_slowdown_distance_m", 0.40))
MOVE_MIN_SPEED = float(ROBOT.get("move_min_speed_mps", 0.06))
MOVE_KP = float(ROBOT.get("move_kp", 1.20))


def _hardware_motion_check(direction: int = 1) -> tuple[bool, str]:
    """Physical-robot motion authorization: never overridden by VLM or toggles."""
    if ROBOT.get("platform_label") != "Hardware":
        return True, ""
    if ROBOT.get("hardware_mode") != "guarded":
        return False, "hardware_diagnostic_mode_no_motion"
    if interrupt_flag.is_set():
        return False, "emergency_stop_latched"
    now = time.monotonic()
    odom_ts = state.get("odom_ts")
    if odom_ts is None or now - float(odom_ts) > ODOM_MAX_AGE_S:
        return False, "hardware_odometry_missing_or_stale"
    img_ts = state.get("image_ts")
    if img_ts is None or now - float(img_ts) > 1.0:
        return False, "hardware_rgb_missing_or_stale"
    r = check_obstacle(1 if direction >= 0 else -1)
    if not r.valid:
        return False, f"hardware_physical_safety_missing:{r.reason}"
    if r.distance is None or not math.isfinite(float(r.distance)) or r.distance < STOP_TRIGGER_DIST:
        return False, "hardware_obstacle_within_stop_trigger"
    return True, ""


def _cmd_pub():
    import ros_node
    return ros_node.cmd_pub


def _publish_twist(linear: float = 0.0, angular: float = 0.0) -> None:
    t = Twist()
    lin, ang = float(linear), float(angular)
    if ROBOT.get("platform_label") == "Hardware" and (abs(lin) > 1e-4 or abs(ang) > 1e-4):
        ok, reason = _hardware_motion_check(1 if lin >= 0 else -1)
        if not ok:
            lin = ang = 0.0
            state["hardware_motion_denial"] = reason
            record_event("HARDWARE_MOTION_DENIED", reason=reason)
        else:
            state["hardware_motion_denial"] = ""
            lin = max(-ROBOT["max_linear_speed"], min(ROBOT["max_linear_speed"], lin))
            ang = max(-ROBOT["max_angular_speed"], min(ROBOT["max_angular_speed"], ang))
    t.linear.x = lin
    t.angular.z = ang
    _cmd_pub().publish(t)


def _drive(linear: float = 0.0, angular: float = 0.0,
           duration: Optional[float] = None) -> None:
    
    linear_cmd = float(linear)
    angular_cmd = float(angular)

    state["motion_active"] = bool(
        abs(linear_cmd) > 1e-4
        or abs(angular_cmd) > 1e-4
    )

    if state["motion_active"]:

        if abs(linear_cmd) > 1e-4:
            state["motion_kind"] = "translation"
        else:
            state["motion_kind"] = "rotation"

    else:
        state["motion_kind"] = ""
    _publish_twist(linear, angular)
    if ROBOT.get("platform_label") == "Hardware" and state.get("hardware_motion_denial"):
        # Publishing a zero command on hardware must not be shown as moving.
        state["motion_active"] = False
        state["motion_kind"] = ""
    if duration is not None:
        time.sleep(max(0.0, float(duration)))
        _stop()


def _wait_until_stopped(direction: int = 1, timeout: float = STOP_WAIT_TIMEOUT_S) -> dict:
    """Wait until odometry reports near-zero velocity for a stable interval.

    Returns zero-velocity time/pose plus the minimum clearance observed while
    braking. If odometry velocity is unavailable, waits only up to the timeout
    and reports confirmed=False rather than inventing a physical stop time.
    """
    t0 = time.monotonic()
    stable_since = None
    min_clearance = None
    have_odom = state.get("odom_ts") is not None

    while time.monotonic() - t0 < timeout:
        # Track minimum clearance during braking for reviewer-facing safety data.
        try:
            r = check_obstacle(direction)
            if r.valid and r.distance is not None and r.distance < 90:
                min_clearance = r.distance if min_clearance is None else min(min_clearance, r.distance)
        except Exception:
            pass

        # A cached zero velocity is not confirmation unless the odometry
        # sample itself is still fresh throughout the stabilization interval.
        ts = state.get("odom_ts")
        odom_fresh = (have_odom and ts is not None
                      and time.monotonic() - float(ts) <= ODOM_MAX_AGE_S)
        if odom_fresh:
            lv = abs(float(state.get("linear_velocity", 0.0)))
            av = abs(float(state.get("angular_velocity", 0.0)))
            if lv <= STOP_VELOCITY_EPS_MPS and av <= STOP_ANGULAR_EPS_RPS:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= STOP_STABLE_TIME_S:
                    return {
                        "confirmed": True,
                        "zero_time": time.monotonic(),
                        "x": float(state["x"]),
                        "y": float(state["y"]),
                        "min_clearance": min_clearance,
                    }
            else:
                stable_since = None
        else:
            stable_since = None
        time.sleep(0.02)

    return {
        "confirmed": False,
        "zero_time": time.monotonic(),
        "x": float(state["x"]),
        "y": float(state["y"]),
        "min_clearance": min_clearance,
    }


def _stop(wait: bool = True, direction: int = 1) -> dict:
    """Publish zero velocity and optionally wait for odometry-confirmed stop."""
    cmd_time = time.monotonic()

    state["motion_active"] = False
    state["motion_kind"] = ""

    _publish_twist(0.0, 0.0)
    if not wait:
        return {
            "stop_command_time": cmd_time,
            "confirmed": False,
            "zero_time": cmd_time,
            "x": float(state["x"]),
            "y": float(state["y"]),
            "min_clearance": None,
        }
    info = _wait_until_stopped(direction=direction)
    info["stop_command_time"] = cmd_time
    return info


def _record_stop_event(*, event_type: str, source: str, direction: int,
                       commanded_speed: float, trigger_distance: Optional[float],
                       reading_age_s: Optional[float], recognition_time: float,
                       x0: float, y0: float, velocity_at_trigger: float,
                       sensor_source: Optional[str] = None) -> dict:
    sensor_time = recognition_time - reading_age_s if reading_age_s is not None else recognition_time
    stop_info = _stop(wait=True, direction=direction)
    stop_confirmed = bool(stop_info.get("confirmed"))
    zero_time = stop_info["zero_time"] if stop_confirmed else None
    stop_distance = math.hypot(stop_info["x"] - x0, stop_info["y"] - y0)
    trigger_d = None if trigger_distance is None else float(trigger_distance)
    min_clearance = stop_info.get("min_clearance")
    threshold_violation = (float(min_clearance) < float(ROBOT["hard_stop_distance"])
                           if min_clearance is not None else None)

    log_stop_measurement(
        event_type=event_type,
        source=source,
        direction=direction,
        commanded_speed_mps=round(commanded_speed, 4),
        trigger_distance_m=trigger_d,
        sensor_source=sensor_source or (source if event_type != "estop" else "none"),
        sensor_age_ms=(reading_age_s * 1000.0 if reading_age_s is not None else None),
        sensor_time_mono=sensor_time,
        recognition_time_mono=recognition_time,
        stop_command_time_mono=stop_info["stop_command_time"],
        zero_velocity_time_mono=zero_time,
        zero_velocity_confirmed=stop_confirmed,
        perception_age_ms=(reading_age_s * 1000.0 if reading_age_s is not None else None),
        decision_to_command_ms=(stop_info["stop_command_time"] - recognition_time) * 1000.0,
        braking_latency_ms=((zero_time - stop_info["stop_command_time"]) * 1000.0
                            if stop_confirmed else None),
        total_stop_latency_ms=((zero_time - sensor_time) * 1000.0
                               if stop_confirmed else None),
        velocity_at_trigger_mps=velocity_at_trigger,
        x_at_trigger=x0,
        y_at_trigger=y0,
        x_at_stop=stop_info["x"],
        y_at_stop=stop_info["y"],
        stop_distance_m=stop_distance,
        minimum_clearance_m=min_clearance,
        threshold_violation=threshold_violation,
    )
    state["safety_stop_count"] = int(state.get("safety_stop_count", 0)) + 1
    return stop_info


def trigger_estop(source: str = "external") -> dict:
    """Single emergency-stop entry point for dashboard/tests/other code."""
    trigger_time = time.monotonic()
    x0, y0 = float(state["x"]), float(state["y"])
    v0 = float(state.get("linear_velocity", 0.0))
    state["estop_source"] = source
    state["estop_trigger_ts"] = trigger_time
    interrupt_flag.set()
    record_event("ESTOP_TRIGGER", source=source, x=x0, y=y0, velocity=v0)
    info = _record_stop_event(
        event_type="estop",
        source=source,
        direction=1 if v0 >= 0 else -1,
        commanded_speed=abs(v0),
        trigger_distance=None,
        reading_age_s=None,
        recognition_time=trigger_time,
        x0=x0,
        y0=y0,
        velocity_at_trigger=v0,
        sensor_source="none",
    )
    log(f"Emergency stop triggered ({source})", "WARN")
    return info



def _sim_visual_clearance_possible(direction: int, require_recent_lidar: bool = True) -> tuple[bool, str]:
    """Is a *controlled Isaac Sim test* eligible to request a visual creep?

    Not a safety sensor, not evidence of actual free-space clearance. It is
    deliberately unavailable for real robots and ordinary launch modes.
    """
    if not ROBOT.get("sim_visual_clearance_test", False) or ROBOT.get("platform_label") != "Isaac Sim":
        return False, "disabled_or_not_simulator"
    if direction != 1 or interrupt_flag.is_set():
        return False, "reverse_or_interrupted"
    d = state.get("nav_visual_distance_m")
    if d is None or not math.isfinite(float(d)) or float(d) <= float(ROBOT.get("sim_visual_min_target_range_m", 3.0)):
        return False, "target_near_or_range_unknown"
    used = float(state.get("sim_visual_fallback_used_m", 0.0))
    if used >= float(ROBOT.get("sim_visual_total_budget_m", 0.80)) - 0.01:
        return False, "visual_fallback_budget_exhausted"
    # A transient gap only: never launch with a permanently absent LiDAR.
    lidar_ts = state.get("lidar_ts")
    if lidar_ts is None or (require_recent_lidar and time.monotonic() - float(lidar_ts) > 2.0):
        return False, "lidar_missing_or_long_stale"
    return True, ""


def _request_sim_visual_clearance() -> bool:
    """Ask about an unobstructed image while the robot is stopped.

    Grants only ONE short/slow forward motion segment. Any uncertainty, missing
    frames, VLM failure, stale odom or new emergency stop denies the request.
    """
    stopped = _stop(wait=True, direction=1)
    if not stopped.get("confirmed", False):
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason="stop_not_confirmed")
        return False
    ok, reason = _sim_visual_clearance_possible(1)
    if not ok:
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason=reason)
        return False
    img = state.get("image_raw")
    image_ts = state.get("image_ts")
    if img is None or image_ts is None or time.monotonic() - float(image_ts) > 0.75:
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason="camera_unavailable_or_stale")
        return False
    try:
        from vlm.manager import vlm_mgr
        prompt = (
            "SAFETY TEST IN A STATIC ISAAC SIM SCENE ONLY. Is the ENTIRE forward driving corridor "
            "for a 0.75m-wide mobile robot, from directly in front up to at least 3 metres, "
            "visibly unobstructed? Treat humans, robots, boxes, pallets, walls, edges, partial "
            "occlusions, or unclear floor as BLOCKED/UNCERTAIN. "
            "You must see free ground space throughout. "
            "Reply with exactly one word: CLEAR, BLOCKED, or UNCERTAIN. "
            "Only reply CLEAR if completely certain from visible evidence."
        )
        answer = vlm_mgr.ask(prompt, img, max_tokens=8)
    except Exception as exc:
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason="vlm_error", detail=str(exc)[:120])
        return False
    if str(answer).strip().upper() != "CLEAR":
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason="vlm_not_clear", answer=str(answer)[:100])
        return False
    if interrupt_flag.is_set():
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason="interrupted_during_query")
        return False
    # The original image may have become old during remote inference. Require
    # a live camera stream and fresh odometry before issuing any velocity.
    image_now = state.get("image_ts")
    odom_now = state.get("odom_ts")
    if (image_now is None or image_now <= image_ts or
            time.monotonic() - float(image_now) > 0.75):
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason="no_new_fresh_camera_frame")
        return False
    if odom_now is None or time.monotonic() - float(odom_now) > ODOM_MAX_AGE_S:
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason="odometry_stale_after_query")
        return False
    ok, reason = _sim_visual_clearance_possible(1, require_recent_lidar=False)
    if not ok:
        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason=reason)
        return False
    record_event("SIM_VISUAL_CLEARANCE_GRANTED", source="vlm_rgb_test_only",
                 max_step_m=ROBOT.get("sim_visual_grant_m", 0.20),
                 max_speed_mps=ROBOT.get("sim_visual_max_speed_mps", 0.08),
                 previous_lidar_age_s=time.monotonic() - float(state["lidar_ts"]))
    return True

def _move_dist(meters: float, speed: float = DEFAULT_SPEED,
               safe_dist: Optional[float] = None) -> float:
    """Move a signed distance with fail-closed safety and endpoint slowdown.

    Distance feedback comes from fresh odometry. The commanded linear speed is
    reduced proportionally as the remaining distance approaches zero, avoiding
    the large coast/latency overshoot seen with constant-speed control.

    `safe_dist` may only make motion MORE conservative:
      effective clearance = max(stop_trigger_distance, requested safe_dist)
    """
    if abs(meters) < 1e-6:
        state["last_motion_outcome"] = "SUCCESS"
        state["last_motion_reason"] = "zero_distance"
        return 0.0

    state["last_motion_outcome"] = "RUNNING"
    state["last_motion_reason"] = ""
    sign = 1 if meters > 0 else -1
    if sign < 0 and not ROBOT.get("rear_obstacle_sensing", False):
        log("Reverse move rejected: no validated rear obstacle sensing", "SAFETY")
        record_event("REVERSE_REJECTED", reason="rear_safety_unavailable")
        state["last_motion_outcome"] = "PRECONDITION_FAILED"
        state["last_motion_reason"] = "rear_safety_unavailable"
        return 0.0

    speed = min(abs(float(speed)), ROBOT["max_linear_speed"])
    effective_safe = max(
        STOP_TRIGGER_DIST,
        float(safe_dist) if safe_dist is not None else STOP_TRIGGER_DIST,
    )
    sx, sy = float(state["x"]), float(state["y"])
    target = abs(float(meters))
    timeout = max(4.0, (target / max(speed, 1e-3)) * 3.0+1)
    t_start = time.monotonic()
    visual_grant_origin = None
    visual_grant_limit = 0.0
    visual_grant_start_ts = None
    visual_total_used = 0.0

    with timed("move_execution", requested_distance_m=meters, speed_mps=speed):
        while True:
            if interrupt_flag.is_set():
                _stop(wait=True, direction=sign)
                state["last_motion_outcome"] = "INTERRUPTED"
                state["last_motion_reason"] = state.get("estop_source", "interrupt")
                break

            # Motion distance is only trustworthy with fresh odometry. Stale
            # odometry can otherwise produce exactly the endpoint overshoot seen
            # in the first Carter test.
            odom_ts = state.get("odom_ts")
            odom_age = None if odom_ts is None else time.monotonic() - float(odom_ts)
            if odom_ts is None or odom_age > ODOM_MAX_AGE_S:
                log(
                    f"Odometry unavailable/stale ({'none' if odom_age is None else f'{odom_age*1000:.0f} ms'}) — STOP",
                    "SAFETY",
                )
                state["last_motion_outcome"] = "ODOM_STALE"
                state["last_motion_reason"] = "odometry_unavailable" if odom_age is None else f"odometry_stale_{odom_age:.3f}s"
                record_event("ODOM_STALE_STOP", odom_age_ms=None if odom_age is None else odom_age * 1000.0)
                _stop(wait=True, direction=sign)
                break

            travelled = math.hypot(float(state["x"]) - sx, float(state["y"]) - sy)
            remaining = max(0.0, target - travelled)

            # Check endpoint BEFORE starting another safety/control cycle. This
            # prevents a completed motion from being reclassified as a stale-
            # sensor failure on the next iteration.
            if remaining <= MOVE_TOLERANCE_M:
                state["last_motion_outcome"] = "SUCCESS"
                state["last_motion_reason"] = "target_distance_reached"
                break

            if ROBOT.get("platform_label") == "Hardware":
                allowed, why = _hardware_motion_check(sign)
                if not allowed:
                    state["last_motion_outcome"] = "HARDWARE_INTERLOCK_STOP"
                    state["last_motion_reason"] = why
                    record_event("HARDWARE_INTERLOCK_STOP", reason=why)
                    break
            recognition_time = time.monotonic()
            reading = check_obstacle(sign)
            if not reading.valid:
                if visual_grant_origin is None:
                    eligible, why = _sim_visual_clearance_possible(sign)
                    if eligible:
                        # Remote VLM calls can take several seconds. Re-read
                        # independent safety sensing after the response EVEN if
                        # the VLM denied the visual-only permission. Its earlier
                        # invalid status may no longer describe the robot.
                        visual_clear = _request_sim_visual_clearance()
                        reading = check_obstacle(sign)
                        if reading.valid:
                            record_event("SIM_VISUAL_SENSOR_RECOVERED_BEFORE_CREEP",
                                         source=reading.source,
                                         distance_m=reading.distance,
                                         visual_clearance_answered_clear=visual_clear)
                        elif visual_clear:
                            visual_grant_origin = (float(state["x"]), float(state["y"]))
                            visual_grant_start_ts = time.monotonic()
                            unused = max(0.0, float(ROBOT.get("sim_visual_total_budget_m", .8))
                                         - float(state.get("sim_visual_fallback_used_m", 0.0)))
                            visual_grant_limit = min(float(ROBOT.get("sim_visual_grant_m", .2)), unused)
                        else:
                            log("VLM did not authorize controlled simulation creep — STOP", "SAFETY")
                    elif ROBOT.get("sim_visual_clearance_test", False):
                        record_event("SIM_VISUAL_CLEARANCE_DENIED", reason=why)
                if visual_grant_origin is not None:
                    grant_moved = math.hypot(float(state["x"]) - visual_grant_origin[0],
                                             float(state["y"]) - visual_grant_origin[1])
                    # Time cap prevents a stalled segment from holding open an
                    # earlier image-based permission indefinitely.
                    if (grant_moved >= visual_grant_limit - 0.01 or
                            time.monotonic() - visual_grant_start_ts > 4.0):
                        visual_total_used += grant_moved
                        state["sim_visual_fallback_used_m"] = float(state.get("sim_visual_fallback_used_m", 0.0)) + grant_moved
                        visual_grant_origin = None
                        state["last_motion_outcome"] = "SIM_VISUAL_SEGMENT_COMPLETED"
                        state["last_motion_reason"] = "limited_visual_test_segment"
                        record_event("SIM_VISUAL_SEGMENT_END", travelled_m=grant_moved,
                                     total_used_m=state["sim_visual_fallback_used_m"])
                        break
                elif not reading.valid:
                    log(f"Safety sensing unavailable ({reading.reason}) — STOP", "SAFETY")
                    state["last_vlm"] = f"SAFETY_SENSOR_UNAVAILABLE: {reading.reason}"
                    record_safety_unknown(sign, reading)
                    state["last_motion_outcome"] = "SAFETY_SENSOR_UNAVAILABLE"
                    state["last_motion_reason"] = reading.reason
                    _record_stop_event(
                        event_type="sensor_fail_stop",
                        source=reading.source or "none",
                        direction=sign,
                        commanded_speed=abs(float(state.get("linear_velocity", 0.0))) or speed,
                        trigger_distance=None,
                        reading_age_s=reading.age_s,
                        recognition_time=recognition_time,
                        x0=float(state["x"]), y0=float(state["y"]),
                        velocity_at_trigger=float(state.get("linear_velocity", 0.0)),
                        sensor_source=reading.source,
                    )
                    break

            if reading.valid and visual_grant_origin is not None:
                visual_total_used += math.hypot(float(state["x"]) - visual_grant_origin[0],
                                                 float(state["y"]) - visual_grant_origin[1])
                state["sim_visual_fallback_used_m"] = float(state.get("sim_visual_fallback_used_m", 0.0)) + visual_total_used
                visual_grant_origin = None
                visual_total_used = 0.0
                record_event("SIM_VISUAL_SENSOR_RECOVERED", travelled_m=state["sim_visual_fallback_used_m"])
            # An invalid safety reading NEVER turns into an asserted 'clear'
            # distance: the special test branch is bounded by its own grant.
            obs = float(reading.distance if reading.valid and reading.distance is not None else 99.0)
            if reading.valid and obs < effective_safe:
                direction_word = "front" if sign > 0 else "rear"
                log(f"Obstacle {obs:.2f} m — stopped ({direction_word})", "SAFETY")
                state["last_vlm"] = f"BLOCKED: obstacle {obs:.2f}m {direction_word}"
                state["last_motion_outcome"] = "BLOCKED"
                state["last_motion_reason"] = f"obstacle_{obs:.3f}m"
                _record_stop_event(
                    event_type="obstacle_stop",
                    source=reading.source,
                    direction=sign,
                    commanded_speed=abs(float(state.get("linear_velocity", 0.0))) or speed,
                    trigger_distance=obs,
                    reading_age_s=reading.age_s,
                    recognition_time=recognition_time,
                    x0=float(state["x"]), y0=float(state["y"]),
                    velocity_at_trigger=float(state.get("linear_velocity", 0.0)),
                    sensor_source=reading.source,
                )
                break

            # Endpoint controller: v = Kp * remaining, capped by the requested
            # maximum speed. Only inside the slowdown region do we permit the
            # reduced command to fall below max speed.
            if remaining < MOVE_SLOWDOWN_M:
                endpoint_speed = min(speed, max(MOVE_MIN_SPEED, MOVE_KP * remaining))
            else:
                endpoint_speed = speed

            # Obstacle slowdown is an independent upper bound on endpoint speed.
            if reading.valid and obs < (effective_safe + SLOWDOWN_MARGIN):
                endpoint_speed = min(endpoint_speed, speed * 0.4)
            if not reading.valid:
                endpoint_speed = min(endpoint_speed, float(ROBOT.get("sim_visual_max_speed_mps", 0.08)))

            _drive(linear=sign * endpoint_speed)
            time.sleep(CONTROL_PERIOD_S)
            travelled = math.hypot(
                float(state["x"]) - sx,
                float(state["y"]) - sy,
            )

            remaining = max(
                0.0,
                target - travelled,
            )

            if remaining <= MOVE_TOLERANCE_M:

                state["last_motion_outcome"] = (
                    "SUCCESS"
                )

                state["last_motion_reason"] = (
                    "target_distance_reached"
                )

                break


            if (
                time.monotonic()
                - t_start
                > timeout
            ):

                log(
                    "Motion timeout — stopping",
                    "WARN",
                )

                record_event(
                    "MOVE_TIMEOUT",
                    requested_distance_m=meters,
                    travelled_m=travelled,
                )

                state["last_motion_outcome"] = (
                    "TIMEOUT"
                )

                state["last_motion_reason"] = (
                    "motion_timeout"
                )

                break
        _stop(wait=True, direction=sign)
        if visual_grant_origin is not None:
            grant_moved = math.hypot(float(state["x"]) - visual_grant_origin[0],
                                     float(state["y"]) - visual_grant_origin[1])
            state["sim_visual_fallback_used_m"] = float(state.get("sim_visual_fallback_used_m", 0.0)) + grant_moved
            record_event("SIM_VISUAL_SEGMENT_END", travelled_m=grant_moved,
                         total_used_m=state["sim_visual_fallback_used_m"],
                         outcome=state.get("last_motion_outcome"))

    actual = math.hypot(float(state["x"]) - sx, float(state["y"]) - sy)
    overshoot = max(0.0, actual - target)
    undershoot = max(0.0, target - actual)
    if state.get("last_motion_outcome") == "RUNNING":
        state["last_motion_outcome"] = "SUCCESS"
        state["last_motion_reason"] = "target_distance_reached"
    record_event(
        "MOTION_RESULT",
        outcome=state["last_motion_outcome"],
        reason=state["last_motion_reason"],
        requested_distance_m=meters,
        actual_distance_m=actual,
        endpoint_error_m=actual - target,
        overshoot_m=overshoot,
        undershoot_m=undershoot,
        odom_hz=state.get("odom_hz_ema"),
        lidar_hz=state.get("lidar_hz_ema"),
    )
    return actual

def _rotate(deg: float, speed: float = DEFAULT_ROTATE_SPEED) -> None:
    """Rotate using odometry feedback.

    Rotational footprint collision checking is not claimed here; experiments
    should describe the validated safety envelope as translational/forward
    obstacle monitoring unless a footprint-aware rotation supervisor is added.
    """
    if abs(deg) < 1.0:
        state["last_rotation_outcome"] = "SUCCESS"
        return

    odom_ts = state.get("odom_ts")
    if odom_ts is None or time.monotonic() - float(odom_ts) > ODOM_MAX_AGE_S:
        _stop(wait=False)
        state["last_rotation_outcome"] = "ODOM_STALE"
        record_event("ROTATION_ABORTED", reason="stale_odometry_at_start", degrees=deg)
        return

    state["last_rotation_outcome"] = "RUNNING"
    start_yaw = float(state["yaw"])
    target = abs(float(deg))
    sign = 1 if deg > 0 else -1
    speed = min(abs(float(speed)), ROBOT["max_angular_speed"])
    timeout = target / 10.0 + 5.0
    t0 = time.monotonic()

    with timed("rotate_execution", requested_degrees=deg, angular_speed=speed):
        while time.monotonic() - t0 < timeout:
            if interrupt_flag.is_set():
                state["last_rotation_outcome"] = "INTERRUPTED"
                log("Rotation interrupted", "WARN")
                break
            odom_ts = state.get("odom_ts")
            if odom_ts is None or time.monotonic() - float(odom_ts) > ODOM_MAX_AGE_S:
                state["last_rotation_outcome"] = "ODOM_STALE"
                record_event("ROTATION_ABORTED", reason="stale_odometry_during_turn", degrees=deg)
                break

            if ROBOT.get("platform_label") == "Hardware":
                allowed, why = _hardware_motion_check(1)
                if not allowed:
                    state["last_rotation_outcome"] = "HARDWARE_INTERLOCK_STOP"
                    record_event("HARDWARE_INTERLOCK_STOP", reason=why, motion="rotation")
                    break
            diff = float(state["yaw"]) - start_yaw
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            travelled = abs(diff)
            if travelled >= target - 2.0:
                state["last_rotation_outcome"] = "SUCCESS"
                break

            rate = speed * 0.4 if (target - travelled) < 20 else speed
            _drive(angular=sign * rate)
            time.sleep(CONTROL_PERIOD_S)

        _stop(wait=True, direction=1)
        if state["last_rotation_outcome"] == "RUNNING":
            state["last_rotation_outcome"] = "ROTATION_TIMEOUT"
        record_event("ROTATION_RESULT", requested_degrees=deg,
                     outcome=state["last_rotation_outcome"],
                     final_yaw=state.get("yaw"))
