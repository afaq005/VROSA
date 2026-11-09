"""
config.py — CLI arguments, environment variables, per-robot profiles, and
experiment constants.
Import this module before rclpy so ROS environment variables are set first.
All platform-specific values live in ROBOT_PROFILES.  Core navigation,
perception, and safety code must not hard-code robot-specific topic names.
"""

from __future__ import annotations

import argparse
import os

# ── ROS environment ───────────────────────────────────────────────
os.environ.setdefault("ROS_DOMAIN_ID", "10")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")

# ══════════════════════════════════════════════════════════════════
# ROBOT PROFILES
# ══════════════════════════════════════════════════════════════════
# Safety invariant used throughout the project:
#   hard_stop_distance is the minimum allowed translational clearance.
# Navigation is allowed to request MORE clearance, but never less.
#
# depth_scale/depth_bias are explicit calibration parameters:
#   calibrated_depth = raw_metric_depth * depth_scale + depth_bias
# Keep them at 1/0 until simulation/physical validation determines otherwise.
ROBOT_PROFILES = {
    "nova_carter_sim": {
        "display_name": "Nova Carter",
        "platform_label": "Isaac Sim",
        "topics": {
            "image": "/carter1/front_stereo_camera/left/image_raw",
            "camera_info": "/carter1/front_stereo_camera/left/camera_info",
            "lidar": "/carter1/front_3d_lidar/lidar_points",
            "odom": "/carter1/chassis/odom",       # VERIFY against the launched Carter namespace
            "cmd_vel": "/carter1/cmd_vel",         # VERIFY against the launched Carter namespace
        },
        "has_lidar": True,
        "rear_obstacle_sensing": True,      # 360° LiDAR supports reverse checks
        "lidar_bbox_projection_validated": False,
        "max_linear_speed": 0.40,
        "max_angular_speed": 0.40,
        "odom_max_age_s": 0.30,
        "lidar_max_age_s": 0.75,
        "depth_max_age_s": 0.75,
        "lidar_store_stride": 4,
        "move_position_tolerance_m": 0.03,
        "move_slowdown_distance_m": 0.40,
        "move_min_speed_mps": 0.06,
        "move_kp": 1.20,
        "hard_stop_distance": 0.75,       # minimum allowed clearance / safety boundary
        "stop_trigger_distance": 1.10,     # set > hard_stop after measured braking validation if needed
        "slowdown_margin": 0.50,
        "side_clearance_required": 1.20,
        "arrive_dist": 2.00,
        "max_legs": 12,
        "sub_step": 0.50,
        "leg_fraction": 0.70,
        "leg_bounds": (0.30, 4.00),
        "direction_angles": {
            "HARD_LEFT": 40,
            "LEFT": 20,
            "CENTER": 0,
            "RIGHT": -20,
            "HARD_RIGHT": -40,
        },
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "depth_scale": 1.0,
        "depth_bias": 0.0,
        # Target-depth calibration is disabled for formal experiments because
        # the earlier exploratory fit used GT-assisted/oracle analysis.
        # Keep raw model output for diagnostics only unless a clean held-out
        # detector-ROI calibration is established.
        "target_depth_calibration_enabled": False,
        "target_depth_cal_scale": 0.548005,
        "target_depth_cal_bias": -0.020028,

        # Highest raw DA-V2 object-distance value within the calibrated
        # short-range regime (~0.8-5 m GT). Beyond this, leave DA raw.
        "target_depth_cal_raw_max_m": 9.50,

        # Visual-agent arrival standoff: keep the complete target visible for
        # VLM identity/arrival confirmation instead of driving into a close crop.
        "runtime_arrive_dist": 2.00,
        "target_verify_dist": 2.00,
        # Allow transient YOLO/depth update gaps without abandoning
        # the already locked physical target.
        "target_reacquire_timeout_s": 2.0,
        # Once a target instance is selected, subsequent same-class
        # detections must remain geometrically consistent with its
        # remembered world location.
        "target_lock_bearing_gate_deg": 28.0,
        "target_lock_range_gate_m": 1.25,

        # When the locked target temporarily disappears while still far,
        # rotate only toward its remembered bearing. Never perform a
        # global same-class re-search.
        "target_memory_turn_max_deg": 20.0,
        "arrival_confirm_frames": 3,
        "arrival_confirm_timeout_s": 2.5,
        "depth_min_m": 0.15,
        "depth_max_m": 20.0,
        "yolo_model": "yolo11n",
        "default_cam_fx": 960.0,
        "camera_desc": "front stereo RGB camera",
        "lidar_desc": "360° 3D LiDAR",
        "embodiment": "a ROS 2 Nova Carter mobile robot in NVIDIA Isaac Sim",
        "footprint": {"length_m": 0.75, "width_m": 0.55, "safety_margin_m": 0.10},
    },
    "scout2": {
        "display_name": "AgileX Scout 2.0",
        "platform_label": "Hardware",
        "topics": {
            "image": "/camera/camera/color/image_raw",
            "camera_info": "/camera/camera/color/camera_info",
            "lidar": None,
            "odom": "/odom",               # VERIFY against your Scout bringup
            "cmd_vel": "/cmd_vel",         # VERIFY against your Scout bringup
        },
        "has_lidar": False,
        "rear_obstacle_sensing": False,     # front camera only: reverse is not autonomously authorized
        "lidar_bbox_projection_validated": False,
        "max_linear_speed": 0.40,
        "max_angular_speed": 0.40,
        "odom_max_age_s": 0.30,
        "lidar_max_age_s": 0.75,
        "depth_max_age_s": 0.75,
        "lidar_store_stride": 4,
        "move_position_tolerance_m": 0.03,
        "move_slowdown_distance_m": 0.40,
        "move_min_speed_mps": 0.06,
        "move_kp": 1.20,
        "hard_stop_distance": 0.75,       # minimum allowed clearance / safety boundary
        "stop_trigger_distance": 1.10,     # conservative STARTING hypothesis; validate on Scout before approach tests
        "slowdown_margin": 0.50,
        "side_clearance_required": 1.20,
        "arrive_dist": 2.00,
        "max_legs": 12,
        "sub_step": 0.50,
        "leg_fraction": 0.70,
        "leg_bounds": (0.30, 4.00),
        "direction_angles": {
            "HARD_LEFT": 40,
            "LEFT": 20,
            "CENTER": 0,
            "RIGHT": -20,
            "HARD_RIGHT": -40,
        },
        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        "depth_scale": 1.0,
        "depth_bias": 0.0,
        "target_depth_calibration_enabled": False,
        "runtime_arrive_dist": 2.00,
        "target_verify_dist": 2.00,
        "arrival_confirm_frames": 3,
        "arrival_confirm_timeout_s": 2.5,
        "depth_min_m": 0.15,
        "depth_max_m": 20.0,
        "yolo_model": "yolo11n",
        "default_cam_fx": 1350.0,            # overwritten by CameraInfo when available
        "camera_desc": "Intel RealSense RGB camera",
        "lidar_desc": "none (camera-only)",
        "embodiment": "a ROS 2 AgileX Scout 2.0 mobile robot",
        "footprint": {"length_m": 0.93, "width_m": 0.70, "safety_margin_m": 0.10},
    },
}

# ── CLI arguments ────────────────────────────────────────────────


# baze_A9d8rpO77E9Ct9XuM1rAexmdbhO460pp0TDqKo  tabish

parser = argparse.ArgumentParser(description="V-ROSA")
parser.add_argument(
    "--robot",
    required=True,
    choices=list(ROBOT_PROFILES.keys()),
    help="Robot profile. Required to avoid accidental launch with wrong topics/safety settings.",
)
parser.add_argument("--vlm", choices=["api", "local", "remote"], default="api")
parser.add_argument("--api-key", default=os.environ.get("VROSA_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
parser.add_argument(
    "--api-base",
    default=os.environ.get("VROSA_API_BASE", "https://factchat-cloud.mindlogic.ai/v1/gateway"),
)
parser.add_argument("--api-model", default=os.environ.get("VROSA_API_MODEL", "claude-sonnet-4-6"))
parser.add_argument("--local-model", default="/home/caim/vrosa_model/vrosa_vlm_final")
parser.add_argument("--local-base", default="llava-hf/llava-1.5-7b-hf")
parser.add_argument("--remote-url", default="")
parser.add_argument("--port", type=int, default=5000)
# Physical Scout: diagnostic startup works without any camera/range stream.
# Guarded operation NEVER permits motion without fresh physical obstacle sensing.
parser.add_argument("--hardware-mode", choices=("diagnostic", "guarded"), default="diagnostic",
                    help="Scout only: diagnostic blocks ALL velocity; guarded checks live sensors")
parser.add_argument("--hardware-lidar-topic", default="", help="Optional real LiDAR topic on Scout")
parser.add_argument("--hardware-range-depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw",
                    help="Aligned RealSense 16UC1 millimeter or 32FC1 meter image")
parser.add_argument("--hardware-use-range-depth", choices=("on", "off"), default="on")
parser.add_argument("--hardware-color-qos", choices=("best_effort", "reliable"), default="best_effort")
parser.add_argument("--hardware-image-topic", default="")
parser.add_argument("--hardware-camera-info-topic", default="")
parser.add_argument("--hardware-odom-topic", default="")
parser.add_argument("--hardware-cmd-topic", default="")
parser.add_argument("--hardware-max-linear", type=float, default=0.15)
parser.add_argument("--hardware-max-angular", type=float, default=0.20)
parser.add_argument("--hardware-stop-trigger", type=float, default=1.10)
parser.add_argument("--hardware-use-model-depth", choices=("on", "off"), default="on",
                    help="Monocular DA depth is diagnostic/semantic only, NOT hardware motion authorization")
parser.add_argument("--hardware-use-yolo", choices=("on", "off"), default="on")
parser.add_argument("--hardware-use-gdino", choices=("on", "off"), default="off")

# Explicit, Isaac-Sim-only research mode. The default remains fail-closed.
parser.add_argument("--sim-visual-clearance-test", action="store_true", default=False,
                    help="Enable short, monitored visual-only creeps during transient LiDAR gaps in Isaac Sim ONLY")

# Experiment identifiers.  These are written into every record and make
# simulation/hardware results joinable without manual log matching.
parser.add_argument("--run-id", default=os.environ.get("VROSA_RUN_ID", ""))
parser.add_argument("--trial-id", default=os.environ.get("VROSA_TRIAL_ID", ""))
parser.add_argument("--scenario-id", default=os.environ.get("VROSA_SCENARIO_ID", ""))
parser.add_argument("--method", default=os.environ.get("VROSA_METHOD", "vrosa"))
parser.add_argument("--seed", type=int, default=int(os.environ.get("VROSA_SEED", "0")))
parser.add_argument("--log-dir", default=os.environ.get("VROSA_LOG_DIR", "logs"))

args, _unknown = parser.parse_known_args()
ROBOT = ROBOT_PROFILES[args.robot]
if args.robot == "scout2":
    if not (0.02 <= args.hardware_max_linear <= 0.25):
        parser.error("Scout pilot linear speed limit must be between 0.02 and 0.25 m/s")
    if not (0.05 <= args.hardware_max_angular <= 0.30):
        parser.error("Scout pilot angular speed limit must be between 0.05 and 0.30 rad/s")
    if not (args.hardware_stop_trigger > ROBOT["hard_stop_distance"]):
        parser.error("Scout stop trigger must be strictly greater than hard stop distance")
    ROBOT["max_linear_speed"] = args.hardware_max_linear
    ROBOT["max_angular_speed"] = args.hardware_max_angular
    ROBOT["stop_trigger_distance"] = args.hardware_stop_trigger
    ROBOT["hardware_mode"] = args.hardware_mode
    ROBOT["hardware_use_range_depth"] = args.hardware_use_range_depth == "on"
    ROBOT["hardware_use_model_depth"] = args.hardware_use_model_depth == "on"
    ROBOT["hardware_use_yolo"] = args.hardware_use_yolo == "on"
    ROBOT["hardware_use_gdino"] = args.hardware_use_gdino == "on"
    ROBOT["camera_qos"] = args.hardware_color_qos
    ROBOT["topics"]["range_depth"] = (args.hardware_range_depth_topic
                                            if ROBOT["hardware_use_range_depth"] else None)
    if args.hardware_lidar_topic:
        ROBOT["topics"]["lidar"] = args.hardware_lidar_topic
        ROBOT["has_lidar"] = True
        ROBOT["rear_obstacle_sensing"] = False  # must be separately validated
    for opt, topic in (("image", args.hardware_image_topic),
                       ("camera_info", args.hardware_camera_info_topic),
                       ("odom", args.hardware_odom_topic),
                       ("cmd_vel", args.hardware_cmd_topic)):
        if topic:
            ROBOT["topics"][opt] = topic
else:
    ROBOT["hardware_mode"] = "not_applicable"

if args.sim_visual_clearance_test and args.robot != "nova_carter_sim":
    parser.error("--sim-visual-clearance-test is permitted only with --robot nova_carter_sim")
ROBOT["sim_visual_clearance_test"] = bool(args.sim_visual_clearance_test)
ROBOT["sim_visual_grant_m"] = 0.20
ROBOT["sim_visual_total_budget_m"] = 0.80
ROBOT["sim_visual_max_speed_mps"] = 0.08
ROBOT["sim_visual_min_target_range_m"] = 3.0

# Backward-compatible aliases.  All are intentionally equal to the single
# hard-stop threshold; callers cannot weaken safety by choosing a smaller value.
ROBOT["safe_dist_motion"] = ROBOT["hard_stop_distance"]
ROBOT["safe_dist_obstacle"] = ROBOT["hard_stop_distance"]

SERVER_PORT = args.port
STREAM_FPS = 15

PERCEPTION_CONFIG = {
    "yolo_conf": 0.25,
    "yolo_target_fps": 15,
    "gdino_conf": 0.30,
    "gdino_query_conf": 0.25,
    "gdino_target_fps": 1,
    "depth_bbox_inner_fraction": 0.60,
    "depth_near_threshold_m": 2.0,
    "depth_mid_threshold_m": 5.0,
    "depth_percentile_near": 10,
    "depth_percentile_mid": 25,
    "depth_percentile_far": 45,
    "target_detection_max_age_s": 1.5,
    "depth_target_fps": 2.0,
}

SAFETY_CONFIG = {
    "control_period_s": 0.05,
    "max_sensor_age_s": 0.75,
    "depth_strip_h": (1 / 3, 0.68),
    "depth_strip_w": (0.35, 0.65),
    "depth_percentile": 8,
    "min_valid_depth_pixels": 50,
    "stop_velocity_eps_mps": 0.02,
    "stop_angular_eps_rps": 0.03,
    "stop_stable_time_s": 0.20,
    "stop_wait_timeout_s": 3.0,
}

CONTROL_PERIOD_S = SAFETY_CONFIG["control_period_s"]
MAX_SENSOR_AGE_S = SAFETY_CONFIG["max_sensor_age_s"]
STOP_VELOCITY_EPS_MPS = SAFETY_CONFIG["stop_velocity_eps_mps"]
STOP_ANGULAR_EPS_RPS = SAFETY_CONFIG["stop_angular_eps_rps"]
STOP_STABLE_TIME_S = SAFETY_CONFIG["stop_stable_time_s"]
STOP_WAIT_TIMEOUT_S = SAFETY_CONFIG["stop_wait_timeout_s"]
