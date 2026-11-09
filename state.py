"""
state.py — Shared runtime state.

The codebase still uses a dict for compatibility with the existing dashboard and
tools, but the fields now include timestamps, velocities, frame identifiers and
experiment counters required for reproducible evaluation.
"""

import threading
from config import ROBOT

state: dict = {
    # Pose / motion
    "x": 0.0,
    "y": 0.0,
    "yaw": 0.0,
    "linear_velocity": 0.0,
    "angular_velocity": 0.0,
    "odom_ts": None,
    "odom_ros_ts": None,
    # Populated ONLY by a separately configured simulator ground-truth topic.
    "gt_robot_x": None,
    "gt_robot_y": None,
    "gt_robot_ts": None,
    "gt_robot_source": None,
    "path_length_m": 0.0,
    "_prev_odom_xy": None,
    "odom_period_s_ema": None,
    "odom_hz_ema": None,

    # Camera
    "image_raw": None,

    # Number of ROS Image messages entering the callback.
    # This increments BEFORE image conversion.
    "image_rx_seq": 0,

    # Number of successfully converted RGB frames.
    "image_seq": 0,

    "image_callback_enter_ts": None,
    "image_ts": None,
    "image_ros_ts": None,

    "image_conversion_errors": 0,

    "image_period_s_ema": None,
    "image_hz_ema": None,
    "image_processing_ms": None,

    "camera_frame": "",
    "camera_width": None,
    "camera_height": None,
    "camera_hfov_deg": None,
    "camera_info_logged": False,

    "cam_fx": ROBOT["default_cam_fx"],
    "cam_fy": ROBOT["default_cam_fx"],
    "cam_cx": None,
    "cam_cy": None,
    # LiDAR
    "lidar": [],
    "lidar_ts": None,
    "lidar_ros_ts": None,
    "lidar_frame": "",
    "has_lidar": False,
    "lidar_period_s_ema": None,
    "lidar_hz_ema": None,
    "lidar_processing_ms": None,
    "lidar_nearest_min": 99.0,
    "lidar_forward_min": 99.0,
    "lidar_reverse_min": 99.0,
    "lidar_left_min": 99.0,
    "lidar_center_min": 99.0,
    "lidar_right_min": 99.0,

    # Perception
    "detections": [],
    "depth_map": None,
    "depth_raw_map": None,
    "depth_source_image": None,
    "depth_ts": None,
    "depth_source_image_ts": None,
    "depth_source_image_seq": None,
    "depth_source_ros_ts": None,
    "perc_ready": False,
    "show_dets": False,
    "use_lidar": ROBOT["has_lidar"],
    "use_depth": ROBOT.get("hardware_use_model_depth", True),
    "use_range_depth": ROBOT.get("hardware_use_range_depth", False),
    "range_depth_map": None,
    "range_depth_ts": None,
    "range_depth_ros_ts": None,
    "range_depth_processing_ms": None,
    "range_depth_hz_ema": None,
    "range_depth_conversion_errors": 0,
    "use_yolo": ROBOT.get("hardware_use_yolo", True),
    "use_gdino": ROBOT.get("hardware_use_gdino", False),

    # Task / UI
    "status": "idle",
    "last_vlm": "—",
    "target": "",
    "last_target": "",
    "last_nav_outcome": "",
    "last_nav_estimated_final_distance": None,
    "last_motion_outcome": "",
    "last_motion_reason": "",
    "nav_visual_distance_m": None,
    "sim_visual_fallback_used_m": 0.0,
    "motion_active": False,
    "motion_kind": "",
    # Experiment counters (reset at trial start)
    "tool_call_count": 0,
    "vlm_query_count": 0,
    "replan_count": 0,
    "safety_stop_count": 0,
    "sequence_error_count": 0,
    "recovery_success_count": 0,

    # Emergency stop bookkeeping
    "estop_source": "",
    "estop_trigger_ts": None,
}

# Kept as a single latch across all motion primitives.
interrupt_flag = threading.Event()

# Optional lock for future atomic snapshot operations. Existing code can keep
# direct dict access, while new logging/snapshot code can use this lock.
state_lock = threading.RLock()
