"""
state.py — Single shared state dict + interrupt flag.
Imported by ros_node, motion, tools, hud, server — keeps everything in sync.
"""

import threading

# ── Robot state (mutated by ROS callbacks + tools) ───────────────
state: dict = {
    "x":         0.0,
    "y":         0.0,
    "yaw":       0.0,
    "image_raw": None,   # latest PIL Image from camera
    "image_hud": None,   # same image with HUD overlay burned in
    "lidar":     [],     # sorted list of {dist, x, y, z} dicts
    "cam_fx":    640.0,  # focal length from CameraInfo
    "status":    "idle",
    "last_vlm":  "—",
    "target":    "",
    # ── Perception pipeline (YOLO + DA-V2 + GroundingDINO) ──
    "detections": [],   # [{label, dist, sector, bbox, conf, source}]
    "depth_map":  None, # numpy H×W float32, metric metres per pixel
    "has_lidar":  False, # flips to True the first time LiDAR data arrives
    "show_dets":  False, # toggle: draw YOLO/GDino boxes on the live stream
    "perc_ready": False, # set True by perception.py once all models load
    # ── Perception source toggles (controllable from dashboard) ──
    "use_lidar":  True,  # use LiDAR for obstacle + distance (when available)
    "use_depth":  False,  # run DA-V2 depth map
    "use_yolo":   True,  # run YOLO fast loop
    "use_gdino":  False,  # run GroundingDINO slow loop
}

# ── Interrupt flag (set by /stop endpoint and stop words) ────────
interrupt_flag = threading.Event()
