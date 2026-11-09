"""
V-ROSA live launcher.

Startup order is deliberate:

1. Parse configuration.
2. Load all perception models completely.
3. Start ROS 2 subscriptions.
4. Start perception inference loops.
5. Initialize the agent/server.

This prevents expensive Torch/CUDA/Transformers model loading from
starving ROS camera callbacks.
"""

import socket
import time

import config
from config import ROBOT, SERVER_PORT

from experiment_logger import write_manifest
from logger import log
from state import state
from vlm.manager import vlm_mgr

# ════════════════════════════════════════════════════════════
# 1. PRELOAD ALL MODELS
# ════════════════════════════════════════════════════════════

from perception import (
    preload_perception_models,
    start_perception_loops,
)

preload_perception_models()

# ════════════════════════════════════════════════════════════
# 2. START ROS AFTER MODEL INITIALIZATION
# ════════════════════════════════════════════════════════════

import ros_node  # noqa: F401

# ════════════════════════════════════════════════════════════
# 3. START PERCEPTION LOOPS
# ════════════════════════════════════════════════════════════

start_perception_loops()

# ════════════════════════════════════════════════════════════
# 4. AGENT + SERVER
# ════════════════════════════════════════════════════════════

import agent  # noqa: F401
from server import app


def get_local_ip() -> str:
    try:
        with socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        ) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]

    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    import rclpy
    from motion import _stop

    write_manifest({
        "active_vlm_backend":
            vlm_mgr.active_name,

        "active_vlm_model":
            vlm_mgr.active_model,
    })

    log(
        "Waiting for initial sensor data...",
        "INFO",
    )

    # Give ROS callbacks a few seconds to populate the state.
    time.sleep(3.0)

    ip = "127.0.0.1" if ROBOT["platform_label"] == "Hardware" else get_local_ip()

    print("=" * 64)
    print("V-ROSA")

    print(
        f"Robot      : "
        f"{ROBOT['display_name']} "
        f"({ROBOT['platform_label']})"
    )

    print(
        f"Dashboard  : "
        f"http://{ip}:{SERVER_PORT}/view"
    )

    print(
        f"Stream     : "
        f"http://{ip}:{SERVER_PORT}/frame.jpg"
    )

    print(
        f"VLM        : "
        f"{vlm_mgr.active_name} / "
        f"{vlm_mgr.active_model}"
    )

    print(
        f"Camera     : "
        f"{'live' if state['image_raw'] is not None else 'waiting'}"
    )

    print(
        f"Camera RX  : "
        f"{state.get('image_rx_seq', 0)}"
    )

    print(
        f"Camera RGB : "
        f"{state.get('image_seq', 0)}"
    )

    print(
        f"LiDAR      : "
        f"{'live' if state['lidar'] else 'not available / waiting'}"
    )

    print(
        f"Depth      : "
        f"{'ON' if state.get('use_depth', False) else 'OFF'} "
        f"(model preloaded)"
    )

    print(
        f"GDINO      : "
        f"{'ON' if state.get('use_gdino', False) else 'OFF'} "
        f"(model preloaded)"
    )

    print("=" * 64)

    try:
        app.run(
            host=("127.0.0.1" if ROBOT["platform_label"] == "Hardware" else "0.0.0.0"),
            port=SERVER_PORT,
            threaded=True,
        )

    except KeyboardInterrupt:
        pass

    finally:
        _stop()
        ros_node.stop_spin()
        rclpy.shutdown()