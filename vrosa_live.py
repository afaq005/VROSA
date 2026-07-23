"""
V-ROSA Live View + Voice Navigation
=====================================
Flexible VLM backend: API (claude/gpt) or Local (trained LLaVA)

Usage:
  python3 vrosa_live.py               # defaults to API mode
  python3 vrosa_live.py --vlm api     # API mode (cloud)
  python3 vrosa_live.py --vlm local   # Local LLaVA mode (GPU)
  python3 vrosa_live.py --vlm remote  # Remote GPU server

Switch live via dashboard or:
  curl -X POST http://localhost:5000/vlm/switch -H 'Content-Type: application/json' -d '{"mode": "local"}'
"""
import sys
print("PYTHON:", sys.executable)

import transformers
print("TRANSFORMERS PATH:", transformers.__file__)
# ── STEP 1: Config + ROS env vars (must be before any rclpy import) ──
import config  # noqa: F401 — sets ROS_DOMAIN_ID, RMW_IMPLEMENTATION, parses args

# ── STEP 2: Standard imports ──────────────────────────────────────
import time
import socket

from config import SERVER_PORT
from logger import log
from state import state

# ── STEP 3: VLM manager (registers backends) ─────────────────────
from vlm.manager import vlm_mgr  # noqa: F401

# ── STEP 4: ROS 2 node + sensor subscriptions ────────────────────
import ros_node  # noqa: F401

# ── STEP 5: Perception pipeline (DA-V2 + YOLO + GroundingDINO) ───
from perception import start_perception
start_perception()   # loads models in background, starts 10-fps loop

# ── STEP 6: ROSA agent ───────────────────────────────────────────
import agent  # noqa: F401

# ── STEP 7: Flask server ─────────────────────────────────────────
from server import app


# ════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════
def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import rclpy
    from motion import _stop

    log("Waiting for sensors (3s)...", "INFO")
    time.sleep(3)

    ip = get_local_ip()
    print("=" * 60)
    print("   V-ROSA Live View")
    print("=" * 60)
    print(f"  Dashboard : http://{ip}:{SERVER_PORT}/view")
    print(f"  Stream    : http://{ip}:{SERVER_PORT}/stream")
    print(f"  VLM mode  : {vlm_mgr.active_name.upper()}")
    print(f"  VLM status: http://{ip}:{SERVER_PORT}/vlm/status")
    print("=" * 60)
    print(f"  Camera    : {'✅ live' if state['image_raw'] else '❌ waiting'}")
    print(f"  LiDAR     : {'✅ live' if state['lidar']     else '❌ waiting'}")
    print("=" * 60)
    print()
    print("  Switch VLM from CLI:")
    print(f"    curl -X POST http://localhost:{SERVER_PORT}/vlm/switch \\")
    print(f"         -H 'Content-Type: application/json' \\")
    print(f"         -d '{{\"mode\": \"local\"}}'")
    print("=" * 60)

    try:
        app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _stop()
        rclpy.shutdown()