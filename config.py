"""
config.py — CLI arguments, environment variables, and global constants.
Must be the first import in vrosa_live.py so ROS env vars are set
before any rclpy import happens.
"""

import os
import argparse

# ── ROS environment (must be set before rclpy is imported) ──────
os.environ["ROS_DOMAIN_ID"]      = ""
os.environ["RMW_IMPLEMENTATION"] = ""

# ── CLI arguments ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="V-ROSA Hardware")

parser.add_argument(
    "--vlm", choices=["api", "local", "remote"], default="api",
    help="VLM backend: api | local | remote",
)
parser.add_argument("--api-key",   default="")
parser.add_argument("--api-base",  default="")
parser.add_argument("--api-model", default="claude-sonnet-4-6")
# parser.add_argument("--api-key",   default=os.environ.get("OPENAI_API_KEY", ""))
# parser.add_argument("--api-base",  default="https://api.openai.com/v1")
# parser.add_argument("--api-model", default="gpt-4o-mini")
parser.add_argument(
    "--local-model", default="",
    help="Path to trained LoRA model dir",
)
parser.add_argument(
    "--local-base", default="llava-hf/llava-1.5-7b-hf",
    help="Base model for local LoRA",
)
parser.add_argument(
    "--remote-url", default="",
    help="Remote inference server ",
)
parser.add_argument("--port", type=int, default=5000)

args = parser.parse_args()

# ── Global constants ─────────────────────────────────────────────
SERVER_PORT = args.port
STREAM_FPS  = 15
