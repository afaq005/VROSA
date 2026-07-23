"""
agent.py — ROSA agent initialization.

Sets up the LangChain LLM, robot system prompts, and the ROSA agent
with all registered tools. Also owns the rosa_lock mutex that serializes
command execution.
"""

import threading

from langchain_openai import ChatOpenAI
from rosa import ROSA, RobotSystemPrompts

from config import args
from logger import log
from tools import (
    get_robot_position,
    move_forward,
    rotate_robot,
    stop_robot,
    describe_current_view,
    look_around,
    find_object,
    navigate_to_object,
    check_path_ahead,
    save_snapshot,
)

# ── LLM ─────────────────────────────────────────────────────────
llm = ChatOpenAI(
    api_key=args.api_key,
    base_url=args.api_base,
    model=args.api_model,
    temperature=0,
    max_tokens=512,
)

# ── System prompts ───────────────────────────────────────────────
prompts = RobotSystemPrompts(
    embodiment_and_persona=(
        "You are V-ROSA, a visual robot agent controlling a mobile robot "
        ". You have a camera and can see the world. "
        "When asked to move toward something, use navigate_to_object. "
        "When asked what you see, use describe_current_view."
    ),
    about_your_environment=(
        "Warehouse simulation. Camera: 1920x1200 front stereo. "
        "LiDAR: 360° 3D scan. Odometry: precise position."
    ),
    constraints_and_guardrails=(
    "Max 0.4 m/s. Stop if obstacle < 0.7m. "
    "NEVER call ros2_node_list or ros2_topic_list — they are unnecessary. "
    "Call navigate_to_object DIRECTLY when asked to navigate. "
    "Call describe_current_view DIRECTLY when asked what you see. "
    "Do NOT gather information before acting — act immediately."
),
    # constraints_and_guardrails=(
    #     "Max 0.4 m/s. Stop if obstacle < 0.7m. "
    #     "Always check_path_ahead before long moves."
    # ),
)

# ── ROSA agent ───────────────────────────────────────────────────
agent = ROSA(
    ros_version=2,
    llm=llm,
    prompts=prompts,
    tools=[
        get_robot_position, move_forward, rotate_robot, stop_robot,
        describe_current_view, look_around, find_object, navigate_to_object,
        check_path_ahead, save_snapshot,
    ],
    verbose=True,
)

# ── Mutex: only one ROSA command runs at a time ──────────────────
rosa_lock = threading.Lock()

log("V-ROSA agent ready", "VROSA")