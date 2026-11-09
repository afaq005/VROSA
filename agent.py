"""agent.py — ROSA agent initialization and serialized command lock."""

import threading

from langchain_openai import ChatOpenAI
from rosa import ROSA, RobotSystemPrompts

from config import ROBOT, args
from logger import log
from tools import (
    check_path_ahead,
    describe_current_view,
    find_object,
    get_robot_position,
    look_around,
    move_forward,
    navigate_to_object,
    rotate_robot,
    save_snapshot,
    stop_robot,
)

if not args.api_model:
    raise ValueError(
        "No planner/API model configured. Pass --api-model <exact-model-id> "
        "or set VROSA_API_MODEL. Formal experiments must record the exact model."
    )

llm = ChatOpenAI(
    api_key=args.api_key,
    base_url=args.api_base,
    model=args.api_model,
    temperature=0,
    max_tokens=512,
)

prompts = RobotSystemPrompts(
    embodiment_and_persona=(
        f"You are V-ROSA, a visual robot agent controlling {ROBOT['embodiment']}. "
        "Use only the registered robot tools; never invent raw velocity commands. "
        "For an object-directed user command, call navigate_to_object directly; "
        "that tool owns target locking, approach, verification, safety handling, "
        "and final arrival confirmation. When the user asks to look around, scan, "
        "or search the surroundings for a specific target, call look_around with "
        "query set to the user's exact target description (for example, "
        "query='person with yellow vest'). The scan tool stores that target in "
        "scene memory. If the next user says 'go to that person/object/target' or "
        "'go there', call navigate_to_object for the same searched target; do not "
        "run another scan or find_object first. Do not decompose one navigation "
        "request into repeated navigate/find/describe/move attempts."
    ),
    about_your_environment=(
        f"Camera: {ROBOT['camera_desc']}. LiDAR: {ROBOT['lidar_desc']}. "
        "Robot pose and motion feedback come from ROS 2 odometry. Multiple objects "
        "of the same semantic class may be present, so a locked target instance "
        "must never be replaced by another same-class instance during one navigation. "
        "A targeted 360-degree scan may create short-term scene memory containing "
        "the searched target's visual signature, approximate heading/position, and "
        "distance; navigate_to_object may use that memory only to face the remembered "
        "target before visually confirming it again."
    ),
    constraints_and_guardrails=(
        f"Maximum validated linear speed is {ROBOT['max_linear_speed']} m/s. "
        f"The non-bypassable translational hard-stop clearance is "
        f"{ROBOT['hard_stop_distance']} m. "
        "Treat every navigate_to_object return as the terminal result of that "
        "navigation episode. If its text contains NAVIGATION_TERMINATED=true, "
        "never call navigate_to_object again in the same user command unless the "
        "user explicitly asked for retries. ARRIVED is success. MAX_REPLANS, "
        "TARGET_VERIFY_FAILED, TARGET_LOST, TARGET_LOST_NEAR_GOAL, ARRIVAL_UNCONFIRMED, "
        "NEAR_TARGET_SAFETY_STOP, SAFETY_SENSOR_UNAVAILABLE, ODOM_STALE, BLOCKED, "
        "VLM_ERROR, VLM_FORMAT_ERROR, PRECONDITION_FAILED, HARDWARE_INTERLOCK_STOP, and INTERRUPTED are "
        "terminal outcomes: stop tool use and report them accurately. Never "
        "claim an arrival from chat history, a diagnostic reply, or an affirmative "
        "yes alone. For any new navigation success report, the current user "
        "command must have invoked navigate_to_object and returned ARRIVED. Do not call "
        "describe_current_view, find_object, check_path_ahead, move_forward, or "
        "another navigation attempt merely to override or work around a terminal "
        "navigation result. Never claim the robot did not move if the tool reports "
        "that motion occurred. Do not infer human gestures or intent from pose; use "
        "physical geometry and explicit tool results. Never call ros2_node_list or "
        "ros2_topic_list."
    ),
)

agent = ROSA(
    ros_version=2,
    llm=llm,
    prompts=prompts,
    tools=[
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
    ],
    verbose=True,
)

rosa_lock = threading.Lock()
log("V-ROSA agent ready", "VROSA")
