"""
tools.py — LangChain tools exposed to the ROSA agent.

Each @tool function is a discrete capability:
  - get_robot_position      : read odometry
  - move_forward            : translate by metres
  - rotate_robot            : rotate by degrees
  - stop_robot              : emergency stop
  - describe_current_view   : VLM scene description
  - find_object             : VLM object localisation
  - navigate_to_object      : full semantic navigation loop
  - check_path_ahead        : LiDAR + VLM path check
  - save_snapshot           : save current frame to disk
  - look_around             : 360° scan with 4 VLM snapshots
"""

import time
import math
import re

import numpy as np
from langchain.tools import tool

from state import state, interrupt_flag
from vlm.manager import vlm_mgr
from motion import _stop, _drive, _move_dist, _rotate, SAFE_DIST, trigger_estop
from logger import log
import perception   # ensures background loop is running
from config import ROBOT, MAX_SENSOR_AGE_S, PERCEPTION_CONFIG
from safety import check_obstacle
from experiment_logger import timed, record_event

# ── Lightweight semantic memory helpers ──────────────────────
# These are intentionally deterministic and cheap.  They are used only to
# preserve user-specified visual attributes (e.g. "yellow vest") across scan
# and navigation tools; they never replace the VLM's semantic reasoning.
_QUERY_FEATURES = {
    "person": {"person", "worker", "human", "operator", "employee", "staff", "man", "woman"},
    "vest": {"vest", "hivis", "highvis", "reflective", "safetyvest"},
    "helmet": {"helmet", "hardhat"},
    "yellow": {"yellow", "lime"},
    "red": {"red"},
    "blue": {"blue", "teal"},
    "green": {"green"},
    "orange": {"orange"},
    "black": {"black"},
    "white": {"white"},
    "purple": {"purple", "violet"},
    "pink": {"pink"},
    "brown": {"brown"},
}


def _semantic_words(text: str) -> set[str]:
    t = (text or "").lower()
    t = t.replace("high-vis", "highvis").replace("hi-vis", "hivis")
    t = t.replace("hard hat", "hardhat").replace("safety vest", "safetyvest")
    return set(re.findall(r"[a-z0-9]+", t))


def _required_query_features(query: str) -> list[tuple[str, set[str]]]:
    words = _semantic_words(query)
    out = []
    for name, synonyms in _QUERY_FEATURES.items():
        if words & synonyms:
            out.append((name, synonyms))
    return out


def _text_satisfies_query(query: str, text: str) -> bool:
    """True when text preserves all explicit visual attributes in query."""
    required = _required_query_features(query)
    if not required:
        return True
    words = _semantic_words(text)
    return all(bool(words & synonyms) for _, synonyms in required)


def _query_equivalent(a: str, b: str) -> bool:
    """Loose semantic equivalence for cross-tool scan/navigation references."""
    if not a or not b:
        return False
    return _text_satisfies_query(a, b) and _text_satisfies_query(b, a)


def _valid_range(value) -> bool:
    """Reject missing, implausible and non-finite VLM ranges for arrival."""
    try:
        return math.isfinite(float(value)) and 0.3 < float(value) < 30.0
    except (ValueError, TypeError):
        return False


def _confirmed_standoff(confirmed: bool, visual_distance, limit_m: float) -> bool:
    """A YES identity alone cannot establish that the robot reached the standoff."""
    return bool(confirmed) and _valid_range(visual_distance) and float(visual_distance) <= float(limit_m)


def _current_scan_reference(memory: dict, *, now=None, ttl_s: float = 300.0) -> bool:
    if not isinstance(memory, dict) or not memory.get("target_query") or not memory.get("best_target"):
        return False
    created = memory.get("created_monotonic")
    try:
        age = (time.monotonic() if now is None else float(now)) - float(created)
    except (ValueError, TypeError):
        return False
    return 0.0 <= age <= ttl_s


def _wrap_deg(angle: float) -> float:
    a = float(angle)
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a

# ════════════════════════════════════════════════════════════
# POSITION / BASIC MOTION
# ════════════════════════════════════════════════════════════
@tool
def get_robot_position() -> str:
    """Get the current position and heading of the robot."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="get_robot_position")
    return (f"x={state['x']:.3f}m, y={state['y']:.3f}m, "
            f"heading={state['yaw']:.1f}deg")


@tool
def move_forward(distance_meters: float) -> str:
    """Move robot forward (positive) or backward (negative) by given metres."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="move_forward", distance_m=distance_meters)
    state["status"] = f"moving {distance_meters}m"
    log(f"Moving {distance_meters}m", "ROBOT")
    d = _move_dist(distance_meters)
    state["status"] = "idle"
    outcome = state.get("last_motion_outcome", "UNKNOWN")
    reason = state.get("last_motion_reason", "")
    record_event("TOOL_RESULT", tool="move_forward", outcome=outcome, reason=reason, actual_distance_m=d)
    return (f"OUTCOME={outcome}; moved={d:.2f}m; reason={reason}; "
            f"pose=({state['x']:.2f},{state['y']:.2f})")


@tool
def rotate_robot(degrees: float) -> str:
    """Rotate robot left (+) or right (-) by degrees."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="rotate_robot", degrees=degrees)
    state["status"] = f"rotating {degrees}deg"
    log(f"Rotating {degrees}deg", "ROBOT")
    _rotate(degrees)
    state["status"] = "idle"
    return f"Rotated {degrees}deg. Facing {state['yaw']:.1f}deg"


@tool
def stop_robot() -> str:
    """Immediately stop the robot."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="stop_robot")
    trigger_estop("agent_stop_tool")
    state["status"] = "idle"
    return "Stopped."


# ════════════════════════════════════════════════════════════
# VISION TOOLS
# ════════════════════════════════════════════════════════════
@tool
def describe_current_view() -> str:
    """Use VLM to describe what the robot's camera currently sees."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="describe_current_view")
    img = state["image_raw"]
    if not img:
        return "No camera image available."
    state["status"] = "analyzing view"
    log(f"VLM scene description [{vlm_mgr.active_name}]...", "VLM")
    with timed("describe_current_view"):
        result = vlm_mgr.ask(
            (
                "Describe only what is visibly present in the current robot camera "
                "frame. Prioritize objects, their relative positions, free space, "
                "and obstacles useful for robot operation. Avoid speculation about "
                "intent, gestures, why a person is posing a certain way, or why the "
                "scene appears synthetic unless the user explicitly asks."
            ),
            img, max_tokens=400,
        )
    state["last_vlm"] = result
    state["status"]   = "idle"
    log(f"VLM: {result[:80]}", "VLM")
    return result


@tool
def find_object(object_name: str) -> str:
    """Find a specific object in the camera view and report its position."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="find_object", object_name=object_name)
    img = state["image_raw"]
    if not img:
        return "No camera image."
    log(f"Finding '{object_name}' [{vlm_mgr.active_name}]", "VLM")
    with timed("vlm_find_object"):
        result = vlm_mgr.ask(
            (
                f"Check whether '{object_name}' is visible in this single image. "
                "Report only visible evidence. If multiple matching instances are "
                "present, mention each separately by left/center/right and visible "
                "distinguishing features; do not choose a navigation target and do "
                "not infer human intent or gestures. If none is visible, say so."
            ),
            img, max_tokens=200,
        )
    state["last_vlm"] = result
    return f"'{object_name}': {result}"


# ════════════════════════════════════════════════════════════
# SEMANTIC NAVIGATION
# ════════════════════════════════════════════════════════════
@tool
def navigate_to_object(object_name: str) -> str:
    """Navigate to a named target using VLM plan-execute-recheck control.

    Final reviewer version:
      * VLM is the PRIMARY semantic navigator: same-instance selection,
        coarse direction, rough distance, re-planning, and final confirmation.
      * YOLO is OPTIONAL supporting evidence. Losing a detector frame never
        aborts navigation by itself.
      * Episode visual memory freezes the VLM-selected instance signature; if
        that instance disappears, another same-class object is never substituted.
      * The platform safety layer remains authoritative for collision/standoff:
        Carter uses LiDAR first; Scout (no LiDAR) uses monocular depth.
      * Motion is executed in meaningful odometry-controlled legs rather than
        frame-by-frame visual servoing, so VLM calls and turns stay low.

    This deliberately follows the original working V-ROSA plan-execute-recheck
    behaviour while retaining the newer fail-closed motion/safety logging.
    """
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1

    # Resolve conversational references through the most recent targeted 360°
    # scan.  This makes "go to that person" refer to the object just found by
    # look_around(query=...), not to whichever same-class object is currently
    # closest to the camera.
    requested_object_name = str(object_name)
    scan_memory = state.get("scene_scan_memory") or {}
    is_reference = requested_object_name.strip().lower() in {
        "that person", "that worker", "that object", "that target",
        "it", "there", "go there",
    }
    if is_reference:
        if not _current_scan_reference(scan_memory):
            _stop()
            record_event("TARGET_REFERENCE_UNRESOLVED", original_query=requested_object_name,
                         reason="no_recent_confirmed_scan_target")
            state["last_nav_outcome"] = "TARGET_REFERENCE_UNRESOLVED"
            state["last_nav_estimated_final_distance"] = None
            return ("OUTCOME=TARGET_REFERENCE_UNRESOLVED; NAVIGATION_TERMINATED=true; "
                    "No recent visually matched scan target; the robot did not navigate.")
        object_name = str(scan_memory["target_query"])
        record_event("TARGET_REFERENCE_RESOLVED", original_query=requested_object_name,
                     resolved_query=object_name, source="scene_scan_memory")

    record_event("TOOL_CALL", tool="navigate_to_object", object_name=object_name)
    state["status"] = f"navigating to {object_name}"
    state["target"] = object_name
    state["last_target"] = object_name
    state["last_nav_outcome"] = "RUNNING"
    state["last_nav_estimated_final_distance"] = None
    state["nav_visual_distance_m"] = None
    state["sim_visual_fallback_used_m"] = 0.0

    log(f"Semantic nav → '{object_name}' [{vlm_mgr.active_name}]", "VROSA")
    record_event(
        "NAV_START", target=object_name,
        x=state["x"], y=state["y"], yaw=state["yaw"],
        controller="vlm_plan_execute_recheck",
        sim_visual_clearance_test=bool(ROBOT.get("sim_visual_clearance_test", False)),
    )

    synonyms = {
        "person": ["person", "worker", "human", "man", "woman", "people",
                   "warehouse worker", "operator", "staff", "employee", "avatar"],
        "forklift": ["forklift", "lift truck", "industrial vehicle", "truck"],
        "shelf": ["shelf", "shelving", "rack", "racking", "storage rack"],
        "box": ["box", "carton", "package", "cardboard", "crate"],
        "door": ["door", "gate", "entrance", "exit"],
        "barrel": ["barrel", "drum", "container", "cylinder"],
    }

    name = object_name.lower().strip()
    target_labels = [name]
    for key, vals in synonyms.items():
        if name == key or name in vals:
            target_labels = vals + [key]
            break

    # Visual-agent terminal standoff.  Two metres keeps the full target visible
    # for semantic confirmation and avoids close-range crop/occlusion failures.
    # For V-ROSA semantic navigation, successful VLM verification at this
    # standoff IS the arrival condition.
    verify_dist = float(ROBOT.get("target_verify_dist", 2.0))
    arrival_dist = min(verify_dist, float(ROBOT.get("runtime_arrive_dist", verify_dist)),
                       float(ROBOT.get("arrive_dist", verify_dist)))
    verify_trigger = verify_dist + 0.20
    arrival_trigger = verify_trigger

    # Preserve the original proven coarse 40/20 degree heading behavior as
    # a fallback when a VLM cannot supply a target image location.  When a
    # TARGET_X coordinate is available, camera geometry drives the heading.
    turn_map = {
        "HARD_LEFT": 40.0,
        "LEFT": 20.0,
        "CENTER": 0.0,
        "RIGHT": -20.0,
        "HARD_RIGHT": -40.0,
        "ARRIVED": 0.0,
    }

    def image_target_heading(target_x):
        """Turn toward a VLM-selected target's image x, NOT a YOLO person box.

        TARGET_X is a normalized horizontal image position in [0, 1].
        Camera intrinsics from the live CameraInfo topic allow a pinhole yaw
        correction; calibration-free fallback uses the configured camera FoV.
        Positive degrees rotate left, negative rotate right.  One correction
        is capped at 30 degrees per VLM planning leg, and zeroed near center.
        """
        if target_x is None:
            return None
        try:
            target_x = float(target_x)
        except (ValueError, TypeError):
            return None
        if not math.isfinite(target_x) or not (0.0 <= target_x <= 1.0):
            return None
        width = float(state.get("camera_width") or 0.0)
        fx = float(state.get("cam_fx") or 0.0)
        cx = state.get("cam_cx")
        if width > 0 and fx > 0 and cx is not None:
            yaw_deg = -math.degrees(math.atan2(target_x * width - float(cx), fx))
        else:
            # Fallback is an angular approximation, NOT calibrated geometry.
            fov = float(state.get("camera_hfov_deg") or 90.0)
            yaw_deg = -(target_x - 0.5) * fov
        if abs(yaw_deg) <= 5.0:
            return 0.0
        return max(-30.0, min(30.0, yaw_deg))

    def side_from_heading(yaw_deg):
        if yaw_deg >= 5.0:
            return "LEFT" if yaw_deg < 25.0 else "HARD_LEFT"
        if yaw_deg <= -5.0:
            return "RIGHT" if yaw_deg > -25.0 else "HARD_RIGHT"
        return "CENTER"

    max_legs = int(ROBOT.get("max_legs", 8))
    identity_verified = False
    target_signature = ""
    last_direction = "CENTER"
    last_vlm_distance = None
    total_moved = 0.0
    transient_odom_recoveries = 0
    transient_sensor_recoveries = 0
    start_vlm_queries = int(state.get("vlm_query_count", 0))

    # Cross-tool scene memory from the latest targeted 360° scan.  It is only a
    # heading/identity hint: the VLM must visually confirm the target after the
    # robot turns toward it before any translation occurs.
    scan_hint = None
    scan_memory = state.get("scene_scan_memory") or {}
    scan_age = (time.monotonic() - float(scan_memory.get("created_monotonic"))) if _current_scan_reference(scan_memory) else None
    candidate = scan_memory.get("best_target")
    if (_current_scan_reference(scan_memory) and candidate
            and _query_equivalent(object_name, str(scan_memory.get("target_query", "")))):
        scan_hint = dict(candidate)

        # Prefer the approximate world point because it remains useful even if
        # the robot translated after the scan.  Fall back to the remembered yaw.
        desired_yaw = scan_hint.get("target_yaw_deg")
        wx, wy = scan_hint.get("world_x"), scan_hint.get("world_y")
        if wx is not None and wy is not None:
            dx = float(wx) - float(state.get("x", 0.0))
            dy = float(wy) - float(state.get("y", 0.0))
            if math.hypot(dx, dy) > 0.25:
                desired_yaw = math.degrees(math.atan2(dy, dx))

        if desired_yaw is not None:
            turn = _wrap_deg(float(desired_yaw) - float(state.get("yaw", 0.0)))
            record_event(
                "SCAN_MEMORY_RECALLED",
                target=object_name,
                scan_label=scan_hint.get("scan_label"),
                target_yaw_deg=desired_yaw,
                relative_turn_deg=turn,
                scan_distance_m=scan_hint.get("distance_m"),
                signature=scan_hint.get("signature"),
                age_s=scan_age,
            )
            # Face the remembered target before the initial VLM scout.  This is
            # a single memory-guided orientation, not blind memory-based motion.
            if abs(turn) > 4.0:
                _rotate(turn)
                if state.get("last_rotation_outcome") != "SUCCESS":
                    _stop()
                    state["last_nav_outcome"] = "SCAN_ALIGNMENT_FAILED"
                    record_event("SCAN_MEMORY_ALIGN_FAILED", outcome=state.get("last_rotation_outcome"))
                    return ("OUTCOME=SCAN_ALIGNMENT_FAILED; NAVIGATION_TERMINATED=true; "
                            "Could not safely align with remembered target; no translation executed.")
                time.sleep(0.20)
                record_event(
                    "SCAN_MEMORY_ALIGN",
                    target=object_name,
                    degrees=turn,
                    resulting_yaw=float(state.get("yaw", 0.0)),
                )

    # Episode-level visual target memory. This is intentionally lightweight:
    # it remembers WHICH semantic instance the VLM selected during this
    # navigation episode, but it does not create a map or override live vision.
    # The identity anchor is immutable after initial selection so a later
    # same-class object can never silently replace it.
    previous_memory = state.get("visual_target_memory") or {}
    target_memory = {
        "active": True,
        "target_query": object_name,
        "initial_signature": "",
        "last_observed_signature": "",
        "last_direction": "CENTER",
        "last_distance_m": None,
        "last_seen_image_seq": state.get("image_seq"),
        "last_seen_pose": {
            "x": float(state.get("x", 0.0)),
            "y": float(state.get("y", 0.0)),
            "yaw": float(state.get("yaw", 0.0)),
        },
        "consecutive_not_visible": 0,
        "identity_confirmed": False,
        "status": "selecting",
        "scan_memory_hint": scan_hint,
        "created_monotonic": time.monotonic(),
    }
    state["visual_target_memory"] = target_memory
    record_event(
        "TARGET_MEMORY_CREATED",
        target=object_name,
        previous_target=previous_memory.get("target_query"),
        policy="immutable_vlm_signature",
    )

    def update_target_memory(*, direction=None, distance=None, observed_signature=None,
                             visible=True, status=None, confirmed=None):
        mem = state.get("visual_target_memory") or target_memory
        if direction in {"HARD_LEFT", "LEFT", "CENTER", "RIGHT", "HARD_RIGHT"}:
            mem["last_direction"] = direction
        if distance is not None:
            mem["last_distance_m"] = float(distance)
        if observed_signature:
            mem["last_observed_signature"] = str(observed_signature)[:180]
        if visible:
            mem["last_seen_image_seq"] = state.get("image_seq")
            mem["last_seen_pose"] = {
                "x": float(state.get("x", 0.0)),
                "y": float(state.get("y", 0.0)),
                "yaw": float(state.get("yaw", 0.0)),
            }
            mem["consecutive_not_visible"] = 0
        else:
            mem["consecutive_not_visible"] = int(mem.get("consecutive_not_visible", 0)) + 1
        if status is not None:
            mem["status"] = status
        if confirmed is not None:
            mem["identity_confirmed"] = bool(confirmed)
        state["visual_target_memory"] = mem
        return mem

    def finish(outcome: str, message: str, dist=None):
        _stop()
        state["status"] = "idle"
        state["target"] = ""
        state["last_nav_outcome"] = outcome
        state["last_nav_estimated_final_distance"] = dist
        state["nav_visual_distance_m"] = None  # never authorize later manual moves
        mem = state.get("visual_target_memory") or target_memory
        mem["active"] = False
        mem["status"] = "arrived" if outcome == "ARRIVED" else f"ended:{outcome}"
        mem["final_distance_m"] = dist
        mem["ended_monotonic"] = time.monotonic()
        state["visual_target_memory"] = mem
        state["last_visual_target_memory"] = dict(mem)
        record_event(
            "TARGET_MEMORY_CLOSED",
            target=object_name,
            outcome=outcome,
            signature=mem.get("initial_signature"),
            last_direction=mem.get("last_direction"),
            last_distance_m=mem.get("last_distance_m"),
            identity_confirmed=mem.get("identity_confirmed", False),
        )
        record_event(
            "NAV_END",
            target=object_name,
            outcome=outcome,
            estimated_final_distance_m=dist,
            distance_source=("vlm_visual_estimate" if dist is not None else None),
            vlm_queries=int(state.get("vlm_query_count", 0)) - start_vlm_queries,
            replans=state.get("replan_count", 0),
            total_moved_m=total_moved,
            sim_visual_fallback_used_m=state.get("sim_visual_fallback_used_m", 0.0),
            x=state["x"], y=state["y"], yaw=state["yaw"],
        )
        dist_text = "unknown" if dist is None else f"{float(dist):.2f}m"
        return (
            f"OUTCOME={outcome}; NAVIGATION_TERMINATED=true; "
            f"FINAL_TARGET_DISTANCE={dist_text}; "
            "Do not automatically call navigate_to_object again for this user command. "
            f"{message}"
        )

    def vlm_failed(text: str) -> bool:
        t = (text or "").lower()
        return (
            not t
            or "api error" in t
            or "remote error" in t
            or "inference error" in t
            or "not ready" in t
            or "timeout" in t
        )

    def parse_vlm_nav(resp: str):
        text = (resp or "").strip()
        match = re.search(
            r"^\s*DIRECTION\s*:\s*(HARD_LEFT|HARD_RIGHT|NOT_VISIBLE|ARRIVED|CENTER|LEFT|RIGHT)\b",
            text, re.IGNORECASE | re.MULTILINE)
        direction = match.group(1).upper() if match else None
        match = re.search(r"^\s*DISTANCE_M\s*:\s*(?:~\s*)?(\d+(?:\.\d+)?)\b",
                          text, re.IGNORECASE | re.MULTILINE)
        distance = float(match.group(1)) if match and _valid_range(match.group(1)) else None

        signature = ""
        for line in text.splitlines():
            if "SIGNATURE" in line.upper() or "DESCRIPTION" in line.upper():
                signature = line.split(":", 1)[-1].strip()
                break
        x_match = re.search(r"^\s*TARGET_X\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)\b",
                            text, re.IGNORECASE | re.MULTILINE)
        target_x = float(x_match.group(1)) if x_match else None
        return direction, distance, signature[:180], target_x

    # The scan-selected identity is a prior anchor, not permission to move.
    # First live VLM view must match it before any translation.
    if scan_hint and scan_hint.get("signature"):
        target_signature = str(scan_hint["signature"])[:180]
        target_memory["initial_signature"] = target_signature
        target_memory["status"] = "scan_identity_pending_live_confirmation"

    def ask_vlm_nav(initial: bool):
        img = state.get("image_raw")
        if img is None:
            return None, None, "", None, "NO_IMAGE"

        if initial:
            scan_context = ""
            if scan_hint:
                scan_context = (
                    f"A recent 360-degree scan specifically found this requested target at "
                    f"{scan_hint.get('scan_label', 'a remembered direction')}, approximately "
                    f"{scan_hint.get('distance_m', 'unknown')} m away, with signature: "
                    f"{scan_hint.get('signature', object_name)}. The robot has already turned toward that memory. "
                    "Use this only as a search hint and visually verify the requested attributes now. "
                )
            prompt = (
                f"Find exactly one instance of the requested target '{object_name}' in this robot camera image. "
                + scan_context +
                "Every explicit attribute in the user request is a HARD identity constraint (for example color, vest, helmet). "
                "If scan memory supplied a signature, select ONLY the visually matching remembered instance, "
                "including its visible discriminating attributes, not another same-class instance. "
                "Never substitute a different same-class object that lacks those attributes. If no matching instance is visible, "
                "return NOT_VISIBLE. If multiple valid matches are visible, choose the nearest / most visually prominent valid one "
                "and keep that single instance as the navigation target. Estimate distance from visual perspective and apparent size; "
                "the estimate only needs to be useful for coarse robot planning.\n"
                "Direction MUST describe where the matching person's horizontal center is in the IMAGE: "
                "LEFT if in the left third, CENTER ONLY if in the middle third, RIGHT if in the right third. "
                "HARD_LEFT / HARD_RIGHT if near a frame edge. Do not say CENTER just because the target is visible. "
                "TARGET_X must be the matching instance's horizontal center / image width, between 0.0 (left edge) "
                "and 1.0 (right edge). Never use another nearby person's horizontal location.\n"
                "Reply in exactly four lines:\n"
                "DIRECTION: HARD_LEFT, LEFT, CENTER, RIGHT, HARD_RIGHT, NOT_VISIBLE, or ARRIVED\n"
                "DISTANCE_M: one number in metres (99 only if not visible)\n"
                "SIGNATURE: at most 12 words describing the visible same instance\n"
                "TARGET_X: fraction 0.0 to 1.0 (or 99 if not visible)"
            )
        else:
            prompt = (
                f"Continue navigating to the SAME previously selected '{object_name}'. "
                f"IMMUTABLE target memory/signature: {target_signature}. "
                f"Last known direction: {target_memory.get('last_direction', last_direction)}. "
                f"Last visual distance: {target_memory.get('last_distance_m', last_vlm_distance)}. "
                "This identity memory belongs to one specific instance and must not be replaced. "
                "If a different object of the same class is visible but the remembered instance is not, output NOT_VISIBLE. "
                "If the remembered target is partly cropped because the robot is close, use visual continuity, clothing/appearance, "
                "last direction, and scene context to recognize it; do not switch targets. "
                "Estimate its current distance using perspective and apparent size.\n"
                "DIRECTION must describe the horizontal image location of the SAME remembered target: "
                "LEFT in left third, CENTER ONLY in middle third, RIGHT in right third; use HARD_LEFT/RIGHT near edges. "
                "Do not confuse visible with centered. Give its normalized horizontal center TARGET_X, "
                "0.0 left edge to 1.0 right edge; do not use the center of another person.\n"
                "Reply in exactly four lines:\n"
                "DIRECTION: HARD_LEFT, LEFT, CENTER, RIGHT, HARD_RIGHT, NOT_VISIBLE, or ARRIVED\n"
                "DISTANCE_M: one number in metres (99 only if not visible)\n"
                "SIGNATURE: repeat/update only the visible signature of that same instance\n"
                "TARGET_X: fraction 0.0 to 1.0 (or 99 if not visible)"
            )

        stage = "vlm_nav_recheck" if not initial else "vlm_nav_scout"
        with timed(stage, target=object_name):
            resp = vlm_mgr.ask(prompt, img, max_tokens=110)
        if vlm_failed(resp):
            # One retry while stopped protects a navigation episode from a
            # transient API/backend hiccup without changing the motion plan.
            time.sleep(0.35)
            img = state.get("image_raw") or img
            record_event("VLM_NAV_RETRY", target=object_name, stage=stage)
            with timed(stage + "_retry", target=object_name):
                resp = vlm_mgr.ask(prompt, img, max_tokens=110)
        if vlm_failed(resp):
            return None, None, "", None, "VLM_ERROR"
        direction, distance, signature, target_x = parse_vlm_nav(resp)
        if direction is None:
            return None, distance, signature, target_x, "MALFORMED"
        if distance is not None and distance >= 90:
            distance = None
        return direction, distance, signature, target_x, "OK"

    def is_target(det: dict) -> bool:
        label = str(det.get("label", "")).lower()
        return any(s in label or label in s for s in target_labels)

    def detector_support(direction: str = ""):
        """Optional YOLO evidence only; never a navigation precondition."""
        dets = [d for d in state.get("detections", []) if is_target(d)]
        if not dets:
            return None

        sector_map = {
            "HARD_LEFT": "left", "LEFT": "left", "CENTER": "center",
            "RIGHT": "right", "HARD_RIGHT": "right",
        }
        sec = sector_map.get(direction)
        if sec:
            guided = [d for d in dets if d.get("sector") == sec]
            if guided:
                dets = guided

        def area(d):
            try:
                x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
                return max(1.0, (x2-x1)*(y2-y1))
            except Exception:
                return 1.0
        return max(dets, key=lambda d: (area(d), float(d.get("conf", 0.0))))

    def record_semantic_target(direction, vlm_dist, signature, initial=False):
        det = detector_support(direction)
        payload = {
            "target": object_name,
            "policy": "vlm_semantic_signature_with_optional_yolo",
            "direction": direction,
            "vlm_distance_m": vlm_dist,
            "signature": signature,
            "yolo_support": det is not None,
        }
        if det is not None:
            payload.update({
                "bbox": det.get("bbox"),
                "confidence": det.get("conf"),
                "image_seq": det.get("image_seq"),
            })
        record_event("TARGET_INSTANCE_SELECTED" if initial else "TARGET_SEMANTIC_RECHECK", **payload)

    def confirm_same_target(distance_hint=None, reason="verification"):
        """VLM confirmation on the current view; YOLO crop is optional, not required."""
        img = state.get("image_raw")
        if img is None:
            return False, None

        # Confirm on the CURRENT full camera frame. YOLO is supporting evidence
        # only; using an older detector bbox to crop a newer frame can create a
        # false identity failure when inference is slow.
        det = detector_support(last_direction)
        view = img
        used_crop = False

        hint = "unknown" if distance_hint is None else f"approximately {float(distance_hint):.2f} m"
        with timed("vlm_target_identity", target=object_name):
            resp = vlm_mgr.ask(
                (
                    f"Confirm the navigation target. Requested class: '{object_name}'. "
                    f"IMMUTABLE target memory/signature: {target_signature or object_name}. "
                    f"Last known direction: {target_memory.get('last_direction', last_direction)}. "
                    f"Current coarse distance hint: {hint}. "
                    "Confirm only this remembered instance. If another same-class object is visible but the remembered target is not, "
                    "reply MATCH: NO. If the remembered target is partly cropped due to proximity, use appearance and scene continuity. "
                    "At about 2 metres, MATCH: YES means visual navigation is complete.\n"
                    "Reply exactly two lines:\nMATCH: YES or NO\nDISTANCE_M: your best visual estimate in metres"
                ),
                view,
                max_tokens=30,
            )
        if vlm_failed(resp):
            # Retry identity confirmation once while the robot remains stopped.
            time.sleep(0.35)
            record_event("VLM_IDENTITY_RETRY", target=object_name, reason=reason)
            with timed("vlm_target_identity_retry", target=object_name):
                resp = vlm_mgr.ask(
                    (
                        f"Confirm the navigation target. Requested class: '{object_name}'. "
                        f"IMMUTABLE target memory/signature: {target_signature or object_name}. "
                        f"Last known direction: {target_memory.get('last_direction', last_direction)}. "
                        f"Current coarse distance hint: {hint}. "
                        "Confirm only this remembered instance at the visual arrival standoff. If a different same-class object is visible, "
                        "reply MATCH: NO rather than substituting it.\n"
                        "Reply exactly two lines:\nMATCH: YES or NO\nDISTANCE_M: your best visual estimate in metres"
                    ),
                    state.get("image_raw") or view,
                    max_tokens=30,
                )
        if vlm_failed(resp):
            record_event("TARGET_IDENTITY_VERIFICATION", target=object_name, reason=reason,
                         response="VLM_ERROR", used_crop=used_crop)
            return False, None
        match_line = re.search(r"^\s*MATCH\s*:\s*(YES|NO)\b", resp,
                               re.IGNORECASE | re.MULTILINE)
        yes = bool(match_line and match_line.group(1).upper() == "YES")
        _, d, _, _ = parse_vlm_nav(resp)
        record_event(
            "TARGET_IDENTITY_VERIFICATION",
            target=object_name,
            reason=reason,
            response=resp.strip()[:160],
            confirmed=yes,
            visual_distance_m=d,
            used_crop=used_crop,
        )
        return yes, d

    def wait_odom_recovery(timeout_s=1.2):
        deadline = time.monotonic() + timeout_s
        max_age = max(0.30, float(ROBOT.get("odom_max_age_s", 0.30)))
        while time.monotonic() < deadline:
            ts = state.get("odom_ts")
            if ts is not None and time.monotonic() - float(ts) <= max_age:
                return True
            time.sleep(0.05)
        return False

    def wait_safety_recovery(timeout_s=1.5):
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            last = check_obstacle(1)
            if last.valid:
                return last
            time.sleep(0.08)
        return last

    for leg in range(max_legs):
        if interrupt_flag.is_set():
            return finish("INTERRUPTED", f"Navigation to '{object_name}' interrupted.", last_vlm_distance)

        initial = (leg == 0)
        direction, vlm_dist, signature, target_x, status = ask_vlm_nav(initial=initial)

        if status == "VLM_ERROR":
            return finish("VLM_ERROR", f"The VLM became unavailable while navigating to '{object_name}'.", last_vlm_distance)
        if status in {"NO_IMAGE", "MALFORMED"}:
            return finish("VLM_FORMAT_ERROR", f"Could not obtain a usable visual navigation plan for '{object_name}'.", last_vlm_distance)

        # Deterministic attribute lock on initial selection.  This specifically
        # prevents e.g. "person with yellow vest" from being satisfied by the
        # nearer red/white-vest worker.  Once identity is frozen, later partial
        # views are handled by the VLM memory prompt instead of lexical checks.
        if initial and direction not in {"NOT_VISIBLE", None}:
            scan_signature = str((scan_hint or {}).get("signature") or "")
            attributes_match = bool(signature.strip()) and _text_satisfies_query(object_name, signature)
            scan_match = (not scan_signature or _text_satisfies_query(scan_signature, signature))
            if not (attributes_match and scan_match):
                record_event(
                    "TARGET_ATTRIBUTE_MISMATCH",
                    target=object_name,
                    vlm_direction=direction,
                    vlm_distance_m=vlm_dist,
                    signature=signature,
                    required_features=[name for name, _ in _required_query_features(object_name)],
                    remembered_scan_signature=(scan_hint or {}).get("signature"),
                )
                direction = "NOT_VISIBLE"
                vlm_dist = None
                signature = ""

        # A pixel-coordinate and coarse direction can disagree. The numeric
        # target position is from the same VLM-selected instance; if the two
        # explicitly point in OPPOSITE directions, do not translate on ambiguous
        # evidence. Recheck while stopped rather than steering to a distractor.
        visual_turn = image_target_heading(target_x) if direction != "NOT_VISIBLE" else None
        visual_side = side_from_heading(visual_turn) if visual_turn is not None else None
        if visual_side is not None and direction in {"LEFT", "HARD_LEFT", "RIGHT", "HARD_RIGHT"}:
            opposite = ((direction.endswith("LEFT") and visual_side.endswith("RIGHT")) or
                        (direction.endswith("RIGHT") and visual_side.endswith("LEFT")))
            disagree = opposite or visual_side == "CENTER"
            if disagree:
                record_event("NAV_DIRECTION_CONFLICT", leg=leg + 1, target=object_name,
                             coarse_direction=direction, target_x=target_x,
                             geometric_direction=visual_side,
                             reason=("opposite_visual_cues" if opposite else "center_vs_side"))
                _stop(wait=True, direction=1)
                state["replan_count"] = int(state.get("replan_count", 0)) + 1
                continue

        if signature and direction != "NOT_VISIBLE":
            if not target_signature:
                # Freeze the first VLM-selected instance description as the
                # identity anchor for this entire navigation episode.
                target_signature = signature[:180]
                target_memory["initial_signature"] = target_signature
            target_memory["last_observed_signature"] = signature[:180]

        if direction in turn_map and direction != "ARRIVED":
            # Memory must store the actual geometric side when it differs
            # from a coarse CENTER response, so recovery can rotate correctly.
            last_direction = visual_side if visual_side not in {None, "CENTER"} else direction
        if vlm_dist is not None:
            last_vlm_distance = float(vlm_dist)
            state["last_nav_estimated_final_distance"] = last_vlm_distance
            state["nav_visual_distance_m"] = last_vlm_distance

        update_target_memory(
            direction=last_direction,
            distance=vlm_dist,
            observed_signature=signature,
            visible=(direction != "NOT_VISIBLE"),
            status="tracking" if direction != "NOT_VISIBLE" else "temporarily_lost",
        )
        if direction != "NOT_VISIBLE":
            record_semantic_target(direction, vlm_dist, target_signature, initial=initial)
        record_event(
            "NAV_RECHECK",
            leg=leg + 1,
            target=object_name,
            direction=direction,
            target_distance_m=vlm_dist,
            distance_source="vlm_visual_estimate",
            identity_verified=identity_verified,
            target_x=target_x,
            effective_direction=last_direction,
        )

        if direction == "NOT_VISIBLE":
            # Never follow another same-class YOLO detection when the remembered
            # target is missing. Recover only from the target memory.
            mem = state.get("visual_target_memory") or target_memory
            misses = int(mem.get("consecutive_not_visible", 0))
            remembered_dir = mem.get("last_direction") or last_direction
            record_event(
                "TARGET_MEMORY_MISS",
                target=object_name,
                misses=misses,
                remembered_direction=remembered_dir,
                signature=target_signature,
            )
            if misses >= 2:
                return finish(
                    "TARGET_LOST",
                    f"The remembered '{object_name}' was not visible in two consecutive VLM rechecks. "
                    "The robot stopped instead of switching to another same-class object.",
                    last_vlm_distance,
                )

            # One small memory-directed look-back while stopped. If the last
            # target was centered, make one bounded search turn rather than
            # leaving the target outside the FoV for a second identical image.
            recovery_turn = {
                "HARD_LEFT": 12.0, "LEFT": 8.0, "CENTER": 10.0,
                "RIGHT": -8.0, "HARD_RIGHT": -12.0,
            }.get(remembered_dir, 0.0)
            if recovery_turn:
                record_event(
                    "NAV_ALIGN", leg=leg + 1, direction="MEMORY_RECOVERY",
                    degrees=recovery_turn, signature=target_signature,
                )
                _rotate(recovery_turn)
                if state.get("last_rotation_outcome") != "SUCCESS":
                    return finish("ODOM_STALE", "Memory recovery rotation could not complete safely.", last_vlm_distance)
                time.sleep(0.15)
            else:
                time.sleep(0.25)
            continue

        if initial and scan_hint:
            _stop(wait=True, direction=1)
            matching, _ = confirm_same_target(vlm_dist, reason="scan_instance_lock")
            if not matching:
                return finish("TARGET_VERIFY_FAILED",
                              "Scan-selected instance could not be confirmed in the live image.", vlm_dist)
            record_event("SCAN_IDENTITY_LOCK_CONFIRMED", target=object_name,
                         scan_signature=(scan_hint or {}).get("signature"))

        # If VLM already judges us close enough for the intermediate stop,
        # verify identity before any further translation.
        if not identity_verified and (
            direction == "ARRIVED" or (vlm_dist is not None and vlm_dist <= verify_trigger)
        ):
            _stop(wait=True, direction=1)
            record_event(
                "TARGET_VERIFY_STOP", target=object_name,
                distance_m=vlm_dist, distance_source="vlm_visual_estimate",
                verify_distance_m=verify_dist,
            )
            ok, visual_d = confirm_same_target(vlm_dist, reason="vlm_verify_standoff")
            if not ok:
                return finish("TARGET_VERIFY_FAILED",
                              f"VLM did not confirm the remembered '{object_name}'.", last_vlm_distance)
            if not _confirmed_standoff(ok, visual_d, arrival_dist):
                record_event("ARRIVAL_UNCONFIRMED", target=object_name,
                             coarse_distance_m=vlm_dist, verified_distance_m=visual_d,
                             threshold_m=arrival_dist, reason="range_not_within_threshold")
                return finish("ARRIVAL_UNCONFIRMED",
                              "Target identity found, but verified visual distance is not within 2.0 m; stopped without closer motion.",
                              visual_d)
            identity_verified = True
            update_target_memory(distance=visual_d, status="verified_arrival", confirmed=True)
            final_d = float(visual_d)
            last_vlm_distance = float(final_d)
            record_event(
                "TARGET_IDENTITY_VERIFIED", target=object_name,
                distance_m=last_vlm_distance,
                distance_source="vlm_visual_estimate",
                signature=target_signature,
            )
            # TERMINAL VISUAL ARRIVAL: once the same target is VLM-confirmed
            # at the ~2 m semantic standoff, do not drive closer.  This preserves
            # the full target in view and avoids unnecessary extra motion.
            return finish(
                "ARRIVED",
                f"Arrived at the VLM-verified '{object_name}' at the {verify_dist:.2f}m visual standoff "
                f"(VLM estimate {last_vlm_distance:.2f}m).",
                last_vlm_distance,
            )

        # Exactly one coarse alignment per planning leg. No frame-by-frame
        # left/right visual servoing.
        rot_deg = visual_turn if visual_turn is not None else turn_map.get(direction, 0.0)
        if visual_turn is not None:
            record_event(
                "TARGET_IMAGE_ALIGNMENT", leg=leg + 1, target=object_name,
                vlm_direction=direction, target_x=target_x,
                corrected_direction=visual_side, calculated_turn_deg=rot_deg,
                camera_width=state.get("camera_width"), camera_fx=state.get("cam_fx"),
                source="vlm_same_instance_target_x",
            )
        if rot_deg:
            record_event(
                "NAV_ALIGN", leg=leg + 1,
                direction=(visual_side if visual_turn is not None else direction),
                vlm_direction=direction, degrees=rot_deg,
                distance_source=("vlm_target_image_geometry" if visual_turn is not None else "vlm_visual_plan"),
            )
            _rotate(rot_deg)
            if state.get("last_rotation_outcome") != "SUCCESS":
                _stop(wait=True, direction=1)
                # Stale odometry midway through a turn must never cause blind
                # repeat of the original degrees. If odometry recovers,
                # re-PLAN from a fresh VLM frame on the next leg instead.
                if (transient_odom_recoveries < 2 and wait_odom_recovery(1.2)):
                    transient_odom_recoveries += 1
                    state["replan_count"] = int(state.get("replan_count", 0)) + 1
                    state["recovery_success_count"] = int(state.get("recovery_success_count", 0)) + 1
                    record_event("NAV_ROTATION_ODOM_RECOVERED", leg=leg + 1,
                                 requested_turn_deg=rot_deg, current_yaw=state.get("yaw"),
                                 action="recheck_vlm_before_any_translation")
                    continue
                return finish("ODOM_STALE", "Navigation alignment rotation failed; stopped before translation.", last_vlm_distance)
            time.sleep(0.15)

        # VLM distance is the PRIMARY planning range. If it is temporarily
        # missing, use a short conservative leg rather than aborting on depth.
        goal_standoff = verify_dist
        if vlm_dist is None:
            leg_dist = 0.40
        else:
            gap = max(0.0, float(vlm_dist) - goal_standoff)
            if gap <= 0.08:
                # Next loop will perform the VLM verification/arrival check.
                continue

            # Original working controller moved ~70% of the remaining gap.
            # Keep that idea, but cap legs to avoid large errors from monocular
            # visual distance estimates.
            leg_dist = 0.70 * gap
            if vlm_dist > 5.0:
                leg_dist = min(2.00, leg_dist)
            elif vlm_dist > 3.5:
                leg_dist = min(1.50, leg_dist)
            else:
                leg_dist = min(0.80, leg_dist)
            leg_dist = max(0.25, leg_dist)

        # One generic safety API works for both platforms:
        #   Carter -> LiDAR first, depth fallback.
        #   Scout  -> depth (no LiDAR configured).
        reading = check_obstacle(1)
        if not reading.valid:
            reading = wait_safety_recovery(1.5)
        if reading is None or not reading.valid:
            # Only the explicitly enabled Isaac Sim test may attempt a short,
            # monitored camera-only creep. The low-level motion primitive will
            # independently request and validate a new VLM clearance decision.
            # Scout, near-target motion, and the default mode remain fail-closed.
            sim_visual_test = (
                ROBOT.get("sim_visual_clearance_test", False)
                and ROBOT.get("platform_label") == "Isaac Sim"
                and last_vlm_distance is not None
                and last_vlm_distance > ROBOT.get("sim_visual_min_target_range_m", 3.0)
            )
            if not sim_visual_test:
                return finish(
                    "SAFETY_SENSOR_UNAVAILABLE",
                    f"Navigation stopped because forward safety sensing is unavailable ({getattr(reading, 'reason', 'unknown')}).",
                    last_vlm_distance,
                )
            record_event("SIM_VISUAL_PREFLIGHT_PENDING", leg=leg + 1,
                         reason=getattr(reading, "reason", "unknown"))

        record_event(
            "NAV_ADVANCE",
            leg=leg + 1,
            planned_distance_m=leg_dist,
            target_distance_m=vlm_dist,
            distance_source="vlm_visual_estimate",
            goal_standoff_m=goal_standoff,
            safety_source=(reading.source if reading is not None and reading.valid else "sim_visual_test_pending"),
            safety_clearance_m=(reading.distance if reading is not None and reading.valid else None),
            identity_verified=identity_verified,
        )

        # Before identity verification, any obstacle at ~2 m becomes a VLM
        # verification opportunity. After verification, the final standoff is
        # enforced the same way. This is platform-neutral; it is not LiDAR logic.
        actual = _move_dist(leg_dist, safe_dist=goal_standoff)
        total_moved += max(0.0, float(actual))
        outcome = state.get("last_motion_outcome", "")

        if outcome == "SUCCESS":
            continue

        if outcome == "BLOCKED":
            _stop(wait=True, direction=1)
            close_reading = check_obstacle(1)
            close_d = (
                float(close_reading.distance)
                if close_reading.valid and close_reading.distance is not None
                else None
            )

            record_event(
                "TARGET_VERIFY_STOP" if not identity_verified else "FINAL_STANDOFF_STOP",
                target=object_name,
                distance_m=close_d,
                distance_source=(close_reading.source if close_reading.valid else "unknown"),
                verify_distance_m=(verify_dist if not identity_verified else arrival_dist),
                trigger="motion_safety_standoff",
            )

            ok, visual_d = confirm_same_target(close_d, reason="safety_standoff")
            if not ok:
                return finish(
                    "BLOCKED",
                    f"A physical obstacle stopped the robot before the target could be confirmed as '{object_name}'.",
                    last_vlm_distance,
                )

            if not _confirmed_standoff(ok, visual_d, arrival_dist):
                record_event("ARRIVAL_UNCONFIRMED", target=object_name,
                             coarse_distance_m=last_vlm_distance,
                             safety_clearance_m=close_d, verified_distance_m=visual_d,
                             threshold_m=arrival_dist, reason="safety_stop_is_not_target_range")
                return finish("NEAR_TARGET_SAFETY_STOP",
                              "Safety stopped the robot, but the target's visual range was not verified within 2.0 m.",
                              visual_d)
            final_d = float(visual_d)
            if not identity_verified:
                identity_verified = True
                update_target_memory(distance=final_d, status="verified_arrival", confirmed=True)
                last_vlm_distance = float(final_d)
                record_event(
                    "TARGET_IDENTITY_VERIFIED", target=object_name,
                    distance_m=last_vlm_distance,
                    distance_source="vlm_visual_estimate",
                    signature=target_signature,
                )

            # A physical safety stop at the ~2 m semantic standoff followed by
            # VLM confirmation is terminal success.  No closer final approach.
            return finish(
                "ARRIVED",
                f"Arrived at the VLM-verified '{object_name}' at the {verify_dist:.2f}m visual standoff "
                f"(VLM estimate {float(final_d):.2f}m).",
                final_d,
            )

        if outcome == "SIM_VISUAL_SEGMENT_COMPLETED":
            # A short visual-only test step is NEVER treated as a full leg or
            # proof of clearance: stop, inspect a fresh frame, and re-plan.
            state["replan_count"] = int(state.get("replan_count", 0)) + 1
            record_event("SIM_VISUAL_RECHECK_REQUIRED", leg=leg + 1,
                         travelled_m=actual)
            continue

        if outcome == "ODOM_STALE":
            _stop(wait=True, direction=1)
            if transient_odom_recoveries < 2 and wait_odom_recovery(1.2):
                transient_odom_recoveries += 1
                state["replan_count"] = int(state.get("replan_count", 0)) + 1
                state["recovery_success_count"] = int(state.get("recovery_success_count", 0)) + 1
                record_event(
                    "NAV_ODOM_RECOVERED", leg=leg + 1,
                    recoveries=transient_odom_recoveries,
                )
                # Re-plan from the current camera image; do not blindly resume
                # the remainder of the interrupted motion.
                continue
            return finish(
                "ODOM_STALE",
                f"Navigation stopped because odometry did not recover ({state.get('last_motion_reason', '')}).",
                last_vlm_distance,
            )

        if outcome == "SAFETY_SENSOR_UNAVAILABLE":
            _stop(wait=True, direction=1)
            if transient_sensor_recoveries < 4:
                recovered = wait_safety_recovery(2.0)
                if recovered is not None and recovered.valid:
                    transient_sensor_recoveries += 1
                    state["replan_count"] = int(state.get("replan_count", 0)) + 1
                    state["recovery_success_count"] = int(state.get("recovery_success_count", 0)) + 1
                    record_event(
                        "NAV_SENSOR_RECOVERED", leg=leg + 1,
                        recoveries=transient_sensor_recoveries,
                        source=recovered.source,
                    )
                    continue
            return finish(
                "SAFETY_SENSOR_UNAVAILABLE",
                f"Navigation stopped because safety sensing did not recover ({state.get('last_motion_reason', '')}).",
                last_vlm_distance,
            )

        if outcome in {"INTERRUPTED", "PRECONDITION_FAILED", "HARDWARE_INTERLOCK_STOP"}:
            return finish(
                outcome,
                f"Navigation stopped while approaching '{object_name}': {state.get('last_motion_reason', '')}.",
                last_vlm_distance,
            )

        # TIMEOUT/short move: stop and let the VLM re-plan once rather than
        # layering more low-level corrective turns.
        _stop(wait=True, direction=1)
        state["replan_count"] = int(state.get("replan_count", 0)) + 1
        record_event(
            "NAV_REPLAN", target=object_name, leg=leg + 1,
            reason=outcome or "short_move",
        )

    return finish(
        "MAX_REPLANS",
        f"Navigation did not complete within {max_legs} VLM plan-execute-recheck legs.",
        last_vlm_distance,
    )


# ════════════════════════════════════════════════════════════
# PATH CHECKING
# ════════════════════════════════════════════════════════════

@tool
def check_path_ahead() -> str:
    """Check if path ahead is clear using sensor data (LiDAR or depth) + camera vision."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="check_path_ahead")
    reading = check_obstacle(1)
    img = state["image_raw"]

    if not reading.valid:
        sensor_status = f"UNKNOWN/STOP ({reading.reason})"
    else:
        near = float(reading.distance if reading.distance is not None else 99.0)
        sensor_status = (
            f"BLOCKED at {near:.2f}m ({reading.source})" if near < SAFE_DIST else
            f"CAUTION at {near:.2f}m ({reading.source})" if near < SAFE_DIST * 2.5 else
            f"CLEAR (nearest {near:.2f}m, {reading.source})"
        )

    vision_status = "No camera"
    if img:
        with timed("vlm_path_check"):
            vision_status = vlm_mgr.ask(
                (
                    "Assess only the physical traversability of the path directly "
                    "ahead. Use visible geometry, not inferred intent or gestures. "
                    "Do not interpret a T-pose, arm position, or body posture as a "
                    "command to stop. A person/object is relevant only if its body "
                    "physically occupies or constrains the robot corridor. Reply one "
                    "line: CLEAR, CAUTION [physical reason], or BLOCKED [physical reason]."
                ),
                img, max_tokens=60,
            )

    return f"Sensor: {sensor_status} | Vision: {vision_status}"


# ════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════
@tool
def save_snapshot(filename: str = "snapshot.png") -> str:
    """Save the current camera view to a PNG file on disk."""
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="save_snapshot", filename=filename)
    img = state["image_raw"]
    if not img:
        return "No image available."
    path = f"/home/caim/{filename}"
    img.save(path)
    return f"Saved {img.size[0]}x{img.size[1]} image to {path}"


@tool("look_around")
def look_around(query: str = "") -> str:
    """Physically scan 360° with four VLM views and remember a searched target.

    Args:
        query: Optional exact object/person description to search for.  When the
               user asks to look/search for something (e.g. "person with yellow
               vest"), pass that phrase here.  A successful scan stores its
               visual signature, approximate world bearing/position and distance
               so a later "go to that person" can immediately face the remembered
               target before normal VLM navigation begins.
    """
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    query = str(query or "").strip()
    record_event("TOOL_CALL", tool="look_around", query=query)
    state["status"] = "scanning 360°"
    log(f"Starting 360° scan{f' for {query!r}' if query else ''}", "VROSA")

    def rotate_to_absolute(target_yaw: float) -> bool:
        if interrupt_flag.is_set():
            return False
        diff = _wrap_deg(float(target_yaw) - float(state["yaw"]))
        if abs(diff) < 3.0:
            return True
        _rotate(diff)
        if state.get("last_rotation_outcome") != "SUCCESS":
            record_event("SCAN_ROTATION_FAILED", target_yaw=target_yaw,
                         outcome=state.get("last_rotation_outcome"))
            return False
        time.sleep(0.35)
        return True

    def parse_target_block(text: str):
        if not query:
            return None
        upper = (text or "").upper()
        m = re.search(r"TARGET_MATCH\s*:\s*(YES|NO)", upper)
        if not m or m.group(1) != "YES":
            return None
        dm = re.search(r"TARGET_DIRECTION\s*:\s*(LEFT|CENTER|RIGHT)", upper)
        distm = re.search(r"TARGET_DISTANCE_M\s*:\s*(\d+(?:\.\d+)?)", upper)
        sigm = re.search(r"TARGET_SIGNATURE\s*:\s*(.+)", text or "", re.IGNORECASE)
        side = dm.group(1) if dm else "CENTER"
        dist = float(distm.group(1)) if distm else None
        if dist is not None and not (0.3 < dist < 30.0):
            dist = None
        signature = sigm.group(1).strip()[:180] if sigm else ""
        if not signature or not _text_satisfies_query(query, signature):
            record_event(
                "SCAN_TARGET_ATTRIBUTE_MISMATCH",
                query=query,
                signature=signature,
                response_preview=(text or "")[:180],
            )
            return None
        return {"side": side, "distance_m": dist, "signature": signature}

    observations = []
    candidates = []

    def scan_direction(label: str) -> str:
        img = state.get("image_raw")
        if img is None:
            return "no image"
        scan_yaw = float(state.get("yaw", 0.0))
        scan_x = float(state.get("x", 0.0))
        scan_y = float(state.get("y", 0.0))

        # LiDAR/depth are safety context only; object identity and target range
        # in the scan are explicitly requested from the visual model.
        if query:
            target_block = (
                f"\nYou are specifically searching for: '{query}'. All explicit attributes are mandatory. "
                "Do not mark a different same-class object as the target. At the END add exactly four lines:\n"
                "TARGET_MATCH: YES or NO\n"
                "TARGET_DIRECTION: LEFT, CENTER, or RIGHT\n"
                "TARGET_DISTANCE_M: visual estimate in metres, or 99 if not matched\n"
                "TARGET_SIGNATURE: <=12 words identifying the matched instance"
            )
        else:
            target_block = ""

        with timed("vlm_scan", query=query or None):
            result = vlm_mgr.ask(
                f"Facing {label}. Use the camera image as the source for object identity and approximate object distance.\n"
                "List UNIQUE visible objects. Format: [color+name] — [left/center/right] — [Xm]\n"
                "No repeats. Max 4 objects. Include color/appearance when visible.\n"
                "End the object list with: Clear: [direction] [Xm]."
                + target_block,
                img,
                max_tokens=220 if query else 150,
            )
        log(f"Scan {label}: {result[:60]}", "VROSA")

        obs = {
            "scan_label": label,
            "heading_yaw_deg": scan_yaw,
            "pose_x": scan_x,
            "pose_y": scan_y,
            "response": result,
        }
        observations.append(obs)

        parsed = parse_target_block(result)
        if parsed:
            side_offset = {"LEFT": 20.0, "CENTER": 0.0, "RIGHT": -20.0}.get(parsed["side"], 0.0)
            target_yaw = _wrap_deg(scan_yaw + side_offset)
            cand = {
                **parsed,
                "scan_label": label,
                "scan_heading_yaw_deg": scan_yaw,
                "target_yaw_deg": target_yaw,
                "pose_x": scan_x,
                "pose_y": scan_y,
            }
            if parsed["distance_m"] is not None:
                r = float(parsed["distance_m"])
                th = math.radians(target_yaw)
                cand["world_x"] = scan_x + r * math.cos(th)
                cand["world_y"] = scan_y + r * math.sin(th)
            candidates.append(cand)
            record_event(
                "SCAN_TARGET_FOUND",
                query=query,
                scan_label=label,
                side=parsed["side"],
                distance_m=parsed["distance_m"],
                signature=parsed["signature"],
                target_yaw_deg=target_yaw,
            )
        return result

    def abort_scan(reason: str) -> str:
        _stop()
        state["scene_scan_memory"] = None  # Never leave a partial scan recallable.
        state["status"] = "idle"
        record_event("SCAN_ABORTED", reason=reason)
        return f"360 scan aborted safely: {reason}. No target scan memory retained."

    start_yaw = float(state["yaw"])
    h0 = start_yaw
    h1 = _wrap_deg(start_yaw + 90.0)
    h2 = _wrap_deg(start_yaw + 180.0)
    h3 = _wrap_deg(start_yaw + 270.0)

    results: dict[str, str] = {}
    results["North (forward)"] = scan_direction("North (forward)")
    if not rotate_to_absolute(h1):
        return abort_scan("rotation_to_west_failed")
    results["West (left)"] = scan_direction("West (left)")
    if not rotate_to_absolute(h2):
        return abort_scan("rotation_to_south_failed")
    results["South (behind)"] = scan_direction("South (behind)")
    if not rotate_to_absolute(h3):
        return abort_scan("rotation_to_east_failed")
    results["East (right)"] = scan_direction("East (right)")
    if not rotate_to_absolute(h0):
        return abort_scan("rotation_to_start_failed")
    time.sleep(0.25)

    final_yaw = float(state["yaw"])
    yaw_error = abs(_wrap_deg(final_yaw - start_yaw))
    log(f"360° scan complete. Yaw error: {yaw_error:.1f}°", "VROSA")

    best_target = None
    if candidates:
        # Prefer the closest visually estimated valid match; unknown-distance
        # matches sort behind measured candidates.
        best_target = min(
            candidates,
            key=lambda c: float(c["distance_m"]) if c.get("distance_m") is not None else 1e9,
        )

    scene_memory = {
        "created_monotonic": time.monotonic(),
        "start_yaw_deg": start_yaw,
        "target_query": query or None,
        "observations": observations,
        "target_candidates": candidates,
        "best_target": best_target,
    }
    state["scene_scan_memory"] = scene_memory
    if query:
        state["last_scan_target_query"] = query
    record_event(
        "SCAN_MEMORY_CREATED",
        query=query or None,
        observation_count=len(observations),
        target_match_count=len(candidates),
        best_scan_label=(best_target or {}).get("scan_label"),
        best_distance_m=(best_target or {}).get("distance_m"),
        best_signature=(best_target or {}).get("signature"),
    )

    state["status"] = "idle"
    state["last_vlm"] = "360° scan complete"
    full = "\n\n".join(f"[{direction}]\n{txt}" for direction, txt in results.items())

    memory_note = ""
    if query and best_target:
        memory_note = (
            f"\n\nRemembered target: '{query}' at {best_target['scan_label']} "
            f"({best_target['side'].lower()} in that view), "
            f"~{best_target.get('distance_m', 'unknown')}m. "
            "A follow-up navigation command will automatically face this remembered target first."
        )
    elif query:
        memory_note = f"\n\nNo visually valid match for '{query}' was stored in scan memory."

    return f"360° Scan Complete (yaw error: {yaw_error:.1f}°):\n\n{full}{memory_note}"


# ════════════════════════════════════════════════════════════
# PERCEPTION TOOL  (GroundingDINO + DA-V2)
# ════════════════════════════════════════════════════════════
@tool
def detect_objects_with_distance(query: str = "") -> str:
    """Detect objects in the current camera view with metric distances.

    Uses GroundingDINO (open-vocabulary) when a query is given,
    or returns the latest YOLO background detections when query is empty.
    Distances come from Depth Anything V2 Metric — no LiDAR needed.

    Args:
        query: Object to search for (e.g. "blue forklift", "shelf").
               Leave empty to list all currently detected objects.

    Returns a formatted list:
        yellow forklift — center — 2.8m (conf: 0.87)
        blue shelf      — left   — 4.1m (conf: 0.79)
    """
    state["tool_call_count"] = int(state.get("tool_call_count", 0)) + 1
    record_event("TOOL_CALL", tool="detect_objects_with_distance", query=query)
    img = state["image_raw"]
    if not img:
        return "No camera image available."

    if query.strip():
        # ── GroundingDINO: find a specific named object ────────
        log(f"GroundingDINO query: '{query}'", "PERC")
        state["status"] = f"detecting '{query}'"
        try:
            from perception import run_grounding_dino
            dets = run_grounding_dino(img, query)
        except Exception as e:
            return f"GroundingDINO error: {e}"
        finally:
            state["status"] = "idle"

        if not dets:
            return f"'{query}' not detected in current view."

        # Merge GroundingDINO results into state detections
        # (keeps YOLO detections, adds / replaces matching labels)
        existing = [d for d in state["detections"]
                    if query.lower() not in d["label"].lower()]
        state["detections"] = existing + dets

        lines = [
            f"{d['label']} — {d['sector']} — {d['dist']:.1f}m (conf: {d['conf']:.2f})"
            for d in sorted(dets, key=lambda x: x["dist"])
        ]
        return "\n".join(lines)

    else:
        # ── Return cached YOLO background detections ───────────
        dets = state.get("detections", [])
        if not dets:
            return "No objects detected yet (perception pipeline may still be loading)."
        lines = [
            f"{d['label']} — {d['sector']} — {d['dist']:.1f}m (conf: {d['conf']:.2f})"
            for d in sorted(dets, key=lambda x: x["dist"])
        ]
        return "\n".join(lines)