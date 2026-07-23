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

import numpy as np
from langchain.tools import tool

from state import state, interrupt_flag
from vlm.manager import vlm_mgr
from motion import _stop, _drive, _move_dist, _rotate, SAFE_DIST
from logger import log
import numpy as np
import perception   # ensures background loop is running


# ════════════════════════════════════════════════════════════
# POSITION / BASIC MOTION
# ════════════════════════════════════════════════════════════
@tool
def get_robot_position() -> str:
    """Get the current position and heading of the Nova Carter robot."""
    return (f"x={state['x']:.3f}m, y={state['y']:.3f}m, "
            f"heading={state['yaw']:.1f}deg")


@tool
def move_forward(distance_meters: float) -> str:
    """Move robot forward (positive) or backward (negative) by given metres."""
    state["status"] = f"moving {distance_meters}m"
    log(f"Moving {distance_meters}m", "ROBOT")
    d = _move_dist(distance_meters)
    state["status"] = "idle"
    return f"Moved {d:.2f}m. Now at x={state['x']:.2f}, y={state['y']:.2f}"


@tool
def rotate_robot(degrees: float) -> str:
    """Rotate robot left (+) or right (-) by degrees."""
    state["status"] = f"rotating {degrees}deg"
    log(f"Rotating {degrees}deg", "ROBOT")
    _rotate(degrees)
    state["status"] = "idle"
    return f"Rotated {degrees}deg. Facing {state['yaw']:.1f}deg"


@tool
def stop_robot() -> str:
    """Immediately stop the robot."""
    _stop()
    state["status"] = "idle"
    return "Stopped."


# ════════════════════════════════════════════════════════════
# VISION TOOLS
# ════════════════════════════════════════════════════════════
@tool
def describe_current_view() -> str:
    """Use VLM to describe what the robot's camera currently sees."""
    img = state["image_raw"]
    if not img:
        return "No camera image available."
    state["status"] = "analyzing view"
    log(f"VLM scene description [{vlm_mgr.active_name}]...", "VLM")
    result = vlm_mgr.ask(
        "Describe what you see in this image in detail.",
        img, max_tokens=400,
    )
    state["last_vlm"] = result
    state["status"]   = "idle"
    log(f"VLM: {result[:80]}", "VLM")
    return result


@tool
def find_object(object_name: str) -> str:
    """Find a specific object in the camera view and report its position."""
    img = state["image_raw"]
    if not img:
        return "No camera image."
    log(f"Finding '{object_name}' [{vlm_mgr.active_name}]", "VLM")
    result = vlm_mgr.ask(
        f"Is there a {object_name} in this image? "
        f"If yes, describe where it is and what it looks like. "
        f"If no, describe what you do see instead.",
        img, max_tokens=200,
    )
    state["last_vlm"] = result
    return f"'{object_name}': {result}"


# ════════════════════════════════════════════════════════════
# SEMANTIC NAVIGATION
# ════════════════════════════════════════════════════════════
@tool
def navigate_to_object(object_name: str) -> str:
    """Navigate to a named object. Plan-Execute-Recheck architecture.

    Architecture (minimal VLM calls):
      1. SCOUT  — 1 VLM call: get direction + rough distance estimate
      2. DRIVE  — pure odometry: move 70% of estimated distance, NO VLM
      3. RECHECK — 1 VLM call: still on course? new distance? → update plan
      4. FINAL  — 1 VLM call: arrival confirmation when within 1.5m

    KEY FIX: obs ≈ obj → obstacle IS the target → arrival check, not avoidance.
    Non-target obstacles (obs << obj) → sidestep around them.

    Total VLM calls for a 5m approach: ~3-4 (was 10-25 before).
    """
    state["status"] = f"navigating to {object_name}"
    state["target"] = object_name
    log(f"Semantic nav → '{object_name}' [{vlm_mgr.active_name}]", "VROSA")

    # ── Synonym table ────────────────────────────────────────
    _SYNONYMS = {
        "person":   ["person","worker","human","man","woman","people",
                     "warehouse worker","operator","staff","employee","avatar"],
        "forklift": ["forklift","lift","truck","vehicle","machine","truck"],
        "shelf":    ["shelf","shelving","rack","racking","storage"],
        "box":      ["box","carton","package","cardboard","pallet","crate"],
        "door":     ["door","gate","entrance","exit"],
        "barrel":   ["barrel","drum","container","cylinder"],
    }
    _ROT_MAP = {
        "HARD_LEFT": 40, 
        "LEFT": 20, 
        "CENTER": 0,
        "RIGHT": -20, 
        "HARD_RIGHT": -40
    }
    def _synonyms(name: str) -> list:
        n = name.lower()
        for key, syns in _SYNONYMS.items():
            if n in key or key in n or any(n in s or s in n for s in syns):
                return syns
        return [n]
    _labels = _synonyms(object_name)
    def _is_target(label: str) -> bool:
        lw = label.lower()
        return any(s in lw or lw in s for s in _labels)

    # ── Distance helpers (depth map — camera only) ────────────
    def _scene_depth(sector="CENTER") -> float:
        """10th-pct of tight centre crop — picks closest real surface."""
        dm = state.get("depth_map")
        if dm is None: return 99.0
        H, W = dm.shape
        if sector == "CENTER":
            strip = dm[H*3//8:H*5//8, W*2//5:W*3//5]
        elif sector == "LEFT":
            strip = dm[H*3//8:H*5//8, W//8:W*3//8]
        else:
            strip = dm[H*3//8:H*5//8, W*5//8:W*7//8]
        v = strip[(strip > 0.3) & (strip < 10.0)]
        return float(np.percentile(v, 10)) if len(v) > 0 else 99.0

    def _obstacle_dist() -> float:
        """8th-pct of narrow centre strip — earliest blocker directly ahead."""
        dm = state.get("depth_map")
        if dm is None: return 99.0
        H, W = dm.shape
        # strip = dm[H//3:2*H//3, W*38//100:W*62//100]
        strip = dm[H//3:H//2, W*38//100:W*62//100]
        v = strip[(strip > 0.15) & (strip < 10.0)]
        return float(np.percentile(v, 8)) if len(v) > 0 else 99.0

    # def _best_target_dist() -> float:
    #     """Target distance from detections first, then depth centre."""
    #     dets  = state.get("detections", [])
    #     found = [d["dist"] for d in dets if _is_target(d["label"]) and d["dist"] < 90]
    #     if found: return min(found)
    #     return _scene_depth("CENTER")
    def _best_target_dist(direction: str = "") -> float:
        """Target distance — filters detections to the scouted direction sector.
        Prevents nearby off-axis detections (person A behind robot)
        from overriding the target the VLM is pointing toward (person B ahead).
        """
        dets = state.get("detections", [])
        if dets and direction:
            # Map VLM direction to detection sector
            sector_map = {
                "HARD_LEFT": "left",  "LEFT": "left",
                "CENTER":    "center",
                "RIGHT":     "right", "HARD_RIGHT": "right",
            }
            target_sector = sector_map.get(direction, "")
            # Try direction-filtered detections first
            if target_sector:
                sector_found = [d["dist"] for d in dets
                                if _is_target(d["label"])
                                and d["sector"] == target_sector
                                and d["dist"] < 90]
                if sector_found:
                    return min(sector_found)
        # Fall back: any matching detection regardless of sector
        all_found = [d["dist"] for d in dets
                     if _is_target(d["label"]) and d["dist"] < 90]
        if all_found:
            return min(all_found)
        return _scene_depth("CENTER")

    def _is_target_the_obstacle(obj_d: float, obs_d: float) -> bool:
        """True when the obstacle depth ≈ object depth — obstacle IS the target."""
        if obj_d >= 90 or obs_d >= 90: return False
        # return abs(obj_d - obs_d) < 0.8   # within 0.8m → same object
        tolerance = 0.5 if obj_d > 3.0 else 1.0
        return abs(obj_d - obs_d) < tolerance

    def _side_clearance():
        dm = state.get("depth_map")
        if dm is None: return 99.0, 99.0
        H, W = dm.shape
        lv = dm[H//4:3*H//4, :W//3]; rv = dm[H//4:3*H//4, 2*W//3:]
        lc = float(np.percentile(lv[(lv>0.2)&(lv<10)], 20)) if (lv>0.2).any() else 99.0
        rc = float(np.percentile(rv[(rv>0.2)&(rv<10)], 20)) if (rv>0.2).any() else 99.0
        return lc, rc

    # ── Ask VLM: direction + coarse distance ─────────────────
    def _vlm_scout() -> tuple:
        """Returns (direction, dist_estimate).
        Asks VLM for direction AND rough distance in one call.
        """
        img = state["image_raw"]
        if not img: return None, 99.0
        resp = vlm_mgr.ask(
            (
                f"Find '{object_name}' in this image. "
                # f"Line 1: ONE WORD direction — LEFT, CENTER, RIGHT, NOT_VISIBLE, or ARRIVED "
                f"Line 1: ONE WORD direction — HARD_LEFT (far left), LEFT (slightly left), "
                f"CENTER (straight ahead), RIGHT (slightly right), HARD_RIGHT (far right), "
                f"NOT_VISIBLE, or ARRIVED "
                # f"Line 2: estimated distance in metres as a number only (e.g. 4.5) "
                f"Line 2: your best estimate of distance in metres, just digits (e.g. 3.2). "
                f"Use visual cues like object size and perspective. "
                f"If not visible write NOT_VISIBLE on line 1 and 99 on line 2. "
                f"REPLY IN EXACTLY 2 LINES:"
            ),
            img,
            max_tokens=20,
        ).strip().upper()
        lines = [l.strip() for l in resp.split("\n") if l.strip()]
        # kws   = {"LEFT","CENTER","RIGHT","NOT_VISIBLE","ARRIVED"}
        # direction = next((t for l in lines for t in l.split() if t in kws), None)
        kws   = {"HARD_LEFT","LEFT","CENTER","RIGHT","HARD_RIGHT","NOT_VISIBLE","ARRIVED"}
        direction = next((t for l in lines for t in l.split() if t in kws), None)
        # Normalise: map 5-way to canonical 3-way for caching, keep angle info
        # _rot_map  = {"HARD_LEFT": 40, "LEFT": 20, "CENTER": 0,
        #              "RIGHT": -20, "HARD_RIGHT": -40}
        dist_est  = 99.0
        for l in lines:
            for tok in l.split():
                try:
                    v = float(tok.replace("M","").replace("METERS","").strip("~<>≈"))
                    if 0.3 < v < 30: dist_est = v; break
                except ValueError: pass
        return direction, dist_est

    def _vlm_direction_only() -> str:
        """Fast single-word direction query for re-checks."""
        img = state["image_raw"]
        if not img: return None
        resp = vlm_mgr.ask(
            # f"Where is '{object_name}'? ONE WORD: LEFT CENTER RIGHT NOT_VISIBLE ARRIVED",
            f"Where is '{object_name}'? ONE WORD: HARD_LEFT LEFT CENTER RIGHT HARD_RIGHT NOT_VISIBLE ARRIVED",
            img, max_tokens=5,
        ).strip().upper()
        kws = {"HARD_LEFT","LEFT","CENTER","RIGHT","HARD_RIGHT","NOT_VISIBLE","ARRIVED"}
        return next((t for t in resp.split() if t in kws), None)

    # ── Arrival confirmation ──────────────────────────────────
    def _confirm_arrival(dist: float) -> bool:
        img = state["image_raw"]
        if not img: return False
        # ans = vlm_mgr.ask(
        #     f"Is '{object_name}' within arm's reach or right in front of you? "
        #     f"YES or NO only.", img, max_tokens=5,
        ans = vlm_mgr.ask(
            f"Is '{object_name}' visible and close (within 1.5 metres)? "
            f"It could be on the floor, ahead, or nearby. "
            f"YES or NO only.", img, max_tokens=5,
        ).strip().upper()
        
        log(f"Arrival confirm: {ans}  dist={dist:.2f}m", "VROSA")
        return "YES" in ans
    
    def _face_target():
        """Rotate to face the target object using a single VLM call."""
        img = state["image_raw"]
        if not img:
            return
        resp = vlm_mgr.ask(
            f"Where exactly is '{object_name}' relative to the centre of the image? "
            f"ONE WORD: LEFT, CENTER, or RIGHT.",
            img, max_tokens=5,
        ).strip().upper()
        kws = {"LEFT", "CENTER", "RIGHT"}
        side = next((t for t in resp.split() if t in kws), "CENTER")
        if side == "LEFT":
            log("Facing target: rotating left 15°", "VROSA")
            _rotate(15)
        elif side == "RIGHT":
            log("Facing target: rotating right 15°", "VROSA")
            _rotate(-15)
        log(f"Facing '{object_name}' (VLM={side})", "VROSA")
    # ════════════════════════════════════════════════════════
    # PLAN-EXECUTE-RECHECK LOOP
    # ════════════════════════════════════════════════════════
    ARRIVE_DIST  = 1.2   # stop when target this close
    SAFE_DIST_OB = 0.70  # hard stop for non-target obstacles
    RECHECK_AT   = 0.70  # re-query VLM when this fraction of leg remains
    MAX_LEGS     = 8     # max plan legs (each = VLM call)

    vlm_calls    = 0
    _total_moved   = 0.0 
    for leg in range(MAX_LEGS):
        if interrupt_flag.is_set(): break

        # ── SCOUT: VLM call to get direction + distance ───────
        log(f"Leg {leg+1}: scouting (VLM call #{vlm_calls+1})", "VROSA")
        direction, vlm_dist = _vlm_scout()
        vlm_calls += 1
        log(f"Scout: dir={direction}  vlm_dist={vlm_dist:.1f}m", "VROSA")
        state["last_vlm"] = f"VLM dir={direction} ~{vlm_dist:.1f}m"

        if direction is None:
            log("Unparseable VLM — scanning 45°", "WARN")
            _rotate(-45 if leg % 2 == 0 else 45)
            time.sleep(0.3); continue

        if direction == "NOT_VISIBLE":
            log(f"'{object_name}' not visible — scanning", "VROSA")
            _rotate(-50 if leg % 2 == 0 else 50)
            time.sleep(0.3); continue

        # ── Also get current sensor-based distance ────────────
        sensor_dist = _best_target_dist(direction)
        # Use the more reliable of the two (prefer sensor if available)
        # obj_dist = sensor_dist if sensor_dist < 90 else vlm_dist
        obj_dist = sensor_dist if sensor_dist < 90 else min(vlm_dist, 6.0)

        if obj_dist >= 90:
            log("No distance estimate — nudging 0.5m", "VROSA")
            _move_dist(0.5); continue

        log(f"Target: {obj_dist:.2f}m away  dir={direction}", "VROSA")

        # ── Arrived right now? ────────────────────────────────
        # if direction == "ARRIVED" or obj_dist < ARRIVE_DIST:
        #     if _confirm_arrival(obj_dist):
        if (direction == "ARRIVED" or obj_dist < ARRIVE_DIST) and _total_moved > 0.5:
            if _confirm_arrival(obj_dist):

                _stop(); _face_target(); state["status"]="idle"; state["target"]=""
                return (f"✅ Arrived at '{object_name}'!  "
                        f"{obj_dist:.2f}m  x={state['x']:.2f} y={state['y']:.2f}")

        # ── Align to direction ────────────────────────────────
        # if direction == "LEFT":
        #     log("Aligning left 20°", "VROSA"); _rotate(20); time.sleep(0.15)
        # elif direction == "RIGHT":
        #     log("Aligning right 20°", "VROSA"); _rotate(-20); time.sleep(0.15)
        rot_deg = _ROT_MAP.get(direction, 0)
        if rot_deg != 0:
            log(f"Aligning {direction} {abs(rot_deg)}°", "VROSA")
            _rotate(rot_deg)
            time.sleep(0.15)
            direction = "CENTER" 
        # ── Compute leg distance ──────────────────────────────
        # Move 70% of estimated distance before re-asking VLM
        # This is the key speed change: covers ground without VLM pauses
        leg_dist    = max(0.3, (obj_dist - ARRIVE_DIST) * 0.70)
        leg_dist    = min(leg_dist, 4.0)   # cap single leg at 4m
        moved_so_far = 0.0
        leg_start_x  = state["x"]
        leg_start_y  = state["y"]

        log(f"Leg dist: {leg_dist:.2f}m  (target={obj_dist:.2f}m)", "VROSA")
        state["last_vlm"] = f"→ {object_name}  {obj_dist:.1f}m  leg={leg_dist:.1f}m"

        # ── EXECUTE: drive the leg in 0.5m sub-steps ─────────
        SUB_STEP = 0.5
        _consec_blocked = 0  
        while moved_so_far < leg_dist - 0.05:
            if interrupt_flag.is_set(): break

            step = min(SUB_STEP, leg_dist - moved_so_far)
            if step < 0.05: break

            # Re-measure distances
            cur_obj = _best_target_dist(direction)
            cur_obs = _obstacle_dist()

            # 1. Arrived during execution? (ONLY check target distance)
            # if cur_obj < ARRIVE_DIST:
            #     if _confirm_arrival(cur_obj):
            if cur_obj < ARRIVE_DIST and _total_moved > 0.5:
                if _confirm_arrival(cur_obj):
                    _stop(); _face_target(); state["status"]="idle"; state["target"]=""
                    return (f"✅ Arrived at '{object_name}'!  "
                            f"{cur_obj:.2f}m  "
                            f"x={state['x']:.2f} y={state['y']:.2f}")
                # VLM said NO (false depth reading) — break leg to re-scout
                break

            # 2. Obstacle check — is it the target or something else?
            if cur_obs < SAFE_DIST_OB:
                if _is_target_the_obstacle(cur_obj, cur_obs):
                    # The obstacle IS the target — stop and confirm arrival
                    log(f"Obstacle IS target ({cur_obs:.2f}m ≈ {cur_obj:.2f}m) — arrival check", "VROSA")
                    if _confirm_arrival(cur_obs):
                        _stop(); _face_target(); state["status"]="idle"; state["target"]=""
                        return (f"✅ Arrived at '{object_name}'!  "
                                f"{cur_obs:.2f}m  "
                                f"x={state['x']:.2f} y={state['y']:.2f}")
                    # False alarm, break leg to re-scout
                    log("Arrival check: NO — re-scouting", "VROSA")
                    break

                else:
                    # Real non-target obstacle
                    _consec_blocked += 1
                    log(f"Non-target obstacle {cur_obs:.2f}m (obj={cur_obj:.2f}m) — blocked #{_consec_blocked}", "VROSA")

                    if _consec_blocked >= 3:
                        # Stuck in a loop — break out for VLM re-scout
                        log("Stuck on obstacle — breaking leg for re-scout", "VROSA")
                        break

                    lc, rc = _side_clearance()
                    if lc > rc and lc > 1.2:
                        log(f"Sidestepping left ({lc:.1f}m clear)", "VROSA")
                        _rotate(50); time.sleep(0.15)
                        _move_dist(0.8, safe_dist=0.40)
                        _rotate(-35); time.sleep(0.15)
                    elif rc > 1.2:
                        log(f"Sidestepping right ({rc:.1f}m clear)", "VROSA")
                        _rotate(-50); time.sleep(0.15)
                        _move_dist(0.8, safe_dist=0.40)
                        _rotate(35); time.sleep(0.15)
                    else:
                        # Truly boxed in — back up and break for re-scout
                        log("No clearance — backing up 0.5m for re-scout", "VROSA")
                        _move_dist(-0.5)
                        _rotate(45 if _consec_blocked % 2 == 0 else -45)
                        break
                    time.sleep(0.2)
                    continue

            # Move sub-step
            # actual = _move_dist(step)
            # actual = _move_dist(step, safe_dist=0.45)
            # moved_so_far += actual
            # _consec_blocked = 0
            actual = _move_dist(step, safe_dist=0.45)
            moved_so_far  += actual
            _total_moved  += actual
            _consec_blocked = 0 
            log(f"  sub-step {step:.2f}m → moved {actual:.2f}m  total={moved_so_far:.2f}m", "VROSA")

            if actual < step * 0.3:
                # Moved much less than requested — likely stopped by obstacle
                log("Short move — checking why", "VROSA")
                break

        # total_moved = math.sqrt((state["x"]-leg_start_x)**2 + (state["y"]-leg_start_y)**2)
        # log(f"Leg {leg+1} complete: moved {total_moved:.2f}m  VLM calls so far: {vlm_calls}", "VROSA")
        leg_moved = math.sqrt((state["x"]-leg_start_x)**2 + (state["y"]-leg_start_y)**2)
        log(f"Leg {leg+1} complete: moved {leg_moved:.2f}m  total={_total_moved:.2f}m  VLM calls: {vlm_calls}", "VROSA")

        # Final distance check before next leg
        final_dist = _best_target_dist(direction)
        if final_dist < ARRIVE_DIST and _total_moved > 0.5:
            if _confirm_arrival(final_dist):
                _stop(); _face_target(); state["status"]="idle"; state["target"]=""
                return (f"✅ Arrived at '{object_name}'!  "
                        f"{final_dist:.2f}m  x={state['x']:.2f} y={state['y']:.2f}")

    _stop(); state["status"]="idle"; state["target"]=""
    final = _best_target_dist(direction)
    return (f"Navigation complete. '{object_name}' est. {final:.2f}m away.  "
            f"x={state['x']:.2f} y={state['y']:.2f}  VLM calls: {vlm_calls}")

# ════════════════════════════════════════════════════════════
# PATH CHECKING
# ════════════════════════════════════════════════════════════
@tool
def check_path_ahead() -> str:
    """Check if path ahead is clear using LiDAR + camera vision."""
    lidar = state["lidar"]
    img   = state["image_raw"]

    if lidar:
        near = lidar[0]["dist"]
        lidar_status = (
            f"BLOCKED at {near:.2f}m"        if near < SAFE_DIST else
            f"CAUTION at {near:.2f}m"        if near < SAFE_DIST * 2.5 else
            f"CLEAR (nearest {near:.2f}m)"
        )
    else:
        lidar_status = "No LiDAR data"

    vision_status = "No camera"
    if img:
        vision_status = vlm_mgr.ask(
            "Is the path directly ahead clear for a robot? "
            "Reply: CLEAR, CAUTION [reason], or BLOCKED [reason]. One line.",
            img, max_tokens=60,
        )

    return f"LiDAR: {lidar_status} | Vision: {vision_status}"


# ════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════
@tool
def save_snapshot(filename: str = "snapshot.png") -> str:
    """Save the current camera view to a PNG file on disk."""
    img = state["image_raw"]
    if not img:
        return "No image available."
    path = f"/home/orin/{filename}"
    img.save(path)
    return f"Saved {img.size[0]}x{img.size[1]} image to {path}"


@tool("look_around")
def look_around() -> str:
    """Physically rotate 360° taking VLM snapshots every 90°, building a full scene map."""
    state["status"] = "scanning 360°"
    log("Starting 360° scan", "VROSA")

    def rotate_to_absolute(target_yaw: float, speed: float = 0.4) -> None:
        """Rotate to an absolute heading using odometry feedback."""
        timeout = 10.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            if interrupt_flag.is_set():
                break
            current = state["yaw"]
            diff    = target_yaw - current
            while diff > 180:  diff -= 360
            while diff < -180: diff += 360
            if abs(diff) < 3.0:
                break
            sign = 1 if diff > 0 else -1
            _drive(angular=sign * speed)
            time.sleep(0.05)
        _stop()
        time.sleep(0.4)

    def scan_direction(label: str) -> str:
        img   = state["image_raw"]
        if not img:
            return "no image"
        lidar = state["lidar"]
        near  = lidar[0]["dist"] if lidar else 99.0
        try:
            sectors = {
                "left":   min([p["dist"] for p in lidar
                               if p["x"] > 0.1 and p["y"] > 0.3 and abs(p["z"]) < 1.5],
                              default=99.0),
                "center": min([p["dist"] for p in lidar
                               if p["x"] > 0.1 and abs(p["y"]) < 0.8 and abs(p["z"]) < 1.5],
                              default=99.0),
                "right":  min([p["dist"] for p in lidar
                               if p["x"] > 0.1 and p["y"] < -0.3 and abs(p["z"]) < 1.5],
                              default=99.0),
            }
            lidar_hint = (f"LiDAR: left={sectors['left']:.1f}m "
                          f"center={sectors['center']:.1f}m "
                          f"right={sectors['right']:.1f}m")
        except Exception:
            lidar_hint = f"LiDAR nearest={near:.1f}m"

        result = vlm_mgr.ask(
            f"Facing {label}. {lidar_hint}\n"
            f"List UNIQUE objects. Format: [color+name] — [left/center/right] — [Xm]\n"
            f"No repeats. Max 4 objects. Include color.\n"
            f"End with: Clear: [direction] [Xm]",
            img, max_tokens=150,
        )
        log(f"Scan {label}: {result[:60]}", "VROSA")
        return result

    # Compute target headings
    start_yaw = state["yaw"]
    h0 = start_yaw
    h1 = start_yaw + 90;  h1 = h1 - 360 if h1 > 180 else h1
    h2 = start_yaw + 180; h2 = h2 - 360 if h2 > 180 else h2
    h3 = start_yaw + 270; h3 = h3 - 360 if h3 > 180 else h3

    results: dict[str, str] = {}
    results["North (forward)"] = scan_direction("North (forward)")
    rotate_to_absolute(h1)
    results["West (left)"]     = scan_direction("West (left)")
    rotate_to_absolute(h2)
    results["South (behind)"]  = scan_direction("South (behind)")
    rotate_to_absolute(h3)
    results["East (right)"]    = scan_direction("East (right)")
    rotate_to_absolute(h0)
    time.sleep(0.3)

    final_yaw = state["yaw"]
    yaw_error = abs(final_yaw - start_yaw)
    if yaw_error > 180:
        yaw_error = 360 - yaw_error
    log(f"360° scan complete. Yaw error: {yaw_error:.1f}°", "VROSA")

    state["status"]   = "idle"
    state["last_vlm"] = "360° scan complete"

    full = "\n\n".join(f"[{direction}]\n{txt}" for direction, txt in results.items())
    return f"360° Scan Complete (yaw error: {yaw_error:.1f}°):\n\n{full}"


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
