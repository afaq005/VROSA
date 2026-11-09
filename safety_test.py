"""
safety_test.py — Quantitative stopping-distance / emergency-stop harness.

Examples:
  python3 safety_test.py --robot scout2 --api-model <model> \
      --mode stopping_distance --speeds 0.2 0.4 --trials 5 --approach 6.0

  python3 safety_test.py --robot scout2 --api-model <model> \
      --mode estop --speeds 0.2 0.4 --trials 5 --approach 6.0

The harness uses the same motion/safety code as VROSA. Actual stopping time is
based on odometry velocity reaching the configured near-zero threshold, not a
fixed sleep after the zero command.
"""

import argparse
import csv
import os
import random
import threading
import time

import config  # parses --robot before ROS import
from config import ROBOT
import perception
import motion
from experiment_logger import SAFETY_PATH, set_context, record_event
from logger import log
from state import interrupt_flag, state

p = argparse.ArgumentParser(description="V-ROSA safety test harness", add_help=False)
p.add_argument("--mode", required=True, choices=["stopping_distance", "estop", "freerun", "dynamic_crossing"])
p.add_argument("--speeds", type=float, nargs="+", default=[ROBOT["max_linear_speed"]])
p.add_argument("--trials", type=int, default=5)
p.add_argument("--approach", type=float, default=6.0)
p.add_argument("--duration", type=float, default=30.0)
args, _ = p.parse_known_args()


def wait_for_perception(timeout=120.0):
    perception.preload_perception_models()
    import ros_node  # starts ROS spin only after heavyweight models are preloaded
    perception.start_perception_loops()
    t0 = time.monotonic()
    while not perception.perception_status()["ready"]:
        if time.monotonic() - t0 > timeout:
            raise RuntimeError("Perception not ready; do not run formal safety trials.")
        time.sleep(0.5)
    if not ROBOT.get("has_lidar"):
        from safety import check_obstacle
        t1 = time.monotonic()
        while not check_obstacle(1).valid:
            if time.monotonic() - t1 > timeout:
                raise RuntimeError("Fresh camera-derived safety depth unavailable; aborting test.")
            time.sleep(0.25)
    log("Perception ready", "SAFETY")


def wait_for_odom(timeout=15.0):
    t0 = time.monotonic()
    while state.get("odom_ts") is None:
        if time.monotonic() - t0 > timeout:
            raise RuntimeError("No odometry timestamp received; check robot profile/topics.")
        time.sleep(0.1)
    log("Odometry ready", "SAFETY")


def count_rows(path) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def last_row(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]


def run_stopping_distance():
    for speed in args.speeds:
        for trial in range(1, args.trials + 1):
            set_context(trial_id=f"SAFETY_STOP_{speed:.2f}_{trial:03d}", scenario_id="static_obstacle", obstacle_type="configured_obstacle", dynamic_obstacle=False)
            input(f"\nPlace obstacle for speed={speed:.2f} m/s, trial={trial}/{args.trials}; press Enter...")
            interrupt_flag.clear()
            n0 = count_rows(SAFETY_PATH)
            record_event("SAFETY_TRIAL_START", test="stopping_distance", speed=speed, trial=trial)
            motion._move_dist(args.approach, speed=speed)
            if count_rows(SAFETY_PATH) > n0:
                r = last_row(SAFETY_PATH)
                log(
                    f"trigger={r.get('trigger_distance_m')}m stop_distance={r.get('stop_distance_m')}m "
                    f"total_latency={r.get('total_stop_latency_ms')}ms min_clearance={r.get('minimum_clearance_m')}",
                    "SAFETY",
                )
            else:
                log("No stop event recorded; trial is invalid for stopping-distance analysis", "WARN")


def run_estop():
    for speed in args.speeds:
        for trial in range(1, args.trials + 1):
            set_context(trial_id=f"SAFETY_ESTOP_{speed:.2f}_{trial:03d}", scenario_id="estop", obstacle_type="none", dynamic_obstacle=False)
            input(f"\nClear path for speed={speed:.2f} m/s, trial={trial}/{args.trials}; press Enter...")
            interrupt_flag.clear()
            trigger_delay = random.uniform(0.30, 0.70) * args.approach / max(speed, 1e-3)

            def delayed_estop():
                time.sleep(trigger_delay)
                motion.trigger_estop("safety_test_timer")

            threading.Thread(target=delayed_estop, daemon=True).start()
            motion._move_dist(args.approach, speed=speed)



def run_dynamic_crossing():
    """Externally scripted Isaac moving box; robot uses unchanged live LiDAR safety.

    Each trial is synchronized manually: arm the Isaac actor for the matching
    trial ID, start simulator playback, THEN press Enter in this harness.
    A non-stop or stale-sensor outcome is logged rather than hidden.
    """
    if not ROBOT.get("has_lidar") or ROBOT.get("platform_label") != "Isaac Sim":
        raise RuntimeError("Dynamic-box test is restricted to Nova Carter Isaac Sim")
    if ROBOT.get("sim_visual_clearance_test"):
        raise RuntimeError("Disable experimental visual-only fallback for formal moving-box tests")
    for speed in args.speeds:
        for trial in range(1, args.trials + 1):
            trial_id = f"SAFETY_DYNAMIC_{speed:.2f}_{trial:03d}"
            set_context(trial_id=trial_id, scenario_id="dynamic_box_crossing",
                        obstacle_type="moving_box", dynamic_obstacle=True)
            input(
                f"\nReset robot and disarm prior box. In Isaac Script Editor call "
                f"VROSA_DYNAMIC_BOX.arm('{trial_id}', {speed}). "
                f"PLAY simulation, then press Enter to start robot..."
            )
            interrupt_flag.clear()
            n0 = count_rows(SAFETY_PATH)
            record_event("SAFETY_TRIAL_START", test="dynamic_crossing",
                         speed=speed, trial=trial,
                         actor_trial_id=trial_id, actor="isaac_dynamic_box.py")
            result = motion._move_dist(args.approach, speed=speed)
            row_count = count_rows(SAFETY_PATH)
            if row_count > n0:
                r = last_row(SAFETY_PATH)
                log(f"Dynamic crossing trial={trial_id} outcome="
                    f"{getattr(result, 'outcome', result)} "
                    f"stop={r.get('stop_distance_m')}m "
                    f"clearance={r.get('minimum_clearance_m')}m", "SAFETY")
            else:
                log(f"Dynamic crossing trial={trial_id}: NO recorded safety stop; "
                    f"inspect motion and GT logs, do not classify as success", "WARN")
            record_event("DYNAMIC_TRIAL_END", trial_id=trial_id,
                         motion_outcome=str(getattr(result, 'outcome', result)),
                         safety_event_recorded=(row_count > n0))



def run_freerun():
    set_context(trial_id="SAFETY_FREERUN", scenario_id="dynamic_obstacle", obstacle_type="pedestrian_or_dynamic_hazard", dynamic_obstacle=True)
    log(f"Free-run safety logging for {args.duration:.1f}s", "SAFETY")
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.duration:
        time.sleep(0.25)


if __name__ == "__main__":
    wait_for_perception()
    wait_for_odom()
    {
        "stopping_distance": run_stopping_distance,
        "estop": run_estop,
        "freerun": run_freerun,
        "dynamic_crossing": run_dynamic_crossing,
    }[args.mode]()
    log(f"Done. Safety data: {SAFETY_PATH}", "SAFETY")
