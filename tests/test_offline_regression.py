"""Dependency-isolated navigation-policy regressions; NO ROS, Isaac or hardware.

Run from project root: python3 -m pytest -q tests/test_offline_regression.py
These tests mock VLM responses, obstacle readings and odometry-driven motion.
They do not assert hardware/simulator functionality.
"""
import contextlib
import importlib.util
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, **kwargs):
    m = types.ModuleType(name)
    m.__dict__.update(kwargs)
    return m


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def nav(monkeypatch):
    st = {
        "x": 0.0, "y": 0.0, "yaw": 0.0, "image_raw": object(),
        "image_seq": 10, "detections": [], "vlm_query_count": 0,
        "tool_call_count": 0, "replan_count": 0, "recovery_success_count": 0,
        "scene_scan_memory": None, "visual_target_memory": None,
        "last_rotation_outcome": "SUCCESS",
    }
    events, asks, turns, moves = [], [], [], []
    answers = []
    cfg = {"target_verify_dist": 2.0, "max_legs": 8, "has_lidar": True}

    def decorator(x=None):
        if callable(x):
            return x
        return lambda f: f

    class FakeVLM:
        active_name = "mock"
        def ask(self, prompt, img, max_tokens=70):
            asks.append(prompt)
            st["vlm_query_count"] += 1
            if not answers:
                raise AssertionError("VLM response queue exhausted")
            return answers.pop(0)

    def turn(deg):
        turns.append(deg)
        st["yaw"] += deg
        st["last_rotation_outcome"] = "SUCCESS"

    def move(d, safe_dist=None):
        moves.append((d, safe_dist))
        st["last_motion_outcome"] = "SUCCESS"
        st["x"] += d
        return d

    @contextlib.contextmanager
    def timed(*a, **kw):
        yield

    def record(event, **kw):
        events.append((event, kw))

    packages = {
        "langchain": module("langchain"),
        "langchain.tools": module("langchain.tools", tool=decorator),
        "state": module("state", state=st, interrupt_flag=types.SimpleNamespace(is_set=lambda: False)),
        "vlm": module("vlm"),
        "vlm.manager": module("vlm.manager", vlm_mgr=FakeVLM()),
        "motion": module("motion", _stop=lambda **kw: None, _drive=lambda **kw: None,
                         _move_dist=move, _rotate=turn, SAFE_DIST=.75,
                         trigger_estop=lambda *a: None),
        "logger": module("logger", log=lambda *a, **kw: None),
        "perception": module("perception"),
        "config": module("config", ROBOT=cfg, MAX_SENSOR_AGE_S=.75, PERCEPTION_CONFIG={}),
        "safety": module("safety", check_obstacle=lambda *a: types.SimpleNamespace(
            valid=True, distance=99.0, source="lidar", reason="")),
        "experiment_logger": module("experiment_logger", timed=timed, record_event=record),
    }
    with monkeypatch.context() as mp:
        for name, value in packages.items():
            mp.setitem(sys.modules, name, value)
        x = load_source("vrosa_test_tools", "tools.py")
        yield types.SimpleNamespace(m=x, state=st, events=events, asks=asks, answers=answers,
                                    turns=turns, moves=moves, cfg=cfg, record=record)
        sys.modules.pop("vrosa_test_tools", None)


def nav_view(direction, dist, sig):
    return f"DIRECTION: {direction}\nDISTANCE_M: {dist}\nSIGNATURE: {sig}"


def verified(dist, match="YES"):
    return f"MATCH: {match}\nDISTANCE_M: {dist}"


def event_names(n):
    return [e for e, _ in n.events]


def test_config_has_correct_thresholds():
    source = (ROOT / "config.py").read_text()
    for name in ("nova_carter_sim", "scout2"):
        assert name in source
    import ast
    tree = ast.parse(source)
    defs = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ROBOT_PROFILES" for t in node.targets))
    cfg = ast.literal_eval(defs)
    for profile in cfg.values():
        for key in ("arrive_dist", "runtime_arrive_dist", "target_verify_dist"):
            assert profile[key] == 2.0
        assert profile["max_legs"] == 8
        assert profile["target_depth_calibration_enabled"] is False
    assert cfg["scout2"]["has_lidar"] is False
    assert cfg["nova_carter_sim"]["has_lidar"] is True


def test_attribute_policy(nav):
    assert nav.m._text_satisfies_query("person in yellow vest", "yellow reflective worker")
    assert not nav.m._text_satisfies_query("person in yellow vest", "red white vest worker")
    assert not nav.m._text_satisfies_query("person in yellow vest", "")


def test_normal_arrival_returns_without_extra_translation(nav):
    nav.answers.extend([nav_view("CENTER", 1.90, "yellow reflective worker"), verified(1.88)])
    result = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=ARRIVED" in result
    assert "TARGET_IDENTITY_VERIFIED" in event_names(nav)
    assert nav.moves == []
    assert nav.state["last_visual_target_memory"]["identity_confirmed"] is True


@pytest.mark.parametrize("distance", [2.01, 2.20, 3.0, 99.0])
def test_no_arrival_outside_verified_standoff(nav, distance):
    nav.answers.extend([nav_view("CENTER", 2.1, "yellow reflective worker"), verified(distance)])
    result = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=ARRIVAL_UNCONFIRMED" in result
    assert "TARGET_IDENTITY_VERIFIED" not in event_names(nav)
    assert nav.moves == []


def test_no_arrival_with_missing_verified_range(nav):
    nav.answers.extend([nav_view("ARRIVED", 99, "yellow reflective worker"), "MATCH: YES\nDISTANCE_M: 99"])
    result = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=ARRIVAL_UNCONFIRMED" in result
    assert not nav.moves


def test_initial_wrong_attributes_never_move(nav):
    nav.answers.extend([nav_view("CENTER", 1.5, "red striped white vest worker"),
                        nav_view("NOT_VISIBLE", 99, "")])
    result = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=TARGET_LOST" in result
    assert "TARGET_ATTRIBUTE_MISMATCH" in event_names(nav)
    assert not nav.moves
    assert "TARGET_IDENTITY_VERIFIED" not in event_names(nav)


def test_initial_blank_signature_never_move(nav):
    nav.answers.extend([nav_view("CENTER", 1.5, ""), nav_view("NOT_VISIBLE", 99, "")])
    result = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=TARGET_LOST" in result
    assert not nav.moves


def test_memory_not_visible_keeps_previous_direction(nav):
    nav.answers.extend([nav_view("LEFT", 5.0, "yellow reflective worker"),
                        nav_view("NOT_VISIBLE", 99, ""),
                        nav_view("NOT_VISIBLE", 99, "")])
    result = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=TARGET_LOST" in result
    assert 8.0 in nav.turns
    assert nav.state["last_visual_target_memory"]["last_direction"] == "LEFT"
    assert sum(name == "TARGET_MEMORY_MISS" for name in event_names(nav)) == 2


def test_unresolved_reference_does_not_move(nav):
    result = nav.m.navigate_to_object("that person")
    assert "OUTCOME=TARGET_REFERENCE_UNRESOLVED" in result
    assert nav.turns == [] and nav.moves == []
    assert "TARGET_REFERENCE_UNRESOLVED" in event_names(nav)


def test_expired_scan_memory_refused(nav):
    nav.state["scene_scan_memory"] = {
        "created_monotonic": time.monotonic() - 400,
        "target_query": "person in yellow vest",
        "best_target": {"target_yaw_deg": 90, "signature": "yellow vest worker"},
    }
    result = nav.m.navigate_to_object("that person")
    assert "OUTCOME=TARGET_REFERENCE_UNRESOLVED" in result
    assert nav.moves == []


def test_scan_memory_alignment_and_identity(nav):
    nav.state["scene_scan_memory"] = {
        "created_monotonic": time.monotonic(),
        "target_query": "person with yellow vest",
        "best_target": {"target_yaw_deg": 90, "signature": "yellow vest green pants worker",
                        "scan_label": "West", "distance_m": 1.9},
    }
    nav.answers.extend([nav_view("CENTER", 1.95, "yellow vest green pants worker"),
                        verified(1.94), verified(1.90)])
    result = nav.m.navigate_to_object("that person")
    assert "OUTCOME=ARRIVED" in result
    assert nav.turns == [90]
    assert not nav.moves
    for e in ("TARGET_REFERENCE_RESOLVED", "SCAN_MEMORY_RECALLED", "SCAN_MEMORY_ALIGN",
              "SCAN_IDENTITY_LOCK_CONFIRMED", "TARGET_IDENTITY_VERIFIED"):
        assert e in event_names(nav)
    assert "IMMUTABLE" in nav.asks[1]


def test_scan_same_color_but_different_signature_refused(nav):
    nav.state["scene_scan_memory"] = {
        "created_monotonic": time.monotonic(),
        "target_query": "person with yellow vest",
        "best_target": {"target_yaw_deg": 0, "signature": "yellow vest green pants worker",
                        "scan_label": "forward", "distance_m": 4.0},
    }
    nav.answers.extend([nav_view("CENTER", 4, "yellow vest blue pants worker"),
                        nav_view("NOT_VISIBLE", 99, "")])
    result = nav.m.navigate_to_object("that person")
    assert "OUTCOME=TARGET_LOST" in result
    assert "TARGET_ATTRIBUTE_MISMATCH" in event_names(nav)
    assert not nav.moves


def test_safety_stop_is_not_target_range(nav, monkeypatch):
    def blocked(d, safe_dist=None):
        nav.moves.append((d, safe_dist))
        nav.state["last_motion_outcome"] = "BLOCKED"
        return 0.5
    monkeypatch.setattr(nav.m, "_move_dist", blocked)
    monkeypatch.setattr(nav.m, "check_obstacle", lambda *a: types.SimpleNamespace(
        valid=True, distance=1.2, source="lidar", reason=""))
    nav.answers.extend([nav_view("CENTER", 5.0, "yellow reflective worker"), verified(3.6)])
    result = nav.m.navigate_to_object("person in yellow vest")
    assert "OUTCOME=NEAR_TARGET_SAFETY_STOP" in result
    assert "TARGET_IDENTITY_VERIFIED" not in event_names(nav)
    assert len(nav.moves) == 1


def test_replan_counter_is_not_perpetually_zero(nav, monkeypatch):
    def timeout(d, safe_dist=None):
        nav.moves.append((d, safe_dist))
        nav.state["last_motion_outcome"] = "TIMEOUT"
        return 0.0
    monkeypatch.setattr(nav.m, "_move_dist", timeout)
    nav.cfg["max_legs"] = 1
    nav.answers.append(nav_view("CENTER", 5.0, "yellow reflective worker"))
    result = nav.m.navigate_to_object("person in yellow vest")
    assert "OUTCOME=MAX_REPLANS" in result
    assert nav.state["replan_count"] == 1


def test_scan_parser_rejects_missing_signature(nav):
    # Four mocked views and no valid scan targets; rotations simulated.
    nav.answers.extend(["TARGET_MATCH: YES\nTARGET_DIRECTION: CENTER\nTARGET_DISTANCE_M: 4.0",
                        "TARGET_MATCH: NO", "TARGET_MATCH: NO", "TARGET_MATCH: NO"])
    result = nav.m.look_around("person in yellow vest")
    assert "No visually valid match" in result
    assert nav.state["scene_scan_memory"]["best_target"] is None
    assert nav.m._wrap_deg(nav.state["yaw"]) == pytest.approx(0, abs=1e-6)


def test_scan_aborts_on_failed_rotation(nav, monkeypatch):
    def fail_rotate(deg):
        nav.turns.append(deg)
        nav.state["last_rotation_outcome"] = "ODOM_STALE"
    monkeypatch.setattr(nav.m, "_rotate", fail_rotate)
    nav.answers.append("TARGET_MATCH: YES\nTARGET_SIGNATURE: yellow vest worker")
    result = nav.m.look_around("person in yellow vest")
    assert "scan aborted" in result
    assert nav.state["scene_scan_memory"] is None
    assert "SCAN_ABORTED" in event_names(nav)


@pytest.fixture
def sensor(monkeypatch):
    st = {"use_lidar": False, "use_depth": True, "depth_map": np.ones((120, 160), dtype=float) * 2,
          "depth_ts": time.monotonic(), "depth_source_image_ts": time.monotonic()}
    cfg = {"has_lidar": False, "rear_obstacle_sensing": False,
           "depth_min_m": .15, "depth_max_m": 20.0, "depth_max_age_s": .75}
    packages = {"config": module("config", ROBOT=cfg, MAX_SENSOR_AGE_S=.75,
                                  SAFETY_CONFIG={"depth_strip_h": (1/3, .68),
                                                 "depth_strip_w": (.35, .65),
                                                 "depth_percentile": 8,
                                                 "min_valid_depth_pixels": 50}),
                "experiment_logger": module("experiment_logger", record_event=lambda *a, **k: None,
                                            record_safety_event=lambda *a, **k: None),
                "logger": module("logger", log=lambda *a, **k: None),
                "state": module("state", state=st)}
    with monkeypatch.context() as mp:
        for name, value in packages.items():
            mp.setitem(sys.modules, name, value)
        s = load_source("vrosa_test_safety", "safety.py")
        yield types.SimpleNamespace(m=s, state=st)
        sys.modules.pop("vrosa_test_safety", None)


def test_scout_depth_fresh_sensor_data_valid(sensor):
    reading = sensor.m.check_obstacle(1)
    assert reading.valid and reading.source == "depth"
    assert reading.distance == pytest.approx(2.0)


def test_scout_old_source_frame_fails_closed_even_with_fresh_completion(sensor):
    sensor.state["depth_source_image_ts"] = time.monotonic() - 2.5
    reading = sensor.m.check_obstacle(1)
    assert not reading.valid
    assert reading.stale
    assert "stale_depth" in reading.reason


def test_scout_missing_source_timestamp_fails_closed(sensor):
    sensor.state["depth_source_image_ts"] = None
    reading = sensor.m.check_obstacle(1)
    assert not reading.valid
    assert "depth_source_timestamp_missing" in reading.reason


def test_scout_rear_motion_has_no_safety(sensor):
    r = sensor.m.check_obstacle(-1)
    assert not r.valid and r.reason == "rear_safety_unavailable"


def test_static_arrival_range_helper(nav):
    assert nav.m._confirmed_standoff(True, 2.0, 2.0)
    assert not nav.m._confirmed_standoff(True, 2.0001, 2.0)
    assert not nav.m._confirmed_standoff(True, float("nan"), 2.0)
    assert not nav.m._confirmed_standoff(False, 1.8, 2.0)

@pytest.fixture
def exp_logger(monkeypatch, tmp_path):
    st = {"x": 0.0, "y": 0.0, "yaw": 0.0, "path_length_m": 0.0,
          "last_visual_target_memory": None, "last_nav_outcome": "",
          "tool_call_count": 0, "vlm_query_count": 0, "replan_count": 0,
          "safety_stop_count": 0, "sequence_error_count": 0, "recovery_success_count": 0}
    cfg = {"arrive_dist": 2.0, "display_name": "Nova Carter", "platform_label": "Isaac Sim",
           "yolo_model": "yolo11n", "depth_model": "DA-V2"}
    arg = types.SimpleNamespace(run_id="OFFLINE_TEST", log_dir=str(tmp_path), trial_id="",
                                scenario_id="", method="vrosa", robot="nova_carter_sim", seed=3407,
                                vlm="api", api_base="unused", api_model="mock", local_base="unused",
                                local_model="", remote_url="")
    vlm = types.SimpleNamespace(active_name="api", active_model="mock")
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    packages = {"torch": module("torch", cuda=fake_cuda),
                "config": module("config", ROBOT=cfg, PERCEPTION_CONFIG={},
                                 SAFETY_CONFIG={}, args=arg),
                "state": module("state", state=st),
                "vlm": module("vlm"),
                "vlm.manager": module("vlm.manager", vlm_mgr=vlm)}
    with monkeypatch.context() as mp:
        for name, value in packages.items():
            mp.setitem(sys.modules, name, value)
        m = load_source("vrosa_test_exp_logger", "experiment_logger.py")
        yield types.SimpleNamespace(m=m, state=st)
        sys.modules.pop("vrosa_test_exp_logger", None)


def test_experiment_success_requires_all_three_criteria(exp_logger):
    import csv
    m, st = exp_logger.m, exp_logger.state

    def run(outcome, identity, distance, source):
        m.begin_trial("go to yellow person", target="yellow person")
        st["last_visual_target_memory"] = {"identity_confirmed": identity}
        st["last_nav_outcome"] = outcome
        m.end_trial(outcome=outcome, ground_truth_final_distance_m=distance,
                    ground_truth_source=source)

    run("ARRIVED", True, 1.9, "independent_sim_pose")
    run("BLOCKED", True, 1.8, "independent_sim_pose")
    run("ARRIVED", False, 1.9, "independent_sim_pose")
    run("ARRIVED", True, 2.1, "independent_sim_pose")
    run("ARRIVED", True, 1.8, "")
    rows = list(csv.DictReader(m.TRIALS_PATH.open()))
    assert [r["experimental_success"] for r in rows] == ["True", "False", "False", "False", ""]
    assert rows[-1]["ground_truth_source"] == "unverified_or_missing"
    assert rows[-1]["ground_truth_final_distance_m"] == ""
    assert rows[-1]["success_evaluable"] == "False"
    assert all(r["gt_arrival_threshold_m"] == "2.0" for r in rows)


def test_experiment_does_not_accept_odom_distance_as_ground_truth(exp_logger):
    import csv
    m, st = exp_logger.m, exp_logger.state
    m.begin_trial("command")
    st["last_visual_target_memory"] = {"identity_confirmed": True}
    m.end_trial(outcome="ARRIVED", ground_truth_final_distance_m=1.0,
                ground_truth_source="odom_to_target")
    row = list(csv.DictReader(m.TRIALS_PATH.open()))[-1]
    assert row["ground_truth_final_distance_m"] == ""
    assert row["experimental_success"] == ""


def test_end_trial_resets_identity_for_next_trial(exp_logger):
    m, st = exp_logger.m, exp_logger.state
    st["last_visual_target_memory"] = {"identity_confirmed": True}
    m.begin_trial("fresh command")
    assert st["last_visual_target_memory"] is None


def test_scan_then_deictic_navigation_full_mock_sequence(nav):
    # One mock 360 scan discovers worker at WEST; follow-up must reuse signature.
    nav.answers.extend([
        "TARGET_MATCH: NO",
        "TARGET_MATCH: YES\nTARGET_DIRECTION: CENTER\nTARGET_DISTANCE_M: 4.0\n"
        "TARGET_SIGNATURE: yellow vest green pants worker",
        "TARGET_MATCH: NO", "TARGET_MATCH: NO",
    ])
    result = nav.m.look_around("person with yellow vest")
    assert "Remembered target" in result
    assert nav.state["scene_scan_memory"]["best_target"] is not None
    nav.answers.extend([
        nav_view("CENTER", 1.9, "yellow vest green pants worker"),
        verified(1.9), verified(1.9),
    ])
    result = nav.m.navigate_to_object("that person")
    assert "OUTCOME=ARRIVED" in result
    assert "SCAN_MEMORY_RECALLED" in event_names(nav)
    assert "SCAN_IDENTITY_LOCK_CONFIRMED" in event_names(nav)
    assert len(nav.moves) == 0


def test_formal_safety_thresholds_not_weakened():
    text = (ROOT / "config.py").read_text()
    import ast
    tree = ast.parse(text)
    profiles = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "ROBOT_PROFILES" for t in n.targets)))
    for cfg in profiles.values():
        assert cfg["hard_stop_distance"] == pytest.approx(0.75)
        assert cfg["stop_trigger_distance"] >= cfg["hard_stop_distance"]
        assert cfg["odom_max_age_s"] == pytest.approx(.30)
        assert cfg["depth_max_age_s"] == pytest.approx(.75)

@pytest.fixture
def motion_unit(monkeypatch):
    published = []
    st = {"x": 0.0, "y": 0.0, "yaw": 0.0, "odom_ts": None,
          "linear_velocity": 0.0, "angular_velocity": 0.0,
          "last_motion_outcome": None, "last_motion_reason": ""}
    cfg = {"hard_stop_distance": .75, "stop_trigger_distance": .75,
           "max_linear_speed": .4, "max_angular_speed": .4,
           "slowdown_margin": .5, "odom_max_age_s": .30,
           "move_position_tolerance_m": .03, "move_slowdown_distance_m": .40,
           "move_min_speed_mps": .06, "move_kp": 1.2,
           "rear_obstacle_sensing": False}

    class Twist:
        def __init__(self):
            self.linear = types.SimpleNamespace(x=0.0)
            self.angular = types.SimpleNamespace(z=0.0)

    @contextlib.contextmanager
    def timed(*a, **kw):
        yield

    record = []
    pkgs = {
        "geometry_msgs": module("geometry_msgs"),
        "geometry_msgs.msg": module("geometry_msgs.msg", Twist=Twist),
        "config": module("config", ROBOT=cfg, CONTROL_PERIOD_S=.001,
                         STOP_ANGULAR_EPS_RPS=.03, STOP_STABLE_TIME_S=.01,
                         STOP_VELOCITY_EPS_MPS=.02, STOP_WAIT_TIMEOUT_S=.05),
        "experiment_logger": module("experiment_logger", record_event=lambda *a, **kw: record.append(a[0]),
                                    timed=timed),
        "logger": module("logger", log=lambda *a, **kw: None),
        "safety": module("safety", check_obstacle=lambda d: types.SimpleNamespace(
            valid=True, distance=99., source="lidar", age_s=0., reason=""),
                         log_stop_measurement=lambda **kw: None,
                         record_safety_unknown=lambda *a: None),
        "state": module("state", state=st, interrupt_flag=types.SimpleNamespace(is_set=lambda: False)),
        "ros_node": module("ros_node", cmd_pub=types.SimpleNamespace(publish=lambda x: published.append(x))),
    }
    with monkeypatch.context() as mp:
        for name, obj in pkgs.items():
            mp.setitem(sys.modules, name, obj)
        obj = load_source("vrosa_test_motion", "motion.py")
        yield types.SimpleNamespace(m=obj, state=st, pub=published, events=record)
        sys.modules.pop("vrosa_test_motion", None)


def test_rotation_aborts_before_motion_on_stale_odometry(motion_unit):
    motion_unit.state["odom_ts"] = time.monotonic() - 3
    motion_unit.m._rotate(90)
    assert motion_unit.state["last_rotation_outcome"] == "ODOM_STALE"
    assert all(t.angular.z == 0.0 for t in motion_unit.pub)
    assert "ROTATION_ABORTED" in motion_unit.events


def test_rotation_aborts_when_odometry_stales_during_turn(motion_unit, monkeypatch):
    st = motion_unit.state
    st["odom_ts"] = time.monotonic()
    call_count = [0]
    def drive(linear=0., angular=0., **kw):
        call_count[0] += 1
        # Stop receiving odometry immediately after the first command.
        st["odom_ts"] = time.monotonic() - 2
    monkeypatch.setattr(motion_unit.m, "_drive", drive)
    motion_unit.m._rotate(90)
    assert call_count[0] == 1
    assert st["last_rotation_outcome"] == "ODOM_STALE"
    assert "ROTATION_RESULT" in motion_unit.events


def test_motion_fails_closed_without_safety_sensor(motion_unit, monkeypatch):
    st = motion_unit.state
    st["odom_ts"] = time.monotonic()
    monkeypatch.setattr(motion_unit.m, "check_obstacle", lambda direction: types.SimpleNamespace(
        valid=False, distance=None, source="none", age_s=None, reason="missing_sensor"))
    moved = motion_unit.m._move_dist(1.0)
    assert moved == pytest.approx(0.0)
    assert st["last_motion_outcome"] == "SAFETY_SENSOR_UNAVAILABLE"
    assert not any(t.linear.x > 0.0 for t in motion_unit.pub)


def test_motion_refuses_blind_reverse_on_scout_policy(motion_unit):
    moved = motion_unit.m._move_dist(-1.0)
    assert moved == 0.0
    assert motion_unit.state["last_motion_outcome"] == "PRECONDITION_FAILED"
    assert "REVERSE_REJECTED" in motion_unit.events


def test_unconfirmed_zero_velocity_does_not_create_fake_latency(motion_unit, monkeypatch):
    log_rows = []
    monkeypatch.setattr(motion_unit.m, "log_stop_measurement", lambda **kw: log_rows.append(kw))
    monkeypatch.setattr(motion_unit.m, "_stop", lambda **kw: {
        "confirmed": False, "zero_time": time.monotonic(),
        "stop_command_time": time.monotonic(), "x": 0.0, "y": 0.0,
        "min_clearance": None,
    })
    motion_unit.m._record_stop_event(
        event_type="sensor_fail_stop", source="none", direction=1,
        commanded_speed=.4, trigger_distance=None, reading_age_s=None,
        recognition_time=time.monotonic(), x0=0.0, y0=0.0,
        velocity_at_trigger=.4,
    )
    assert log_rows[0]["zero_velocity_confirmed"] is False
    assert log_rows[0]["zero_velocity_time_mono"] is None
    assert log_rows[0]["braking_latency_ms"] is None
    assert log_rows[0]["total_stop_latency_ms"] is None
    assert log_rows[0]["threshold_violation"] is None

# Heading regressions from October 23 real Isaac logs: the VLM repeatedly
# returned CENTER while the visually selected person drifted in the frame.
# Tests use the same 1920x1200 / fx 957.81 intrinsics as that log.
def view_with_x(direction, dist, signature, target_x):
    return (f"DIRECTION: {direction}\nDISTANCE_M: {dist}\n"
            f"SIGNATURE: {signature}\nTARGET_X: {target_x}")


@pytest.mark.parametrize("target_x,expected_sign", [(0.20, 1), (0.80, -1)])
def test_center_word_cannot_suppress_image_heading_correction(nav, target_x, expected_sign):
    nav.state.update(camera_width=1920, cam_fx=957.8115559493878, cam_cx=960.0)
    nav.cfg["max_legs"] = 1
    nav.answers.append(view_with_x("CENTER", 8.0, "yellow reflective worker", target_x))
    result = nav.m.navigate_to_object("person in yellow vest")
    assert "OUTCOME=MAX_REPLANS" in result
    assert len(nav.turns) == 1
    assert nav.turns[0] * expected_sign > 15.0
    assert abs(nav.turns[0]) <= 30.0
    assert "TARGET_IMAGE_ALIGNMENT" in event_names(nav)
    assert "NAV_ALIGN" in event_names(nav)
    assert len(nav.moves) == 1  # move only after a successful heading correction


def test_center_of_frame_requires_no_spurious_yaw(nav):
    nav.state.update(camera_width=1920, cam_fx=957.8115559493878, cam_cx=960.0)
    nav.cfg["max_legs"] = 1
    nav.answers.append(view_with_x("CENTER", 8, "yellow reflective worker", 0.50))
    nav.m.navigate_to_object("person in yellow vest")
    assert nav.turns == []


def test_opposite_direction_and_x_stops_for_recheck(nav):
    nav.state.update(camera_width=1920, cam_fx=957.8115559493878, cam_cx=960.0)
    nav.cfg["max_legs"] = 1
    nav.answers.append(view_with_x("LEFT", 8, "yellow reflective worker", 0.85))
    result = nav.m.navigate_to_object("person in yellow vest")
    assert "OUTCOME=MAX_REPLANS" in result
    assert "NAV_DIRECTION_CONFLICT" in event_names(nav)
    assert nav.moves == []
    assert nav.turns == []


def test_missing_image_x_preserves_old_coarse_20_degree_turn(nav):
    nav.cfg["max_legs"] = 1
    nav.answers.append(nav_view("LEFT", 8, "yellow reflective worker"))
    nav.m.navigate_to_object("person with yellow vest")
    assert nav.turns == [20.0]


def test_missing_target_is_not_steered_from_unverified_yolo_person(nav):
    nav.cfg["max_legs"] = 1
    nav.state["detections"] = [{"label": "person", "bbox": [1650, 300, 1850, 650],
                                "sector": "right", "conf": .99, "image_seq": 10}]
    nav.answers.append(nav_view("CENTER", 8, "yellow reflective worker"))
    nav.m.navigate_to_object("person with yellow vest")
    # The raw detector box could belong to the red/white distractor; the VLM
    # didn't select an image x for the yellow target, so do NOT turn toward it.
    assert nav.turns == []


def test_stale_odom_during_alignment_cannot_cause_blind_translation(nav, monkeypatch):
    nav.cfg["max_legs"] = 1
    def abort_rotation(deg):
        nav.turns.append(deg)
        nav.state["last_rotation_outcome"] = "ODOM_STALE"
    monkeypatch.setattr(nav.m, "_rotate", abort_rotation)
    nav.answers.append(view_with_x("CENTER", 10, "yellow reflective worker", 0.15))
    result = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=ODOM_STALE" in result
    assert nav.moves == []


def test_scan_align_then_correct_frame_left_before_motion(nav):
    nav.state.update(camera_width=1920, cam_fx=957.8115559493878, cam_cx=960.0)
    nav.cfg["max_legs"] = 1
    nav.state["scene_scan_memory"] = {
        "created_monotonic": time.monotonic(),
        "target_query": "person with yellow vest",
        "best_target": {"target_yaw_deg": 90, "signature": "yellow vest green pants worker",
                        "scan_label": "West", "distance_m": 15.0},
    }
    nav.answers.extend([view_with_x("CENTER", 15.0, "yellow vest green pants worker", 0.25),
                        verified(15.0)])
    result = nav.m.navigate_to_object("that person")
    assert "OUTCOME=MAX_REPLANS" in result
    assert len(nav.turns) == 2
    assert nav.turns[0] == pytest.approx(90.0)
    assert nav.turns[1] > 20.0  # left correction after scan alignment
    assert len(nav.moves) == 1
    assert "SCAN_IDENTITY_LOCK_CONFIRMED" in event_names(nav)
    assert "TARGET_IMAGE_ALIGNMENT" in event_names(nav)


def test_side_word_with_center_x_requires_recheck(nav):
    nav.state.update(camera_width=1920, cam_fx=957.8115559493878, cam_cx=960.0)
    nav.cfg["max_legs"] = 1
    nav.answers.append(view_with_x("LEFT", 8.0, "yellow reflective worker", 0.50))
    nav.m.navigate_to_object("person in yellow vest")
    assert "NAV_DIRECTION_CONFLICT" in event_names(nav)
    assert nav.moves == []


def test_sim_visual_nav_preflight_allows_only_controlled_mock_step(nav):
    """Missing LiDAR can pass preflight only when explicitly in Isaac Sim test mode."""
    nav.cfg.update(max_legs=1, platform_label="Isaac Sim",
                   sim_visual_clearance_test=True, sim_visual_min_target_range_m=3.0)
    nav.m.check_obstacle = lambda *a: types.SimpleNamespace(valid=False, distance=None,
                                                            source="none", reason="stale_lidar;no_depth")
    nav.answers.append(nav_view("CENTER", 7., "person wearing yellow reflective vest"))
    def move(d, safe_dist=None):
        nav.moves.append((d, safe_dist))
        nav.state["last_motion_outcome"] = "SIM_VISUAL_SEGMENT_COMPLETED"
        nav.state["x"] += .15
        return .15
    nav.m._move_dist = move
    out = nav.m.navigate_to_object("person with yellow vest")
    assert "SIM_VISUAL_PREFLIGHT_PENDING" in event_names(nav)
    assert "SIM_VISUAL_RECHECK_REQUIRED" in event_names(nav)
    assert len(nav.moves) == 1
    assert nav.state["nav_visual_distance_m"] is None
    assert "OUTCOME=MAX_REPLANS" in out


def test_sim_visual_nav_preflight_remains_blocked_on_hardware(nav):
    nav.cfg.update(max_legs=1, platform_label="Hardware",
                   sim_visual_clearance_test=True, sim_visual_min_target_range_m=3.0)
    nav.m.check_obstacle = lambda *a: types.SimpleNamespace(valid=False, distance=None,
                                                            source="none", reason="no_depth")
    nav.answers.append(nav_view("CENTER", 7., "person wearing yellow reflective vest"))
    out = nav.m.navigate_to_object("person with yellow vest")
    assert "OUTCOME=SAFETY_SENSOR_UNAVAILABLE" in out
    assert not nav.moves
    assert nav.state["nav_visual_distance_m"] is None
