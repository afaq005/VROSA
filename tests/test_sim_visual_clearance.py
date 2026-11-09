"""Offline controls for opt-in, strictly Isaac-Sim-only visual clearance tests.

The VLM and motion are mocked. No physical safety is established by these tests.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def m(name, **fields):
    obj = types.ModuleType(name)
    obj.__dict__.update(fields)
    return obj


@pytest.fixture
def v(monkeypatch):
    class World:
        now = 100.0
        velocity = 0.0
        vlm_answer = "CLEAR"
        events = []
        asks = []
        cmds = []
        estop = False
        sensor = "stale"
        sensor_after_vlm = None
        odom_after_vlm = True
        advance_camera = True

    w = World()
    state = dict(x=0., y=0., yaw=0., odom_ts=99.99, image_ts=99.99,
                 image_raw=object(), lidar_ts=99.0,
                 nav_visual_distance_m=7.0,
                 sim_visual_fallback_used_m=0., linear_velocity=0., angular_velocity=0.,
                 safety_stop_count=0, vlm_query_count=0)
    cfg = dict(platform_label="Isaac Sim", sim_visual_clearance_test=True,
               sim_visual_grant_m=.20, sim_visual_total_budget_m=.8,
               sim_visual_max_speed_mps=.08, sim_visual_min_target_range_m=3.,
               max_linear_speed=.4, max_angular_speed=.4, odom_max_age_s=.3,
               hard_stop_distance=.75, stop_trigger_distance=.75,
               slowdown_margin=.50, rear_obstacle_sensing=False,
               move_position_tolerance_m=.03, move_slowdown_distance_m=.4,
               move_min_speed_mps=.06, move_kp=1.2)

    def sleep(seconds):
        w.now += seconds
        state["x"] += w.velocity * seconds
        state["odom_ts"] = w.now
        state["image_ts"] = w.now

    class VLM:
        def ask(self, prompt, image, max_tokens=8):
            w.asks.append(prompt)
            if w.sensor_after_vlm is not None:
                w.sensor = w.sensor_after_vlm
            state["vlm_query_count"] += 1
            w.now += (.50 if not w.odom_after_vlm else .02)
            if w.advance_camera:
                state["image_ts"] = w.now
            if w.odom_after_vlm:
                state["odom_ts"] = w.now
            return w.vlm_answer

    def stop(*args, **kwargs):
        w.velocity = 0.0
        return {"confirmed": True, "x": state["x"], "y": state["y"]}

    def drive(linear=0., angular=0., **kwargs):
        w.cmds.append(linear)
        w.velocity = linear

    def reading(*args):
        if w.sensor == "clear":
            return types.SimpleNamespace(valid=True, distance=9., source="lidar", reason="", age_s=.01)
        if w.sensor == "blocked":
            return types.SimpleNamespace(valid=True, distance=.5, source="lidar", reason="", age_s=.01)
        return types.SimpleNamespace(valid=False, distance=None, source="none", reason="stale_lidar;no_depth", age_s=.8)

    @__import__('contextlib').contextmanager
    def timed(*args, **kwargs):
        yield

    libs = {
        "geometry_msgs": m("geometry_msgs"),
        "geometry_msgs.msg": m("geometry_msgs.msg", Twist=type("Twist", (), {})),
        "config": m("config", ROBOT=cfg, CONTROL_PERIOD_S=.05,
                    STOP_ANGULAR_EPS_RPS=.03, STOP_STABLE_TIME_S=.20,
                    STOP_VELOCITY_EPS_MPS=.02, STOP_WAIT_TIMEOUT_S=3.0),
        "state": m("state", state=state, interrupt_flag=types.SimpleNamespace(is_set=lambda: w.estop)),
        "experiment_logger": m("experiment_logger", record_event=lambda name, **kw: w.events.append((name, kw)), timed=timed),
        "logger": m("logger", log=lambda *args, **kwargs: None),
        "safety": m("safety", check_obstacle=reading,
                    record_safety_unknown=lambda *args: w.events.append(("SENSOR_UNAVAILABLE", {})),
                    log_stop_measurement=lambda **kwargs: None),
        "vlm": m("vlm"), "vlm.manager": m("vlm.manager", vlm_mgr=VLM()),
    }
    with monkeypatch.context() as patch:
        for name, dep in libs.items():
            patch.setitem(sys.modules, name, dep)
        spec = importlib.util.spec_from_file_location("vrosa_test_motion_visual", ROOT / "motion.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.time = types.SimpleNamespace(monotonic=lambda: w.now, sleep=sleep)
        module._stop = stop
        module._drive = drive
        module._record_stop_event = lambda **kwargs: stop()
        yield types.SimpleNamespace(m=module, state=state, cfg=cfg, w=w)
        sys.modules.pop(spec.name, None)


def events(v):
    return [n for n, kw in v.w.events]


def test_default_mode_rejects_missing_sensors(v):
    v.cfg["sim_visual_clearance_test"] = False
    assert v.m._move_dist(1.0) == 0
    assert v.state["last_motion_outcome"] == "SAFETY_SENSOR_UNAVAILABLE"
    assert not v.w.asks


def test_sim_visual_clear_issues_only_bounded_creep(v):
    d = v.m._move_dist(1.0)
    assert 0.0 < d <= .22
    assert v.state["last_motion_outcome"] == "SIM_VISUAL_SEGMENT_COMPLETED"
    assert len(v.w.asks) == 1
    assert max(v.w.cmds) <= .08
    assert v.state["sim_visual_fallback_used_m"] <= .22
    assert "SIM_VISUAL_CLEARANCE_GRANTED" in events(v)


@pytest.mark.parametrize("answer", ["BLOCKED", "UNCERTAIN", "probably CLEAR", "", "CLEAR BUT NOT CERTAIN"])
def test_vlm_must_return_exact_clear(v, answer):
    v.w.vlm_answer = answer
    assert v.m._move_dist(1.0) == 0
    assert v.state["last_motion_outcome"] == "SAFETY_SENSOR_UNAVAILABLE"
    assert not v.w.cmds or max(v.w.cmds) <= 0
    assert "SIM_VISUAL_CLEARANCE_DENIED" in events(v)


def test_no_camera_frame_during_query_denies(v):
    v.w.advance_camera = False
    assert not v.m._request_sim_visual_clearance()
    assert "SIM_VISUAL_CLEARANCE_DENIED" in events(v)


def test_no_camera_at_start_denies(v):
    v.state["image_raw"] = None
    assert v.m._move_dist(1.0) == 0
    assert not v.w.asks


def test_stale_odom_after_vlm_denies(v):
    v.w.odom_after_vlm = False
    assert not v.m._request_sim_visual_clearance()


def test_sim_mode_never_allowed_on_scout(v):
    v.cfg["platform_label"] = "Hardware"
    assert v.m._move_dist(1.) == 0
    assert not v.w.asks


def test_near_target_no_visual_only_forward(v):
    v.state["nav_visual_distance_m"] = 2.4
    assert v.m._move_dist(1.) == 0
    assert not v.w.asks


def test_lidar_entirely_absent_does_not_initiate_fallback(v):
    v.state["lidar_ts"] = None
    assert v.m._move_dist(1.) == 0
    assert not v.w.asks


def test_lidar_stale_more_than_two_seconds_cannot_initiate(v):
    v.state["lidar_ts"] = 90.0
    assert v.m._move_dist(1.) == 0
    assert not v.w.asks


def test_visual_budget_exhausted(v):
    v.state["sim_visual_fallback_used_m"] = 0.80
    assert v.m._move_dist(1.) == 0
    assert not v.w.asks


def test_estop_blocks_visual_mode(v):
    v.w.estop = True
    assert v.m._move_dist(1.) == 0
    assert not v.w.asks


def test_fresh_lidar_obstacle_wins_even_in_test_mode(v):
    v.w.sensor = "blocked"
    assert v.m._move_dist(1.) == 0
    assert v.state["last_motion_outcome"] == "BLOCKED"
    assert not v.w.asks


def test_forward_sensor_recovery_during_creep_restores_normal_monitor(v):
    original_drive = v.m._drive
    def drive_and_recover(*args, **kwargs):
        original_drive(*args, **kwargs)
        if kwargs.get("linear", 0) > 0:
            v.w.sensor = "clear"
    v.m._drive = drive_and_recover
    moved = v.m._move_dist(.50)
    assert moved >= .45
    assert v.state["last_motion_outcome"] == "SUCCESS"
    assert "SIM_VISUAL_SENSOR_RECOVERED" in events(v)
    assert v.state["sim_visual_fallback_used_m"] >= 0


def test_estop_while_creeping_immediately_stops(v):
    original_drive = v.m._drive
    def drive_and_estop(*args, **kwargs):
        original_drive(*args, **kwargs)
        if kwargs.get("linear", 0) > 0:
            v.w.estop = True
    v.m._drive = drive_and_estop
    v.m._move_dist(1.)
    assert v.state["last_motion_outcome"] == "INTERRUPTED"
    assert "SIM_VISUAL_CLEARANCE_GRANTED" in events(v)


def test_fresh_lidar_returning_during_clear_query_resumes_sensor_navigation(v):
    """Reproduce real run: VLM says CLEAR, LiDAR returns before answer."""
    v.w.sensor_after_vlm = "clear"
    moved = v.m._move_dist(.50)
    assert moved >= .45
    assert v.state["last_motion_outcome"] == "SUCCESS"
    assert "SENSOR_UNAVAILABLE" not in events(v)
    assert "SIM_VISUAL_CLEARANCE_GRANTED" in events(v)
    assert max(v.w.cmds) > .08  # full sensor-regulated motion, not a creep


def test_fresh_lidar_obstacle_returning_during_clear_query_blocks(v):
    """Sensor recovery must not override a physically present obstacle."""
    v.w.sensor_after_vlm = "blocked"
    assert v.m._move_dist(.50) == 0
    assert v.state["last_motion_outcome"] == "BLOCKED"
    assert not v.w.cmds


def test_lidar_returning_during_denied_visual_query_resumes_sensor_navigation(v):
    """Fresh range sensing takes priority even if image-based grant is denied."""
    v.w.vlm_answer = "BLOCKED"
    v.w.sensor_after_vlm = "clear"
    moved = v.m._move_dist(.50)
    assert moved >= .45
    assert v.state["last_motion_outcome"] == "SUCCESS"
    assert "SIM_VISUAL_CLEARANCE_DENIED" in events(v)
    assert "SENSOR_UNAVAILABLE" not in events(v)
