"""Hardware invariants; no robot/ROS/Isaac required for the offline tests."""
import ast
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_hardware_defaults_safe_and_physical_depth_required():
    cfg = (ROOT / "config.py").read_text()
    assert 'default="diagnostic"' in cfg
    assert '--hardware-mode' in cfg
    assert '--hardware-use-range-depth' in cfg
    assert '--hardware-max-linear' in cfg
    assert '"stop_trigger_distance": 1.10' in cfg
    assert '"hard_stop_distance": 0.75' in cfg
    assert 'baze_' not in cfg
    assert '"OPENAI_API_KEY", ""' in cfg
    assert 'sim-visual-clearance-test is permitted only with --robot nova_carter_sim' in cfg


def test_preflight_is_read_only_and_does_not_import_motion():
    source = (ROOT / "hardware_preflight.py").read_text()
    tree = ast.parse(source)
    imports = [x.module for x in ast.walk(tree) if isinstance(x, ast.ImportFrom)]
    assert 'motion' not in imports
    assert 'tools' not in imports
    assert 'create_publisher' not in source
    assert 'motion_authorized": False' in source


def test_scout_diag_launch_parsing_and_reject_unsafe_args():
    proc = subprocess.run([sys.executable, "config.py", "--robot", "scout2"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run([sys.executable, "config.py", "--robot", "scout2",
                           "--hardware-stop-trigger", "0.70"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert 'strictly greater than hard stop' in proc.stderr
    proc = subprocess.run([sys.executable, "config.py", "--robot", "scout2",
                           "--sim-visual-clearance-test"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0


def test_physical_depth_reading_and_absent_sensor_fails_closed(monkeypatch):
    import importlib
    fake_cfg = types.ModuleType('config')
    fake_cfg.ROBOT = dict(platform_label='Hardware', has_lidar=False,
                          depth_min_m=.15, depth_max_m=20., depth_max_age_s=.75,
                          lidar_max_age_s=.75, rear_obstacle_sensing=False)
    fake_cfg.MAX_SENSOR_AGE_S = .75
    fake_cfg.SAFETY_CONFIG = dict(depth_strip_h=(1/3,.68), depth_strip_w=(.35,.65),
                                  depth_percentile=8, min_valid_depth_pixels=50)
    state = {'use_range_depth':True,'use_depth':True,'use_lidar':False,
             'range_depth_map':None,'range_depth_ts':None, 'depth_map':np.full((64,64),20.),
             'depth_ts':time.monotonic(),'depth_source_image_ts':time.monotonic()}
    fake_state=types.ModuleType('state'); fake_state.state=state
    fake_logs=types.ModuleType('logger'); fake_logs.log=lambda *a,**kw: None
    fake_events=types.ModuleType('experiment_logger')
    fake_events.record_event=lambda *a,**kw: None
    fake_events.record_safety_event=lambda *a,**kw: None
    with monkeypatch.context() as m:
        m.setitem(sys.modules,'config',fake_cfg)
        m.setitem(sys.modules,'state',fake_state)
        m.setitem(sys.modules,'logger',fake_logs)
        m.setitem(sys.modules,'experiment_logger',fake_events)
        # load using file location so module cache from other tests is untouched
        import importlib.util
        spec=importlib.util.spec_from_file_location('_isolated_hw_safety',ROOT/'safety.py')
        safety=importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = safety
        spec.loader.exec_module(safety)
        assert not safety.check_obstacle(1).valid  # model depth must NOT permit motion
        state['range_depth_map']=np.full((72,96), 2.5, dtype=np.float32)
        state['range_depth_ts']=time.monotonic()
        result=safety.check_obstacle(1)
        assert result.valid and result.source=='realsense_depth' and result.distance==2.5
        assert not safety.check_obstacle(-1).valid
        state['range_depth_ts']=time.monotonic()-2.0
        assert not safety.check_obstacle(1).valid
        state['use_range_depth']=False
        state['range_depth_ts']=time.monotonic()
        assert not safety.check_obstacle(1).valid
        del sys.modules[spec.name]


def test_ros_realsense_decode_rejects_bad_encoding():
    src=(ROOT/'ros_node.py').read_text()
    assert '16uc1' in src and '32fc1' in src
    assert 'Unsupported physical depth encoding' in src
    assert 'range_depth_conversion_errors' in src
    assert 'LIDAR_QOS, callback_group=self._range_depth_group' in src


def test_hardware_command_gate_and_capped_velocity():
    src=(ROOT/'motion.py').read_text()
    assert 'hardware_diagnostic_mode_no_motion' in src
    assert 'hardware_rgb_missing_or_stale' in src
    assert 'hardware_physical_safety_missing' in src
    assert 'hardware_odometry_missing_or_stale' in src
    assert 'if not allowed:' in src
    assert 'lin = ang = 0.0' in src


def test_hardware_estop_cannot_auto_rearm_by_new_command():
    src=(ROOT/'server.py').read_text()
    assert 'Hardware software E-stop latched' in src
    assert 'RESET_SOFTWARE_ESTOP' in src
    assert '@app.route("/hardware/reset-estop", methods=["POST"])' in src
    assert 'ROBOT.get("platform_label") != "Hardware":\n        interrupt_flag.clear()' in src


def test_scout_dashboard_local_only():
    src=(ROOT/'vrosa_live.py').read_text()
    assert '"127.0.0.1" if ROBOT["platform_label"] == "Hardware"' in src


def test_real_hardware_motion_preconditions_via_extracted_function():
    """Execute the production authorization function with injected real conditions."""
    import math
    import ast
    import types
    text = (ROOT/'motion.py').read_text()
    fn = next(n for n in ast.parse(text).body
              if isinstance(n, ast.FunctionDef) and n.name == '_hardware_motion_check')
    code = compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                   str(ROOT/'motion.py'), 'exec')
    now = time.monotonic()
    sensor = types.SimpleNamespace(valid=True, distance=3.0, reason='')
    profile = {'platform_label':'Hardware', 'hardware_mode':'diagnostic'}
    state = {'odom_ts':now, 'image_ts':now}
    stopped = types.SimpleNamespace(is_set=lambda:False)
    scope = dict(ROBOT=profile, state=state, interrupt_flag=stopped, time=time,
                 ODOM_MAX_AGE_S=.30, STOP_TRIGGER_DIST=1.1, math=math,
                 check_obstacle=lambda direction:sensor)
    exec(code,scope)
    gate=scope['_hardware_motion_check']
    assert gate()[0] is False and 'diagnostic' in gate()[1]
    profile['hardware_mode']='guarded'
    assert gate() == (True,'')
    sensor.valid=False
    sensor.reason='no_physical_depth'
    assert gate()[0] is False
    sensor.valid=True
    sensor.distance=0.85
    assert gate()[0] is False
    sensor.distance=3.
    state['image_ts']=now-3.
    assert 'rgb_missing_or_stale' in gate()[1]
    state['image_ts']=now
    state['odom_ts']=now-3.
    assert 'odometry_missing_or_stale' in gate()[1]
    state['odom_ts']=now
    scope['interrupt_flag']=types.SimpleNamespace(is_set=lambda:True)
    assert 'emergency_stop_latched' in gate()[1]
