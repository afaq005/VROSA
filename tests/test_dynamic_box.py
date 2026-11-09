"""Offline validation of continuous crossing geometry and test-harness wiring."""
from __future__ import annotations

import ast
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dynamic_box_math import (box_lateral, project_xy, rectangle_clearance,
                              world_xy, unit_xy)


def test_crossing_keeps_velocity_as_simulation_steps_advance():
    dt = 1/60
    speed = 0.35
    values = [box_lateral(1.4, -1.4, speed, i * dt) for i in range(200)]
    assert all(b <= a for a, b in zip(values, values[1:]))
    for a, b in zip(values[:50], values[1:51]):
        assert b-a == pytest.approx(-speed*dt, abs=1e-7)


def test_crossing_end_cannot_overshoot():
    assert box_lateral(1.4, -1.4, .5, 100) == pytest.approx(-1.4)
    assert box_lateral(-1.4, 1.4, .5, 100) == pytest.approx(1.4)


def test_paused_simulation_does_not_advance_box():
    assert box_lateral(1.4, -1.4, .3, 0) == pytest.approx(1.4)


def test_invalid_speed_or_time_is_rejected():
    with pytest.raises(ValueError): box_lateral(1, -1, 0, 2)
    with pytest.raises(ValueError): box_lateral(1, -1, 0.3, -1)
    with pytest.raises(ValueError): box_lateral(1, -1, .3, math.inf)


def test_world_projection_inverts_for_rotated_robot():
    origin = (2.4, -3.0)
    fwd = unit_xy(.6, .8)
    world = world_xy(origin, fwd, 3, 1.4)
    along, left = project_xy(origin, fwd, world)
    assert along == pytest.approx(3)
    assert left == pytest.approx(1.4)


def test_invalid_robot_forward_disallowed():
    with pytest.raises(ValueError): unit_xy(0.0, 0.0)


def test_independent_gt_clearance_proxy():
    assert rectangle_clearance(2, 0, .375, .275, .4, .4) == pytest.approx(1.225)
    assert rectangle_clearance(.2, 0, .375, .275, .4, .4) == 0
    assert rectangle_clearance(0, 2, .375, .275, .4, .4) == pytest.approx(1.325)


def test_box_off_corridor_initially_and_crosses_corridor():
    robot_half_width = .275
    box_half_width = .4
    assert abs(1.4) > robot_half_width+box_half_width
    assert abs(box_lateral(1.4, -1.4, .3, 5)) <= robot_half_width + box_half_width


def test_script_creates_collidable_kinematic_cube_and_events():
    txt = (ROOT/'isaac_dynamic_box.py').read_text()
    assert 'UsdPhysics.CollisionAPI.Apply(prim)' in txt
    assert 'UsdPhysics.RigidBodyAPI.Apply(prim)' in txt
    assert 'rigid.CreateKinematicEnabledAttr(True)' in txt
    assert 'subscribe_physics_step_events' in txt
    assert 'BOX_ENTERED_ROBOT_CORRIDOR' in txt
    assert 'BOX_CROSSING_START' in txt
    assert 'monotonic_time' in txt
    assert 'dynamic_box_gt.csv' in txt
    assert 'self.elapsed += dt' in txt


def test_harness_marks_dynamic_true_and_rejects_hardware():
    txt = (ROOT/'safety_test.py').read_text()
    tree = ast.parse(txt)
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=='run_dynamic_crossing')
    fntext = ast.get_source_segment(txt, fn)
    assert 'dynamic_obstacle=True' in fntext
    assert 'if not ROBOT.get("has_lidar")' in fntext
    assert 'ROBOT.get("sim_visual_clearance_test")' in fntext
    assert 'motion._move_dist' in fntext
    assert '"dynamic_crossing": run_dynamic_crossing' in txt


def test_margin_keeps_hard_stop_and_scout_unchanged():
    tree=ast.parse((ROOT/'config.py').read_text())
    profiles=next(x.value for x in tree.body if isinstance(x, ast.Assign) and
         any(isinstance(t,ast.Name) and t.id=='ROBOT_PROFILES' for t in x.targets))
    vals=ast.literal_eval(profiles)
    assert vals['nova_carter_sim']['stop_trigger_distance'] == 1.1
    assert vals['nova_carter_sim']['hard_stop_distance'] == .75
    assert vals['scout2']['stop_trigger_distance'] > vals['scout2']['hard_stop_distance']


def test_physics_gt_not_just_commanded_path_and_mismatch_detected():
    txt = (ROOT/'isaac_dynamic_box.py').read_text()
    assert 'physical_pose = self._dc.get_rigid_body_pose(self._box_handle)' in txt
    assert 'box_pose_mismatch_steps > 5' in txt
    assert 'PhysX box did not follow commanded motion' in txt
    assert 'self._physics_robot_pose()' in txt


# The actor runs inside Isaac Sim, which is unavailable in offline CI. Execute
# only the discovery helpers from the AST against small USD/PhysX stubs.

def _discovery_with_fake_stage():
    import types
    src = (ROOT/'isaac_dynamic_box.py').read_text()
    tree = ast.parse(src)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ('_robot_candidates', '_robot_prim_and_handle')]
    module = ast.Module(body=functions, type_ignores=[])
    class RigidBodyAPI: pass
    ns = {'ROBOT_PRIM_PATH': None,
          'BOX_PATH': '/World/VROSA_Dynamic_Crossing_Box',
          'UsdPhysics': types.SimpleNamespace(RigidBodyAPI=RigidBodyAPI)}
    exec(compile(ast.fix_missing_locations(module), '<discovery>', 'exec'), ns)
    class Prim:
        def __init__(self, path, rigid=True, type_name='Xform'):
            self.path, self.rigid, self.type_name = path, rigid, type_name
        def GetPath(self): return self.path
        def IsValid(self): return True
        def HasAPI(self, api):
            assert api is RigidBodyAPI
            return self.rigid
        def GetTypeName(self): return self.type_name
    class MissingPrim(Prim):
        def __init__(self): super().__init__('(missing)', False)
        def IsValid(self): return False
    class Stage:
        def __init__(self, prims): self.prims = prims
        def Traverse(self): return iter(self.prims)
        def GetPrimAtPath(self, path):
            return next((p for p in self.prims if p.path == path), MissingPrim())
    class DC:
        def __init__(self, handles): self.handles = handles
        def get_rigid_body(self, path): return self.handles.get(path, 0)
    return ns, Prim, Stage, DC


def test_robot_detection_does_not_require_world_or_usd_xform():
    ns, Prim, Stage, DC = _discovery_with_fake_stage()
    target = Prim('/Simulation/robot_arm/NovaCarter/chassis_link', rigid=True,
                  type_name='Scope')
    stage = Stage([Prim('/Simulation/robot_arm/NovaCarter', False), target,
                   Prim('/World/VROSA_Dynamic_Crossing_Box', True)])
    assert ns['_robot_prim_and_handle'](stage, DC({target.path: 17})) == (target, 17)


def test_robot_detection_prefers_carter_over_unrelated_chassis():
    ns, Prim, Stage, DC = _discovery_with_fake_stage()
    target = Prim('/World/NovaCarter/chassis_link')
    forklift = Prim('/World/Forklift/chassis_link')
    stage = Stage([forklift, target])
    assert ns['_robot_prim_and_handle'](stage, DC({target.path: 9, forklift.path: 3}))[0] is target


def test_robot_detection_tries_other_candidates_if_not_physics_ready():
    ns, Prim, Stage, DC = _discovery_with_fake_stage()
    a = Prim('/World/nova_carter/chassis_link')
    b = Prim('/World/NovaCarter_2/chassis_link')
    stage = Stage([a,b])
    assert ns['_robot_prim_and_handle'](stage, DC({b.path: 22}))[0] is b


def test_robot_detection_fails_cleanly_if_scene_not_loaded():
    ns, Prim, Stage, DC = _discovery_with_fake_stage()
    with pytest.raises(RuntimeError, match='not present'):
        ns['_robot_prim_and_handle'](Stage([]), DC({}))


def test_robot_detection_does_not_accept_nonrigid_chassis():
    ns, Prim, Stage, DC = _discovery_with_fake_stage()
    with pytest.raises(RuntimeError, match='not present'):
        ns['_robot_prim_and_handle'](Stage([Prim('/World/NovaCarter/chassis_link', rigid=False)]),
                                     DC({'/World/NovaCarter/chassis_link': 9}))


def test_robot_detection_rejects_missing_explicit_override():
    ns, Prim, Stage, DC = _discovery_with_fake_stage()
    ns['ROBOT_PRIM_PATH'] = '/World/no_such_robot/chassis_link'
    with pytest.raises(RuntimeError, match='does not exist'):
        ns['_robot_prim_and_handle'](Stage([]), DC({}))


def test_robot_detection_requires_active_physx_body():
    ns, Prim, Stage, DC = _discovery_with_fake_stage()
    with pytest.raises(RuntimeError, match='no active PhysX rigid body'):
        ns['_robot_prim_and_handle'](Stage([Prim('/World/NovaCarter/chassis_link')]), DC({}))


def test_robot_lookup_delayed_until_arm():
    src = (ROOT/'isaac_dynamic_box.py').read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='MovingBox')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='__init__')
    arm = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='arm')
    init_calls = [n for n in ast.walk(init) if isinstance(n, ast.Call) and
                  isinstance(n.func, ast.Name) and n.func.id=='_robot_prim_and_handle']
    arm_calls = [n for n in ast.walk(arm) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id=='_robot_prim_and_handle']
    assert init_calls == []
    assert len(arm_calls) == 1
