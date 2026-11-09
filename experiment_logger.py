"""
experiment_logger.py — Structured, publication-oriented experiment records.

Every record carries run_id / trial_id / scenario_id / method so data from
simulation, baselines and hardware can be joined directly for statistics.

Files per run:
  manifest.json      frozen software/model/configuration metadata
  events.jsonl       navigation/tool/safety state transitions
  latency.jsonl      per-stage timing
  safety_events.csv  stopping-distance and emergency-stop measurements
  trials.csv         one summary row per natural-language command/trial
  depth_validation.csv optional ground-truth depth samples
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ROBOT, PERCEPTION_CONFIG, SAFETY_CONFIG, args

_LOCK = threading.RLock()
_RUN_ID = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_DIR = Path(args.log_dir).expanduser().resolve() / _RUN_ID
_BASE_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = _BASE_DIR

_CONTEXT = {
    "run_id": _RUN_ID,
    "trial_id": args.trial_id.strip(),
    "scenario_id": args.scenario_id.strip(),
    "method": args.method,
}

_TRIAL_BASELINE: dict[str, Any] = {}

EVENTS_PATH = _BASE_DIR / "events.jsonl"
LATENCY_PATH = _BASE_DIR / "latency.jsonl"
SAFETY_PATH = _BASE_DIR / "safety_events.csv"
TRIALS_PATH = _BASE_DIR / "trials.csv"
SCENARIOS_PATH = _BASE_DIR / "scenarios.csv"
DEPTH_VALIDATION_PATH = _BASE_DIR / "depth_validation.csv"
MANIFEST_PATH = _BASE_DIR / "manifest.json"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _json_default(obj):
    try:
        return float(obj)
    except Exception:
        return str(obj)


def context() -> dict[str, str]:
    with _LOCK:
        return dict(_CONTEXT)


def set_context(*, trial_id: str | None = None, scenario_id: str | None = None,
                method: str | None = None, **extra) -> None:
    with _LOCK:
        if trial_id is not None:
            _CONTEXT["trial_id"] = str(trial_id)
        if scenario_id is not None:
            _CONTEXT["scenario_id"] = str(scenario_id)
        if method is not None:
            _CONTEXT["method"] = str(method)
        for key, value in extra.items():
            _CONTEXT[key] = value


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")


def record_event(event: str, **data) -> None:
    rec = {
        **context(),
        "event": event,
        "wall_time": time.time(),
        "monotonic_time": time.monotonic(),
        **data,
    }
    _append_jsonl(EVENTS_PATH, rec)


def record_latency(stage: str, duration_ms: float, *, success: bool = True, **extra) -> None:
    rec = {
        **context(),
        "stage": stage,
        "duration_ms": round(float(duration_ms), 3),
        "success": bool(success),
        "wall_time": time.time(),
        **extra,
    }
    _append_jsonl(LATENCY_PATH, rec)


@contextmanager
def timed(stage: str, **extra):
    t0 = time.perf_counter_ns()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        dt_ms = (time.perf_counter_ns() - t0) / 1e6
        record_latency(stage, dt_ms, success=ok, **extra)


def _write_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with _LOCK:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                w.writeheader()
            w.writerow(row)


SAFETY_FIELDS = [
    "run_id", "trial_id", "scenario_id", "method", "wall_time",
    "event_type", "source", "robot", "obstacle_type", "dynamic_obstacle",
    "lighting_condition", "direction", "commanded_speed_mps",
    "trigger_distance_m", "stop_trigger_distance_m", "hard_stop_distance_m", "sensor_source", "sensor_age_ms",
    "sensor_time_mono", "recognition_time_mono", "stop_command_time_mono",
    "zero_velocity_time_mono", "zero_velocity_confirmed",
    "perception_age_ms", "decision_to_command_ms",
    "braking_latency_ms", "total_stop_latency_ms", "velocity_at_trigger_mps",
    "x_at_trigger", "y_at_trigger", "x_at_stop", "y_at_stop",
    "stop_distance_m", "minimum_clearance_m", "threshold_violation",
]


def record_safety_event(**row) -> None:
    from config import ROBOT
    base = {
        **context(),
        "wall_time": time.time(),
        "robot": ROBOT["display_name"],
        "hard_stop_distance_m": ROBOT["hard_stop_distance"],
        "stop_trigger_distance_m": ROBOT["stop_trigger_distance"],
    }
    base.update(row)
    _write_csv_row(SAFETY_PATH, SAFETY_FIELDS, base)


TRIAL_FIELDS = [
    "run_id", "trial_id", "scenario_id", "method", "wall_time",
    "robot", "platform", "command", "target", "command_language", "paraphrase_id",
    "heldout_category", "start_x", "start_y", "start_yaw", "end_x", "end_y", "end_yaw",
    "path_length_m", "completion_time_s", "outcome", "runtime_outcome",
    "experimental_success", "gt_arrival_threshold_m", "failure_reason",
    "estimated_final_distance_m", "ground_truth_final_distance_m",
    "ground_truth_source", "runtime_identity_confirmed", "success_evaluable",
    "tool_calls", "vlm_queries", "replans", "safety_stops", "sequence_errors",
    "recovery_successes", "vlm_backend", "vlm_model", "yolo_model", "depth_model",
    "seed", "notes",
]


def begin_trial(command: str, *, target: str = "", trial_id: str | None = None,
                scenario_id: str | None = None, method: str | None = None,
                **metadata) -> str:
    from state import state
    if trial_id is None:
        trial_id = f"T{int(time.time() * 1000)}"
    set_context(trial_id=trial_id, scenario_id=scenario_id, method=method)
    state["tool_call_count"] = 0
    state["vlm_query_count"] = 0
    state["replan_count"] = 0
    state["safety_stop_count"] = 0
    state["sequence_error_count"] = 0
    state["recovery_success_count"] = 0
    state["last_nav_outcome"] = ""
    state["last_visual_target_memory"] = None
    state["last_nav_estimated_final_distance"] = None
    state["last_target"] = ""
    with _LOCK:
        _TRIAL_BASELINE.clear()
        _TRIAL_BASELINE.update({
            "start_mono": time.monotonic(),
            "start_x": state["x"],
            "start_y": state["y"],
            "start_yaw": state["yaw"],
            "path0": state.get("path_length_m", 0.0),
            "command": command,
            "target": target,
            **metadata,
        })
    record_event("TRIAL_START", command=command, target=target, **metadata)
    return trial_id


def end_trial(*, outcome: str = "", failure_reason: str = "",
              ground_truth_final_distance_m: float | None = None,
              ground_truth_source: str = "",
              notes: str = "", **metadata) -> None:
    from state import state
    try:
        from vlm.manager import vlm_mgr
        backend = vlm_mgr.active_name
        model = vlm_mgr.active_model
    except Exception:
        backend, model = "", ""

    with _LOCK:
        b = dict(_TRIAL_BASELINE)
    if not b:
        return

    completion = time.monotonic() - b["start_mono"]
    final_outcome = outcome or state.get("last_nav_outcome", "") or "completed"
    identity_confirmed = bool((state.get("last_visual_target_memory") or {}).get("identity_confirmed"))
    gt_valid = (ground_truth_source in {"independent_sim_pose", "independent_field_measurement"}
                and ground_truth_final_distance_m is not None
                and 0.0 <= float(ground_truth_final_distance_m) < float("inf"))
    experimental_success = (bool(final_outcome == "ARRIVED" and identity_confirmed
                                 and float(ground_truth_final_distance_m) <= float(ROBOT["arrive_dist"]))
                            if gt_valid else "")
    row = {
        **context(),
        "wall_time": time.time(),
        "robot": ROBOT["display_name"],
        "platform": ROBOT["platform_label"],
        "command": b.get("command", ""),
        "target": b.get("target") or state.get("last_target") or state.get("target", ""),
        "command_language": b.get("command_language", ""),
        "paraphrase_id": b.get("paraphrase_id", ""),
        "heldout_category": b.get("heldout_category", ""),
        "start_x": round(b["start_x"], 4),
        "start_y": round(b["start_y"], 4),
        "start_yaw": round(b["start_yaw"], 3),
        "end_x": round(state["x"], 4),
        "end_y": round(state["y"], 4),
        "end_yaw": round(state["yaw"], 3),
        "path_length_m": round(state.get("path_length_m", 0.0) - b.get("path0", 0.0), 4),
        "completion_time_s": round(completion, 3),
        "outcome": final_outcome,
        "runtime_outcome": final_outcome,
        "experimental_success": experimental_success,
        "gt_arrival_threshold_m": ROBOT["arrive_dist"],
        "failure_reason": failure_reason,
        "estimated_final_distance_m": state.get("last_nav_estimated_final_distance"),
        "ground_truth_final_distance_m": ground_truth_final_distance_m if gt_valid else None,
        "ground_truth_source": ground_truth_source if gt_valid else "unverified_or_missing",
        "runtime_identity_confirmed": identity_confirmed,
        "success_evaluable": gt_valid,
        "tool_calls": state.get("tool_call_count", 0),
        "vlm_queries": state.get("vlm_query_count", 0),
        "replans": state.get("replan_count", 0),
        "safety_stops": state.get("safety_stop_count", 0),
        "sequence_errors": state.get("sequence_error_count", 0),
        "recovery_successes": state.get("recovery_success_count", 0),
        "vlm_backend": backend,
        "vlm_model": model,
        "yolo_model": ROBOT["yolo_model"],
        "depth_model": ROBOT["depth_model"],
        "seed": args.seed,
        "notes": notes,
        **metadata,
    }
    _write_csv_row(TRIALS_PATH, TRIAL_FIELDS, row)
    record_event("TRIAL_END", outcome=final_outcome, failure_reason=failure_reason)
    with _LOCK:
        _TRIAL_BASELINE.clear()



SCENARIO_FIELDS = [
    "run_id", "scenario_id", "method", "wall_time", "seed", "scene",
    "robot_start_x", "robot_start_y", "robot_start_yaw",
    "target_class", "target_x", "target_y", "target_z",
    "initial_gt_distance_m", "initial_view_angle_deg",
    "distractor_count", "distractor_classes", "obstacle_configuration",
    "lighting_condition", "dynamic_hazard", "notes",
]


def record_scenario(**row) -> None:
    base = {
        "run_id": _RUN_ID,
        "scenario_id": row.get("scenario_id") or context().get("scenario_id", ""),
        "method": row.get("method") or context().get("method", ""),
        "wall_time": time.time(),
        "seed": row.get("seed", args.seed),
    }
    base.update(row)
    _write_csv_row(SCENARIOS_PATH, SCENARIO_FIELDS, base)


DEPTH_FIELDS = [
    "run_id", "trial_id", "scenario_id", "method", "wall_time", "frame_id", "target",
    "gt_distance_m", "raw_depth_m", "calibrated_depth_m", "lidar_distance_m",
    "abs_error_m", "relative_error", "bbox_width_px", "bbox_height_px",
    "valid_pixel_count", "valid_pixel_ratio", "percentile_used", "lighting",
    "surface_type", "target_size", "range_bin",
]


def record_depth_validation(**row) -> None:
    base = {**context(), "wall_time": time.time()}
    base.update(row)
    _write_csv_row(DEPTH_VALIDATION_PATH, DEPTH_FIELDS, base)


def write_manifest(extra: dict[str, Any] | None = None) -> None:
    manifest = {
        "run_id": _RUN_ID,
        "created_at": datetime.now().isoformat(),
        "robot_profile": args.robot,
        "robot": ROBOT,
        "perception_config": PERCEPTION_CONFIG,
        "safety_config": SAFETY_CONFIG,
        "method": args.method,
        "seed": args.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "ultralytics": _pkg_version("ultralytics"),
            "openai": _pkg_version("openai"),
            "langchain": _pkg_version("langchain"),
        },
        "vlm": {
            "backend_requested": args.vlm,
            "api_base": args.api_base,
            "api_model": args.api_model,
            "local_base": args.local_base,
            "local_model": args.local_model,
            "remote_url": args.remote_url,
        },
        "git_commit": os.environ.get("VROSA_GIT_COMMIT", "unknown"),
    }
    try:
        import torch
        manifest["compute"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    except Exception:
        manifest["compute"] = {"cuda_available": False, "gpu": "unknown"}
    if extra:
        manifest.update(extra)
    with _LOCK:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")


# Write a baseline manifest as soon as the module is imported. vrosa_live.py can
# call write_manifest() again after all components are initialized.
write_manifest()
