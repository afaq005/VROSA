"""
safety.py — Canonical obstacle sensing and safety-record helpers.

Important invariant: UNKNOWN/STALE safety sensing is never represented as a
large distance. Translational motion must fail closed when check_obstacle()
returns valid=False.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import MAX_SENSOR_AGE_S, ROBOT, SAFETY_CONFIG
from experiment_logger import record_event, record_safety_event
from logger import log
from state import state

_DEPTH_STRIP_H = SAFETY_CONFIG["depth_strip_h"]
_DEPTH_STRIP_W = SAFETY_CONFIG["depth_strip_w"]
_DEPTH_PERCENTILE = SAFETY_CONFIG["depth_percentile"]
_MIN_VALID_DEPTH_PIXELS = SAFETY_CONFIG["min_valid_depth_pixels"]
_LIDAR_MAX_AGE_S = float(ROBOT.get("lidar_max_age_s", MAX_SENSOR_AGE_S))
_DEPTH_MAX_AGE_S = float(ROBOT.get("depth_max_age_s", MAX_SENSOR_AGE_S))

# Throttle repeated stale warnings while a stop monitor is polling at 50 Hz.
_last_warning_ts = {}


def _warn_throttled(key: str, message: str, period_s: float = 1.0) -> None:
    now = time.monotonic()
    if now - float(_last_warning_ts.get(key, 0.0)) >= period_s:
        log(message, "WARN")
        _last_warning_ts[key] = now


@dataclass(frozen=True)
class ObstacleReading:
    valid: bool
    distance: Optional[float]
    source: str                 # lidar | depth | none
    age_s: Optional[float]
    stale: bool
    reason: str = ""

    @property
    def is_clear(self) -> bool:
        return self.valid and (self.distance is None or self.distance >= 90.0)


def _age(ts) -> Optional[float]:
    return None if ts is None else max(0.0, time.monotonic() - float(ts))


def _lidar_obstacle(direction: int) -> ObstacleReading:
    ts = state.get("lidar_ts")
    age = _age(ts)
    if ts is None:
        return ObstacleReading(False, None, "lidar", None, False, "no_lidar_timestamp")
    if age is not None and age > _LIDAR_MAX_AGE_S:
        return ObstacleReading(False, None, "lidar", age, True, "stale_lidar")

    key = "lidar_forward_min" if direction > 0 else "lidar_reverse_min"
    d = state.get(key)
    if d is None:
        return ObstacleReading(False, None, "lidar", age, False, "lidar_corridor_unavailable")
    return ObstacleReading(True, float(d), "lidar", age, False, "")


def _depth_obstacle(direction: int) -> ObstacleReading:
    if direction != 1:
        return ObstacleReading(False, None, "depth", None, False, "no_rear_camera_coverage")

    dm = state.get("depth_map")
    ts = state.get("depth_ts")
    source_ts = state.get("depth_source_image_ts")
    ages = [_age(t) for t in (ts, source_ts)]
    age = max((a for a in ages if a is not None), default=None)
    if dm is None or ts is None:
        return ObstacleReading(False, None, "depth", age, False, "no_depth")
    if source_ts is None:
        return ObstacleReading(False, None, "depth", age, False, "depth_source_timestamp_missing")
    if age is not None and age > _DEPTH_MAX_AGE_S:
        return ObstacleReading(False, None, "depth", age, True, "stale_depth")

    H, W = dm.shape
    h0, h1 = int(H * _DEPTH_STRIP_H[0]), int(H * _DEPTH_STRIP_H[1])
    w0, w1 = int(W * _DEPTH_STRIP_W[0]), int(W * _DEPTH_STRIP_W[1])
    strip = dm[h0:h1, w0:w1]
    dmin, dmax = ROBOT["depth_min_m"], ROBOT["depth_max_m"]
    valid = strip[np.isfinite(strip) & (strip >= dmin) & (strip <= dmax)]
    if valid.size < _MIN_VALID_DEPTH_PIXELS:
        return ObstacleReading(False, None, "depth", age, False, "insufficient_valid_depth")

    d = float(np.percentile(valid, _DEPTH_PERCENTILE))
    return ObstacleReading(True, d, "depth", age, False, "")


def _hardware_range_depth_obstacle(direction: int) -> ObstacleReading:
    if direction != 1:
        return ObstacleReading(False, None, "realsense_depth", None, False, "no_rear_range_coverage")
    age = _age(state.get("range_depth_ts"))
    dm = state.get("range_depth_map")
    if dm is None or age is None:
        return ObstacleReading(False, None, "realsense_depth", age, False, "no_physical_depth")
    if age > _DEPTH_MAX_AGE_S:
        return ObstacleReading(False, None, "realsense_depth", age, True, "stale_physical_depth")
    if not isinstance(dm, np.ndarray) or dm.ndim != 2 or dm.size == 0:
        return ObstacleReading(False, None, "realsense_depth", age, False, "invalid_physical_depth_geometry")
    h, w = dm.shape
    strip = dm[int(h * _DEPTH_STRIP_H[0]):int(h * _DEPTH_STRIP_H[1]),
               int(w * _DEPTH_STRIP_W[0]):int(w * _DEPTH_STRIP_W[1])]
    good = strip[np.isfinite(strip) & (strip >= ROBOT["depth_min_m"]) &
                 (strip <= ROBOT["depth_max_m"])]
    if good.size < _MIN_VALID_DEPTH_PIXELS:
        return ObstacleReading(False, None, "realsense_depth", age, False, "insufficient_physical_depth")
    return ObstacleReading(True, float(np.percentile(good, _DEPTH_PERCENTILE)),
                           "realsense_depth", age, False, "")


def check_obstacle(direction: int = 1) -> ObstacleReading:
    """Return the best fresh obstacle observation for the requested direction."""
    if direction not in (-1, 1):
        return ObstacleReading(False, None, "none", None, False, "invalid_direction")

    if direction < 0 and not ROBOT.get("rear_obstacle_sensing", False):
        return ObstacleReading(False, None, "none", None, False, "rear_safety_unavailable")

    reasons = []
    stale = False
    ages = []

    # Hardware: physical range sensing is mandatory for motion authorization.
    # Inferred DA-V2 monocular depth may still run for target perception, but
    # cannot authorize physical translation if physical range sensing is lost.
    if ROBOT.get("platform_label") == "Hardware":
        if state.get("use_lidar", False) and ROBOT.get("has_lidar", False):
            lidar = _lidar_obstacle(direction)
            if lidar.valid:
                return lidar
            reasons.append(lidar.reason)
            stale |= lidar.stale
            if lidar.age_s is not None:
                ages.append(lidar.age_s)
        if state.get("use_range_depth", False):
            r = _hardware_range_depth_obstacle(direction)
            if r.valid:
                return r
            reasons.append(r.reason)
            stale |= r.stale
            if r.age_s is not None:
                ages.append(r.age_s)
        else:
            reasons.append("physical_depth_disabled")
        return ObstacleReading(False, None, "none", max(ages) if ages else None,
                               stale, ";".join(reasons) or "no_verified_physical_safety_sensor")

    if state.get("use_lidar", True) and ROBOT.get("has_lidar", False):
        r = _lidar_obstacle(direction)
        if r.valid:
            return r
        reasons.append(r.reason)
        stale = stale or r.stale
        if r.age_s is not None:
            ages.append(r.age_s)
        if r.stale:
            _warn_throttled("lidar", f"LiDAR stale ({r.age_s*1000:.0f} ms); trying fallback sensor")

    if state.get("use_depth", True) and direction > 0:
        r = _depth_obstacle(direction)
        if r.valid:
            return r
        reasons.append(r.reason)
        stale = stale or r.stale
        if r.age_s is not None:
            ages.append(r.age_s)
        if r.stale:
            _warn_throttled("depth", f"Depth stale ({r.age_s*1000:.0f} ms); motion will fail closed")

    return ObstacleReading(
        valid=False,
        distance=None,
        source="none",
        age_s=max(ages) if ages else None,
        stale=stale,
        reason=";".join(x for x in reasons if x) or "no_safety_sensor",
    )


def safety_ready(direction: int = 1) -> bool:
    return check_obstacle(direction).valid


def log_stop_measurement(**kwargs) -> None:
    record_safety_event(**kwargs)


def record_safety_unknown(direction: int, reading: ObstacleReading) -> None:
    record_event(
        "SAFETY_SENSOR_UNAVAILABLE",
        direction=direction,
        sensor_source=reading.source,
        sensor_age_ms=(reading.age_s * 1000.0 if reading.age_s is not None else None),
        reason=reading.reason,
        x=state.get("x"), y=state.get("y"), yaw=state.get("yaw"),
    )
