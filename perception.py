"""
perception.py — YOLO + Depth Anything V2 + GroundingDINO perception pipeline.

RUNTIME PERCEPTION SCHEDULING:
  YOLO: continuous best-effort GPU object detection.
  DA-V2: on-demand while the robot is stationary on LiDAR-equipped platforms;
         continuous only on camera-only platforms that require it for safety.
  GroundingDINO: optional slow CPU semantic detector.

YOLO and DA-V2 share one GPU inference lock so they never contend on CUDA.
Detection sources write to state["detections"] with deduplication (IoU merge).
GroundingDINO lives on CPU to preserve VRAM alongside Isaac Sim.

YOLO model hot-swappable at runtime via /yolo/switch endpoint.
"""

import math
import time
import threading

import numpy as np
from PIL import Image as PILImage

from state import state, state_lock
from logger import log
from config import (
    ROBOT,
    PERCEPTION_CONFIG,
    MAX_SENSOR_AGE_S,
)
_DEFAULT_YOLO = ROBOT["yolo_model"]
import warnings
from experiment_logger import timed, record_event
warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")
# ════════════════════════════════════════════════════════════
# AVAILABLE YOLO MODELS
# ════════════════════════════════════════════════════════════
YOLO_MODELS = {
    # ── YOLO26 (latest, 2025 — NMS-free, small-object aware) ─
    "yolo26n": {"file": "yolo26n.pt", "desc": "YOLO26 Nano   — fastest,  NMS-free"},
    "yolo26s": {"file": "yolo26s.pt", "desc": "YOLO26 Small  — fast,     NMS-free"},
    "yolo26m": {"file": "yolo26m.pt", "desc": "YOLO26 Medium — balanced, NMS-free"},
    "yolo26l": {"file": "yolo26l.pt", "desc": "YOLO26 Large  — accurate, NMS-free"},
    "yolo26x": {"file": "yolo26x.pt", "desc": "YOLO26 XLarge — best,     NMS-free"},
    # ── YOLO11 (2024) ─────────────────────────────────────────
    "yolo11n": {"file": "yolo11n.pt", "desc": "YOLO11 Nano   — fastest"},
    "yolo11s": {"file": "yolo11s.pt", "desc": "YOLO11 Small  — fast"},
    "yolo11m": {"file": "yolo11m.pt", "desc": "YOLO11 Medium — balanced"},
    "yolo11l": {"file": "yolo11l.pt", "desc": "YOLO11 Large  — accurate"},
    "yolo11x": {"file": "yolo11x.pt", "desc": "YOLO11 XLarge — best"},
    # ── YOLOv8 (stable) ───────────────────────────────────────
    "yolov8n": {"file": "yolov8n.pt", "desc": "YOLOv8 Nano   — fastest"},
    "yolov8s": {"file": "yolov8s.pt", "desc": "YOLOv8 Small  — fast"},
    "yolov8m": {"file": "yolov8m.pt", "desc": "YOLOv8 Medium — balanced"},
}
# _DEFAULT_YOLO = "yolo26n"


# ════════════════════════════════════════════════════════════
# MODEL SINGLETONS
# ════════════════════════════════════════════════════════════
_depth_pipe   = None
_gdino_model  = None   # CPU only — preserves VRAM for Isaac Sim
_gdino_proc   = None
_yolo_model   = None
_yolo_key     = _DEFAULT_YOLO
_yolo_lock    = threading.Lock()

_models_ready = False
_load_error   = None
_device       = "cpu"

# Separate detection caches — merged on read
_yolo_dets    = []    # updated by fast loop
_gdino_dets   = []    # updated by slow loop
_dets_lock    = threading.Lock()
_gpu_inference_lock = threading.Lock()
_depth_priority = threading.Event()
_yolo_frame_cache_lock = threading.Lock()
_yolo_frame_cache = {}
_YOLO_FRAME_CACHE_SIZE = 3

# ════════════════════════════════════════════════════════════
# SHARED GEOMETRY HELPERS
# ════════════════════════════════════════════════════════════
def _calibrate_target_distance(raw_m: float) -> tuple[float, bool]:
    """Apply configured short-range object-distance calibration when enabled.

    Formal experiments must use coefficients fitted without GT-assisted ROI
    selection; disabling calibration returns the raw metric-model estimate.
    """
    raw_m = float(raw_m)

    if not ROBOT.get("target_depth_calibration_enabled", False):
        return raw_m, False

    raw_max = float(
        ROBOT.get(
            "target_depth_cal_raw_max_m",
            0.0,
        )
    )

    # Do not extrapolate the short-range fit into far-range navigation.
    if raw_max > 0.0 and raw_m > raw_max:
        return raw_m, False

    a = float(
        ROBOT.get(
            "target_depth_cal_scale",
            1.0,
        )
    )

    b = float(
        ROBOT.get(
            "target_depth_cal_bias",
            0.0,
        )
    )

    calibrated = (
        a * raw_m
        + b
    )

    return max(0.0, calibrated), True

def depth_stats_from_bbox(depth_map, bbox: list, pct: int = None) -> dict:
    """Return distance plus validity metadata for a bbox centre crop."""
    if depth_map is None:
        return {"distance": 99.0, "valid_pixel_count": 0, "valid_pixel_ratio": 0.0, "percentile": pct}
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = depth_map.shape
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    half_frac = float(PERCEPTION_CONFIG["depth_bbox_inner_fraction"]) / 2.0
    hw = max(4, int((x2 - x1) * half_frac))
    hh = max(4, int((y2 - y1) * half_frac))
    cx1, cy1 = max(0, mx - hw), max(0, my - hh)
    cx2, cy2 = min(W, mx + hw), min(H, my + hh)
    patch = depth_map[cy1:cy2, cx1:cx2]
    if patch.size == 0:
        return {"distance": 99.0, "valid_pixel_count": 0, "valid_pixel_ratio": 0.0, "percentile": pct}
    valid = patch[np.isfinite(patch) & (patch >= ROBOT["depth_min_m"]) & (patch <= ROBOT["depth_max_m"])]
    ratio = float(valid.size / patch.size)
    if valid.size == 0:
        return {"distance": 99.0, "valid_pixel_count": 0, "valid_pixel_ratio": ratio, "percentile": pct}
    rough_raw = float(
        np.median(valid)
    )

    rough_for_band, rough_calibrated = (
        _calibrate_target_distance(
            rough_raw
        )
    )

    if pct is None:

        if (
            rough_for_band
            < PERCEPTION_CONFIG[
                "depth_near_threshold_m"
            ]
        ):
            pct = PERCEPTION_CONFIG[
                "depth_percentile_near"
            ]

        elif (
            rough_for_band
            < PERCEPTION_CONFIG[
                "depth_mid_threshold_m"
            ]
        ):
            pct = PERCEPTION_CONFIG[
                "depth_percentile_mid"
            ]

        else:
            pct = PERCEPTION_CONFIG[
                "depth_percentile_far"
            ]


    raw_distance = float(
        np.percentile(
            valid,
            pct,
        )
    )

    distance, calibrated = (
        _calibrate_target_distance(
            raw_distance
        )
    )


    return {
        "distance": distance,
        "raw_distance": raw_distance,
        "calibrated": calibrated,
        "valid_pixel_count": int(valid.size),
        "valid_pixel_ratio": ratio,
        "percentile": int(pct),
        "rough_median": rough_for_band,
        "rough_raw_median": rough_raw,
    }


def _dist_from_bbox(depth_map, bbox: list, pct: int = None) -> float:
    return float(depth_stats_from_bbox(depth_map, bbox, pct)["distance"])


def _sector(cx: int, W: int) -> str:
    if cx < W // 3:      return "left"
    if cx > 2 * W // 3:  return "right"
    return "center"


def _bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _merge_dets(primary: list, secondary: list) -> list:
    """Merge two detection lists; prefer primary, skip secondary duplicates."""
    merged = primary[:]
    p_bboxes = [d["bbox"] for d in merged]
    for det in secondary:
        if not any(_bbox_iou(det["bbox"], b) > 0.4 for b in p_bboxes):
            merged.append(det)
    return sorted(merged, key=lambda d: d["dist"])



def _lidar_dist_for_bbox(bbox: list, img_w: int) -> float | None:
    """Legacy angular bbox→LiDAR association helper.

    This approximation does NOT apply an explicit camera↔LiDAR extrinsic
    transform. It is therefore disabled by default through
    ROBOT["lidar_bbox_projection_validated"] and must not be used for formal
    object-distance fusion until calibration/synchronization are validated.
    """
    if not state.get("use_lidar", True):
        return None
    lidar = state.get("lidar", [])
    if not lidar:
        return None

    cam_fx  = state.get("cam_fx", 640.0)
    img_cx  = img_w / 2.0
    x1, y1, x2, y2 = bbox

    # Horizontal angles for left and right edge of bbox
    # Camera: pixel right = positive x = robot right = negative LiDAR y
    angle_left  = math.atan2((x1 - img_cx), cam_fx)   # radians, neg = left
    angle_right = math.atan2((x2 - img_cx), cam_fx)   # radians, pos = right

    dists = []
    for p in lidar:
        if p["x"] <= 0.35:           # ignore points behind, beside, or on the robot itself
            continue
        # LiDAR y=left, camera x=right → flip sign
        bearing = math.atan2(-p["y"], p["x"])
        if angle_left <= bearing <= angle_right:
            dists.append(p["dist"])

    if not dists:
        return None
    return float(np.percentile(dists, 20))

def _fresh_depth_map():
    """
    Return the latest metric depth map only while it satisfies
    the active robot profile's freshness requirement.

    A stale map must never be attached to a fresh object detection.
    """
    dm = state.get("depth_map")
    ts = state.get("depth_ts")
    source_ts = state.get("depth_source_image_ts")

    if dm is None or ts is None or source_ts is None:
        return None

    max_age = float(
        ROBOT.get(
            "depth_max_age_s",
            MAX_SENSOR_AGE_S,
        )
    )

    age = max(time.monotonic() - float(ts), time.monotonic() - float(source_ts))

    if age > max_age:
        return None

    return dm
def _best_dist(
    bbox: list,
    img_w: int,
    depth_map=None,
) -> float:
    """
    Return a detector-associated metric distance.

    Object-level LiDAR projection remains disabled unless explicitly
    validated. DA-V2 may be used only while its map is fresh.
    """
    if (
        ROBOT.get(
            "lidar_bbox_projection_validated",
            False,
        )
        and state.get(
            "use_lidar",
            True,
        )
    ):
        d = _lidar_dist_for_bbox(
            bbox,
            img_w,
        )

        if d is not None:
            return round(
                d,
                2,
            )


    if not state.get(
        "use_depth",
        False,
    ):
        return 99.0


    # Background detections must use the canonical freshness gate.
    fresh_depth = (
        _fresh_depth_map()
    )

    if fresh_depth is None:
        return 99.0


    try:
        d = _dist_from_bbox(
            fresh_depth,
            bbox,
        )

        if not np.isfinite(d):
            return 99.0

        return round(
            float(d),
            2,
        )

    except Exception:
        return 99.0
# ════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════
def _load_yolo_model(key: str) -> bool:
    global _yolo_model, _yolo_key
    if key not in YOLO_MODELS:
        log(f"Unknown YOLO model: {key}", "PERC")
        return False
    try:
        from ultralytics import YOLO
        mfile = YOLO_MODELS[key]["file"]
        log(f"Loading YOLO model: {mfile}...", "PERC")
        new_model = YOLO(mfile)
        if _device == "cuda":
            new_model.to("cuda")
        with _yolo_lock:
            _yolo_model = new_model
            _yolo_key   = key
        state["yolo_model"] = key
        log(f"✅ YOLO → {key}", "PERC")
        return True
    except Exception as e:
        log(f"❌ YOLO load failed ({key}): {e}", "PERC")
        return False


def _load_models() -> None:
    global _depth_pipe, _gdino_model, _gdino_proc
    global _models_ready, _load_error, _device

    log("Loading perception models ...", "PERC")
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Perception GPU: {_device} | GroundingDINO: CPU (saves VRAM for Isaac Sim)", "PERC")

        # ── DA-V2 → GPU ───────────────────────────────────────
        log(f"Loading depth model: {ROBOT['depth_model']} ({ROBOT['display_name']}, GPU)...", "PERC")
        from transformers import pipeline as hf_pipeline
        _depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=ROBOT["depth_model"],
            device=0 if _device == "cuda" else -1,
        )
        log("✅ DA-V2 ready", "PERC")

        # ── GroundingDINO → CPU ───────────────────────────────
        log("Loading GroundingDINO (CPU)...", "PERC")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            _gdino_proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-base"
            )  # intentionally CPU — no .to(_device)
            _gdino_model.eval()
            log("✅ GroundingDINO ready (CPU)", "PERC")
        except Exception as e:
            log(f"⚠ GroundingDINO load failed: {e}", "PERC")

        # ── YOLO → GPU ──────────────────────────────────────
        _load_yolo_model(state.get("yolo_model", _DEFAULT_YOLO))

        _models_ready       = True
        state["perc_ready"] = True
        log(
            "✅ All perception models loaded "
            "(YOLO active; DA-V2 and GroundingDINO runtime-controlled)",
            "PERC",
        )

    except ImportError as e:
        _load_error = str(e)
        log(f"❌ Package missing: {e}", "PERC")
        log("   pip install transformers accelerate ultralytics", "PERC")
    except Exception as e:
        _load_error = str(e)
        log(f"❌ Perception load failed: {e}", "PERC")


def switch_yolo(key: str) -> str:
    """Hot-swap YOLO model. Called by /yolo/switch endpoint."""
    if key not in YOLO_MODELS:
        return f"Unknown model '{key}'. Available: {list(YOLO_MODELS)}"
    threading.Thread(target=_load_yolo_model, args=(key,), daemon=True).start()
    return f"Loading {key} ({YOLO_MODELS[key]['file']}) in background..."


# ════════════════════════════════════════════════════════════
# DEPTH
# ════════════════════════════════════════════════════════════
def _run_depth(img: PILImage.Image):
    """
    Returns a float32 H×W numpy array in METRIC METRES.

    IMPORTANT: result["depth"] is a PIL Image normalized 0-255 for display
               — NOT metric values. Using it directly was the cause of
               3m objects reading as 25m.
    result["predicted_depth"] is the raw model output tensor in real metres.
    """
    if _depth_pipe is None:
        return None
    try:
        result = _depth_pipe(img)
        # Raw model output. Calibration is explicit in config.py rather than
        # hidden as an unexplained multiplier.
        depth_tensor = result["predicted_depth"].squeeze()
        raw = depth_tensor.detach().cpu().numpy().astype(np.float32)

        # Align the raw metric output to the RGB image before bbox indexing.
        if raw.shape != (img.height, img.width):
            raw_img = PILImage.fromarray(raw, mode="F").resize(
                (img.width, img.height), PILImage.Resampling.BICUBIC
            )
            raw = np.asarray(raw_img, dtype=np.float32)
        raw[~np.isfinite(raw)] = np.nan
        state["depth_raw_map"] = raw.copy()

        depth_np = raw * float(ROBOT.get("depth_scale", 1.0)) + float(ROBOT.get("depth_bias", 0.0))
        depth_np[~np.isfinite(depth_np)] = np.nan
        return depth_np
    except Exception as e:
        log(f"DA-V2 error: {e}", "PERC")
        return None



def _cache_yolo_frame(image_seq, image: PILImage.Image) -> None:
    """Keep a few exact RGB frames so bbox and on-demand depth stay synchronized."""
    if image_seq is None or image is None:
        return
    with _yolo_frame_cache_lock:
        _yolo_frame_cache[int(image_seq)] = image.copy()
        for old_seq in sorted(_yolo_frame_cache)[:-_YOLO_FRAME_CACHE_SIZE]:
            _yolo_frame_cache.pop(old_seq, None)


def get_detection_image(det: dict):
    """Return the exact frozen RGB frame that produced a YOLO detection."""
    if det is None:
        return None
    seq = det.get("image_seq")
    if seq is None:
        return None
    with _yolo_frame_cache_lock:
        img = _yolo_frame_cache.get(int(seq))
        return None if img is None else img.copy()


def measure_detection_distance(det: dict):
    """Run one synchronized DA-V2 measurement for a detector bbox.

    This call is intended for a STOPPED robot. It gives depth priority, waits
    for any in-flight YOLO CUDA call to finish, and runs DA-V2 on the exact RGB
    frame that produced ``det``. The result is returned as a copied detection
    with a metric ``dist``; it is NOT published as a continuously fresh safety
    depth map.
    """
    if det is None or _depth_pipe is None:
        return None

    img = get_detection_image(det)
    if img is None:
        record_event(
            "ON_DEMAND_DEPTH_FAILED",
            reason="source_frame_not_cached",
            image_seq=det.get("image_seq"),
        )
        return None

    bbox = det.get("bbox")
    if not bbox:
        return None

    seq = det.get("image_seq")
    started = time.monotonic()
    _depth_priority.set()
    try:
        with _gpu_inference_lock:
            with timed("depth_estimation_on_demand", image_seq=seq):
                depth_map = _run_depth(img)
    finally:
        _depth_priority.clear()

    if depth_map is None:
        record_event(
            "ON_DEMAND_DEPTH_FAILED",
            reason="depth_inference_failed",
            image_seq=seq,
        )
        return None

    stats = depth_stats_from_bbox(depth_map, bbox)
    distance = float(stats.get("distance", 99.0))
    if not np.isfinite(distance) or not (0.0 < distance < 90.0):
        record_event(
            "ON_DEMAND_DEPTH_FAILED",
            reason="invalid_bbox_depth",
            image_seq=seq,
            valid_pixel_count=stats.get("valid_pixel_count", 0),
        )
        return None

    result = dict(det)
    result["dist"] = round(distance, 3)
    result["depth_source_seq"] = seq
    result["metric_source"] = "da_v2_on_demand"
    result["depth_valid_pixel_count"] = stats.get("valid_pixel_count", 0)
    result["depth_valid_pixel_ratio"] = stats.get("valid_pixel_ratio", 0.0)
    result["depth_percentile"] = stats.get("percentile")
    # Preserve provenance so navigation can reject out-of-regime target
    # measurements instead of treating an uncalibrated far estimate as a
    # trustworthy world-coordinate lock.
    result["raw_distance_m"] = stats.get("raw_distance")
    result["depth_calibrated"] = bool(stats.get("calibrated", False))
    result["rough_raw_median_m"] = stats.get("rough_raw_median")

    elapsed_ms = (time.monotonic() - started) * 1000.0
    state["last_ondemand_depth_ms"] = elapsed_ms
    state["last_ondemand_depth_seq"] = seq

    record_event(
        "ON_DEMAND_TARGET_DEPTH",
        image_seq=seq,
        distance_m=result["dist"],
        raw_distance_m=stats.get("raw_distance"),
        calibrated=stats.get("calibrated", False),
        duration_ms=elapsed_ms,
        valid_pixel_count=stats.get("valid_pixel_count", 0),
    )
    return result


# ════════════════════════════════════════════════════════════
# FAST LOOP — YOLO + DA-V2  (target rate; actual timing is logged)
# ════════════════════════════════════════════════════════════
_YOLO_CONF = PERCEPTION_CONFIG[
    "yolo_conf"
]

_YOLO_FPS = float(
    PERCEPTION_CONFIG[
        "yolo_target_fps"
    ]
)

_DEPTH_FPS = float(
    PERCEPTION_CONFIG.get(
        "depth_target_fps",
        2.0,
    )
)


def _yolo_loop() -> None:
    """Best-effort YOLO loop with DA-V2 priority and serialized CUDA access."""
    global _yolo_dets
    interval = 1.0 / max(_YOLO_FPS, 0.1)

    while True:
        loop_t0 = time.monotonic()

        if not _models_ready:
            time.sleep(0.5)
            continue
        if not state.get("use_yolo", True):
            time.sleep(interval)
            continue
        if _depth_priority.is_set():
            time.sleep(0.05)
            continue

        img = state.get("image_raw")
        image_seq = state.get("image_seq")
        image_ros_ts = state.get("image_ros_ts")
        image_recv_ts = state.get("image_ts")
        if img is None:
            time.sleep(interval)
            continue

        try:
            with _yolo_lock:
                model = _yolo_model
            if model is None:
                time.sleep(interval)
                continue

            W, _ = img.size

            # Depth requests get priority. The second check closes the race
            # between the first Event check and acquiring the CUDA lock.
            with _gpu_inference_lock:
                if _depth_priority.is_set():
                    time.sleep(0.02)
                    continue
                with timed("yolo_detection", image_seq=image_seq):
                    results = model(img, verbose=False, conf=_YOLO_CONF)[0]

            # Cache exactly the frame whose bboxes we are about to publish.
            _cache_yolo_frame(image_seq, img)

            dets = []
            for box in results.boxes:
                bbox = [round(v) for v in box.xyxy[0].tolist()]
                cx = (bbox[0] + bbox[2]) // 2
                label = results.names[int(box.cls)]

                # Continuous object detection does not require DA-V2. A fresh
                # background depth map is used only where one legitimately
                # exists (e.g. camera-only platform); Carter normally gets 99
                # here and requests synchronized depth only while stopped.
                distance = _best_dist(bbox, W)

                dets.append({
                    "label": label,
                    "dist": distance,
                    "sector": _sector(cx, W),
                    "bbox": bbox,
                    "conf": round(float(box.conf), 3),
                    "source": "yolo",
                    "image_seq": image_seq,
                    "image_ros_ts": image_ros_ts,
                    "image_recv_ts": image_recv_ts,
                    "depth_source_seq": (
                        state.get("depth_source_image_seq")
                        if distance < 90.0 else None
                    ),
                    "created_ts": time.monotonic(),
                })

            with _dets_lock:
                _yolo_dets = dets
            _update_state_detections()

        except Exception as e:
            log(f"YOLO loop error: {e}", "PERC")

        elapsed = time.monotonic() - loop_t0
        time.sleep(max(0.0, interval - elapsed))


def _depth_loop() -> None:
    """
    Independent DA-V2 metric-depth loop.

    Depth may run more slowly than YOLO without blocking object
    detection. Safety and navigation explicitly reject stale maps.
    """
    interval = (
        1.0
        / max(
            _DEPTH_FPS,
            0.1,
        )
    )


    while True:

        loop_t0 = (
            time.monotonic()
        )


        if not _models_ready:

            time.sleep(
                0.5
            )

            continue


        if not state.get(
            "use_depth",
            False,
        ):

            state[
                "depth_map"
            ] = None

            state[
                "depth_ts"
            ] = None
            state["depth_source_image_ts"] = None

            time.sleep(
                0.25
            )

            continue


        # Consistent snapshot: the image, receive time and sequence must belong
        # to the SAME ROS callback, even while image callbacks are concurrent.
        with state_lock:
            img = state.get("image_raw")
            image_seq = state.get("image_seq")
            image_ros_ts = state.get("image_ros_ts")
            image_recv_ts = state.get("image_ts")


        if img is None:

            time.sleep(
                interval
            )

            continue


        # Freeze this RGB frame for the depth computation.
        depth_image = (
            img.copy()
        )


        try:
            with _gpu_inference_lock:
                with timed("depth_estimation"):
                    depth_map = _run_depth(depth_image)


            if depth_map is not None:

                completed_ts = (
                    time.monotonic()
                )

                state[
                    "depth_map"
                ] = depth_map

                state[
                    "depth_ts"
                ] = completed_ts

                state[
                    "depth_source_image"
                ] = depth_image

                state[
                    "depth_source_image_seq"
                ] = image_seq

                state[
                    "depth_source_ros_ts"
                ] = image_ros_ts

                state["depth_source_image_ts"] = image_recv_ts

                state[
                    "depth_inference_ms"
                ] = (
                    completed_ts
                    - loop_t0
                ) * 1000.0


        except Exception as e:

            log(
                f"Depth loop error: {e}",
                "PERC",
            )


        elapsed = (
            time.monotonic()
            - loop_t0
        )

        time.sleep(
            max(
                0.0,
                interval
                - elapsed,
            )
        )
# ════════════════════════════════════════════════════════════
# SLOW LOOP — GroundingDINO  (~1 fps, CPU)
# ════════════════════════════════════════════════════════════
_GDINO_FPS = PERCEPTION_CONFIG["gdino_target_fps"]
_GDINO_CONF = PERCEPTION_CONFIG["gdino_conf"]

# Broad scene query — catches things YOLO misses (domain-specific objects)
_SCENE_QUERY = (
    "person . shelf . box . cart . forklift . door . wall . cone . "
    "pallet . bin . table . chair . robot . vehicle . barrel . fence . "
    "pillar . machine . equipment . package . container . sign . rack"
)

def _run_gdino_query(img: PILImage.Image, query: str, depth_map,
                     threshold: float = _GDINO_CONF, frame_seq=None, image_ros_ts=None) -> list:
    if _gdino_model is None or _gdino_proc is None:
        return []
    try:
        import torch
        W, H  = img.size
        text  = query.rstrip(".") + "."
        inputs = _gdino_proc(images=img, text=text, return_tensors="pt")
        # CPU — no device transfer
        with torch.no_grad():
            with timed("gdino_detection"):
                outputs = _gdino_model(**inputs)
        results = _gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=threshold, text_threshold=threshold,
            target_sizes=[(H, W)],
        )[0]
        dets = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            bbox = [round(v) for v in box.tolist()]
            cx   = (bbox[0] + bbox[2]) // 2
            dets.append({
                "label":  label,
                "dist":   round(_dist_from_bbox(depth_map, bbox), 2),
                "sector": _sector(cx, W),
                "bbox":   bbox,
                "conf":   round(float(score), 3),
                "source": "gdino",
                "image_seq": frame_seq if frame_seq is not None else state.get("image_seq"),
                "image_ros_ts": image_ros_ts if image_ros_ts is not None else state.get("image_ros_ts"),
                "depth_source_seq": state.get("depth_source_image_seq"),
                "created_ts": time.monotonic(),
            })
        return dets
    except Exception as e:
        log(f"GroundingDINO error: {e}", "PERC")
        return []


def _slow_loop() -> None:
    """GroundingDINO at ~1fps on CPU. Updates _gdino_dets."""
    global _gdino_dets
    interval = 1.0 / _GDINO_FPS

    while True:
        if not _models_ready or _gdino_model is None:
            time.sleep(1.0)
            continue

        # Skip if GDINO disabled
        if not state.get("use_gdino", False):
            time.sleep(1.0)
            continue

        img = state.get("depth_source_image") or state.get("image_raw")
        depth_map = state.get("depth_map")
        if img is None:
            time.sleep(interval)
            continue

        try:
            dets = _run_gdino_query(
                img, _SCENE_QUERY, depth_map,
                frame_seq=state.get("depth_source_image_seq"),
                image_ros_ts=state.get("depth_source_ros_ts"),
            )
            # Upgrade distances using LiDAR if available
            if (ROBOT.get("lidar_bbox_projection_validated", False)
                    and state.get("use_lidar", True) and state["lidar"]):
                W = img.size[0]
                for d in dets:
                    ld = _lidar_dist_for_bbox(d["bbox"], W)
                    if ld is not None:
                        d["dist"] = round(ld, 2)
            with _dets_lock:
                _gdino_dets = dets
            _update_state_detections()
        except Exception as e:
            log(f"Slow loop error: {e}", "PERC")

        time.sleep(interval)


def _update_state_detections() -> None:
    """Merge active detection sources into state["detections"]."""
    with _dets_lock:
        y = _yolo_dets[:] if state.get("use_yolo", True) else []
        g = _gdino_dets[:] if state.get("use_gdino", False) else []
    # GroundingDINO primary (richer labels), YOLO fills gaps
    state["detections"] = _merge_dets(g, y)


# ════════════════════════════════════════════════════════════
# PUBLIC: targeted query (tool call)
# ════════════════════════════════════════════════════════════
def query_object(object_name: str, img: PILImage.Image) -> list:
    """
    Return matching detections from enabled perception sources.

    GroundingDINO is already loaded at startup, but it is not executed
    unless use_gdino is enabled from the dashboard.
    """
    now = time.monotonic()

    cached = [
        d for d in state["detections"]
        if object_name.lower() in d["label"].lower()
        and now - float(d.get("created_ts", now))
        <= PERCEPTION_CONFIG["target_detection_max_age_s"]
    ]

    if cached:
        return cached

    # DINO model may be loaded, but do not execute it while disabled.
    if not state.get("use_gdino", False):
        return []

    depth = (
        _fresh_depth_map()
        if state.get(
            "use_depth",
            False,
        )
        else None
    )

    return _run_gdino_query(
        img,
        object_name,
        depth,
        threshold=PERCEPTION_CONFIG["gdino_query_conf"],
        frame_seq=state.get("image_seq"),
        image_ros_ts=state.get("image_ros_ts"),
    )


def run_grounding_dino(img: PILImage.Image, query: str) -> list:
    """
    Public targeted GroundingDINO query.

    The model remains loaded in memory but is not executed while
    GroundingDINO is disabled.
    """
    if not state.get("use_gdino", False):
        return []

    depth = (
        _fresh_depth_map()
        if state.get(
            "use_depth",
            False,
        )
        else None
    )
    return _run_gdino_query(
        img,
        query,
        depth,
        threshold=PERCEPTION_CONFIG["gdino_query_conf"],
        frame_seq=state.get("image_seq"),
        image_ros_ts=state.get("image_ros_ts"),
    )

# ════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════
def preload_perception_models() -> None:
    """
    Load all perception models synchronously.

    This is intentionally done BEFORE the ROS node starts so expensive
    Torch/CUDA/Transformers initialization cannot starve high-rate ROS
    sensor callbacks.

    Models are loaded into memory, but use_depth/use_gdino still control
    whether inference is executed.
    """
    state["yolo_model"] = _DEFAULT_YOLO

    log(
        "Preloading perception models before ROS startup...",
        "PERC",
    )

    _load_models()

    if not _models_ready:
        raise RuntimeError(
            "Perception model preload failed: "
            + str(_load_error)
        )

    log(
        "✅ Perception models preloaded",
        "PERC",
    )
def start_perception_loops() -> None:
    """Start perception with platform-appropriate depth scheduling.

    LiDAR-equipped robots keep DA-V2 on-demand so slow depth inference never
    competes continuously with Isaac/YOLO. Camera-only robots retain the
    continuous depth loop because depth is part of their safety sensing.
    """
    threading.Thread(
        target=_yolo_loop,
        daemon=True,
        name="vrosa-yolo-perception",
    ).start()

    if not ROBOT.get("has_lidar", False):
        threading.Thread(
            target=_depth_loop,
            daemon=True,
            name="vrosa-depth-perception",
        ).start()
        depth_mode = "continuous (camera-only platform)"
    else:
        # Clear any inherited/stale map. Carter safety is LiDAR-first and
        # navigation obtains target depth explicitly while stationary.
        state["depth_map"] = None
        state["depth_ts"] = None
        state["depth_source_image_ts"] = None
        depth_mode = "on-demand while stopped"

    threading.Thread(
        target=_slow_loop,
        daemon=True,
        name="vrosa-gdino-perception",
    ).start()

    log(
        f"Perception loops started (YOLO continuous / DA-V2 {depth_mode} / "
        "GroundingDINO runtime-controlled)",
        "PERC",
    )


def perception_status() -> dict:
    return {
        "ready":        _models_ready,
        "error":        _load_error,
        "device":       _device,
        "gdino_device": "cpu",
        "yolo_model":   _yolo_key,
        "has_lidar":    state.get("has_lidar", False),
        "n_yolo":       len(_yolo_dets),
        "n_gdino":      len(_gdino_dets),
        "n_total":      len(state["detections"]),
        "depth_ready":  state["depth_map"] is not None,
    }