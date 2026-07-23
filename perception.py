"""
perception.py — YOLO26 + Depth Anything V2 + GroundingDINO perception pipeline.

TWO DECOUPLED LOOPS for max speed:
  Fast loop  (~15 fps, GPU): YOLO26 + DA-V2  → real-time bbox + depth
  Slow loop  (~1  fps, CPU): GroundingDINO    → semantic labels for known objects

Both write to state["detections"] with deduplication (IoU merge).
GroundingDINO lives on CPU to preserve VRAM alongside Isaac Sim.

YOLO model hot-swappable at runtime via /yolo/switch endpoint.
"""

import math
import time
import threading

import numpy as np
from PIL import Image as PILImage

from state import state
from logger import log

import warnings
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
_DEFAULT_YOLO = "yolo26n"


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


# ════════════════════════════════════════════════════════════
# SHARED GEOMETRY HELPERS
# ════════════════════════════════════════════════════════════
def _dist_from_bbox(depth_map, bbox: list, pct: int = None) -> float:
    """Robust metric distance from depth map inside a bounding box.

    Uses a centre-crop (inner 60%) of the bbox to avoid noisy depth at edges.
    Adaptive percentile:
      < 2 m  → 10th pct  (grab nearest surface, very close)
      2–5 m  → 25th pct  (stable but still near-biased)
      > 5 m  → 45th pct  (median-ish — far-range depth is less certain)
    """
    if depth_map is None:
        return 99.0
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = depth_map.shape

    # Centre-crop: inner 60% of bbox width/height — avoids depth bleeding at edges
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    hw = max(4, int((x2 - x1) * 0.30))   # 30% half-width
    hh = max(4, int((y2 - y1) * 0.30))   # 30% half-height
    cx1, cy1 = max(0, mx - hw), max(0, my - hh)
    cx2, cy2 = min(W, mx + hw), min(H, my + hh)

    patch = depth_map[cy1:cy2, cx1:cx2]
    valid = patch[patch > 0.1]
    if len(valid) == 0:
        return 99.0

    # Adaptive percentile
    rough = float(np.median(valid))
    if pct is None:
        if   rough < 2.0: pct = 10
        elif rough < 5.0: pct = 25
        else:             pct = 45

    return float(np.percentile(valid, pct))


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
    """
    Project a pixel bbox into LiDAR space and return the closest point.

    How it works:
      - Camera focal length (cam_fx) converts pixel offset → horizontal angle
      - Each LiDAR point has (x=forward, y=left, z=up) in robot frame
      - We compute each point's bearing angle and check if it falls inside
        the angular span of the bbox
      - Returns the 20th-percentile distance of matching points (robust)
      - Returns None if LiDAR is disabled or no points found in that sector

    This is the most accurate distance source — LiDAR is ±2cm at any range.
    DA-V2 depth degrades beyond ~8m, so LiDAR fusion fixes far objects.
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


def _best_dist(bbox: list, img_w: int, depth_map) -> float:
    """
    Return the best available distance for a detection, in priority order:
      1. LiDAR  — most accurate at ALL ranges (enabled via use_lidar flag)
      2. DA-V2  — camera-only fallback (enabled via use_depth flag)
      3. 99.0   — unknown
    """
    # Try LiDAR first — fixes far-object inaccuracy
    if state.get("use_lidar", True):
        d = _lidar_dist_for_bbox(bbox, img_w)
        if d is not None:
            return round(d, 2)

    # Fallback: depth map
    if state.get("use_depth", True) and depth_map is not None:
        return round(_dist_from_bbox(depth_map, bbox), 2)

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

    log("Loading perception models (background)...", "PERC")
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Perception GPU: {_device} | GroundingDINO: CPU (saves VRAM for Isaac Sim)", "PERC")

        # ── DA-V2 → GPU ───────────────────────────────────────
        log("Loading DA-V2 Metric Outdoor Small (GPU)...", "PERC")
        from transformers import pipeline as hf_pipeline
        _depth_pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
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

        # ── YOLO26 → GPU ──────────────────────────────────────
        _load_yolo_model(state.get("yolo_model", _DEFAULT_YOLO))

        _models_ready       = True
        state["perc_ready"] = True
        log("✅ All perception models loaded — dual-loop active", "PERC")

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
        # predicted_depth: torch.Tensor, shape [H, W], values in metres
        depth_tensor = result["predicted_depth"].squeeze()
        depth_np = depth_tensor.cpu().numpy().astype(np.float32) * 10.0
        # Sanity clamp: DA-V2 Metric Indoor trained for 0.1–20m range
        depth_np = np.clip(depth_np, 0.1, 20.0)
        return depth_np
    except Exception as e:
        log(f"DA-V2 error: {e}", "PERC")
        return None


# ════════════════════════════════════════════════════════════
# FAST LOOP — YOLO26 + DA-V2  (~15 fps, GPU)
# ════════════════════════════════════════════════════════════
_YOLO_CONF    = 0.25   # lower = more detections
_YOLO_FPS     = 15

def _fast_loop() -> None:
    """YOLO + depth at ~15fps. Updates _yolo_dets and state["depth_map"]."""
    global _yolo_dets
    interval = 1.0 / _YOLO_FPS

    while True:
        if not _models_ready:
            time.sleep(0.5)
            continue

        img = state["image_raw"]
        if img is None:
            time.sleep(interval)
            continue

        try:
            # ── LiDAR presence tracking ───────────────────────
            if state["lidar"]:
                state["has_lidar"] = True
                state["_lidar_ts"] = time.time()
            elif time.time() - state.get("_lidar_ts", 0) > 5.0:
                state["has_lidar"] = False

            # ── DA-V2 depth map (only if enabled) ────────────
            if state.get("use_depth", True):
                depth_map = _run_depth(img)
                if depth_map is not None:
                    state["depth_map"] = depth_map

            # ── YOLO (only if enabled) ────────────────────────
            if not state.get("use_yolo", True):
                time.sleep(interval)
                continue

            with _yolo_lock:
                model = _yolo_model
            if model is None:
                time.sleep(interval)
                continue

            W, H    = img.size
            results = model(img, verbose=False, conf=_YOLO_CONF)[0]
            dets    = []
            for box in results.boxes:
                bbox  = [round(v) for v in box.xyxy[0].tolist()]
                cx    = (bbox[0] + bbox[2]) // 2
                label = results.names[int(box.cls)]
                # Use best available distance: LiDAR > depth > 99
                dets.append({
                    "label":  label,
                    "dist":   _best_dist(bbox, W, state["depth_map"]),
                    "sector": _sector(cx, W),
                    "bbox":   bbox,
                    "conf":   round(float(box.conf), 3),
                    "source": "yolo",
                })

            with _dets_lock:
                _yolo_dets = dets
            _update_state_detections()

        except Exception as e:
            log(f"Fast loop error: {e}", "PERC")

        time.sleep(interval)


# ════════════════════════════════════════════════════════════
# SLOW LOOP — GroundingDINO  (~1 fps, CPU)
# ════════════════════════════════════════════════════════════
_GDINO_FPS  = 1
_GDINO_CONF = 0.30

# Broad scene query — catches things YOLO misses (domain-specific objects)
_SCENE_QUERY = (
    "person . shelf . box . cart . forklift . door . wall . cone . "
    "pallet . bin . table . chair . robot . vehicle . barrel . fence . "
    "pillar . machine . equipment . package . container . sign . rack"
)

def _run_gdino_query(img: PILImage.Image, query: str, depth_map,
                     threshold: float = _GDINO_CONF) -> list:
    if _gdino_model is None or _gdino_proc is None:
        return []
    try:
        import torch
        W, H  = img.size
        text  = query.rstrip(".") + "."
        inputs = _gdino_proc(images=img, text=text, return_tensors="pt")
        # CPU — no device transfer
        with torch.no_grad():
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
        if not state.get("use_gdino", True):
            time.sleep(1.0)
            continue

        img       = state["image_raw"]
        depth_map = state["depth_map"]
        if img is None:
            time.sleep(interval)
            continue

        try:
            dets = _run_gdino_query(img, _SCENE_QUERY, depth_map)
            # Upgrade distances using LiDAR if available
            if state.get("use_lidar", True) and state["lidar"]:
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
        y = _yolo_dets[:] if state.get("use_yolo",  True) else []
        g = _gdino_dets[:] if state.get("use_gdino", True) else []
    # GroundingDINO primary (richer labels), YOLO fills gaps
    state["detections"] = _merge_dets(g, y)


# ════════════════════════════════════════════════════════════
# PUBLIC: targeted query (tool call)
# ════════════════════════════════════════════════════════════
def query_object(object_name: str, img: PILImage.Image) -> list:
    """On-demand GroundingDINO search for a specific named object."""
    cached = [d for d in state["detections"]
              if object_name.lower() in d["label"].lower()]
    if cached:
        return cached
    return _run_gdino_query(img, object_name, state["depth_map"], threshold=0.25)


# ════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════
def start_perception() -> None:
    """Load models then start both inference loops in background threads."""
    state["yolo_model"] = _DEFAULT_YOLO

    def _boot():
        _load_models()
        # Start both loops after models are ready
        threading.Thread(target=_slow_loop, daemon=True).start()
        _fast_loop()   # fast loop blocks this thread

    threading.Thread(target=_boot, daemon=True).start()
    log("Perception pipeline starting (background)...", "PERC")


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
