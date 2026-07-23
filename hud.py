"""
hud.py — Builds the HUD overlay burned into each camera frame.

Draws:
  - Top info bar (position, heading, status, VLM mode, detection count)
  - Bottom LiDAR/depth distance bar + last VLM text
  - Crosshair
  - Detection boxes from YOLO / GroundingDINO with label + metric distance
"""

from PIL import Image as PILImage, ImageDraw

from state import state
from vlm.manager import vlm_mgr

# ── Colour palette per detection source ─────────────────────────
_COLOUR = {
    "yolo":      (0,   212, 255),   # cyan  — YOLO
    "grounding": (165, 214, 167),   # green — GroundingDINO
}
_LABEL_BG = {
    "yolo":      (0,   60,  80),
    "grounding": (20,  60,  20),
}


def _build_hud(img: PILImage.Image) -> PILImage.Image:
    """Burn telemetry + detection overlay onto a copy of the camera image."""
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # ════════════════════════════════════════════════════════
    # DETECTION BOXES  (only when toggle is ON)
    # ════════════════════════════════════════════════════════
    detections = state.get("detections", []) if state.get("show_dets") else []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        src    = det.get("source", "yolo")
        colour = _COLOUR.get(src, (255, 255, 255))
        bg     = _LABEL_BG.get(src, (30, 30, 30))
        dist   = det["dist"]
        label  = det["label"]

        # ── Bounding box ─────────────────────────────────────
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

        # ── Distance fill bar at bottom of box ───────────────
        bar_w  = max(1, x2 - x1)
        fill_w = min(bar_w, int(bar_w * min(1.0, 2.0 / max(dist, 0.1))))
        draw.rectangle([x1, y2 - 4, x1 + fill_w, y2], fill=colour)

        # ── Label chip: "person  1.8m" ───────────────────────
        dist_str = f"{dist:.1f}m" if dist < 90 else "—"
        text     = f" {label}  {dist_str} "
        tw       = len(text) * 6
        th       = 13
        chip_y   = max(0, y1 - th - 2)
        draw.rectangle([x1, chip_y, x1 + tw, chip_y + th], fill=bg)
        draw.text((x1 + 2, chip_y + 2), text, fill=colour)

    # ════════════════════════════════════════════════════════
    # TOP INFO BAR
    # ════════════════════════════════════════════════════════
    draw.rectangle([0, 0, W, 44], fill=(0, 0, 0))
    n_dets        = len(detections)
    perc_mode     = "LiDAR+Depth" if state.get("has_lidar") else "Depth-only"
    draw.text(
        (12, 12),
        f"V-ROSA  |  x={state['x']:.2f}m  y={state['y']:.2f}m  "
        f"hdg={state['yaw']:.1f}°  |  [{state['status'].upper()}]  "
        f"|  VLM:{vlm_mgr.active_name.upper()}  "
        f"|  {n_dets} obj  {perc_mode}",
        fill=(0, 212, 255),
    )

    # ════════════════════════════════════════════════════════
    # BOTTOM DISTANCE BAR
    # ════════════════════════════════════════════════════════
    lidar = state["lidar"]
    if lidar:
        near      = lidar[0]["dist"]
        src_label = "LiDAR"
    elif detections:
        visible = [d["dist"] for d in detections if d["dist"] < 90]
        near      = min(visible) if visible else 99.0
        src_label = "Depth"
    else:
        near      = 99.0
        src_label = "—"

    color = (255,  60,  60) if near < 1.0 else \
            (255, 200,   0) if near < 2.5 else \
            (  0, 220,  80)

    draw.rectangle([0, H - 80, W, H], fill=(0, 0, 0))
    draw.text(
        (12, H - 68),
        f"[{src_label}] closest: {near:.2f}m  |  Target: {state['target'] or 'none'}",
        fill=color,
    )
    bar = int(min(1.0, 3.0 / max(near, 0.1)) * (W - 24))
    draw.rectangle([12, H - 20, 12 + bar, H - 6], fill=color)

    # ── VLM last response ─────────────────────────────────────────
    draw.text((12, H - 46), f"VLM: {state['last_vlm'][:120]}", fill=(200, 200, 200))

    # ════════════════════════════════════════════════════════
    # CROSSHAIR
    # ════════════════════════════════════════════════════════
    cx, cy = W // 2, H // 2
    draw.line([cx - 30, cy,      cx + 30, cy],      fill=(255, 255, 255), width=2)
    draw.line([cx,      cy - 30, cx,      cy + 30], fill=(255, 255, 255), width=2)
    draw.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], outline=(255, 255, 255), width=1)

    return img