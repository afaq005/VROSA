"""
stream.py — MJPEG frame generator for the /stream Flask route.
Sends HUD-overlaid frames (or a placeholder) at STREAM_FPS.
"""

import io
import time

from PIL import Image as PILImage, ImageDraw

from config import STREAM_FPS
from state import state


def generate_stream():
    """Yields MJPEG boundary frames at the configured FPS."""
    interval = 1.0 / STREAM_FPS
    while True:
        img = state["image_hud"] or state["image_raw"]
        if img:
            thumb = img.resize((960, 600), PILImage.LANCZOS)
            buf   = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=80)
        else:
            # Placeholder while waiting for camera
            ph   = PILImage.new("RGB", (960, 600), (15, 15, 25))
            draw = ImageDraw.Draw(ph)
            draw.text((340, 280), "Waiting for camera...", fill=(0, 212, 255))
            buf = io.BytesIO()
            ph.save(buf, format="JPEG")

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.getvalue() + b"\r\n")
        time.sleep(interval)