"""
stream.py — MJPEG camera stream.

Always starts from the latest raw ROS camera frame.
The HUD is rebuilt for every streamed frame so a stale image_hud
can never freeze the dashboard video.
"""

import io
import time

from PIL import Image as PILImage, ImageDraw

from config import STREAM_FPS
from state import state
from hud import _build_hud


def generate_stream():
    """Yield continuously updated MJPEG frames."""
    interval = 1.0 / max(1, STREAM_FPS)
    # frame_count = 0
    while True:
        # frame_count += 1
        raw = state.get("image_raw")
        # if frame_count % 30 == 0:
        #     print(
        #         f"[STREAM] generated={frame_count}, "
        #         f"image_seq={state.get('image_seq', 0)}, "
        #         f"image_ts={state.get('image_ts')}"
        #     )
        if raw is not None:
            try:
                # IMPORTANT:
                # Work on a copy so HUD drawing never modifies image_raw.
                frame = raw.copy()

                # Build HUD from CURRENT telemetry/detections.
                frame = _build_hud(frame)

                # Resize only for web streaming.
                frame = frame.resize(
                    (960, 600),
                    PILImage.Resampling.LANCZOS,
                )

                buf = io.BytesIO()
                frame.save(
                    buf,
                    format="JPEG",
                    quality=80,
                    optimize=False,
                )

                jpeg = buf.getvalue()

            except Exception as exc:
                # Do not kill the MJPEG generator because of one bad frame.
                ph = PILImage.new(
                    "RGB",
                    (960, 600),
                    (15, 15, 25),
                )
                draw = ImageDraw.Draw(ph)
                draw.text(
                    (250, 280),
                    f"Camera frame error: {exc}",
                    fill=(255, 120, 120),
                )

                buf = io.BytesIO()
                ph.save(buf, format="JPEG", quality=80)
                jpeg = buf.getvalue()

        else:
            ph = PILImage.new(
                "RGB",
                (960, 600),
                (15, 15, 25),
            )
            draw = ImageDraw.Draw(ph)
            draw.text(
                (340, 280),
                "Waiting for camera...",
                fill=(0, 212, 255),
            )

            buf = io.BytesIO()
            ph.save(buf, format="JPEG", quality=80)
            jpeg = buf.getvalue()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Pragma: no-cache\r\n"
            b"\r\n"
            + jpeg
            + b"\r\n"
        )

        time.sleep(interval)