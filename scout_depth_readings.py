#!/usr/bin/env python3
"""Read-only ROS 2 aligned-depth sampler for Scout/RealSense static board tests.

Example:
  python3 scout_depth_readings.py --distance 1.25 --samples 5

Prints a board-patch median and a forward ROI 8th-percentile (P8) in meters.
All readings are also appended to a CSV file. NEVER publishes robot commands.
"""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

DEFAULT_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
DEFAULT_OUTPUT = str(Path.home() / 'final_vrosa_review_candidate' / 'logs' /
                     'SCOUT_DEPTH_ACC_001' / 'depth_measurements.csv')
COLUMNS = [
    'utc_time', 'tape_m', 'sample', 'topic', 'frame_id', 'encoding',
    'board_patch_median_m', 'board_patch_p08_m', 'board_patch_valid_pct',
    'forward_roi_p08_m', 'forward_roi_median_m', 'forward_roi_valid_pct',
    'board_patch_error_m', 'board_patch_absolute_error_m',
]


def stats(pixels):
    valid = pixels[np.isfinite(pixels) & (pixels >= 0.20) & (pixels <= 10.0)]
    valid_pct = 100.0 * (valid.size / pixels.size) if pixels.size else 0.0
    if valid.size == 0:
        return (float('nan'), float('nan'), valid_pct)
    return (float(np.median(valid)), float(np.percentile(valid, 8)), valid_pct)


def crop(img, y0, y1, x0, x1):
    h, w = img.shape
    r0 = max(0, min(h - 1, int(h * y0)))
    r1 = max(r0 + 1, min(h, int(h * y1)))
    c0 = max(0, min(w - 1, int(w * x0)))
    c1 = max(c0 + 1, min(w, int(w * x1)))
    return img[r0:r1, c0:c1]


def meters_from_image(msg):
    enc = msg.encoding.upper()
    if enc in ('16UC1', 'MONO16'):
        dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
        factor = 0.001  # measured D435i scale: 16-bit millimeters -> meters
    elif enc == '32FC1':
        dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
        factor = 1.0  # 32-bit floating-point depth already in meters
    else:
        raise ValueError(f'Unsupported encoding {msg.encoding!r}; expected 16UC1 or 32FC1')
    if msg.step < msg.width * dtype.itemsize:
        raise ValueError('Invalid image row stride (step)')
    raw = memoryview(msg.data)
    required = (msg.height - 1) * msg.step + msg.width * dtype.itemsize
    if len(raw) < required:
        raise ValueError('Depth image payload shorter than expected')
    raw_img = np.ndarray(shape=(msg.height, msg.width), dtype=dtype,
                         buffer=raw, strides=(msg.step, dtype.itemsize))
    return raw_img.astype(np.float32) * factor


class Sampler(Node):
    def __init__(self, args):
        super().__init__('scout_depth_accuracy_sampler')
        self.args = args
        self.samples = []
        self.last_capture = -1e20
        self.last_error = None
        self.path = Path(args.output).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self.csv_file = self.path.open('a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=COLUMNS)
        if new_file:
            self.writer.writeheader()
            self.csv_file.flush()
        self.sub = self.create_subscription(Image, args.topic, self.callback, 10)

    def callback(self, msg):
        now = time.monotonic()
        if len(self.samples) >= self.args.samples or now - self.last_capture < self.args.interval:
            return
        try:
            img = meters_from_image(msg)
            # Approximate documented VROSA forward safety ROI: rows 33-68%, cols 35-65%.
            roi = crop(img, 1.0/3.0, 0.68, 0.35, 0.65)
            # Board resting on the floor is BELOW the lens optical axis (~54-cm mount).
            # This patch is positioned below image center to avoid sampling background.
            patch = crop(img, self.args.patch_y0, self.args.patch_y1,
                         self.args.patch_x0, self.args.patch_x1)
            pmed, pp8, pv = stats(patch)
            rmed, rp8, rv = stats(roi)
        except (ValueError, TypeError) as e:
            if str(e) != self.last_error:
                print(f'[ERROR] {e}', file=sys.stderr, flush=True)
                self.last_error = str(e)
            return
        self.last_capture = now
        num = len(self.samples) + 1
        err = pmed - self.args.distance if math.isfinite(pmed) else float('nan')
        record = dict(
            utc_time=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
            tape_m=f'{self.args.distance:.4f}', sample=num,
            topic=self.args.topic, frame_id=msg.header.frame_id,
            encoding=msg.encoding,
            board_patch_median_m=f'{pmed:.4f}',
            board_patch_p08_m=f'{pp8:.4f}',
            board_patch_valid_pct=f'{pv:.1f}',
            forward_roi_p08_m=f'{rp8:.4f}',
            forward_roi_median_m=f'{rmed:.4f}',
            forward_roi_valid_pct=f'{rv:.1f}',
            board_patch_error_m=f'{err:.4f}',
            board_patch_absolute_error_m=f'{abs(err):.4f}',
        )
        self.samples.append(record)
        self.writer.writerow(record)
        self.csv_file.flush()
        print(f'{num}/{self.args.samples}  '
              f'board_patch={pmed:6.3f} m (valid={pv:5.1f}%)  '
              f'ROI_P8={rp8:6.3f} m (valid={rv:5.1f}%)  '
              f'board_error={err:+.3f} m', flush=True)
        if not math.isfinite(pmed) or abs(err) > 0.15:
            print('    CHECK: board patch may include background, another object, '
                  'or the taped distance/reference may be incorrect.', flush=True)

    def close(self):
        self.csv_file.flush()
        self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(description='Read-only Scout RealSense depth calibration sampler')
    parser.add_argument('--distance', type=float, required=True,
                        help='Tape-measured camera-lens-to-board distance, in meters')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Minimum seconds between captured frames')
    parser.add_argument('--topic', default=DEFAULT_TOPIC)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--patch-y0', type=float, default=0.56)
    parser.add_argument('--patch-y1', type=float, default=0.64)
    parser.add_argument('--patch-x0', type=float, default=0.46)
    parser.add_argument('--patch-x1', type=float, default=0.54)
    args = parser.parse_args()
    if not (0.2 <= args.distance <= 10.0):
        parser.error('distance must be between 0.2 and 10 meters')
    if args.samples < 1 or args.samples > 100:
        parser.error('samples must be in 1..100')
    if args.interval < 0:
        parser.error('interval must be >= 0')
    if not (0 <= args.patch_y0 < args.patch_y1 <= 1 and
            0 <= args.patch_x0 < args.patch_x1 <= 1):
        parser.error('patch bounds must be normalized and strictly increasing')

    rclpy.init()
    node = Sampler(args)
    print(f'Topic: {args.topic}')
    print(f'Tape distance: {args.distance:.3f} m | samples: {args.samples}')
    print('READ-ONLY: no velocity or robot commands will be published.')
    print(f'CSV: {node.path}\n', flush=True)
    started = time.monotonic()
    try:
        while rclpy.ok() and len(node.samples) < args.samples:
            rclpy.spin_once(node, timeout_sec=0.3)
            if time.monotonic() - started > max(20.0, args.samples * (args.interval + 2.0)):
                print('[TIMEOUT] Not enough depth messages. Check the camera topic '
                      'and ROS_DOMAIN_ID.', file=sys.stderr)
                break
        good = [s for s in node.samples if math.isfinite(float(s['board_patch_median_m']))]
        if good:
            bp = np.array([float(s['board_patch_median_m']) for s in good])
            rp = np.array([float(s['forward_roi_p08_m']) for s in node.samples], dtype=float)
            print('\n--- Summary ---')
            print(f'Board-patch depth: mean={np.mean(bp):.4f} m; '
                  f'absolute error vs tape={abs(np.mean(bp)-args.distance):.4f} m')
            if np.isfinite(rp).any():
                print(f'Forward safety ROI P8: mean={np.nanmean(rp):.4f} m')
            print(f'Saved {len(node.samples)} readings to {node.path}')
        else:
            print('No valid board depth; do not use as accuracy evidence.')
    except KeyboardInterrupt:
        print('\nInterrupted; collected readings are already saved.')
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if len(node.samples) != args.samples:
        sys.exit(2)


if __name__ == '__main__':
    main()
