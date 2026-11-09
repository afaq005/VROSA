"""
ros_node.py — ROS 2 I/O layer.

All topic names come from the active robot profile. Sensor callbacks are split
across callback groups and executed with a MultiThreadedExecutor so high-rate
camera/PointCloud2 work cannot starve odometry. LiDAR safety corridor minima are
computed once per scan; the 50 ms motion safety loop therefore does not rescan
an entire point cloud on every iteration.
"""

from __future__ import annotations

import math
import threading
import time
import os

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from config import ROBOT
from experiment_logger import record_event
from logger import log
from state import state, state_lock

TOPICS = ROBOT["topics"]
# Isaac Sim RGB writer is normally RELIABLE.
# Use RELIABLE here so large 1920x1200 images are not silently dropped
# under CPU/GPU load.
CAMERA_QOS = QoSProfile(
    reliability=(ReliabilityPolicy.BEST_EFFORT if ROBOT.get("camera_qos") == "best_effort"
                 else ReliabilityPolicy.RELIABLE),
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
)

CAMERA_INFO_QOS = QoSProfile(
    reliability=(ReliabilityPolicy.BEST_EFFORT if ROBOT.get("camera_qos") == "best_effort"
                 else ReliabilityPolicy.RELIABLE),
    history=HistoryPolicy.KEEP_LAST,
    depth=2,
)

# LiDAR remains sensor-style BEST_EFFORT because only the newest scan matters.
LIDAR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_HALF_ROBOT_WIDTH = ROBOT.get("footprint", {}).get("width_m", 0.70) / 2.0
_SAFETY_MARGIN = ROBOT.get("footprint", {}).get("safety_margin_m", 0.10)
_LIDAR_HALF_WIDTH = _HALF_ROBOT_WIDTH + _SAFETY_MARGIN
_LIDAR_Z_MIN, _LIDAR_Z_MAX = -0.10, 2.0
_LIDAR_X_MIN = 0.05
_LIDAR_STORE_STRIDE = max(1, int(ROBOT.get("lidar_store_stride", 4)))


def _ros_stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _update_rate(prefix: str, recv_ts: float) -> None:
    """Maintain an EWMA receive period/rate for diagnostics and the paper."""
    key_last = f"_{prefix}_last_recv_ts"
    key_period = f"{prefix}_period_s_ema"
    key_hz = f"{prefix}_hz_ema"
    last = state.get(key_last)
    state[key_last] = recv_ts
    if last is None:
        return
    period = recv_ts - float(last)
    if period <= 0:
        return
    old = state.get(key_period)
    ema = period if old is None else (0.85 * float(old) + 0.15 * period)
    state[key_period] = ema
    state[key_hz] = 1.0 / ema if ema > 1e-9 else None


class VRosaNode(Node):
    def __init__(self):
        super().__init__("vrosa_node")

        # Independent callback groups prevent a large image/point-cloud callback
        # from blocking odometry updates in the ROS executor.
        self._image_group = MutuallyExclusiveCallbackGroup()
        self._caminfo_group = MutuallyExclusiveCallbackGroup()
        self._lidar_group = MutuallyExclusiveCallbackGroup()
        self._odom_group = MutuallyExclusiveCallbackGroup()
        self._range_depth_group = MutuallyExclusiveCallbackGroup()

        self.cmd_pub = self.create_publisher(Twist, TOPICS["cmd_vel"], 10)
        self.create_subscription(
            Image,
            TOPICS["image"],
            self._cb_image,
            CAMERA_QOS,
            callback_group=self._image_group,
        )
        self.create_subscription(
            CameraInfo,
            TOPICS["camera_info"],
            self._cb_caminfo,
            CAMERA_INFO_QOS,
            callback_group=self._caminfo_group,
        )
        if TOPICS.get("lidar"):
            self.create_subscription(
                PointCloud2, TOPICS["lidar"], self._cb_lidar, LIDAR_QOS,
                callback_group=self._lidar_group,
            )
        else:
            log(f"{ROBOT['display_name']}: no LiDAR topic configured", "ROS")
        if ROBOT.get("platform_label") == "Hardware" and TOPICS.get("range_depth"):
            self.create_subscription(
                Image, TOPICS["range_depth"], self._cb_range_depth,
                LIDAR_QOS, callback_group=self._range_depth_group,
            )
            log(f"Hardware aligned metric depth: {TOPICS['range_depth']}", "ROS")
        self.create_subscription(
            Odometry, TOPICS["odom"], self._cb_odom, 10,
            callback_group=self._odom_group,
        )
        # Optional independent simulator pose. NEVER point this at the normal
        # wheel/robot odometry topic and call it GT. Check in the simulator first.
        gt_topic = os.environ.get("VROSA_GT_ODOM_TOPIC", "").strip()
        self._gt_group = MutuallyExclusiveCallbackGroup()  # must retain callback group
        if gt_topic:
            if gt_topic == TOPICS["odom"]:
                raise ValueError("VROSA_GT_ODOM_TOPIC must differ from robot odometry")
            self.create_subscription(Odometry, gt_topic, self._cb_gt_odom, 10,
                                     callback_group=self._gt_group)
            log(f"Independent GT pose input configured: {gt_topic}", "ROS")
        log(f"ROS 2 node started — {ROBOT['display_name']} ({ROBOT['platform_label']})", "ROS")

    def _cb_gt_odom(self, msg: Odometry):
        state["gt_robot_x"] = float(msg.pose.pose.position.x)
        state["gt_robot_y"] = float(msg.pose.pose.position.y)
        state["gt_robot_ts"] = time.monotonic()
        state["gt_robot_source"] = os.environ.get("VROSA_GT_ODOM_TOPIC", "")

    def _cb_image(self, msg: Image):
        """
        Receive one ROS Image message and convert it to PIL RGB.

        image_rx_seq increments immediately on callback entry.
        image_seq increments only after successful RGB conversion.

        Therefore:
            rx_seq increases, image_seq does not
                -> conversion problem

            neither increases
                -> ROS/DDS/QoS delivery problem
        """
        recv_ts = time.monotonic()

        # Count callback entry BEFORE doing any processing.
        rx_seq = int(state.get("image_rx_seq", 0)) + 1
        state["image_rx_seq"] = rx_seq
        state["image_callback_enter_ts"] = recv_ts

        _update_rate("image", recv_ts)

        # if rx_seq % 30 == 0:
        #     log(
        #         f"Camera callbacks: rx={rx_seq}, "
        #         f"converted={state.get('image_seq', 0)}, "
        #         f"rate={state.get('image_hz_ema') or 0.0:.2f} Hz",
        #         "ROS",
        #     )

        try:
            height = int(msg.height)
            width = int(msg.width)
            step = int(msg.step)

            if height <= 0 or width <= 0 or step <= 0:
                raise ValueError(
                    f"Invalid image geometry: "
                    f"{width}x{height}, step={step}"
                )

            enc = str(msg.encoding).lower()

            # Map common ROS image encodings explicitly.
            if enc in ("rgb8", "bgr8"):
                channels = 3
            elif enc in ("rgba8", "bgra8"):
                channels = 4
            elif enc in ("mono8", "8uc1"):
                channels = 1
            else:
                # Fallback based on row stride.
                channels = max(
                    1,
                    int(step // max(1, width)),
                )

            raw = np.frombuffer(
                msg.data,
                dtype=np.uint8,
            )

            expected_bytes = height * step

            if raw.size < expected_bytes:
                raise ValueError(
                    f"Image buffer too small: "
                    f"{raw.size} < {expected_bytes}"
                )

            # Respect ROS row stride/padding.
            rows = raw[:expected_bytes].reshape(
                height,
                step,
            )

            row_pixels = width * channels

            if row_pixels > step:
                raise ValueError(
                    f"Encoding/step mismatch: "
                    f"width={width}, channels={channels}, step={step}"
                )

            pixels = rows[:, :row_pixels]

            if channels == 1:
                arr = pixels.reshape(
                    height,
                    width,
                )

                arr = np.repeat(
                    arr[..., None],
                    3,
                    axis=2,
                )

            else:
                arr = pixels.reshape(
                    height,
                    width,
                    channels,
                )

                # Remove alpha channel if present.
                arr = arr[..., :3]

                # ROS BGR/BGRA -> RGB.
                if enc in ("bgr8", "bgra8"):
                    arr = arr[..., ::-1]

            # Make an independent contiguous copy.
            arr = np.ascontiguousarray(arr)

            image = PILImage.fromarray(
                arr,
                mode="RGB",
            )

            # Publish the new complete frame atomically.
            with state_lock:
                state["image_raw"] = image
                state["image_ts"] = recv_ts
                state["image_ros_ts"] = _ros_stamp_sec(msg.header.stamp)
                # Use callback count as processed sequence number.
                state["image_seq"] = rx_seq

            state["camera_frame"] = (
                msg.header.frame_id
            )

            state["image_processing_ms"] = (
                time.monotonic() - recv_ts
            ) * 1000.0

        except Exception as exc:
            state["image_conversion_errors"] = (
                int(
                    state.get(
                        "image_conversion_errors",
                        0,
                    )
                )
                + 1
            )

            log(
                f"Image callback conversion error "
                f"(rx={rx_seq}, encoding={msg.encoding}, "
                f"size={msg.width}x{msg.height}, "
                f"step={msg.step}): {exc}",
                "ROS",
            )

    def _cb_caminfo(self, msg: CameraInfo):
        if len(msg.k) >= 9 and msg.k[0] > 0:
            state["cam_fx"] = float(msg.k[0])
            state["cam_fy"] = float(msg.k[4])
            state["cam_cx"] = float(msg.k[2])
            state["cam_cy"] = float(msg.k[5])
            state["camera_width"] = int(msg.width)
            state["camera_height"] = int(msg.height)
            state["camera_frame"] = msg.header.frame_id or state.get("camera_frame", "")
            if msg.width > 0:
                state["camera_hfov_deg"] = math.degrees(
                    2.0 * math.atan(msg.width / (2.0 * float(msg.k[0])))
                )
            if not state.get("camera_info_logged", False):
                record_event(
                    "CAMERA_CALIBRATION", frame=state["camera_frame"], width=msg.width, height=msg.height,
                    fx=state["cam_fx"], fy=state["cam_fy"], cx=state["cam_cx"], cy=state["cam_cy"],
                    horizontal_fov_deg=state.get("camera_hfov_deg"),
                )
                state["camera_info_logged"] = True

    def _cb_range_depth(self, msg: Image):
        """Read physical RealSense aligned depth; 16UC1 mm or 32FC1 m only.

        This is NOT an inferred monocular-depth image. Never convert color images
        or an unknown encoding into a synthetic safe range.
        """
        recv_ts = time.monotonic()
        _update_rate("range_depth", recv_ts)
        try:
            height, width, step = int(msg.height), int(msg.width), int(msg.step)
            enc = str(msg.encoding).lower()
            if enc in ("16uc1", "mono16"):
                dtype, scale = np.dtype("u2"), 0.001
            elif enc == "32fc1":
                dtype, scale = np.dtype("f4"), 1.0
            else:
                raise ValueError(f"Unsupported physical depth encoding: {msg.encoding}")
            dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
            row_nbytes = width * dtype.itemsize
            if width <= 0 or height <= 0 or step < row_nbytes:
                raise ValueError("Invalid physical depth image geometry")
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if buf.size < height * step:
                raise ValueError("Truncated physical depth image")
            view = buf[:height * step].reshape(height, step)[:, :row_nbytes]
            metric = np.ascontiguousarray(view).view(dtype).reshape(height, width).astype(np.float32)
            metric *= scale
            # A zero pixel means no measurement; it is never evidence of free space.
            state["range_depth_map"] = metric
            state["range_depth_ts"] = recv_ts
            state["range_depth_ros_ts"] = _ros_stamp_sec(msg.header.stamp)
            state["range_depth_processing_ms"] = (time.monotonic() - recv_ts) * 1000
        except Exception as exc:
            state["range_depth_conversion_errors"] = int(state.get("range_depth_conversion_errors", 0)) + 1
            log(f"Hardware depth conversion rejected: {exc}", "ROS")

    def _cb_lidar(self, msg: PointCloud2):
        recv_ts = time.monotonic()
        _update_rate("lidar", recv_ts)
        try:
            sampled = []
            nearest = math.inf
            fwd = math.inf
            rear = math.inf
            left = math.inf
            center = math.inf
            right = math.inf

            # Safety minima are computed from ALL valid points. Only the list kept
            # for UI/VLM hints is downsampled, which greatly reduces Python memory
            # and sorting cost without weakening the safety corridor.
            for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                d = math.sqrt(x * x + y * y + z * z)
                if d <= 0.1:
                    continue
                nearest = min(nearest, d)

                if _LIDAR_Z_MIN < z < _LIDAR_Z_MAX and abs(y) < _LIDAR_HALF_WIDTH:
                    if x > _LIDAR_X_MIN:
                        fwd = min(fwd, d)
                    elif x < -_LIDAR_X_MIN:
                        rear = min(rear, d)

                if x > 0.1 and abs(z) < 1.5:
                    if y > 0.3:
                        left = min(left, d)
                    if abs(y) < 0.8:
                        center = min(center, d)
                    if y < -0.3:
                        right = min(right, d)

                if i % _LIDAR_STORE_STRIDE == 0:
                    sampled.append({"x": x, "y": y, "z": z, "dist": d})

            sampled.sort(key=lambda q: q["dist"])
            state["lidar"] = sampled
            state["lidar_nearest_min"] = 99.0 if not math.isfinite(nearest) else nearest
            state["lidar_forward_min"] = 99.0 if not math.isfinite(fwd) else fwd
            state["lidar_reverse_min"] = 99.0 if not math.isfinite(rear) else rear
            state["lidar_left_min"] = 99.0 if not math.isfinite(left) else left
            state["lidar_center_min"] = 99.0 if not math.isfinite(center) else center
            state["lidar_right_min"] = 99.0 if not math.isfinite(right) else right
            state["lidar_ts"] = recv_ts
            state["lidar_ros_ts"] = _ros_stamp_sec(msg.header.stamp)
            state["lidar_frame"] = msg.header.frame_id
            state["has_lidar"] = True
            state["lidar_processing_ms"] = (time.monotonic() - recv_ts) * 1000.0
        except Exception as e:
            log(f"LiDAR callback error: {e}", "ROS")

    def _cb_odom(self, msg: Odometry):
        recv_ts = time.monotonic()
        _update_rate("odom", recv_ts)
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        prev = state.get("_prev_odom_xy")
        if prev is not None:
            step = math.hypot(x - prev[0], y - prev[1])
            # Reject improbable discontinuities caused by reset/teleport between trials.
            if step < 2.0:
                state["path_length_m"] = float(state.get("path_length_m", 0.0)) + step
        state["_prev_odom_xy"] = (x, y)
        state["x"], state["y"] = x, y

        q = msg.pose.pose.orientation
        state["yaw"] = math.degrees(
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        )
        state["linear_velocity"] = float(msg.twist.twist.linear.x)
        state["angular_velocity"] = float(msg.twist.twist.angular.z)
        state["odom_ts"] = recv_ts
        state["odom_ros_ts"] = _ros_stamp_sec(msg.header.stamp)


rclpy.init()
_node = VRosaNode()
cmd_pub = _node.cmd_pub
_executor = MultiThreadedExecutor(num_threads=4)
_executor.add_node(_node)


def spin():
    _executor.spin()


_spin_thread = threading.Thread(target=spin, daemon=True)
_spin_thread.start()


def stop_spin():
    """Shutdown executor before rclpy.shutdown to avoid background spin crash."""
    try:
        _executor.shutdown(timeout_sec=2.0)
    except Exception as exc:
        log(f"ROS executor shutdown warning: {exc}", "WARN")
    if _spin_thread.is_alive():
        _spin_thread.join(timeout=2.0)
    _node.destroy_node()


record_event("ROS_NODE_STARTED", robot=ROBOT["display_name"], topics=TOPICS, executor_threads=4)
log("ROS 2 multi-threaded spin started (4 threads)", "ROS")
