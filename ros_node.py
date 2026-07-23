"""
ros_node.py — ROS 2 subscriber/publisher node for real Nova Carter hardware.
"""

import threading
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from PIL import Image as PILImage
import io
import time
from state import state
from logger import log

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ── Hardware topic names ──────────────────────────────────────────
#TOPIC_IMAGE    = "/front_stereo_camera/left/image_raw"
#TOPIC_CAMINFO  = "/front_stereo_camera/left/camera_info"
TOPIC_LIDAR    = "/front_3d_lidar/lidar_points"
#TOPIC_ODOM     = "/chassis/odom"
#TOPIC_CMDVEL   = "/cmd_vel"

TOPIC_IMAGE   = "/camera/camera/color/image_raw"
TOPIC_CAMINFO = "/camera/camera/color/camera_info"
TOPIC_DEPTH   = "/camera/camera/depth/image_rect_raw"
TOPIC_DEPTHINFO = "/camera/camera/depth/camera_info"


TOPIC_CMDVEL  = "/cmd_vel"
TOPIC_ODOM    = "/odom"

class VRosaNode(Node):
    def __init__(self):
        super().__init__("vrosa_node")

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, TOPIC_CMDVEL, 10)
        self._img_count = 0
        # Subscribers
        self.create_subscription(Image,       TOPIC_IMAGE,   self._cb_image,   SENSOR_QOS)
        self.create_subscription(CameraInfo,  TOPIC_CAMINFO, self._cb_caminfo, SENSOR_QOS)
        self.create_subscription(PointCloud2, TOPIC_LIDAR,   self._cb_lidar,   SENSOR_QOS)
        self.create_subscription(Odometry,    TOPIC_ODOM,    self._cb_odom,    10)

        log("ROS 2 node started (hardware mode)", "ROS")

    def _cb_image(self, msg: Image):
        try:
            enc = msg.encoding.lower()

            arr = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding.lower() in ["rgb8", "bgr8"]:
                arr = arr.reshape(msg.height, msg.width, 3)
            elif msg.encoding.lower() in ["rgba8", "bgra8"]:
                arr = arr.reshape(msg.height, msg.width, 4)
                arr = arr[:, :, :3]
            elif msg.encoding.lower() in ["mono8"]:
                arr = arr.reshape(msg.height, msg.width)
            else:
                arr = arr.reshape(msg.height, msg.width, -1)

            if "bgr" in enc:
                arr = arr[..., ::-1]

            state["image_raw"] = PILImage.fromarray(arr.copy())
            state["camera_live"] = True
            state["last_image_time"] = time.time()

            self._img_count += 1
            if self._img_count % 3000 == 0:
                log(
                    f"Image received: {msg.width}x{msg.height}, "
                    f"enc={msg.encoding}, count={self._img_count}",
                    "ROS"
                )

        except Exception as e:
            log(f"Image cb error: {e}", "ROS")

    def _cb_caminfo(self, msg: CameraInfo):
        if msg.k[0] > 0:
            state["cam_fx"] = float(msg.k[0])

    def _cb_lidar(self, msg: PointCloud2):
        try:
            pts = []
            for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True):
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                d = math.sqrt(x*x + y*y + z*z)
                if d > 0.1:
                    pts.append({"x": x, "y": y, "z": z, "dist": round(d, 3)})
            pts.sort(key=lambda p: p["dist"])
            state["lidar"] = pts
        except Exception as e:
            log(f"LiDAR cb error: {e}", "ROS")

    def _cb_odom(self, msg: Odometry):
        state["x"] = msg.pose.pose.position.x
        state["y"] = msg.pose.pose.position.y
        # Quaternion → yaw
        q  = msg.pose.pose.orientation
        yaw = math.degrees(
            math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        )
        state["yaw"] = yaw


# ── Singleton ─────────────────────────────────────────────────────
rclpy.init()
_node    = VRosaNode()
cmd_pub  = _node.cmd_pub   # motion.py imports this

def spin():
    rclpy.spin(_node)

threading.Thread(target=spin, daemon=True).start()
log("ROS 2 spin thread started", "ROS")
