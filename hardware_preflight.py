"""Scout 2.0 sensor diagnostic. Never publishes velocity, including if all sensors fail.

Allows paper-based supervised preflight even without RGB, physical depth, or
odometry. The output describes stream presence ONLY, never physical clearance.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--run-id", default="SCOUT_PREFLIGHT_001")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--range-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--odom-topic", default="/odom")
    a = p.parse_args()
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import ReentrantCallbackGroup

    class Watch(Node):
        def __init__(self):
            super().__init__("vrosa_scout_preflight_read_only")
            self.last = {"rgb": None, "range": None, "odom": None}
            self.count = {name: 0 for name in self.last}
            cb = ReentrantCallbackGroup()
            for name, kind, topic, qos in (("rgb", Image, a.image_topic, qos_profile_sensor_data),
                                           ("range", Image, a.range_topic, qos_profile_sensor_data),
                                           ("odom", Odometry, a.odom_topic, qos_profile_sensor_data)):
                self.create_subscription(kind, topic, lambda msg, n=name: self.note(n), qos, callback_group=cb)
        def note(self, name):
            self.last[name] = time.monotonic()
            self.count[name] += 1

    rclpy.init()
    n = Watch()
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(n)
    folder = Path(a.log_dir) / a.run_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "hardware_preflight.jsonl"
    try:
        start = time.monotonic()
        last_print = 0.0
        with target.open("a", encoding="utf-8") as f:
            while time.monotonic() - start < a.duration:
                ex.spin_once(timeout_sec=0.15)
                now = time.monotonic()
                if now - last_print >= 1.0:
                    ages = {name: (None if ts is None else round(now - ts, 3)) for name, ts in n.last.items()}
                    record = {"run_id": a.run_id, "wall_time": time.time(), "stream_age_s": ages,
                              "received_count": dict(n.count), "motion_authorized": False}
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    print(record)
                    last_print = now
    finally:
        ex.shutdown()
        n.destroy_node()
        rclpy.shutdown()
    print("Saved:", target)

if __name__ == "__main__":
    main()
