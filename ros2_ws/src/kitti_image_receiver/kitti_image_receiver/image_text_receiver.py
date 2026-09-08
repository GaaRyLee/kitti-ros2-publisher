#!/usr/bin/env python3
"""Save a fixed number of ROS Image messages as plain-text pixel data."""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image


class ImageTextReceiver(Node):
    def __init__(self, topic, output_dir, frame_count):
        super().__init__("image_text_receiver")
        self.bridge = CvBridge()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = frame_count
        self.received = 0
        self.subscription = self.create_subscription(Image, topic, self.receive_image, 10)
        self.get_logger().info(
            f"Waiting for {frame_count} image frames on {topic}; saving to {self.output_dir}"
        )

    def receive_image(self, message):
        if self.received >= self.frame_count:
            return

        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        frame = np.asarray(frame)
        filename = self.output_dir / f"frame_{self.received:03d}.txt"
        stamp = message.header.stamp
        header = (
            f"stamp_sec={stamp.sec} stamp_nanosec={stamp.nanosec} "
            f"frame_id={message.header.frame_id} encoding={message.encoding} "
            f"shape={frame.shape} dtype={frame.dtype}\n"
            "Each row is one pixel. Columns are channel values in the ROS image encoding order."
        )
        pixels = frame.reshape(-1, 1) if frame.ndim == 2 else frame.reshape(-1, frame.shape[-1])
        np.savetxt(filename, pixels, fmt="%d", header=header)
        self.received += 1
        self.get_logger().info(f"Saved {filename.name} ({self.received}/{self.frame_count})")

        if self.received == self.frame_count:
            self.get_logger().info("Received the requested 10 frames; stopping.")
            self.destroy_subscription(self.subscription)
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/kitti/camera_02/image_raw")
    parser.add_argument("--output-dir", default="/workspace/output/image_text")
    parser.add_argument("--frame-count", type=int, default=10)
    args = parser.parse_args(remove_ros_args()[1:])
    if args.frame_count <= 0:
        parser.error("--frame-count must be positive")

    rclpy.init()
    node = ImageTextReceiver(args.topic, args.output_dir, args.frame_count)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
