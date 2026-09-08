#!/usr/bin/env python3
"""Save timestamp-matched DrivingStereo ROS image pairs as plain-text pixels."""
import argparse
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image


class StereoImageTextReceiver(Node):
    def __init__(self, left_topic, right_topic, output_dir, frame_count):
        super().__init__("drivingstereo_image_receiver")
        self.bridge = CvBridge()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = frame_count
        self.saved = 0
        self.left_messages = {}
        self.right_messages = {}
        self.left_subscription = self.create_subscription(Image, left_topic, self.receive_left, 10)
        self.right_subscription = self.create_subscription(Image, right_topic, self.receive_right, 10)
        self.get_logger().info(
            f"Waiting for {frame_count} matched stereo pairs on {left_topic} and {right_topic}; "
            f"writing to {self.output_dir}"
        )

    @staticmethod
    def stamp_key(message):
        stamp = message.header.stamp
        return stamp.sec, stamp.nanosec

    def receive_left(self, message):
        self.left_messages[self.stamp_key(message)] = message
        self.save_matches()

    def receive_right(self, message):
        self.right_messages[self.stamp_key(message)] = message
        self.save_matches()

    def save_matches(self):
        if self.saved >= self.frame_count:
            return
        for stamp in sorted(self.left_messages.keys() & self.right_messages.keys()):
            left = self.left_messages.pop(stamp)
            right = self.right_messages.pop(stamp)
            self.save_pair(left, right)
            self.saved += 1
            self.get_logger().info(f"Saved stereo pair {self.saved}/{self.frame_count} at {stamp[0]}.{stamp[1]:09d}")
            if self.saved == self.frame_count:
                self.get_logger().info("Received requested stereo pairs; stopping.")
                self.destroy_subscription(self.left_subscription)
                self.destroy_subscription(self.right_subscription)
                rclpy.shutdown()
                return

    def save_pair(self, left_message, right_message):
        for side, message in (("left", left_message), ("right", right_message)):
            image = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"))
            stamp = message.header.stamp
            filename = self.output_dir / f"pair_{self.saved:03d}_{side}.txt"
            header = (
                f"stamp_sec={stamp.sec} stamp_nanosec={stamp.nanosec} "
                f"frame_id={message.header.frame_id} encoding={message.encoding} "
                f"shape={image.shape} dtype={image.dtype}\n"
                "Each row is one pixel. Columns are channel values in the ROS image encoding order."
            )
            pixels = image.reshape(-1, 1) if image.ndim == 2 else image.reshape(-1, image.shape[-1])
            np.savetxt(filename, pixels, fmt="%d", header=header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-topic", default="/drivingstereo/left/image_rect")
    parser.add_argument("--right-topic", default="/drivingstereo/right/image_rect")
    parser.add_argument("--output-dir", default="/workspace/output/drivingstereo_image_text")
    parser.add_argument("--frame-count", type=int, default=10)
    args = parser.parse_args(remove_ros_args()[1:])
    if args.frame_count <= 0:
        parser.error("--frame-count must be positive")
    rclpy.init()
    node = StereoImageTextReceiver(**vars(args))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
