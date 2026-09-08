#!/usr/bin/env python3
"""Replay paired DrivingStereo images as rectified stereo ROS 2 topics."""
import argparse
import datetime as dt
from pathlib import Path

import cv2
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


LEFT_CAMERA = "101"
RIGHT_CAMERA = "103"


def parse_calibration(filename):
    values = {}
    for line in filename.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            values[key.strip()] = [float(item) for item in value.split()]
        except ValueError:
            continue
    return values


def camera_info(values, camera):
    projection = values[f"P_rect_{camera}"]
    rectification = values.get(f"R_rect_{camera}", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    size = values[f"S_rect_{camera}"]
    info = CameraInfo()
    info.width, info.height = int(size[0]), int(size[1])
    info.k = [projection[0], 0.0, projection[2], 0.0, projection[5], projection[6], 0.0, 0.0, 1.0]
    info.r = rectification
    info.p = projection
    info.distortion_model = "plumb_bob"
    return info


def stamp_from_filename(filename):
    """DrivingStereo names end in YYYY-MM-DD-HH-MM-SS-mmm.jpg."""
    timestamp = filename.stem.rsplit("_", 1)[-1]
    parsed = dt.datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S-%f")
    return int(parsed.replace(tzinfo=dt.timezone.utc).timestamp() * 1_000_000_000)


class DrivingStereoPublisher(Node):
    def __init__(self, dataset, sequence, calibration, rate, loop):
        super().__init__("drivingstereo_publisher")
        self.dataset = Path(dataset)
        self.sequence, self.rate, self.loop = sequence, rate, loop
        self.bridge = CvBridge()
        self.left_pub = self.create_publisher(Image, "/drivingstereo/left/image_rect", 10)
        self.right_pub = self.create_publisher(Image, "/drivingstereo/right/image_rect", 10)
        self.left_info_pub = self.create_publisher(CameraInfo, "/drivingstereo/left/camera_info", 10)
        self.right_info_pub = self.create_publisher(CameraInfo, "/drivingstereo/right/camera_info", 10)
        self.frame_metadata_pub = self.create_publisher(Header, "/drivingstereo/frame_metadata", 10)

        calibration_file = self.dataset / "calib" / calibration / f"{sequence}.txt"
        if not calibration_file.is_file():
            raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
        values = parse_calibration(calibration_file)
        self.left_info = camera_info(values, LEFT_CAMERA)
        self.right_info = camera_info(values, RIGHT_CAMERA)

        left = {path.name: path for path in (self.dataset / "left").glob(f"{sequence}_*.jpg")}
        right = {path.name: path for path in (self.dataset / "right").glob(f"{sequence}_*.jpg")}
        names = sorted(left.keys() & right.keys())
        if not names:
            raise RuntimeError(f"No paired images found for sequence {sequence}")
        self.frames = [(left[name], right[name], stamp_from_filename(left[name])) for name in names]
        missing_left, missing_right = len(right) - len(names), len(left) - len(names)
        if missing_left or missing_right:
            self.get_logger().warning(f"Skipping unpaired images: left-only={missing_right}, right-only={missing_left}")

        self.index = 0
        self.published_frames = 0
        self.timer = None
        self.schedule_next(0.0)
        self.get_logger().info(
            f"Replaying {len(self.frames)} stereo pairs from {sequence}; "
            "DrivingStereo directory contains no IMU files, so no IMU topic is published."
        )

    @staticmethod
    def to_time(nanoseconds):
        result = Time()
        result.sec, result.nanosec = divmod(nanoseconds, 1_000_000_000)
        return result

    def schedule_next(self, seconds):
        # A one-shot timer must be destroyed before replacing it.  Merely
        # cancelling it left the old callback registered after the first frame
        # in rclpy, so replay appeared alive but published no further images.
        if self.timer is not None:
            self.timer.cancel()
            self.destroy_timer(self.timer)
        self.timer = self.create_timer(max(seconds, 0.0001), self.publish_frame)

    def publish_frame(self):
        left_path, right_path, nanoseconds = self.frames[self.index]
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"Failed to read stereo pair: {left_path.name}")

        stamp = self.to_time(nanoseconds)
        left_image = self.bridge.cv2_to_imgmsg(left, encoding="bgr8")
        right_image = self.bridge.cv2_to_imgmsg(right, encoding="bgr8")
        left_image.header.stamp, left_image.header.frame_id = stamp, "drivingstereo_left_optical"
        right_image.header.stamp, right_image.header.frame_id = stamp, "drivingstereo_right_optical"
        self.left_info.header, self.right_info.header = left_image.header, right_image.header
        self.left_pub.publish(left_image)
        self.right_pub.publish(right_image)
        self.left_info_pub.publish(self.left_info)
        self.right_info_pub.publish(self.right_info)
        # Header.stamp associates this source frame number with the stereo pair.
        self.frame_metadata_pub.publish(Header(stamp=stamp, frame_id=f"{self.index:06d}"))
        self.published_frames += 1
        if self.published_frames == 1:
            self.get_logger().info(f"Published first stereo pair: {left_path.name}")

        current = self.index
        self.index += 1
        if self.index == len(self.frames):
            if not self.loop:
                self.get_logger().info("Playback complete")
                rclpy.shutdown()
                return
            self.index = 0
        delta = (self.frames[self.index][2] - self.frames[current][2]) / 1_000_000_000 / self.rate
        self.schedule_next(delta if delta > 0 else 0.1 / self.rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--calibration", choices=("half-image-calib", "full-image-calib"), default="half-image-calib")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--loop", nargs="?", const=True, default=True,
                        type=lambda value: str(value).lower() in ("1", "true", "yes", "on"))
    args = parser.parse_args(remove_ros_args()[1:])
    if args.rate <= 0:
        parser.error("--rate must be positive")
    rclpy.init()
    node = DrivingStereoPublisher(**vars(args))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
