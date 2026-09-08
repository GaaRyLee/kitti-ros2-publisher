#!/usr/bin/env python3
"""Replay a KITTI Raw synced drive as ROS 2 camera, IMU, and clock topics."""
import argparse
import math
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo, Image, Imu
from rosgraph_msgs.msg import Clock


CAMERA_ENCODINGS = {"00": "mono8", "01": "mono8", "02": "bgr8", "03": "bgr8"}


def stamp_from_kitti(line):
    # Timestamp precision is nanoseconds; avoid platform-dependent datetime parsing.
    date, clock = line.strip().split()
    year, month, day = map(int, date.split("-"))
    hour, minute, second_fraction = clock.split(":")
    second, fraction = second_fraction.split(".")
    import datetime
    epoch = datetime.datetime(year, month, day, int(hour), int(minute), int(second),
                              tzinfo=datetime.timezone.utc).timestamp()
    return int(epoch) * 1_000_000_000 + int(fraction[:9].ljust(9, "0"))


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class KittiPublisher(Node):
    def __init__(self, sequence, calibration, rate, loop):
        super().__init__("kitti_publisher")
        self.sequence, self.rate, self.loop = Path(sequence), rate, loop
        self.bridge = CvBridge()
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.imu_pub = self.create_publisher(Imu, "/kitti/imu", 10)
        self.image_pubs = {c: self.create_publisher(Image, f"/kitti/camera_{c}/image_raw", 10)
                           for c in CAMERA_ENCODINGS}
        self.info_pubs = {c: self.create_publisher(CameraInfo, f"/kitti/camera_{c}/camera_info", 10)
                          for c in CAMERA_ENCODINGS}
        self.timestamps = [stamp_from_kitti(x) for x in
                           (self.sequence / "image_00" / "timestamps.txt").read_text().splitlines()]
        self.imu_files = sorted((self.sequence / "oxts" / "data").glob("*.txt"))
        self.camera_info = self.load_camera_info(Path(calibration) / "calib_cam_to_cam.txt")
        if not self.timestamps or len(self.timestamps) != len(self.imu_files):
            raise RuntimeError("Camera timestamps and OXTS frames must be present and have equal lengths")
        self.index = 0
        self.timer = None
        self.schedule_next(0.0)
        self.get_logger().info(f"Replaying {len(self.timestamps)} frames from {self.sequence}")

    def load_camera_info(self, filename):
        values = {}
        for line in filename.read_text().splitlines():
            if ":" in line:
                key, data = line.split(":", 1)
                try:
                    values[key] = [float(x) for x in data.split()]
                except ValueError:
                    # KITTI calibration files also contain text metadata such as
                    # the calibration date; it is not a camera parameter.
                    continue
        result = {}
        for camera in CAMERA_ENCODINGS:
            info = CameraInfo()
            size = values.get(f"S_rect_{camera}", values.get(f"S_{camera}"))
            matrix = values.get(f"P_rect_{camera}", values.get(f"P_{camera}"))
            if size and matrix:
                info.width, info.height = int(size[0]), int(size[1])
                info.k = [matrix[0], 0.0, matrix[2], 0.0, matrix[5], matrix[6], 0.0, 0.0, 1.0]
                info.p = matrix
                info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                info.distortion_model = "plumb_bob"
            result[camera] = info
        return result

    @staticmethod
    def to_time(nanoseconds):
        msg = Time()
        msg.sec, msg.nanosec = divmod(nanoseconds, 1_000_000_000)
        return msg

    def schedule_next(self, seconds):
        self.timer = self.create_timer(max(seconds, 0.0001), self.publish_frame)

    def publish_frame(self):
        self.timer.cancel()
        i, nanos = self.index, self.timestamps[self.index]
        stamp = self.to_time(nanos)
        self.clock_pub.publish(Clock(clock=stamp))
        for camera, encoding in CAMERA_ENCODINGS.items():
            frame = cv2.imread(str(self.sequence / f"image_{camera}" / "data" / f"{i:010d}.png"),
                               cv2.IMREAD_GRAYSCALE if encoding == "mono8" else cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Missing camera {camera} frame {i}")
            image = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
            image.header.stamp, image.header.frame_id = stamp, f"camera_{camera}"
            info = self.camera_info[camera]
            info.header = image.header
            self.image_pubs[camera].publish(image)
            self.info_pubs[camera].publish(info)
        data = [float(x) for x in self.imu_files[i].read_text().split()]
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, "imu"
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = quaternion_from_rpy(*data[3:6])
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = data[11:14]
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = data[20:23]
        self.imu_pub.publish(imu)
        self.index += 1
        if self.index == len(self.timestamps):
            if not self.loop:
                self.get_logger().info("Playback complete")
                rclpy.shutdown()
                return
            self.index = 0
        delta = (self.timestamps[self.index] - nanos) / 1_000_000_000 / self.rate
        if delta <= 0:
            delta = 0.1 / self.rate
        self.schedule_next(delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument(
        "--loop", nargs="?", const=True, default=False,
        type=lambda value: str(value).lower() in ("1", "true", "yes", "on"),
        help="Repeat playback (optionally pass true/false).",
    )
    # A launch file appends ROS-specific arguments (for example, --ros-args).
    # argparse should receive only this node's command-line arguments.
    args = parser.parse_args(remove_ros_args()[1:])
    if args.rate <= 0:
        parser.error("--rate must be positive")
    rclpy.init()
    node = KittiPublisher(args.sequence, args.calibration, args.rate, args.loop)
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
