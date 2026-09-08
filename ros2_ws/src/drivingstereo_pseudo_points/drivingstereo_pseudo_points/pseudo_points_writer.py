#!/usr/bin/env python3
"""DrivingStereo pseudo-points, ORB/PnP odometry, and RViz map publishing."""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from torch_cluster import knn

from kitti_image_receiver.gitHubModel import PSMNet
from kitti_image_receiver.MuiltiScale import MuiltiScaleModelDual


MAX_DISP = 192


def pad_to_multiple_of_16(left, right):
    _, _, height, width = left.shape
    top_pad, right_pad = (-height) % 16, (-width) % 16
    return F.pad(left, (0, right_pad, top_pad, 0)), F.pad(right, (0, right_pad, top_pad, 0)), top_pad, right_pad


def disparity_from_psmnet(model, left, right, device):
    left, right, top_pad, right_pad = pad_to_multiple_of_16(left, right)
    with torch.no_grad():
        disparity = model(left.to(device), right.to(device))[0, 0].detach().cpu().numpy()
    if top_pad:
        disparity = disparity[top_pad:, :]
    if right_pad:
        disparity = disparity[:, :-right_pad]
    valid = (disparity > 0) & (disparity < MAX_DISP)
    return np.where(valid, disparity, 0).astype(np.float32)


def depth_from_disparity(disparity, fx, baseline):
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = fx * baseline / disparity[valid]
    return depth


def disparity_to_filter_points(disparity, fx, fy, cx, cy, baseline):
    """Return the learned filter's existing [forward, left, up] coordinate convention."""
    valid = disparity > 0
    depth = depth_from_disparity(disparity, fx, baseline)
    height, width = disparity.shape
    u = np.broadcast_to(np.arange(width), (height, width))
    v = np.broadcast_to(np.arange(height)[:, None], (height, width))
    x_camera, y_camera = (u - cx) * depth / fx, (v - cy) * depth / fy
    return np.stack((depth[valid], -x_camera[valid], -y_camera[valid]), axis=-1).astype(np.float32)


def filter_to_optical(points):
    """[forward, left, up] -> ROS optical [right, down, forward]."""
    return np.column_stack((-points[:, 1], -points[:, 2], points[:, 0])).astype(np.float32, copy=False) if len(points) else points.reshape(0, 3)


def color_points_from_lidar(image_bgr, points, fx, fy, cx, cy):
    """Project [forward, left, up] points into the left image and return RGB uint8."""
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    if not len(points):
        return colors
    x_camera, y_camera = -points[:, 1], -points[:, 2]
    z_camera = points[:, 0]
    valid = z_camera > 1e-6
    u = np.rint(x_camera[valid] * fx / z_camera[valid] + cx).astype(np.int32)
    v = np.rint(y_camera[valid] * fy / z_camera[valid] + cy).astype(np.int32)
    height, width = image_bgr.shape[:2]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_indices = np.flatnonzero(valid)[inside]
    # cv_bridge supplies BGR; RViz's packed rgb field requires RGB order.
    colors[valid_indices] = image_bgr[v[inside], u[inside], ::-1]
    return colors


def colored_cloud(header, points, colors):
    """Create an XYZ + packed-RGB PointCloud2 without Python point-by-point loops."""
    cloud = PointCloud2()
    cloud.header, cloud.height, cloud.width = header, 1, len(points)
    cloud.fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)]
    cloud.is_bigendian, cloud.point_step, cloud.row_step, cloud.is_dense = False, 16, 16 * len(points), True
    records = np.empty(len(points), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")]))
    records["x"], records["y"], records["z"] = points[:, 0], points[:, 1], points[:, 2]
    records["rgb"] = ((colors[:, 0].astype(np.uint32) << 16) |
                      (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2].astype(np.uint32))
    cloud.data = records.tobytes()
    return cloud


def fast_knn(points, k):
    return points[knn(points, points, k)[1].view(points.size(0), k)]


def filter_points(model, pseudo_points, batch_size, confidence):
    kept = []
    with torch.no_grad():
        for start in range(0, pseudo_points.size(0), batch_size):
            batch = pseudo_points[start:start + batch_size]
            if batch.size(0) < 24:
                continue
            neighbors = fast_knn(batch, 24)
            scales = {12: neighbors[:, :12], 16: neighbors[:, :16], 20: neighbors[:, :20], 24: neighbors}
            classification, regression = model(batch, scales)
            mask = classification.squeeze().squeeze(-1) >= confidence
            if mask.any():
                kept.append((regression.squeeze(0) + batch)[mask])
    return torch.cat(kept).cpu().numpy().astype(np.float32) if kept else np.empty((0, 3), dtype=np.float32)


def quaternion_from_rotation(rotation):
    """Matrix to xyzw quaternion without an additional transform dependency."""
    trace = np.trace(rotation)
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        return ((rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale)
    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        scale = 2 * np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        return (0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale)
    if index == 1:
        scale = 2 * np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        return ((rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale)
    scale = 2 * np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    return ((rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale)


class PseudoPointsWriter(Node):
    def __init__(self, left_topic, right_topic, output_dir, psm_weights, filter_weights, frame_count,
                 batch_size, confidence, fx, fy, cx, cy, baseline, map_frame, camera_frame,
                 voxel_size, max_depth, min_pnp_inliers):
        super().__init__("drivingstereo_mapping")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for PSMNet; start the Docker service with GPU support.")
        self.device, self.bridge = torch.device("cuda"), CvBridge()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pose_log_file = (self.output_dir / "pose_log.csv").open("w", newline="")
        self.pose_log = csv.writer(self.pose_log_file)
        self.pose_log.writerow(("frame_index", "stamp_sec", "stamp_nanosec", "pose_accepted", "orb_matches",
                                "pnp_inliers", "relative_tx_m", "relative_ty_m", "relative_tz_m",
                                "relative_translation_m", "relative_rotation_deg", "map_camera_x_m",
                                "map_camera_y_m", "map_camera_z_m"))
        self.frame_count, self.batch_size, self.confidence = frame_count, batch_size, confidence
        self.fx, self.fy, self.cx, self.cy, self.baseline = fx, fy, cx, cy, baseline
        self.map_frame, self.camera_frame = map_frame, camera_frame
        self.voxel_size, self.max_depth, self.min_pnp_inliers = voxel_size, max_depth, min_pnp_inliers
        self.left_messages, self.right_messages, self.frame_indices = {}, {}, {}
        self.written, self.shutdown_timer = 0, None
        self.previous_gray, self.previous_depth = None, None
        # OpenCV/ROS optical camera coordinates are [right, down, forward].
        # RViz's conventional map ground plane is [forward, left, up].  Make
        # that conversion the initial map pose, then compose PnP poses in the
        # same optical coordinate basis on every subsequent frame.
        self.map_from_camera = np.eye(4)
        self.map_from_camera[:3, :3] = np.array(((0., 0., 1.),
                                                  (-1., 0., 0.),
                                                  (0., -1., 0.)))
        self.map_points = np.empty((0, 3), dtype=np.float32)
        self.map_colors = np.empty((0, 3), dtype=np.uint8)
        self.orb = cv2.ORB_create(nfeatures=2500, scaleFactor=1.2, nlevels=8, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.psmnet = PSMNet(MAX_DISP).to(self.device)
        self.psmnet.load_state_dict(torch.load(psm_weights, map_location=self.device))
        self.psmnet.eval()
        self.filter_model = MuiltiScaleModelDual().to(self.device)
        self.filter_model.load_state_dict(torch.load(filter_weights, map_location=self.device))
        self.filter_model.eval()
        self.filtered_pub = self.create_publisher(PointCloud2, "/drivingstereo/filtered_points", 10)
        # Retain the newest accumulated map so RViz2 can be opened after the
        # mapper and still render a cloud immediately.
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(PointCloud2, "/drivingstereo/map_points", map_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.left_subscription = self.create_subscription(Image, left_topic, self.receive_left, 10)
        self.right_subscription = self.create_subscription(Image, right_topic, self.receive_right, 10)
        self.metadata_subscription = self.create_subscription(Header, "/drivingstereo/frame_metadata", self.receive_metadata, 10)
        self.get_logger().info(f"Mapping ready: {frame_count} frames, ORB + PnP-RANSAC, map frame '{map_frame}'.")

    @staticmethod
    def stamp_key(message):
        return message.header.stamp.sec, message.header.stamp.nanosec

    def receive_left(self, message):
        self.left_messages[self.stamp_key(message)] = message
        self.process_matches()

    def receive_right(self, message):
        self.right_messages[self.stamp_key(message)] = message
        self.process_matches()

    def receive_metadata(self, message):
        try:
            self.frame_indices[(message.stamp.sec, message.stamp.nanosec)] = int(message.frame_id)
        except ValueError:
            self.get_logger().warning(f"Ignoring invalid frame metadata: {message.frame_id!r}")
            return
        self.process_matches()

    def process_matches(self):
        if self.written >= self.frame_count:
            return
        for stamp in sorted(self.left_messages.keys() & self.right_messages.keys() & self.frame_indices.keys()):
            left, right = self.left_messages.pop(stamp), self.right_messages.pop(stamp)
            self.process_pair(left, right, self.frame_indices.pop(stamp))
            self.written += 1
            if self.written == self.frame_count:
                self.get_logger().info("Finished requested mapping frames; stopping.")
                self.shutdown_timer = self.create_timer(0.01, self.finish)
                return

    def finish(self):
        self.shutdown_timer.cancel()
        self.pose_log_file.close()
        self.get_logger().info("Mapping node shutdown complete.")
        rclpy.shutdown()

    def estimate_pose(self, current_gray):
        """Find T_current_previous by using previous dense stereo depth at ORB keypoints."""
        previous_keys, previous_desc = self.orb.detectAndCompute(self.previous_gray, None)
        current_keys, current_desc = self.orb.detectAndCompute(current_gray, None)
        if previous_desc is None or current_desc is None:
            return None, 0, 0
        pairs = self.matcher.knnMatch(previous_desc, current_desc, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        object_points, image_points = [], []
        height, width = self.previous_depth.shape
        for match in matches:
            u, v = np.rint(previous_keys[match.queryIdx].pt).astype(int)
            if not (0 <= u < width and 0 <= v < height):
                continue
            depth = self.previous_depth[v, u]
            if 0.1 < depth < self.max_depth:
                object_points.append(((u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth))
                image_points.append(current_keys[match.trainIdx].pt)
        if len(object_points) < self.min_pnp_inliers:
            return None, len(matches), 0
        camera = np.array(((self.fx, 0., self.cx), (0., self.fy, self.cy), (0., 0., 1.)))
        success, rvec, translation, inliers = cv2.solvePnPRansac(
            np.asarray(object_points, np.float32), np.asarray(image_points, np.float32), camera, None,
            iterationsCount=200, reprojectionError=2.5, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        inlier_count = 0 if inliers is None else len(inliers)
        if not success or inlier_count < self.min_pnp_inliers:
            return None, len(matches), inlier_count
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation.reshape(3)
        return transform, len(matches), inlier_count

    def add_to_map(self, optical_points, colors):
        if not len(optical_points):
            return
        transformed = optical_points @ self.map_from_camera[:3, :3].T + self.map_from_camera[:3, 3]
        candidates = np.vstack((self.map_points, transformed.astype(np.float32, copy=False)))
        candidate_colors = np.vstack((self.map_colors, colors))
        keys = np.floor(candidates / self.voxel_size).astype(np.int32)
        _, indices = np.unique(keys, axis=0, return_index=True)
        indices = np.sort(indices)
        self.map_points, self.map_colors = candidates[indices], candidate_colors[indices]

    def publish(self, optical_points, colors, stamp):
        self.filtered_pub.publish(colored_cloud(Header(stamp=stamp, frame_id=self.camera_frame), optical_points, colors))
        header = Header(stamp=stamp, frame_id=self.map_frame)
        self.map_pub.publish(colored_cloud(header, self.map_points, self.map_colors))
        transform = TransformStamped(header=header, child_frame_id=self.camera_frame)
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = self.map_from_camera[:3, 3]
        quaternion = quaternion_from_rotation(self.map_from_camera[:3, :3])
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = quaternion
        self.tf_broadcaster.sendTransform(transform)

    def process_pair(self, left_message, right_message, frame_index):
        left_bgr = self.bridge.imgmsg_to_cv2(left_message, desired_encoding="bgr8")
        right_bgr = self.bridge.imgmsg_to_cv2(right_message, desired_encoding="bgr8")
        to_tensor = lambda image: torch.from_numpy(image[:, :, ::-1].copy()).float().permute(2, 0, 1).div_(255.).unsqueeze(0)
        disparity = disparity_from_psmnet(self.psmnet, to_tensor(left_bgr), to_tensor(right_bgr), self.device)
        depth, current_gray = depth_from_disparity(disparity, self.fx, self.baseline), cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        accepted, matches, inliers = True, 0, 0
        current_from_previous = np.eye(4)
        if self.previous_gray is not None:
            current_from_previous, matches, inliers = self.estimate_pose(current_gray)
            accepted = current_from_previous is not None
            if accepted:
                self.map_from_camera = self.map_from_camera @ np.linalg.inv(current_from_previous)
            else:
                self.get_logger().warning(f"Frame {frame_index:06d}: PnP rejected; ORB matches={matches}, inliers={inliers}. Not added to map.")
        filtered = filter_points(self.filter_model, torch.from_numpy(
            disparity_to_filter_points(disparity, self.fx, self.fy, self.cx, self.cy, self.baseline)).to(self.device),
            self.batch_size, self.confidence)
        optical = filter_to_optical(filtered)
        colors = color_points_from_lidar(left_bgr, filtered, self.fx, self.fy, self.cx, self.cy)
        # filtered.tofile(self.output_dir / f"{frame_index:06d}.bin")
        if accepted:
            self.add_to_map(optical, colors)
        self.publish(optical, colors, left_message.header.stamp)
        self.previous_gray, self.previous_depth = current_gray, depth
        xyz = self.map_from_camera[:3, 3]
        relative_translation = np.full(3, np.nan) if not accepted else current_from_previous[:3, 3]
        relative_distance = np.nan if not accepted else np.linalg.norm(relative_translation)
        if accepted:
            cosine = np.clip((np.trace(current_from_previous[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            relative_rotation_deg = np.degrees(np.arccos(cosine))
        else:
            relative_rotation_deg = np.nan
        stamp = left_message.header.stamp
        self.pose_log.writerow((frame_index, stamp.sec, stamp.nanosec, accepted, matches, inliers,
                                *relative_translation, relative_distance, relative_rotation_deg, *xyz))
        self.pose_log_file.flush()
        self.get_logger().info(f"Frame {frame_index:06d}: filtered={len(optical)}, ORB={matches}, PnP_inliers={inliers}, "
                               f"relative_t={relative_distance:.2f}m, relative_R={relative_rotation_deg:.2f}deg, "
                               f"map_points={len(self.map_points)}, map_camera_xyz=({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})")


def main():
    model_dir = "/ros2_ws/model_folder"
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-topic", default="/drivingstereo/left/image_rect")
    parser.add_argument("--right-topic", default="/drivingstereo/right/image_rect")
    parser.add_argument("--output-dir", default="/workspace/output/pseudo_points")
    parser.add_argument("--psm-weights", default=f"{model_dir}/psm_fly_correctlr_slicnormal_100.pt")
    parser.add_argument("--filter-weights", default=f"{model_dir}/MuiltiScale_dual_v2.pt")
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--fx", type=float, default=1003.556)
    parser.add_argument("--fy", type=float, default=1003.556)
    parser.add_argument("--cx", type=float, default=455.689)
    parser.add_argument("--cy", type=float, default=197.663)
    parser.add_argument("--baseline", type=float, default=0.5446)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--camera-frame", default="drivingstereo_left_optical")
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=80.)
    parser.add_argument("--min-pnp-inliers", type=int, default=20)
    args = parser.parse_args(remove_ros_args()[1:])
    if args.frame_count <= 0 or args.batch_size < 24 or args.voxel_size <= 0 or args.min_pnp_inliers < 4:
        parser.error("frame-count > 0, batch-size >= 24, voxel-size > 0, min-pnp-inliers >= 4 are required")
    rclpy.init()
    node = PseudoPointsWriter(**vars(args))
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
