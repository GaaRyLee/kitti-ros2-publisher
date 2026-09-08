"""Start DrivingStereo playback after the GPU mapping node has initialized."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    dataset = LaunchConfiguration("dataset")
    sequence = LaunchConfiguration("sequence")
    calibration = LaunchConfiguration("calibration")
    rate = LaunchConfiguration("rate")
    frame_count = LaunchConfiguration("frame_count")
    mapping = Node(
        package="drivingstereo_pseudo_points",
        executable="pseudo_points_writer",
        name="drivingstereo_mapping",
        output="screen",
        arguments=["--frame-count", frame_count],
    )
    playback = Node(
        package="drivingstereo_dataset_publisher",
        executable="drivingstereo_publisher",
        name="drivingstereo_publisher",
        output="screen",
        arguments=["--dataset", dataset, "--sequence", sequence, "--calibration", calibration,
                   "--rate", rate, "--loop", "true"],
    )
    return LaunchDescription([
        DeclareLaunchArgument("dataset", default_value="/workspace/drivingstereo"),
        DeclareLaunchArgument("sequence", default_value="2018-07-09-16-11-56"),
        DeclareLaunchArgument("calibration", default_value="full-image-calib"),
        DeclareLaunchArgument("rate", default_value="0.1"),
        DeclareLaunchArgument("frame_count", default_value="30"),
        mapping,
        TimerAction(period=10.0, actions=[playback]),
    ])
