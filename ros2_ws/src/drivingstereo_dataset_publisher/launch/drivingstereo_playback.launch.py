from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("dataset", description="DrivingStereo root containing left/, right/, and calib/"),
        DeclareLaunchArgument("sequence", description="Sequence prefix, e.g. 2018-07-09-16-11-56"),
        DeclareLaunchArgument("calibration", default_value="half-image-calib"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("loop", default_value="true"),
        Node(
            package="drivingstereo_dataset_publisher",
            executable="drivingstereo_publisher",
            name="drivingstereo_publisher",
            output="screen",
            arguments=[
                "--dataset", LaunchConfiguration("dataset"),
                "--sequence", LaunchConfiguration("sequence"),
                "--calibration", LaunchConfiguration("calibration"),
                "--rate", LaunchConfiguration("rate"),
                "--loop", LaunchConfiguration("loop"),
            ],
        ),
    ])
