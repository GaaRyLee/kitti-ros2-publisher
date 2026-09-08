from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sequence'),
        DeclareLaunchArgument('calibration'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('loop', default_value='false'),
        Node(
            package='kitti_dataset_publisher',
            executable='kitti_publisher',
            name='kitti_publisher',
            output='screen',
            arguments=[
                '--sequence', LaunchConfiguration('sequence'),
                '--calibration', LaunchConfiguration('calibration'),
                '--rate', LaunchConfiguration('rate'),
                '--loop', LaunchConfiguration('loop'),
            ],
        ),
    ])
