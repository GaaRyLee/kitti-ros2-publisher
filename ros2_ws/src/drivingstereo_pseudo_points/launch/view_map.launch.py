"""Open RViz2 preconfigured for the DrivingStereo colored map."""
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = get_package_share_directory("drivingstereo_pseudo_points") + "/rviz/drivingstereo_map.rviz"
    return LaunchDescription([
        Node(
            package="rviz2",
            executable="rviz2",
            name="drivingstereo_rviz",
            output="screen",
            arguments=["-d", config],
        ),
    ])
