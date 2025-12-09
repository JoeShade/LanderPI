import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Mirror the minimal bringup: controller (odom/imu/motors), depth camera, and lidar.
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('controller'), 'launch', 'controller.launch.py')
        )
    )

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('peripherals'), 'launch', 'depth_camera.launch.py')
        )
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('peripherals'), 'launch', 'lidar.launch.py')
        )
    )

    # Scenario runner orchestrates line_following -> green_nav -> HRI.
    runner = Node(
        package='scenario_pkg',
        executable='scenario_runner',
        name='scenario_runner',
        output='screen',
        parameters=[{'use_sim_time': False}],
    )

    return LaunchDescription([
        controller_launch,
        depth_camera_launch,
        lidar_launch,
        runner,
    ])
