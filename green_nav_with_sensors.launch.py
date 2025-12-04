import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Toggle between installed packages and local source tree depending on need_compile.
    compiled = os.environ.get('need_compile', 'True')

    if compiled == 'True':
        controller_package_path = get_package_share_directory('controller')
        peripherals_package_path = get_package_share_directory('peripherals')
        xf_mic_asr_offline_package_path = get_package_share_directory('xf_mic_asr_offline')
    else:
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
        xf_mic_asr_offline_package_path = '/home/ubuntu/ros2_ws/src/xf_mic_asr_offline'

    # Base chassis and IMU bring-up.
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )

    # RGB-D camera pipeline.
    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    # Lidar pipeline.
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/lidar.launch.py')),
    )

    # Microphone / ASR initialization.
    mic_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(xf_mic_asr_offline_package_path, 'launch/mic_init.launch.py')),
    )

    # Green beacon navigation node.
    green_nav_node = Node(
        package='green_nav_pkg',
        executable='green_nav',
        output='screen',
        parameters=[{'debug': False}],
        remappings=[
            ('~/image_result', 'green_nav/image_result'),
        ],
    )

    # Set initial pose for the controller stack.
    init_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(controller_package_path, 'launch/init_pose.launch.py')),
        launch_arguments={
            'namespace': '',
            'use_namespace': 'false',
            'action_name': 'init',
        }.items(),
    )

    # Optional visualization/bridge components you can enable if needed.
    rosbridge_websocket_launch = ExecuteProcess(
            cmd=['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'],
            output='screen'
        )

    web_video_server_node = Node(
        package='web_video_server',
        executable='web_video_server',
        output='screen',
    )

    return LaunchDescription([
        controller_launch,
        depth_camera_launch,
        lidar_launch,
        mic_launch,
        green_nav_node,
        init_pose_launch,
        # rosbridge_websocket_launch,
        # web_video_server_node,
    ])
