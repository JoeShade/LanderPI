import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 获取各个功能包的路径
    controller_package_path = get_package_share_directory('controller')
    peripherals_package_path = get_package_share_directory('peripherals')

    # 1. 启动底盘控制 (包含里程计发布、电机控制、IMU发布)
    # 对应源码: src/driver/controller/launch/controller.launch.py
    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(controller_package_path, 'launch/controller.launch.py')),
    )
    
    # 2. 启动深度相机 (RGB + Depth)
    # 对应源码: src/peripherals/launch/depth_camera.launch.py
    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    # 3. 启动激光雷达
    # 对应源码: src/peripherals/launch/lidar.launch.py
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/lidar.launch.py')),
    )

    # 4. (可选) 启动 ROSBridge 和 视频服务器
    # 如果学生需要用 手机APP 或 网页 查看图像/地图，请保留下面这两块。
    # 如果只用 Rviz 和 命令行，可以注释掉以节省资源。
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
        # rosbridge_websocket_launch, # 建议保留，方便调试
        # web_video_server_node,      # 建议保留，方便调试
    ])