from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='scenario_pkg',
            executable='scenario_runner',
            name='scenario_runner',
            output='screen',
            parameters=[{'use_sim_time': False}],
        ),
    ])
