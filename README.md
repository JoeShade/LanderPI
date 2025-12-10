# Useful commands

## Line following program
- Launch with debug enabled:
```bash
ros2 launch app line_following_node.launch.py debug:=true
```
- Enter the line-following routine:
```bash
ros2 service call /line_following/enter std_srvs/srv/Trigger {}
```
- Allow the node to drive:
```bash
ros2 service call /line_following/set_running std_srvs/srv/SetBool "{data: True}"
```

## Green nav program
- Launch required topics:
```bash
ros2 launch green_nav_pkg green_nav_with_sensors.launch.py
```
- Launch `green_nav` with debug features:
```bash
ros2 run green_nav_pkg green_nav --ros-args -p debug:=true
```
- Start the program:
```bash
ros2 service call /green_nav/enter std_srvs/srv/Trigger {}
```
- Rebuild the package:
```bash
cd ~/ros2_ws
colcon build --packages-select green_nav_pkg --symlink-install
```
- Replace the program file with a new build:
```bash
cp /home/ubuntu/shared/green_nav.py ~/ros2_ws/src/green_nav_pkg/green_nav_pkg/
```

## HRI program
- Launch the gesture-based controller:
```bash
ros2 launch HRI_pkg HRI_control.launch.py
```

## Scenario manager (scenario_pkg)
- Build and source:
```bash
cp -r /home/ubuntu/shared/scenario_pkg ~/ros2_ws/src/
cd ~/ros2_ws
rm -rf build/scenario_pkg install/scenario_pkg log/latest
colcon build --packages-select scenario_pkg --symlink-install
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```
- Launch the orchestrated mission (line following -> green nav -> HRI):
```bash
ros2 launch scenario_pkg scenario.launch.py
```
- Run the node directly (e.g., in a dev shell):
```bash
ros2 run scenario_pkg scenario_runner
```
- Copy audio prompts into the scenario manager feedback_voice directory:
```bash
cp /home/ubuntu/shared/warning.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
cp /home/ubuntu/shared/start_track_green.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
cp /home/ubuntu/shared/find_target.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
cp /home/ubuntu/shared/Danger.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
cp /home/ubuntu/shared/Survivor.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
```
- Useful run-time knobs:
```bash
# Adjust camera tuning used for transitions (default DEPTH_CAMERA_TYPE=aurora)
DEPTH_CAMERA_TYPE=usb_cam ros2 launch scenario_pkg scenario.launch.py

# Quiet or boost voice prompts (default VOICE_VOLUME=90)
VOICE_VOLUME=70 ros2 run scenario_pkg scenario_runner

# Confirm node is alive and reading images
ros2 topic echo /camera/image_raw --no-arr

# Enable verbose debugging and pipe child stdout/stderr into runner logs
ros2 run scenario_pkg scenario_runner --ros-args -p debug_mode:=true

# Save transition snapshots to /tmp/scenario_debug
ros2 run scenario_pkg scenario_runner --ros-args -p save_debug_images:=true

# Query live status
ros2 service call /get_status std_srvs/srv/Trigger {}

# Override stage (LINE/GREEN/HRI/NEXT)
ros2 service call /set_stage interfaces/srv/SetString "{data: 'GREEN'}"
```
