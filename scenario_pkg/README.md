# Scenario Package

This ROS 2 Python package orchestrates the full `line_following.py`, `green_nav.py`, and `HRI.py` scripts so they run back-to-back without user input. The runner keeps the original nodes intact by launching the source scripts as subprocesses, handing off from line following, to green beacon navigation, and finally to gesture-based HRI.

## Prerequisites

- ROS 2 (tested with rclpy-based nodes)
- `colcon` build tool
- Dependencies listed in [`package.xml`](package.xml) (e.g., `rclpy`, `cv_bridge`, `sensor_msgs`, `geometry_msgs`, `std_srvs`, `mediapipe`, `servo_controller_msgs`)
- The original `line_following.py`, `green_nav.py`, and `HRI.py` scripts present in the same workspace (this repository already includes them)

> The runner resolves those scripts relative to the package source tree, so keep the repository layout intact when adding it to a workspace.

## Build the package

1) Place this repository (or at least the `scenario_pkg` folder plus its sibling scripts) inside your ROS 2 workspace `src/` directory (example on target device):
```bash
cp -r /home/ubuntu/shared/scenario_pkg ~/ros2_ws/src/
```

2) Build and source:
```bash
cd ~/ros2_ws
rm -rf build/scenario_pkg install/scenario_pkg log/latest
colcon build --packages-select scenario_pkg --symlink-install
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

3) Ensure the `scenario_runner` entrypoint exists in the install space (only needed if it did not get generated):
```bash
mkdir -p ~/ros2_ws/install/scenario_pkg/lib/scenario_pkg
cat <<'EOF' > ~/ros2_ws/install/scenario_pkg/lib/scenario_pkg/scenario_runner
#!/usr/bin/env bash
exec /usr/bin/env python3 -m scenario_pkg.scenario_runner "$@"
EOF
chmod +x ~/ros2_ws/install/scenario_pkg/lib/scenario_pkg/scenario_runner
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

## Launch the scenario

Run the launch file to start the orchestrated mission:
```bash
ros2 launch scenario_pkg scenario.launch.py
```
View camera topics:
```bash
ros2 run rqt_image_view rqt_image_view
```

This brings up `scenario_runner`, which automatically:
- Starts `line_following.py` with the built-in black line profile
- Switches to `green_nav.py` when the line disappears for multiple frames
- Hands off to `HRI.py` once the green beacon fills the camera view

## Useful commands and knobs

- Run the node directly (e.g., in a dev shell):
  ```bash
  ros2 run scenario_pkg scenario_runner
  ```
- Adjust the camera tuning used for transition detection (defaults to `aurora`):
  ```bash
  DEPTH_CAMERA_TYPE=usb_cam ros2 launch scenario_pkg scenario.launch.py
  ```
- Quiet or boost voice prompts from the child nodes (default `VOICE_VOLUME=90`):
  ```bash
  VOICE_VOLUME=70 ros2 run scenario_pkg scenario_runner
  ```
- Confirm the node is alive and reading images:
  ```bash
  ros2 topic echo /camera/image_raw --no-arr
  ```
- Rebuild after edits without rebuilding the whole workspace:
  ```bash
  colcon build --packages-select scenario_pkg --symlink-install
  ```
- Enable verbose debugging and pipe child stdout/stderr into the runner logs:
  ```bash
  ros2 run scenario_pkg scenario_runner --ros-args -p debug_mode:=true
  ```
- Save transition snapshots for offline triage (images land in `/tmp/scenario_debug` by default):
  ```bash
  ros2 run scenario_pkg scenario_runner --ros-args -p save_debug_images:=true
  ```
- Query the runner status while it is live:
  ```bash
  ros2 service call /get_status std_srvs/srv/Trigger {}
  ```
- Override a stall manually (force stage): jump to a specific stage or the next one if things get stuck.
  ```bash
  ros2 service call /set_stage interfaces/srv/SetString "{data: 'GREEN'}"  # or LINE/HRI/NEXT
  ```
- Stall watchdog: the runner monitors `/odom` to detect when motion commands are being published but the robot is not actually moving, and will advance LINE→GREEN or GREEN→HRI after the timeout (default 3s). It now waits for live `/odom` data before acting. Tweak via parameters:
  ```bash
  ros2 run scenario_pkg scenario_runner --ros-args -p motion_timeout:=3.0 -p motion_check_period:=0.5
  ```
  Make sure `/odom` is available; otherwise reduce the timeout or disable if testing in a headless environment.
- Thread safety (green_nav): service callbacks and image/lidar callbacks are guarded by a re-entrant lock to avoid races on `is_running`, `stop`, and related flags.
- Servo logging: servo commands in HRI/green_nav/line_following still execute immediately but their console logs are throttled to every 0.5s to keep output readable.
- Transition safety: stage launches are serialized to avoid overlapping transitions when manual overrides and watchdogs fire together.

The scenario is hands-free once launched; no manual color picking or stage toggling is required.

## Audio prompts

Copy audio prompts into the scenario manager feedback_voice directory:
```bash
cp /home/ubuntu/shared/warning.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
cp /home/ubuntu/shared/start_track_green.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
cp /home/ubuntu/shared/find_target.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
```

Copy HRI gesture audio prompts into the same feedback_voice directory:
```bash
cp /home/ubuntu/shared/Danger.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
cp /home/ubuntu/shared/Survivor.wav ~/ros2_ws/src/scenario_pkg/scenario_pkg/feedback_voice/
# add any other .wav cues you want HRI.py to play into the same folder
```
