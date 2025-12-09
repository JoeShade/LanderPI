# Scenario Package

This ROS 2 Python package orchestrates the full `line_following.py`, `green_nav.py`, and `HRI.py` scripts so they run back-to-back without user input. The runner keeps the original nodes intact by launching the source scripts as subprocesses, handing off from line following, to green beacon navigation, and finally to gesture-based HRI.

## Prerequisites

- ROS 2 (tested with rclpy-based nodes)
- `colcon` build tool
- Dependencies listed in [`package.xml`](package.xml) (e.g., `rclpy`, `cv_bridge`, `sensor_msgs`, `geometry_msgs`, `std_srvs`, `mediapipe`, `servo_controller_msgs`)
- The original `line_following.py`, `green_nav.py`, and `HRI.py` scripts present in the same workspace (this repository already includes them)

> The runner resolves those scripts relative to the package source tree, so keep the repository layout intact when adding it to a workspace.

## Build the package

1. Place this repository (or at least the `scenario_pkg` folder plus its sibling scripts) inside your ROS 2 workspace `src/` directory.
2. From the workspace root, build with `colcon`:
   ```bash
   colcon build --packages-select scenario_pkg
   ```
3. Source the overlay (replace `~/ws` with your workspace path if different):
   ```bash
   source install/setup.bash
   ```

## Launch the scenario

Run the launch file to start the orchestrated mission:
```bash
ros2 launch scenario_pkg scenario.launch.py
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
- Stall watchdog (new): the runner now monitors `/odom` to detect when motion commands are being published but the robot is not actually moving, and will advance LINE→GREEN or GREEN→HRI after the timeout (default 3s). Tweak via parameters:
  ```bash
  ros2 run scenario_pkg scenario_runner --ros-args -p motion_timeout:=3.0 -p motion_check_period:=0.5
  ```
  Make sure `/odom` is available; otherwise reduce the timeout or disable if testing in a headless environment.
- Thread safety (green_nav): service callbacks and image/lidar callbacks are now guarded by a re-entrant lock to avoid races on `is_running`, `stop`, and related flags. No behavior change expected, just safer concurrent handling.

The scenario is hands-free once launched; no manual color picking or stage toggling is required.
