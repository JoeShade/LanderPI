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

The scenario is hands-free once launched; no manual color picking or stage toggling is required.
