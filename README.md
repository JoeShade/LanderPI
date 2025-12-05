Useful commands

// Green Nav program commands

// Launch required topics
ros2 launch green_nav_pkg green_nav_with_sensors.launch.py 

// Launch green_nav node with debug features
ros2 run green_nav_pkg green_nav --ros-args -p debug:=true 

// Launch program
ros2 service call /green_nav/enter std_srvs/srv/Trigger {}


// Rebuild package
Navigate to ros2_ws
run 
colcon build --packages-select green_nav_pkg --symlink-install

// Replace program file with new build 
cp /home/ubuntu/shared/green_nav.py ~/ros2_ws/src/green_nav_pkg/green_nav_pkg/
