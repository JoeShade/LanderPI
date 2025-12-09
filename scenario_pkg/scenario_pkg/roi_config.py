"""Shared ROI definitions used by line_following and scenario_runner for consistency."""

# Percentages: (y_min, y_max, x_min, x_max, weight)
# These entries describe what portion of the camera image each region of
# interest covers. The weight lets downstream code favor the band closest to
# the robot when averaging the detected line position.
ROI_TABLE = {
    "ascamera": ((0.9, 0.95, 0, 1, 0.7), (0.8, 0.85, 0, 1, 0.2), (0.7, 0.75, 0, 1, 0.1)),
    "aurora": ((0.81, 0.83, 0, 1, 0.7), (0.69, 0.71, 0, 1, 0.2), (0.57, 0.59, 0, 1, 0.1)),
    "usb_cam": ((0.79, 0.81, 0, 1, 0.7), (0.67, 0.69, 0, 1, 0.2), (0.55, 0.57, 0, 1, 0.1)),
}


def get_rois(camera_type: str):
    """Return ROI tuple for the given camera type, defaulting to aurora tuning."""
    return ROI_TABLE.get(camera_type, ROI_TABLE["aurora"])
