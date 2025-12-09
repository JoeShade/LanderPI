"""Shared helpers for scenario_pkg nodes."""

import os


def resolve_camera_topic(camera_type: str, default: str = "/camera/image_raw") -> str:
    """Pick a camera topic that matches the active sensor; env IMAGE_TOPIC overrides."""
    env_topic = os.environ.get("IMAGE_TOPIC") or os.environ.get("CAMERA_TOPIC")
    if env_topic:
        return env_topic
    if camera_type == "usb_cam":
        return "/camera/image"
    if camera_type in ("ascamera", "aurora"):
        return "/ascamera/camera_publisher/rgb0/image"
    return default
