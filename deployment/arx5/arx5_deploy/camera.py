from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _import_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is unavailable; run this command in an environment with RealSense dependencies installed"
        ) from exc
    return rs


def list_realsense_cameras() -> None:
    rs = _import_realsense()
    devices = list(rs.context().query_devices())
    if not devices:
        print("No RealSense cameras detected.")
    for device in devices:
        print(
            f"{device.get_info(rs.camera_info.serial_number)}\t"
            f"{device.get_info(rs.camera_info.name)}"
        )


class RealSenseColorCamera:
    def __init__(
        self,
        serial: str,
        label: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.serial = serial
        self.label = label
        self.width = width
        self.height = height
        self.rs = _import_realsense()
        self.pipeline = self.rs.pipeline()
        config = self.rs.config()
        config.enable_device(serial)
        config.enable_stream(
            self.rs.stream.color,
            width,
            height,
            self.rs.format.rgb8,
            fps,
        )
        logger.info("Starting %s camera serial=%s", label, serial)
        self.pipeline.start(config)
        for _ in range(10):
            self.pipeline.wait_for_frames(timeout_ms=5000)

    def read_rgb(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        frame = frames.get_color_frame()
        if not frame:
            raise RuntimeError(f"{self.label} camera returned no color frame")
        image = np.asanyarray(frame.get_data())
        expected = (self.height, self.width, 3)
        if image.shape != expected:
            raise RuntimeError(f"{self.label} camera shape {image.shape}, expected {expected}")
        return image.copy()

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception as exc:
            logger.warning("Failed to stop %s camera: %s", self.label, exc)

