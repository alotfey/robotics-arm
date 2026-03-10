from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraFrame:
    frame_bgr: np.ndarray
    timestamp_ms: int


class UvcCamera:
    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        """Initialize a USB camera wrapper.

        Args:
            index: OpenCV camera index.
            width: Requested frame width in pixels.
            height: Requested frame height in pixels.
            fps: Requested frames per second.
        """
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        """Open the camera and apply capture settings.

        Raises:
            RuntimeError: If the camera index cannot be opened.
        """
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._capture = cap

    def read(self) -> CameraFrame:
        """Read one frame from the active camera stream.

        Returns:
            CameraFrame: Captured BGR image with monotonic timestamp.

        Raises:
            RuntimeError: If the camera was not started or frame capture fails.
        """
        if self._capture is None:
            raise RuntimeError("Camera not started")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read camera frame")
        timestamp_ms = int(time.monotonic() * 1000)
        return CameraFrame(frame_bgr=frame, timestamp_ms=timestamp_ms)

    def stop(self) -> None:
        """Release the camera if it is currently open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
