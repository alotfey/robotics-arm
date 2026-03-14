from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraFrame:
    frame_bgr: np.ndarray
    timestamp_ms: int


def is_stereo_side_by_side_frame(frame_bgr: np.ndarray) -> bool:
    """Return True when a frame likely contains left/right side-by-side views."""
    if frame_bgr.ndim != 3:
        return False
    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return False
    # Typical synchronized stereo UVC streams are much wider than tall.
    return (width % 2) == 0 and (float(width) / float(height)) >= 1.8


def stereo_left_view(frame_bgr: np.ndarray) -> np.ndarray:
    """Extract the left camera view when a side-by-side stereo frame is detected."""
    if not is_stereo_side_by_side_frame(frame_bgr):
        return frame_bgr
    half = frame_bgr.shape[1] // 2
    return frame_bgr[:, :half].copy()


class UvcCamera:
    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._capture = cap

    def read(self) -> CameraFrame:
        if self._capture is None:
            raise RuntimeError("Camera not started")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read camera frame")
        timestamp_ms = int(time.monotonic() * 1000)
        return CameraFrame(frame_bgr=frame, timestamp_ms=timestamp_ms)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
