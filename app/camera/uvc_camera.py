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


class DualUvcStereoCamera:
    """Capture stereo frames from two camera indices and stitch side-by-side."""

    def __init__(self, left_index: int, right_index: int, width: int, height: int, fps: int) -> None:
        if width < 2 or (width % 2) != 0:
            raise ValueError("Stereo width must be an even number when using dual camera indices")
        self._left_index = left_index
        self._right_index = right_index
        self._stitched_width = width
        self._height = height
        self._fps = fps
        self._single_width = width // 2
        self._left_capture: cv2.VideoCapture | None = None
        self._right_capture: cv2.VideoCapture | None = None

    @staticmethod
    def _set_capture_format(cap: cv2.VideoCapture, *, width: int, height: int, fps: int) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    def start(self) -> None:
        left = cv2.VideoCapture(self._left_index)
        if not left.isOpened():
            raise RuntimeError(f"Could not open left camera index {self._left_index}")
        self._set_capture_format(left, width=self._single_width, height=self._height, fps=self._fps)

        right = cv2.VideoCapture(self._right_index)
        if not right.isOpened():
            left.release()
            raise RuntimeError(f"Could not open right camera index {self._right_index}")
        self._set_capture_format(right, width=self._single_width, height=self._height, fps=self._fps)

        self._left_capture = left
        self._right_capture = right

    def read(self) -> CameraFrame:
        if self._left_capture is None or self._right_capture is None:
            raise RuntimeError("Stereo camera not started")

        left_ok, left_frame = self._left_capture.read()
        right_ok, right_frame = self._right_capture.read()
        if not left_ok or left_frame is None:
            raise RuntimeError(f"Failed to read left camera frame (index {self._left_index})")
        if not right_ok or right_frame is None:
            raise RuntimeError(f"Failed to read right camera frame (index {self._right_index})")

        if left_frame.shape[:2] != right_frame.shape[:2]:
            right_frame = cv2.resize(
                right_frame,
                (left_frame.shape[1], left_frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        frame = np.concatenate((left_frame, right_frame), axis=1)
        timestamp_ms = int(time.monotonic() * 1000)
        return CameraFrame(frame_bgr=frame, timestamp_ms=timestamp_ms)

    def stop(self) -> None:
        if self._left_capture is not None:
            self._left_capture.release()
            self._left_capture = None
        if self._right_capture is not None:
            self._right_capture.release()
            self._right_capture = None
