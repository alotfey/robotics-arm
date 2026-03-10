from __future__ import annotations

import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight test runs
    cv2 = None  # type: ignore[assignment]

from app.camera.uvc_camera import CameraFrame
from app.models import GestureEvent, GestureName
from app.vision.gestures import GestureDetection


class DemoCamera:
    """Synthetic camera stream used for demo mode without hardware."""

    def __init__(self, width: int, height: int, fps: int, duration_sec: float) -> None:
        """Initialize the synthetic camera generator.

        Args:
            width: Generated frame width in pixels.
            height: Generated frame height in pixels.
            fps: Target frames per second.
            duration_sec: Total demo duration before auto-stop.
        """
        self._width = width
        self._height = height
        self._fps = max(1, fps)
        self._duration_sec = duration_sec
        self._started_at: float | None = None
        self._next_frame_at: float = 0.0
        self._frame_idx = 0

    def start(self) -> None:
        """Start demo timing and frame counters."""
        started = time.monotonic()
        self._started_at = started
        self._next_frame_at = started
        self._frame_idx = 0

    def read(self) -> CameraFrame:
        """Generate and return the next synthetic frame.

        Returns:
            CameraFrame: Generated frame and monotonic timestamp.

        Raises:
            RuntimeError: If the demo camera is read before start.
            KeyboardInterrupt: When configured demo duration has elapsed.
        """
        if self._started_at is None:
            raise RuntimeError("Demo camera not started")

        now = time.monotonic()
        if now >= (self._started_at + self._duration_sec):
            raise KeyboardInterrupt("Demo duration elapsed")

        if now < self._next_frame_at:
            time.sleep(self._next_frame_at - now)

        timestamp_ms = int(time.monotonic() * 1000)
        frame = self._make_frame()

        self._frame_idx += 1
        frame_interval = 1.0 / float(self._fps)
        self._next_frame_at = max(self._next_frame_at + frame_interval, time.monotonic())
        return CameraFrame(frame_bgr=frame, timestamp_ms=timestamp_ms)

    def stop(self) -> None:
        """Stop the demo stream."""
        self._started_at = None

    def _make_frame(self) -> np.ndarray:
        """Render one synthetic demo frame."""
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        phase = self._frame_idx % 255
        frame[:, :, 0] = (phase * 3) % 255
        frame[:, :, 1] = (phase * 2) % 255

        bar_x = int((self._frame_idx * max(1, self._width // 80)) % self._width)
        frame[:, max(0, bar_x - 3) : min(self._width, bar_x + 3), :] = (20, 220, 220)

        if cv2 is not None:
            cv2.putText(
                frame,
                "DEMO CAMERA",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame


class DemoGestureClassifier:
    """Deterministic gesture source for demo mode without MediaPipe."""

    def __init__(self, stable_frames_required: int = 5) -> None:
        """Initialize deterministic gesture sequence state.

        Args:
            stable_frames_required: Stable-frame count applied to generated events.
        """
        self._stable_frames_required = stable_frames_required
        self._tick = 0

    def close(self) -> None:
        """No-op close method for interface compatibility."""
        return

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> GestureDetection:
        """Return the next synthetic gesture detection result.

        Args:
            frame_bgr: Input frame used as annotation background.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            GestureDetection: Deterministic synthetic gesture output.
        """
        annotated = frame_bgr.copy()
        event, hand_detected, label = self._next_event(timestamp_ms)

        if cv2 is not None:
            cv2.putText(
                annotated,
                f"DEMO GESTURE: {label}",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        self._tick += 1
        return GestureDetection(event=event, annotated_frame=annotated, hand_detected=hand_detected)

    def _next_event(self, timestamp_ms: int) -> tuple[GestureEvent | None, bool, str]:
        """Compute the next event in the repeating demo gesture sequence.

        Args:
            timestamp_ms: Timestamp assigned to emitted events.

        Returns:
            tuple[GestureEvent | None, bool, str]: Event, hand-presence flag, and label text.
        """
        slot = self._tick % 12

        if slot in (0, 1):
            return self._event(GestureName.RIGHT, 0.9, timestamp_ms), True, GestureName.RIGHT.value
        if slot in (2, 3):
            return self._event(GestureName.UP, 0.9, timestamp_ms), True, GestureName.UP.value
        if slot == 4:
            return (
                self._event(GestureName.GRIP_TOGGLE, 0.85, timestamp_ms),
                True,
                GestureName.GRIP_TOGGLE.value,
            )
        if slot == 6:
            return None, False, "NO_HAND"
        if slot in (7, 8):
            return self._event(GestureName.LEFT, 0.9, timestamp_ms), True, GestureName.LEFT.value
        if slot in (9, 10):
            return self._event(GestureName.DOWN, 0.9, timestamp_ms), True, GestureName.DOWN.value
        return None, True, "IDLE"

    def _event(self, name: GestureName, confidence: float, timestamp_ms: int) -> GestureEvent:
        """Create a synthetic gesture event payload.

        Args:
            name: Gesture identifier.
            confidence: Confidence value to embed in the event.
            timestamp_ms: Event timestamp in milliseconds.

        Returns:
            GestureEvent: Constructed gesture event instance.
        """
        return GestureEvent(
            name=name,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            stable_frames=self._stable_frames_required,
        )
