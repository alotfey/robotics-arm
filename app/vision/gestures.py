from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight test runs
    cv2 = None  # type: ignore[assignment]

from app.camera.uvc_camera import stereo_left_view
from app.models import GestureEvent, GestureName


@dataclass
class GestureDetection:
    event: GestureEvent | None
    annotated_frame: np.ndarray
    hand_detected: bool


class GestureClassifier:
    """Deterministic hand-gesture classifier on top of MediaPipe landmarks."""

    def __init__(self, stable_frames_required: int = 5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mediapipe is required for gesture detection. Install project dependencies."
            ) from exc

        if cv2 is None:
            raise RuntimeError("opencv-python is required for gesture detection.")

        self._stable_frames_required = stable_frames_required
        self._last_label: GestureName | None = None
        self._stable_count = 0
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._drawer = mp.solutions.drawing_utils

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> GestureDetection:
        frame_for_detection = stereo_left_view(frame_bgr)
        rgb = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        annotated = frame_for_detection.copy()

        if not result.multi_hand_landmarks:
            self._stable_count = 0
            self._last_label = None
            return GestureDetection(event=None, annotated_frame=annotated, hand_detected=False)

        hand_landmarks = result.multi_hand_landmarks[0]
        self._drawer.draw_landmarks(
            annotated,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
        )

        points = hand_landmarks.landmark
        label, confidence = self._classify(points)

        if label is None:
            self._stable_count = 0
            self._last_label = None
            return GestureDetection(event=None, annotated_frame=annotated, hand_detected=True)

        if label == self._last_label:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._last_label = label

        cv2.putText(
            annotated,
            f"{label.value} ({self._stable_count}/{self._stable_frames_required})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if self._stable_count < self._stable_frames_required:
            return GestureDetection(event=None, annotated_frame=annotated, hand_detected=True)

        event = GestureEvent(
            name=label,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            stable_frames=self._stable_count,
        )
        return GestureDetection(event=event, annotated_frame=annotated, hand_detected=True)

    @staticmethod
    def _classify(points: Sequence[object]) -> tuple[GestureName | None, float]:
        def y(idx: int) -> float:
            return float(points[idx].y)

        def x(idx: int) -> float:
            return float(points[idx].x)

        wrist = np.array([x(0), y(0)])
        middle_tip = np.array([x(12), y(12)])
        index_tip = np.array([x(8), y(8)])
        thumb_tip = np.array([x(4), y(4)])

        finger_up = {
            "index": y(8) < y(6),
            "middle": y(12) < y(10),
            "ring": y(16) < y(14),
            "pinky": y(20) < y(18),
        }

        open_count = sum(int(v) for v in finger_up.values())

        # Grip toggle: closed fist (all fingers folded).
        if open_count == 0:
            return GestureName.GRIP_TOGGLE, 0.85

        direction = middle_tip - wrist
        dx = float(direction[0])
        dy = float(direction[1])

        # Vertical commands with open hand.
        if open_count >= 3:
            if dy < -0.15:
                return GestureName.UP, min(1.0, abs(dy) + 0.4)
            if dy > 0.15:
                return GestureName.DOWN, min(1.0, abs(dy) + 0.4)
            if dx < -0.12:
                return GestureName.LEFT, min(1.0, abs(dx) + 0.4)
            if dx > 0.12:
                return GestureName.RIGHT, min(1.0, abs(dx) + 0.4)

        # Pointing gestures as horizontal fallback.
        if finger_up["index"] and not finger_up["middle"]:
            if index_tip[0] < wrist[0] - 0.08:
                return GestureName.LEFT, 0.75
            if index_tip[0] > wrist[0] + 0.08:
                return GestureName.RIGHT, 0.75

        # Thumb orientation fallback for up/down.
        if thumb_tip[1] < wrist[1] - 0.10:
            return GestureName.UP, 0.70
        if thumb_tip[1] > wrist[1] + 0.10:
            return GestureName.DOWN, 0.70

        return None, 0.0
