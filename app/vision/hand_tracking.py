from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight test runs
    cv2 = None  # type: ignore[assignment]

from app.camera.uvc_camera import stereo_left_view

@dataclass
class HandTrackingDetection:
    hand_detected: bool
    annotated_frame: np.ndarray
    hand_x: float | None
    hand_y: float | None
    pinch_distance: float | None
    confidence: float


class HandTracker:
    """Continuous hand pose tracker based on MediaPipe hand landmarks."""

    def __init__(self, min_detection_confidence: float = 0.55, min_tracking_confidence: float = 0.55) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mediapipe is required for hand tracking. Install project dependencies."
            ) from exc

        if cv2 is None:
            raise RuntimeError("opencv-python is required for hand tracking.")

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._drawer = mp.solutions.drawing_utils

    def close(self) -> None:
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray) -> HandTrackingDetection:
        frame_for_detection = stereo_left_view(frame_bgr)
        rgb = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        annotated = frame_for_detection.copy()

        if not result.multi_hand_landmarks:
            return HandTrackingDetection(
                hand_detected=False,
                annotated_frame=annotated,
                hand_x=None,
                hand_y=None,
                pinch_distance=None,
                confidence=0.0,
            )

        hand_landmarks = result.multi_hand_landmarks[0]
        self._drawer.draw_landmarks(
            annotated,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
        )

        points = hand_landmarks.landmark

        wrist = np.array([float(points[0].x), float(points[0].y)], dtype=np.float32)
        index_mcp = np.array([float(points[5].x), float(points[5].y)], dtype=np.float32)
        middle_mcp = np.array([float(points[9].x), float(points[9].y)], dtype=np.float32)
        pinky_mcp = np.array([float(points[17].x), float(points[17].y)], dtype=np.float32)
        palm_center = (wrist + index_mcp + middle_mcp + pinky_mcp) / 4.0

        thumb_tip = np.array([float(points[4].x), float(points[4].y)], dtype=np.float32)
        index_tip = np.array([float(points[8].x), float(points[8].y)], dtype=np.float32)
        pinch_distance = float(np.linalg.norm(index_tip - thumb_tip))

        hand_x = float(max(0.0, min(1.0, palm_center[0])))
        hand_y = float(max(0.0, min(1.0, palm_center[1])))
        confidence = float(max(0.0, min(1.0, 1.0 - min(1.0, pinch_distance))))

        cv2.putText(
            annotated,
            f"x={hand_x:.2f} y={hand_y:.2f} pinch={pinch_distance:.3f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            annotated,
            (
                int(hand_x * annotated.shape[1]),
                int(hand_y * annotated.shape[0]),
            ),
            6,
            (255, 200, 0),
            -1,
        )

        return HandTrackingDetection(
            hand_detected=True,
            annotated_frame=annotated,
            hand_x=hand_x,
            hand_y=hand_y,
            pinch_distance=pinch_distance,
            confidence=confidence,
        )
