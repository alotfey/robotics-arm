from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight test runs
    cv2 = None  # type: ignore[assignment]

from app.models import GestureEvent, GestureName


@dataclass
class GestureDetection:
    event: GestureEvent | None
    annotated_frame: np.ndarray
    hand_detected: bool


class GestureClassifier:
    """Deterministic hand-gesture classifier on top of MediaPipe landmarks."""

    def __init__(self, stable_frames_required: int = 5) -> None:
        """Initialize MediaPipe-backed gesture classification state.

        Args:
            stable_frames_required: Consecutive frames required before emitting an event.

        Raises:
            RuntimeError: If required runtime dependencies are unavailable.
        """
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mediapipe is required for gesture detection. Install project dependencies."
            ) from exc

        if cv2 is None:
            raise RuntimeError("opencv-python is required for gesture detection.")

        if not hasattr(mp, "solutions"):
            mp_version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "Installed mediapipe does not expose the legacy 'solutions' API "
                f"(found version {mp_version}). Install mediapipe==0.10.14."
            )

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
        """Release MediaPipe resources held by the classifier."""
        self._hands.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> GestureDetection:
        """Run hand landmark detection and classify the current gesture.

        Args:
            frame_bgr: Input frame in BGR color space.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            GestureDetection: Annotated frame and optional stable gesture event.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        annotated = frame_bgr.copy()

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
        """Classify raw landmark coordinates into a gesture label.

        Args:
            points: Sequence of MediaPipe normalized landmarks.

        Returns:
            tuple[GestureName | None, float]: Gesture label and confidence.
        """
        def y(idx: int) -> float:
            """Return normalized y coordinate for a landmark index."""
            return float(points[idx].y)

        def x(idx: int) -> float:
            """Return normalized x coordinate for a landmark index."""
            return float(points[idx].x)

        def point(idx: int) -> np.ndarray:
            """Return a 2D point array for a landmark index."""
            return np.array([x(idx), y(idx)], dtype=np.float32)

        def mean_point(indices: tuple[int, ...]) -> np.ndarray:
            """Return the centroid for a set of landmark indices."""
            return np.mean([point(idx) for idx in indices], axis=0)

        def dist(a: int, b: int) -> float:
            """Return Euclidean distance between two landmark indices."""
            return float(np.linalg.norm(point(a) - point(b)))

        def is_finger_extended(tip: int, pip: int, mcp: int) -> bool:
            """Estimate whether a finger is extended from three joints.

            Args:
                tip: Landmark index for the fingertip.
                pip: Landmark index for the proximal interphalangeal joint.
                mcp: Landmark index for the metacarpophalangeal joint.

            Returns:
                bool: ``True`` when geometry indicates finger extension.
            """
            tip_to_mcp = dist(tip, mcp)
            pip_to_mcp = max(dist(pip, mcp), 1e-4)
            # Distance-based extension is rotation invariant; the y-check helps with upright palms.
            return (tip_to_mcp >= pip_to_mcp * 1.25) or (y(tip) < y(pip) and tip_to_mcp >= pip_to_mcp * 1.10)

        def classify_axis(
            dx: float,
            dy: float,
            *,
            min_magnitude: float,
            dominance_ratio: float,
            base_confidence: float,
        ) -> tuple[GestureName | None, float]:
            """Classify a direction vector into one cardinal gesture axis.

            Args:
                dx: Horizontal vector component.
                dy: Vertical vector component.
                min_magnitude: Minimum vector strength required to classify.
                dominance_ratio: Axis dominance threshold to avoid diagonal ambiguity.
                base_confidence: Base confidence to use once vector is classifiable.

            Returns:
                tuple[GestureName | None, float]: Axis label and confidence.
            """
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            strength = max(abs_dx, abs_dy)
            if strength < min_magnitude:
                return None, 0.0

            if abs_dx >= abs_dy * dominance_ratio:
                label = GestureName.RIGHT if dx > 0 else GestureName.LEFT
                confidence = min(1.0, base_confidence + min(0.40, (strength - min_magnitude) * 0.30))
                return label, confidence

            if abs_dy >= abs_dx * dominance_ratio:
                label = GestureName.DOWN if dy > 0 else GestureName.UP
                confidence = min(1.0, base_confidence + min(0.40, (strength - min_magnitude) * 0.30))
                return label, confidence

            # Diagonal / ambiguous hand direction: ignore instead of forcing one axis.
            return None, 0.0

        wrist = point(0)
        knuckle_center = mean_point((5, 9, 13, 17))
        tip_center = mean_point((8, 12, 16, 20))
        index_tip = point(8)

        palm_scale = float(np.mean([dist(0, 5), dist(0, 9), dist(0, 13), dist(0, 17)]))
        if palm_scale < 0.06:
            # Fallback for synthetic/edge cases where knuckle points collapse.
            palm_scale = max(0.10, dist(0, 12), dist(0, 8))

        finger_extended = {
            "index": is_finger_extended(8, 6, 5),
            "middle": is_finger_extended(12, 10, 9),
            "ring": is_finger_extended(16, 14, 13),
            "pinky": is_finger_extended(20, 18, 17),
        }

        open_count = sum(int(v) for v in finger_extended.values())
        curled_count = 4 - open_count

        finger_tip_dist_norm = [dist(idx, 0) / palm_scale for idx in (8, 12, 16, 20)]
        avg_tip_dist_norm = float(np.mean(finger_tip_dist_norm))
        index_tip_dist_norm = finger_tip_dist_norm[0]
        tip_spread_norm = float(
            np.mean(
                [
                    dist(8, 12),
                    dist(12, 16),
                    dist(16, 20),
                    dist(8, 16),
                    dist(12, 20),
                ]
            )
            / palm_scale
        )

        # Grip toggle: mostly curled fingers and compact fingertip cluster near wrist.
        is_index_pointing_candidate = finger_extended["index"] and index_tip_dist_norm > 1.35
        if (
            curled_count >= 3
            and avg_tip_dist_norm < 1.55
            and tip_spread_norm < 0.95
            and not is_index_pointing_candidate
        ):
            confidence = min(
                1.0,
                0.70 + max(0.0, 1.55 - avg_tip_dist_norm) * 0.16 + max(0.0, 0.95 - tip_spread_norm) * 0.12,
            )
            return GestureName.GRIP_TOGGLE, confidence

        # Open hand direction from palm to fingertip centroid.
        if open_count >= 3:
            direction = (tip_center - knuckle_center) / palm_scale
            label, confidence = classify_axis(
                float(direction[0]),
                float(direction[1]),
                min_magnitude=0.65,
                dominance_ratio=1.15,
                base_confidence=0.55,
            )
            if label is not None:
                return label, confidence

        # Index-pointing fallback for deliberate directional commands.
        if (
            finger_extended["index"]
            and not finger_extended["middle"]
            and not finger_extended["ring"]
            and not finger_extended["pinky"]
        ):
            pointing = (index_tip - wrist) / palm_scale
            dx = float(pointing[0])
            dy = float(pointing[1])

            if abs(dx) >= 0.75 and abs(dx) >= abs(dy) * 0.75:
                confidence = min(1.0, 0.65 + min(0.35, (abs(dx) - 0.75) * 0.50))
                return (GestureName.RIGHT if dx > 0 else GestureName.LEFT), confidence

            label, confidence = classify_axis(
                dx,
                dy,
                min_magnitude=0.75,
                dominance_ratio=1.20,
                base_confidence=0.60,
            )
            if label is not None:
                return label, confidence

        return None, 0.0
