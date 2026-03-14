from __future__ import annotations

from dataclasses import dataclass
from math import tan, radians

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight test runs
    cv2 = None  # type: ignore[assignment]


@dataclass
class StereoHandDetection:
    hand_detected: bool
    annotated_frame: np.ndarray
    hand_x: float | None
    hand_y: float | None
    pinch_distance: float | None
    depth_m: float | None
    depth_confidence: float


class StereoDepthHandTracker:
    """Stereo hand tracker using MediaPipe landmarks + OpenCV disparity."""

    def __init__(
        self,
        *,
        baseline_mm: float = 60.0,
        hfov_deg: float = 100.0,
        disparity_downscale: float = 0.5,
        swap_halves: bool = False,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "mediapipe is required for stereo hand tracking. Install project dependencies."
            ) from exc

        if cv2 is None:
            raise RuntimeError("opencv-python is required for stereo hand tracking.")

        self._swap_halves = bool(swap_halves)
        self._baseline_m = max(0.001, float(baseline_mm) / 1000.0)
        self._hfov_deg = float(hfov_deg)
        self._downscale = max(0.2, min(1.0, float(disparity_downscale)))

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._drawer = mp.solutions.drawing_utils

        block_size = 7
        num_disp = 16 * 8
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=60,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def close(self) -> None:
        self._hands.close()

    def detect(self, stereo_frame_bgr: np.ndarray) -> StereoHandDetection:
        left, right = self._split_stereo_frame(stereo_frame_bgr)
        left_for_draw = left.copy()

        rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            cv2.putText(
                left_for_draw,
                "No hand",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return StereoHandDetection(
                hand_detected=False,
                annotated_frame=left_for_draw,
                hand_x=None,
                hand_y=None,
                pinch_distance=None,
                depth_m=None,
                depth_confidence=0.0,
            )

        landmarks = result.multi_hand_landmarks[0]
        self._drawer.draw_landmarks(
            left_for_draw,
            landmarks,
            self._mp_hands.HAND_CONNECTIONS,
        )
        points = landmarks.landmark

        wrist = np.array([float(points[0].x), float(points[0].y)], dtype=np.float32)
        index_mcp = np.array([float(points[5].x), float(points[5].y)], dtype=np.float32)
        middle_mcp = np.array([float(points[9].x), float(points[9].y)], dtype=np.float32)
        pinky_mcp = np.array([float(points[17].x), float(points[17].y)], dtype=np.float32)
        palm_center = (wrist + index_mcp + middle_mcp + pinky_mcp) / 4.0
        hand_x = float(max(0.0, min(1.0, palm_center[0])))
        hand_y = float(max(0.0, min(1.0, palm_center[1])))

        thumb_tip = np.array([float(points[4].x), float(points[4].y)], dtype=np.float32)
        index_tip = np.array([float(points[8].x), float(points[8].y)], dtype=np.float32)
        pinch_distance = float(np.linalg.norm(index_tip - thumb_tip))

        depth_m, depth_conf = self._estimate_depth(left, right, hand_x, hand_y)

        depth_label = "n/a" if depth_m is None else f"{depth_m:.2f}m"
        cv2.putText(
            left_for_draw,
            f"x={hand_x:.2f} y={hand_y:.2f} pinch={pinch_distance:.3f} depth={depth_label}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            left_for_draw,
            (int(hand_x * left_for_draw.shape[1]), int(hand_y * left_for_draw.shape[0])),
            6,
            (255, 180, 0),
            -1,
        )

        return StereoHandDetection(
            hand_detected=True,
            annotated_frame=left_for_draw,
            hand_x=hand_x,
            hand_y=hand_y,
            pinch_distance=pinch_distance,
            depth_m=depth_m,
            depth_confidence=depth_conf,
        )

    def _split_stereo_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frame.ndim != 3:
            raise RuntimeError("Stereo frame must be a color image")
        height, width = frame.shape[:2]
        if width < 2 or (width % 2) != 0:
            raise RuntimeError(
                "Stereo camera frame width must be an even side-by-side image. "
                "Try --width 2560 --height 960 for this camera."
            )
        half = width // 2
        left = frame[:, :half]
        right = frame[:, half:]
        if self._swap_halves:
            left, right = right, left
        return left, right

    def _estimate_depth(self, left: np.ndarray, right: np.ndarray, hand_x: float, hand_y: float) -> tuple[float | None, float]:
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        if self._downscale < 0.999:
            left_gray = cv2.resize(
                left_gray,
                None,
                fx=self._downscale,
                fy=self._downscale,
                interpolation=cv2.INTER_AREA,
            )
            right_gray = cv2.resize(
                right_gray,
                None,
                fx=self._downscale,
                fy=self._downscale,
                interpolation=cv2.INTER_AREA,
            )

        disparity = self._matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        h, w = disparity.shape[:2]
        cx = int(max(0, min(w - 1, round(hand_x * (w - 1)))))
        cy = int(max(0, min(h - 1, round(hand_y * (h - 1)))))
        r = max(8, int(min(h, w) * 0.04))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)
        roi = disparity[y0:y1, x0:x1]
        if roi.size == 0:
            return None, 0.0

        valid = roi[(roi > 0.5) & np.isfinite(roi)]
        if valid.size < 20:
            return None, 0.0

        disparity_px = float(np.median(valid))
        if disparity_px <= 0.01:
            return None, 0.0

        focal_px = self._focal_px_for_width(float(w))
        depth_m = (self._baseline_m * focal_px) / disparity_px
        coverage = float(valid.size) / float(roi.size)
        confidence = max(0.0, min(1.0, coverage))
        return float(depth_m), confidence

    def _focal_px_for_width(self, width_px: float) -> float:
        hfov = max(10.0, min(170.0, self._hfov_deg))
        return float(width_px) / (2.0 * tan(radians(hfov) / 2.0))
