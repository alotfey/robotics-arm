from __future__ import annotations

import time

try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight test runs
    cv2 = None  # type: ignore[assignment]
from rich.console import Console

from app.camera.uvc_camera import UvcCamera
from app.models import AxisName, RuntimeConfig
from app.robot.driver import RobotArmDriver
from app.vision.stereo_depth_hand_tracking import StereoDepthHandTracker, StereoHandDetection


class StereoHandTrackingLoop:
    """3D hand tracking loop driven by stereo disparity depth."""

    def __init__(
        self,
        camera: UvcCamera,
        tracker: StereoDepthHandTracker,
        driver: RobotArmDriver,
        config: RuntimeConfig,
        *,
        x_deadzone: float = 0.07,
        y_deadzone: float = 0.07,
        depth_deadzone_m: float = 0.03,
        smoothing_alpha: float = 0.35,
        depth_scale_deg_per_m: float = 120.0,
        elbow_max_deg: float = 60.0,
        depth_neutral_m: float = -1.0,
        invert_depth: bool = False,
        grip_close_threshold: float = 0.055,
        grip_open_threshold: float = 0.095,
        min_depth_confidence: float = 0.15,
        console: Console | None = None,
    ) -> None:
        self._camera = camera
        self._tracker = tracker
        self._driver = driver
        self._config = config
        self._console = console or Console()

        self._x_deadzone = max(0.0, min(0.45, float(x_deadzone)))
        self._y_deadzone = max(0.0, min(0.45, float(y_deadzone)))
        self._depth_deadzone_m = max(0.0, float(depth_deadzone_m))
        self._smoothing_alpha = max(0.01, min(1.0, float(smoothing_alpha)))
        self._depth_scale_deg_per_m = max(1.0, float(depth_scale_deg_per_m))
        self._elbow_max_deg = max(5.0, float(elbow_max_deg))
        self._neutral_depth_m: float | None = (
            float(depth_neutral_m) if float(depth_neutral_m) > 0.0 else None
        )
        self._invert_depth = bool(invert_depth)
        self._grip_close_threshold = float(grip_close_threshold)
        self._grip_open_threshold = float(grip_open_threshold)
        self._min_depth_confidence = max(0.0, min(1.0, float(min_depth_confidence)))

        if self._grip_close_threshold >= self._grip_open_threshold:
            raise ValueError("--grip-close-threshold must be less than --grip-open-threshold")

        self._last_hand_seen_ms = 0
        self._last_command_at = 0.0
        self._gripper_open = True
        self._base_pos = 0.0
        self._shoulder_pos = 0.0
        self._elbow_pos = 0.0
        self._smoothed_x: float | None = None
        self._smoothed_y: float | None = None
        self._smoothed_depth_m: float | None = None

    def run(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required to run stereo hand tracking control.")

        driver_connected = False
        camera_started = False
        try:
            self._driver.connect()
            driver_connected = True
            if self._config.robot.home_on_startup:
                self._driver.home()

            self._camera.start()
            camera_started = True
            self._last_hand_seen_ms = self._now_ms()
            self._console.log("Press 'q' to exit 3D hand tracking, 'n' to reset neutral depth")

            while True:
                frame = self._camera.read()
                detection = self._tracker.detect(frame.frame_bgr)

                if detection.hand_detected:
                    self._last_hand_seen_ms = frame.timestamp_ms
                    self._handle_detection(detection)
                else:
                    self._maybe_timeout_stop(frame.timestamp_ms)

                if self._config.preview:
                    cv2.imshow("Stereo Hand Tracking 3D", detection.annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self._console.log("Stereo hand tracking stopped by keyboard")
                        self._driver.stop_all()
                        break
                    if key == ord("n"):
                        self._neutral_depth_m = None
                        self._console.log("Neutral depth reset requested")
        finally:
            self._tracker.close()
            if camera_started:
                self._camera.stop()
            if driver_connected:
                self._driver.disconnect()
            cv2.destroyAllWindows()

    def _handle_detection(self, detection: StereoHandDetection) -> None:
        if detection.hand_x is None or detection.hand_y is None:
            return

        x = detection.hand_x if self._smoothed_x is None else self._lerp(self._smoothed_x, detection.hand_x)
        y = detection.hand_y if self._smoothed_y is None else self._lerp(self._smoothed_y, detection.hand_y)
        self._smoothed_x = x
        self._smoothed_y = y

        depth_m: float | None = None
        if detection.depth_m is not None and detection.depth_confidence >= self._min_depth_confidence:
            depth_m = (
                detection.depth_m
                if self._smoothed_depth_m is None
                else self._lerp(self._smoothed_depth_m, detection.depth_m)
            )
            self._smoothed_depth_m = depth_m
            if self._neutral_depth_m is None:
                self._neutral_depth_m = depth_m
                self._console.log(f"Depth neutral locked at {depth_m:.2f}m")

        if self._command_allowed():
            self._apply_xy_tracking(x=x, y=y)
            if depth_m is not None:
                self._apply_depth_tracking(depth_m=depth_m)
            self._last_command_at = time.monotonic()

        self._update_gripper(detection.pinch_distance)

    def _apply_xy_tracking(self, x: float, y: float) -> None:
        base_target = self._offset_target_from_xy(
            value=x,
            deadzone=self._x_deadzone,
            negative_limit=abs(self._config.safety.base_min_deg),
            positive_limit=abs(self._config.safety.base_max_deg),
            invert=False,
        )
        shoulder_target = self._offset_target_from_xy(
            value=y,
            deadzone=self._y_deadzone,
            negative_limit=abs(self._config.safety.shoulder_min_deg),
            positive_limit=abs(self._config.safety.shoulder_max_deg),
            invert=True,
        )
        self._move_base_towards(base_target)
        self._move_shoulder_towards(shoulder_target)

    def _apply_depth_tracking(self, depth_m: float) -> None:
        if self._neutral_depth_m is None:
            return
        delta_m = self._neutral_depth_m - float(depth_m)
        if self._invert_depth:
            delta_m = -delta_m
        if abs(delta_m) <= self._depth_deadzone_m:
            target = 0.0
        else:
            target = delta_m * self._depth_scale_deg_per_m
            target = self._clamp(target, -self._elbow_max_deg, self._elbow_max_deg)
        self._move_elbow_towards(target)

    def _update_gripper(self, pinch_distance: float | None) -> None:
        if pinch_distance is None:
            return
        if self._gripper_open and pinch_distance <= self._grip_close_threshold:
            self._gripper_open = False
            self._driver.set_gripper(open=False)
            self._console.log(f"Grip close (pinch={pinch_distance:.3f})")
        elif (not self._gripper_open) and pinch_distance >= self._grip_open_threshold:
            self._gripper_open = True
            self._driver.set_gripper(open=True)
            self._console.log(f"Grip open (pinch={pinch_distance:.3f})")

    def _command_allowed(self) -> bool:
        min_interval = 1.0 / self._config.safety.max_command_hz
        return (time.monotonic() - self._last_command_at) >= min_interval

    def _move_base_towards(self, target: float) -> None:
        delta = self._clamp(
            target - self._base_pos,
            -self._config.safety.max_joint_step_deg,
            self._config.safety.max_joint_step_deg,
        )
        new_pos = self._clamp(
            self._base_pos + delta,
            self._config.safety.base_min_deg,
            self._config.safety.base_max_deg,
        )
        applied_delta = new_pos - self._base_pos
        if abs(applied_delta) <= 1e-6:
            return
        self._driver.move_axis(AxisName.BASE.value, applied_delta)
        self._base_pos = new_pos

    def _move_shoulder_towards(self, target: float) -> None:
        delta = self._clamp(
            target - self._shoulder_pos,
            -self._config.safety.max_joint_step_deg,
            self._config.safety.max_joint_step_deg,
        )
        new_pos = self._clamp(
            self._shoulder_pos + delta,
            self._config.safety.shoulder_min_deg,
            self._config.safety.shoulder_max_deg,
        )
        applied_delta = new_pos - self._shoulder_pos
        if abs(applied_delta) <= 1e-6:
            return
        self._driver.move_axis(AxisName.SHOULDER.value, applied_delta)
        self._shoulder_pos = new_pos

    def _move_elbow_towards(self, target: float) -> None:
        delta = self._clamp(
            target - self._elbow_pos,
            -self._config.safety.max_joint_step_deg,
            self._config.safety.max_joint_step_deg,
        )
        new_pos = self._clamp(
            self._elbow_pos + delta,
            -self._elbow_max_deg,
            self._elbow_max_deg,
        )
        applied_delta = new_pos - self._elbow_pos
        if abs(applied_delta) <= 1e-6:
            return
        self._driver.move_axis("elbow", applied_delta)
        self._elbow_pos = new_pos

    @staticmethod
    def _offset_target_from_xy(
        *,
        value: float,
        deadzone: float,
        negative_limit: float,
        positive_limit: float,
        invert: bool,
    ) -> float:
        centered = float(value) - 0.5
        if invert:
            centered = -centered
        if abs(centered) <= deadzone:
            return 0.0

        span = max(0.001, 0.5 - deadzone)
        normalized = (abs(centered) - deadzone) / span
        normalized = max(0.0, min(1.0, normalized))
        magnitude = normalized * (positive_limit if centered >= 0.0 else negative_limit)
        return magnitude if centered >= 0.0 else -magnitude

    def _maybe_timeout_stop(self, timestamp_ms: int) -> None:
        if (timestamp_ms - self._last_hand_seen_ms) >= self._config.safety.no_hand_timeout_ms:
            self._driver.stop_all()
            self._last_hand_seen_ms = timestamp_ms

    def _lerp(self, current: float, observed: float) -> float:
        a = self._smoothing_alpha
        return (1.0 - a) * float(current) + a * float(observed)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)
