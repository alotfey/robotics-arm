from __future__ import annotations

import time

try:
    import cv2
except ImportError:  # pragma: no cover - optional during lightweight test runs
    cv2 = None  # type: ignore[assignment]
from rich.console import Console

from app.camera.uvc_camera import UvcCamera
from app.models import AxisName, GestureEvent, GestureName, RuntimeConfig
from app.robot.driver import RobotArmDriver
from app.vision.gestures import GestureClassifier


class ControlLoop:
    def __init__(
        self,
        camera: UvcCamera,
        classifier: GestureClassifier,
        driver: RobotArmDriver,
        config: RuntimeConfig,
        console: Console | None = None,
    ) -> None:
        """Initialize the live gesture-control runtime loop.

        Args:
            camera: Camera source providing frames.
            classifier: Gesture detector/classifier implementation.
            driver: Robot driver receiving movement commands.
            config: Runtime configuration and safety limits.
            console: Optional Rich console for logs.
        """
        self._camera = camera
        self._classifier = classifier
        self._driver = driver
        self._config = config
        self._console = console or Console()

        self._last_hand_seen_ms = 0
        self._last_command_at = 0.0
        self._gripper_open = True
        self._base_pos = 0.0
        self._shoulder_pos = 0.0

    def run(self) -> None:
        """Run the control loop until interrupted or preview exit."""
        if cv2 is None:
            raise RuntimeError("opencv-python is required to run the control loop.")
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
            self._console.log("Press 'q' in preview window to trigger emergency stop and exit")

            while True:
                frame = self._camera.read()
                detection = self._classifier.detect(frame.frame_bgr, frame.timestamp_ms)

                if detection.hand_detected:
                    self._last_hand_seen_ms = frame.timestamp_ms
                else:
                    self._maybe_timeout_stop(frame.timestamp_ms)

                if detection.event is not None:
                    self._handle_event(detection.event)

                if self._config.preview:
                    cv2.imshow("Gesture Control", detection.annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self._console.log("Emergency stop triggered by keyboard")
                        self._driver.stop_all()
                        break

        finally:
            self._classifier.close()
            if camera_started:
                self._camera.stop()
            if driver_connected:
                self._driver.disconnect()
            cv2.destroyAllWindows()

    def _handle_event(self, event: GestureEvent) -> None:
        """Translate a gesture event into robot commands.

        Args:
            event: Stable gesture event emitted by the classifier.
        """
        if not self._command_allowed():
            return

        step = min(
            self._config.safety.max_joint_step_deg,
            self._config.safety.max_joint_step_deg * max(0.4, event.confidence),
        )

        if event.name == GestureName.UP:
            self._move_shoulder(step)
        elif event.name == GestureName.DOWN:
            self._move_shoulder(-step)
        elif event.name == GestureName.LEFT:
            self._move_base(-step)
        elif event.name == GestureName.RIGHT:
            self._move_base(step)
        elif event.name == GestureName.GRIP_TOGGLE:
            # Toggle once when the gesture first becomes stable; ignore continued hold.
            if event.stable_frames > self._config.safety.debounce_frames:
                return
            self._gripper_open = not self._gripper_open
            self._driver.set_gripper(self._gripper_open)

        self._last_command_at = time.monotonic()

    def _command_allowed(self) -> bool:
        """Return whether a new command can be sent under rate limits."""
        min_interval = 1.0 / self._config.safety.max_command_hz
        return (time.monotonic() - self._last_command_at) >= min_interval

    def _move_base(self, delta: float) -> None:
        """Move the base axis while enforcing configured joint bounds.

        Args:
            delta: Requested relative movement in degrees.
        """
        new_pos = self._clamp(
            self._base_pos + delta,
            self._config.safety.base_min_deg,
            self._config.safety.base_max_deg,
        )
        applied_delta = new_pos - self._base_pos
        if abs(applied_delta) > 1e-6:
            self._driver.move_axis(AxisName.BASE.value, applied_delta)
            self._base_pos = new_pos

    def _move_shoulder(self, delta: float) -> None:
        """Move the shoulder axis while enforcing configured joint bounds.

        Args:
            delta: Requested relative movement in degrees.
        """
        new_pos = self._clamp(
            self._shoulder_pos + delta,
            self._config.safety.shoulder_min_deg,
            self._config.safety.shoulder_max_deg,
        )
        applied_delta = new_pos - self._shoulder_pos
        if abs(applied_delta) > 1e-6:
            self._driver.move_axis(AxisName.SHOULDER.value, applied_delta)
            self._shoulder_pos = new_pos

    def _maybe_timeout_stop(self, timestamp_ms: int) -> None:
        """Stop the robot if no hand has been detected for too long.

        Args:
            timestamp_ms: Current frame timestamp in milliseconds.
        """
        if (timestamp_ms - self._last_hand_seen_ms) >= self._config.safety.no_hand_timeout_ms:
            self._driver.stop_all()
            self._last_hand_seen_ms = timestamp_ms

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        """Clamp a numeric value into the inclusive range ``[low, high]``."""
        return max(low, min(high, value))

    @staticmethod
    def _now_ms() -> int:
        """Return current monotonic time in milliseconds."""
        return int(time.monotonic() * 1000)
