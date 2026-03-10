from __future__ import annotations

import time

from app.models import CameraConfig, GestureEvent, GestureName, RobotConfig, RuntimeConfig, SafetyConfig
from app.runtime.control_loop import ControlLoop


class DummyCamera:
    def start(self) -> None:
        """No-op camera start used by unit tests."""
        pass

    def stop(self) -> None:
        """No-op camera stop used by unit tests."""
        pass


class DummyClassifier:
    def close(self) -> None:
        """No-op classifier close used by unit tests."""
        pass


class DummyDriver:
    def __init__(self) -> None:
        """Initialize in-memory call recording state."""
        self.moves: list[tuple[str, float]] = []
        self.gripper: list[bool] = []
        self.stops = 0

    def connect(self) -> None:
        """No-op driver connect used by unit tests."""
        pass

    def home(self) -> None:
        """No-op driver home used by unit tests."""
        pass

    def move_axis(self, axis: str, delta: float) -> None:
        """Record movement commands for assertions.

        Args:
            axis: Axis name.
            delta: Relative movement value.
        """
        self.moves.append((axis, delta))

    def set_gripper(self, open: bool) -> None:
        """Record gripper state changes for assertions.

        Args:
            open: Gripper open/closed state.
        """
        self.gripper.append(open)

    def stop_all(self) -> None:
        """Record emergency stop calls."""
        self.stops += 1

    def disconnect(self) -> None:
        """No-op driver disconnect used by unit tests."""
        pass


def _build_loop(max_step: float = 5.0) -> tuple[ControlLoop, DummyDriver]:
    """Build a control loop and dummy driver with test-safe defaults.

    Args:
        max_step: Maximum joint step configured for the loop.

    Returns:
        tuple[ControlLoop, DummyDriver]: Constructed loop and backing driver.
    """
    config = RuntimeConfig(
        camera=CameraConfig(),
        robot=RobotConfig(),
        safety=SafetyConfig(max_joint_step_deg=max_step, max_command_hz=60.0, no_hand_timeout_ms=800),
        preview=False,
    )
    driver = DummyDriver()
    loop = ControlLoop(DummyCamera(), DummyClassifier(), driver, config)
    return loop, driver


def test_move_clamped_by_joint_limits() -> None:
    """Movement commands are clamped at configured joint bounds."""
    loop, driver = _build_loop(max_step=10.0)
    loop._base_pos = 89.0  # near max bound

    event = GestureEvent(name=GestureName.RIGHT, confidence=1.0, timestamp_ms=1, stable_frames=5)
    loop._handle_event(event)

    assert driver.moves
    axis, delta = driver.moves[-1]
    assert axis == "base"
    assert round(delta, 5) == 1.0  # clamped to max bound (90 - 89)


def test_grip_toggle_emits_command() -> None:
    """Two separate grip toggles flip the gripper state twice."""
    loop, driver = _build_loop()

    event = GestureEvent(name=GestureName.GRIP_TOGGLE, confidence=1.0, timestamp_ms=1, stable_frames=5)
    loop._handle_event(event)
    loop._last_command_at = 0.0
    loop._handle_event(event)

    assert driver.gripper == [False, True]


def test_grip_toggle_ignores_held_gesture_after_first_toggle() -> None:
    """Held grip gesture emits only one toggle command."""
    loop, driver = _build_loop()

    first = GestureEvent(name=GestureName.GRIP_TOGGLE, confidence=1.0, timestamp_ms=1, stable_frames=5)
    held = GestureEvent(name=GestureName.GRIP_TOGGLE, confidence=1.0, timestamp_ms=2, stable_frames=6)

    loop._handle_event(first)
    loop._last_command_at = 0.0
    loop._handle_event(held)

    assert driver.gripper == [False]


def test_no_hand_timeout_triggers_stop() -> None:
    """No-hand timeout triggers a stop command."""
    loop, driver = _build_loop()
    loop._last_hand_seen_ms = 1000

    loop._maybe_timeout_stop(2000)

    assert driver.stops == 1


def test_rate_limit_blocks_rapid_commands() -> None:
    """Command rate limiting blocks commands sent too quickly."""
    config = RuntimeConfig(
        camera=CameraConfig(),
        robot=RobotConfig(),
        safety=SafetyConfig(max_joint_step_deg=5.0, max_command_hz=2.0, no_hand_timeout_ms=800),
        preview=False,
    )
    driver = DummyDriver()
    loop = ControlLoop(DummyCamera(), DummyClassifier(), driver, config)

    event = GestureEvent(name=GestureName.UP, confidence=1.0, timestamp_ms=1, stable_frames=5)
    loop._handle_event(event)
    first_count = len(driver.moves)
    loop._handle_event(event)

    assert first_count == 1
    assert len(driver.moves) == 1

    time.sleep(0.55)
    loop._handle_event(event)
    assert len(driver.moves) == 2
