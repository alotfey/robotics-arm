from __future__ import annotations

import argparse

from app import cli


def _demo_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "width": 320,
        "height": 240,
        "fps": 24,
        "duration_sec": 0.15,
        "no_preview": True,
        "no_home": True,
        "debounce_frames": 5,
        "no_hand_timeout_ms": 800,
        "max_command_hz": 20.0,
        "max_joint_step_deg": 5.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _demo_robot_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "port": "/dev/tty.demo",
        "baud_rate": 9600,
        "protocol": "default-program",
        "dry_run": False,
        "live": False,
        "prefer_bluetooth": False,
        "cycles": 1,
        "step_deg": 2.0,
        "pause_ms": 0,
        "moves_per_direction": 3,
        "servo_hold_ms": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _demo_dance_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "port": "/dev/tty.demo",
        "baud_rate": 9600,
        "protocol": "default-program",
        "dry_run": False,
        "live": False,
        "prefer_bluetooth": False,
        "cycles": 1,
        "routine_file": "",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _track_hand_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "camera_index": 0,
        "width": 320,
        "height": 240,
        "fps": 20,
        "port": "/dev/tty.demo",
        "baud_rate": 9600,
        "protocol": "default-program",
        "dry_run": True,
        "prefer_bluetooth": False,
        "no_home": True,
        "debounce_frames": 5,
        "no_hand_timeout_ms": 800,
        "max_command_hz": 20.0,
        "max_joint_step_deg": 4.0,
        "no_preview": True,
        "x_deadzone": 0.08,
        "y_deadzone": 0.08,
        "smoothing_alpha": 0.35,
        "grip_close_threshold": 0.055,
        "grip_open_threshold": 0.095,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _track_hand_3d_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "camera_index": 0,
        "width": 2560,
        "height": 960,
        "fps": 30,
        "port": "/dev/tty.demo",
        "baud_rate": 9600,
        "protocol": "default-program",
        "dry_run": True,
        "prefer_bluetooth": False,
        "no_home": True,
        "debounce_frames": 5,
        "no_hand_timeout_ms": 800,
        "max_command_hz": 20.0,
        "max_joint_step_deg": 4.0,
        "no_preview": True,
        "x_deadzone": 0.07,
        "y_deadzone": 0.07,
        "depth_deadzone_m": 0.03,
        "smoothing_alpha": 0.35,
        "stereo_baseline_mm": 60.0,
        "stereo_hfov_deg": 100.0,
        "swap_stereo_halves": False,
        "depth_scale_deg_per_m": 120.0,
        "elbow_max_deg": 60.0,
        "depth_neutral_m": -1.0,
        "invert_depth": False,
        "grip_close_threshold": 0.055,
        "grip_open_threshold": 0.095,
        "min_depth_confidence": 0.15,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_parser_supports_demo_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["demo", "--duration-sec", "0.1", "--no-preview"])
    assert args.command == "demo"
    assert args.duration_sec == 0.1


def test_run_demo_completes_without_hardware() -> None:
    exit_code = cli.run_demo(_demo_args())
    assert exit_code == 0


def test_build_parser_supports_demo_robot_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["demo-robot", "--cycles", "2", "--step-deg", "4", "--moves-per-direction", "5", "--live"]
    )
    assert args.command == "demo-robot"
    assert args.cycles == 2
    assert args.step_deg == 4
    assert args.moves_per_direction == 5
    assert args.live is True


def test_build_parser_supports_default_program_protocol() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["demo-robot", "--protocol", "default-program", "--baud-rate", "9600", "--dry-run"]
    )
    assert args.protocol == "default-program"
    assert args.baud_rate == 9600


def test_build_parser_defaults_to_default_program_protocol() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["demo-robot"])
    assert args.protocol == "default-program"
    assert args.baud_rate == 9600


def test_build_parser_supports_detect_hardware_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["detect-hardware", "--camera-max-index", "4"])
    assert args.command == "detect-hardware"
    assert args.camera_max_index == 4


def test_build_parser_supports_demo_dance_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["demo-dance", "--cycles", "2", "--live"])
    assert args.command == "demo-dance"
    assert args.cycles == 2
    assert args.live is True


def test_build_parser_supports_track_hand_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["track-hand", "--camera-index", "0", "--no-preview"])
    assert args.command == "track-hand"
    assert args.camera_index == 0
    assert args.no_preview is True


def test_build_parser_supports_track_hand_3d_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["track-hand-3d", "--camera-index", "0", "--no-preview"])
    assert args.command == "track-hand-3d"
    assert args.camera_index == 0
    assert args.no_preview is True


def test_run_demo_robot_completes_without_hardware() -> None:
    exit_code = cli.run_demo_robot(_demo_robot_args())
    assert exit_code == 0


def test_run_demo_dance_completes_without_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_load_dance_routine",
        lambda _: {
            "name": "tiny",
            "movements": [
                {"phase": 1, "name": "p1", "steps": [{"base": 100, "gripper": 100, "delay_ms": 0}]}
            ],
        },
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli.run_demo_dance(_demo_dance_args())
    assert exit_code == 0


def test_run_demo_robot_rejects_live_and_dry_run_together() -> None:
    args = _demo_robot_args(live=True, dry_run=True)
    try:
        cli.run_demo_robot(args)
        raise AssertionError("Expected ValueError when both --live and --dry-run are set")
    except ValueError as exc:
        assert "Choose either --live or --dry-run, not both" in str(exc)


def test_run_demo_robot_rejects_nonpositive_moves_per_direction() -> None:
    args = _demo_robot_args(moves_per_direction=0)
    try:
        cli.run_demo_robot(args)
        raise AssertionError("Expected ValueError for --moves-per-direction <= 0")
    except ValueError as exc:
        assert "--moves-per-direction must be greater than 0" in str(exc)


def test_run_demo_robot_live_uses_live_driver(monkeypatch) -> None:
    state: dict[str, object] = {}

    class FakeDriver:
        def __init__(self, config, console=None) -> None:
            state["config"] = config
            state["calls"] = []

        def connect(self) -> None:
            state["calls"].append("connect")

        def home(self) -> None:
            state["calls"].append("home")

        def read_positions(self) -> dict[str, int]:
            state["calls"].append("read_positions")
            return {"base": 500, "shoulder": 500, "elbow": 500, "wrist": 500, "gripper": 500}

        def center_all(self) -> None:
            state["calls"].append("center_all")

        def move_axis(self, axis: str, delta: float) -> None:
            state["calls"].append(("move", axis, delta))

        def set_gripper(self, open: bool) -> None:
            state["calls"].append(("grip", open))

        def stop_all(self) -> None:
            state["calls"].append("stop")

        def disconnect(self) -> None:
            state["calls"].append("disconnect")

    monkeypatch.setattr(cli, "LewanSoulMiniArmDriver", FakeDriver)

    exit_code = cli.run_demo_robot(
        _demo_robot_args(live=True, dry_run=False, cycles=1, step_deg=2.0, moves_per_direction=2)
    )

    assert exit_code == 0
    assert state["config"].dry_run is False
    calls = state["calls"]
    assert calls.count("read_positions") == 1
    assert calls.count("center_all") == 2
    # sweep = step_deg * moves_per_direction = 2 * 2 = 4
    # Each axis does: +4, -4
    assert calls.count(("move", "base", 4.0)) == 1
    assert calls.count(("move", "base", -4.0)) == 1
    assert calls.count(("move", "shoulder", 4.0)) == 1
    assert calls.count(("move", "shoulder", -4.0)) == 1
    assert calls.count(("move", "elbow", 4.0)) == 1
    assert calls.count(("move", "elbow", -4.0)) == 1
    assert calls.count(("move", "wrist", 4.0)) == 1
    assert calls.count(("move", "wrist", -4.0)) == 1
    assert calls.count(("grip", False)) == 2
    assert calls.count(("grip", True)) == 2


def test_run_demo_dance_live_uses_live_driver(monkeypatch) -> None:
    state: dict[str, object] = {}

    class FakeDriver:
        def __init__(self, config, console=None) -> None:
            state["config"] = config
            state["calls"] = []

        def connect(self) -> None:
            state["calls"].append("connect")

        def home(self) -> None:
            state["calls"].append("home")

        def move_axis(self, axis: str, delta: float) -> None:
            state["calls"].append(("move", axis, delta))

        def set_gripper(self, open: bool) -> None:
            state["calls"].append(("grip", open))

        def stop_all(self) -> None:
            state["calls"].append("stop")

        def disconnect(self) -> None:
            state["calls"].append("disconnect")

    monkeypatch.setattr(cli, "LewanSoulMiniArmDriver", FakeDriver)
    monkeypatch.setattr(
        cli,
        "_load_dance_routine",
        lambda _: {
            "name": "tiny",
            "movements": [
                {
                    "phase": 1,
                    "name": "p1",
                    "steps": [
                        {"base": 100, "gripper": 100, "delay_ms": 0},
                        {"base": 80, "gripper": 0, "delay_ms": 0},
                    ],
                }
            ],
        },
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli.run_demo_dance(_demo_dance_args(live=True, dry_run=False, cycles=1))

    assert exit_code == 0
    assert state["config"].dry_run is False
    calls = state["calls"]
    assert calls.count("connect") == 1
    assert calls.count("home") == 2
    assert calls.count(("move", "base", 10.0)) == 1
    assert calls.count(("move", "base", -20.0)) == 1
    assert calls.count(("grip", True)) == 1
    assert calls.count(("grip", False)) == 1
    assert calls.count("stop") == 1
    assert calls.count("disconnect") == 1


def test_run_track_hand_wires_tracking_loop(monkeypatch) -> None:
    state: dict[str, object] = {}

    class FakeCamera:
        def __init__(self, index: int, width: int, height: int, fps: int) -> None:
            state["camera_args"] = (index, width, height, fps)

    class FakeTracker:
        def __init__(self, min_detection_confidence: float, min_tracking_confidence: float) -> None:
            state["tracker_args"] = (min_detection_confidence, min_tracking_confidence)

    class FakeDriver:
        def __init__(self, config, console=None) -> None:
            state["driver_config"] = config

    class FakeLoop:
        def __init__(self, camera, tracker, driver, config, **kwargs) -> None:
            state["loop_inputs"] = (camera, tracker, driver, config)
            state["loop_kwargs"] = kwargs
            state["run_called"] = False

        def run(self) -> None:
            state["run_called"] = True

    monkeypatch.setattr(cli, "UvcCamera", FakeCamera)
    monkeypatch.setattr(cli, "HandTracker", FakeTracker)
    monkeypatch.setattr(cli, "LewanSoulMiniArmDriver", FakeDriver)
    monkeypatch.setattr(cli, "HandTrackingLoop", FakeLoop)

    exit_code = cli.run_track_hand(_track_hand_args())
    assert exit_code == 0
    assert state["camera_args"] == (0, 320, 240, 20)
    assert state["tracker_args"] == (0.55, 0.55)
    assert state["driver_config"].dry_run is True
    assert state["run_called"] is True


def test_run_track_hand_3d_wires_tracking_loop(monkeypatch) -> None:
    state: dict[str, object] = {}

    class FakeCamera:
        def __init__(self, index: int, width: int, height: int, fps: int) -> None:
            state["camera_args"] = (index, width, height, fps)

    class FakeTracker:
        def __init__(
            self,
            baseline_mm: float,
            hfov_deg: float,
            swap_halves: bool,
            min_detection_confidence: float,
            min_tracking_confidence: float,
        ) -> None:
            state["tracker_args"] = (
                baseline_mm,
                hfov_deg,
                swap_halves,
                min_detection_confidence,
                min_tracking_confidence,
            )

    class FakeDriver:
        def __init__(self, config, console=None) -> None:
            state["driver_config"] = config

    class FakeLoop:
        def __init__(self, camera, tracker, driver, config, **kwargs) -> None:
            state["loop_inputs"] = (camera, tracker, driver, config)
            state["loop_kwargs"] = kwargs
            state["run_called"] = False

        def run(self) -> None:
            state["run_called"] = True

    monkeypatch.setattr(cli, "UvcCamera", FakeCamera)
    monkeypatch.setattr(cli, "StereoDepthHandTracker", FakeTracker)
    monkeypatch.setattr(cli, "LewanSoulMiniArmDriver", FakeDriver)
    monkeypatch.setattr(cli, "StereoHandTrackingLoop", FakeLoop)

    exit_code = cli.run_track_hand_3d(_track_hand_3d_args())
    assert exit_code == 0
    assert state["camera_args"] == (0, 2560, 960, 30)
    assert state["tracker_args"] == (60.0, 100.0, False, 0.55, 0.55)
    assert state["driver_config"].dry_run is True
    assert state["run_called"] is True
