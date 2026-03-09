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
        "baud_rate": 115200,
        "dry_run": False,
        "cycles": 1,
        "step_deg": 2.0,
        "pause_ms": 0,
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
    args = parser.parse_args(["demo-robot", "--cycles", "2", "--step-deg", "4"])
    assert args.command == "demo-robot"
    assert args.cycles == 2
    assert args.step_deg == 4


def test_run_demo_robot_completes_without_hardware() -> None:
    exit_code = cli.run_demo_robot(_demo_robot_args())
    assert exit_code == 0
