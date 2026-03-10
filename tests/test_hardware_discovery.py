from __future__ import annotations

from pathlib import Path

from app import cli
from app.runtime import hardware_discovery as hd


def test_load_hardware_config_missing_file_returns_empty(tmp_path: Path) -> None:
    """Missing config files load as empty dictionaries."""
    config_path = tmp_path / "missing.json"
    assert hd.load_hardware_config(config_path) == {}


def test_detect_and_save_hardware_paths_writes_config(tmp_path: Path, monkeypatch) -> None:
    """Discovery writes expected fields to config output."""
    config_path = tmp_path / "hardware_paths.json"

    monkeypatch.setattr(hd, "detect_arm_path", lambda: "/dev/tty.usbmodemTEST")
    monkeypatch.setattr(hd, "detect_stereo_camera", lambda camera_max_index: ("/dev/video2", 2))

    data = hd.detect_and_save_hardware_paths(config_path=config_path, camera_max_index=9)

    assert data["arm_path"] == "/dev/tty.usbmodemTEST"
    assert data["stereo_camera_path"] == "/dev/video2"
    assert data["stereo_camera_index"] == 2
    assert data["updated_at"]
    assert config_path.exists()

    loaded = hd.load_hardware_config(config_path)
    assert loaded["arm_path"] == "/dev/tty.usbmodemTEST"
    assert loaded["stereo_camera_path"] == "/dev/video2"
    assert loaded["stereo_camera_index"] == 2


def test_run_detect_hardware_writes_requested_file(tmp_path: Path) -> None:
    """CLI detect-hardware command writes to requested output path."""
    config_path = tmp_path / "detected.json"
    args = type("Args", (), {"config_file": str(config_path), "camera_max_index": 0})()

    exit_code = cli.run_detect_hardware(args)
    assert exit_code == 0
    assert config_path.exists()
