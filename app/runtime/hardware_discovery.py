from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore[assignment]

DEFAULT_CONFIG_PATH = Path("config/hardware_paths.json")


def detect_arm_path() -> str | None:
    """Detect a likely serial path for the LewanSoul arm."""
    try:
        from serial.tools import list_ports
    except ImportError:  # pragma: no cover - environment dependent
        return None

    candidates: list[tuple[int, str]] = []
    for port in list_ports.comports():
        device = str(getattr(port, "device", "")).strip()
        if not device:
            continue
        haystack = " ".join(
            [
                str(getattr(port, "description", "")),
                str(getattr(port, "hwid", "")),
                device,
            ]
        ).lower()
        score = _serial_score(haystack, device)
        if score > 0:
            candidates.append((score, device))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def detect_stereo_camera(camera_max_index: int = 8) -> tuple[str | None, int | None]:
    """Detect a working stereo camera source by probing camera indices."""
    if cv2 is None:  # pragma: no cover - environment dependent
        return None, None

    working_indices: list[int] = []
    for idx in range(camera_max_index + 1):
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                working_indices.append(idx)
        finally:
            cap.release()

    if not working_indices:
        return None, None

    # Many stereo USB cameras expose 2 nodes. Prefer the first index from a pair.
    if len(working_indices) >= 2:
        selected_idx = working_indices[0]
    else:
        selected_idx = working_indices[0]

    camera_path = _camera_path_for_index(selected_idx)
    return camera_path, selected_idx


def load_hardware_config(config_path: Path) -> dict[str, Any]:
    """Load hardware configuration JSON if the file exists.

    Args:
        config_path: Path to the hardware config JSON file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary, or empty dict when missing/blank.
    """
    if not config_path.exists():
        return {}
    raw = config_path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    return dict(json.loads(raw))


def detect_and_save_hardware_paths(
    config_path: Path = DEFAULT_CONFIG_PATH,
    camera_max_index: int = 8,
) -> dict[str, Any]:
    """Detect hardware paths and persist them to configuration JSON.

    Args:
        config_path: Destination config file path.
        camera_max_index: Highest camera index to probe.

    Returns:
        dict[str, Any]: Final configuration dictionary that was written.
    """
    arm_path = detect_arm_path()
    camera_path, camera_index = detect_stereo_camera(camera_max_index=camera_max_index)

    config = load_hardware_config(config_path)
    config["arm_path"] = arm_path
    config["stereo_camera_path"] = camera_path
    config["stereo_camera_index"] = camera_index
    config["updated_at"] = datetime.now(timezone.utc).isoformat()
    config["platform"] = platform.platform()

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def _serial_score(haystack: str, device: str) -> int:
    """Score a serial candidate path based on likely USB-arm characteristics.

    Args:
        haystack: Lowercased descriptive metadata for a port.
        device: Port device path string.

    Returns:
        int: Heuristic relevance score (higher is better).
    """
    score = 0
    if "usb" in haystack:
        score += 2
    if "tty.usb" in device or "cu.usb" in device:
        score += 4
    if any(tag in haystack for tag in ("wch", "cp210", "ftdi", "ch340", "silicon labs", "usbmodem")):
        score += 6
    if "bluetooth" in haystack:
        score -= 5
    return score


def _camera_path_for_index(index: int) -> str:
    """Resolve a user-friendly camera path string for an index.

    Args:
        index: OpenCV camera index.

    Returns:
        str: Concrete device path if available, otherwise ``index:{index}``.
    """
    # Linux-style video device path, with fallback for macOS/Windows.
    candidate = Path(f"/dev/video{index}")
    if candidate.exists():
        return str(candidate)
    return f"index:{index}"


def main() -> int:
    """CLI entrypoint for hardware auto-discovery.

    Returns:
        int: Process-style status code.

    Raises:
        ValueError: If camera-max-index is negative.
    """
    parser = argparse.ArgumentParser(description="Detect arm/camera and write hardware config")
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--camera-max-index", type=int, default=8)
    args = parser.parse_args()

    if args.camera_max_index < 0:
        raise ValueError("--camera-max-index must be >= 0")

    result = detect_and_save_hardware_paths(
        config_path=Path(args.config_file),
        camera_max_index=args.camera_max_index,
    )
    print(f"Hardware config updated: {args.config_file}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
