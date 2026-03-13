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


def detect_arm_path(prefer_bluetooth: bool = False) -> str | None:
    """Detect a likely serial path for the LewanSoul arm.

    Args:
        prefer_bluetooth: Whether Bluetooth serial adapters should be ranked
            above USB adapters when both are available.

    Returns:
        Best-matching serial device path or ``None`` when nothing matches.
    """
    candidates = _serial_candidates_from_pyserial_compat(prefer_bluetooth=prefer_bluetooth)

    # Fallback for environments without pyserial: probe /dev directly.
    if not candidates:
        candidates = _serial_candidates_from_dev_compat(prefer_bluetooth=prefer_bluetooth)

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _serial_candidates_from_pyserial(prefer_bluetooth: bool = False) -> list[tuple[int, str]]:
    """Collect scored serial candidates via pyserial, if available."""
    try:
        from serial.tools import list_ports
    except ImportError:  # pragma: no cover - environment dependent
        return []

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
        score = _serial_score(
            haystack,
            device,
            prefer_bluetooth=prefer_bluetooth,
        )
        if score > 0:
            candidates.append((score, device))
    return candidates


def _serial_candidates_from_dev(prefer_bluetooth: bool = False) -> list[tuple[int, str]]:
    """Collect scored serial candidates by scanning ``/dev`` patterns."""
    dev = Path("/dev")
    patterns = (
        "cu.usb*",
        "tty.usb*",
        "cu.wchusb*",
        "tty.wchusb*",
        "ttyACM*",
        "ttyUSB*",
        # Bluetooth SPP/BLE serial adapters on macOS often show up as /dev/cu.*
        # with names such as HC-05-DevB, HMSoft, or custom module names.
        "cu.*",
        "tty.*",
    )
    seen: set[str] = set()
    candidates: list[tuple[int, str]] = []
    for pattern in patterns:
        for path in sorted(dev.glob(pattern)):
            device = str(path)
            if device in seen:
                continue
            seen.add(device)
            haystack = f"{path.name} {device}".lower()
            score = _serial_score(
                haystack,
                device,
                prefer_bluetooth=prefer_bluetooth,
            )
            if score > 0:
                candidates.append((score, device))
    return candidates


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
    if not config_path.exists():
        return {}
    raw = config_path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    return dict(json.loads(raw))


def detect_and_save_hardware_paths(
    config_path: Path = DEFAULT_CONFIG_PATH,
    camera_max_index: int = 8,
    prefer_bluetooth: bool = False,
) -> dict[str, Any]:
    arm_path = _detect_arm_path_compat(prefer_bluetooth=prefer_bluetooth)
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


def _serial_score(haystack: str, device: str, prefer_bluetooth: bool = False) -> int:
    score = 0
    if "usb" in haystack:
        score += 2
    if "tty.usb" in device or "cu.usb" in device:
        score += 4
    if any(tag in haystack for tag in ("wch", "cp210", "ftdi", "ch340", "silicon labs", "usbmodem", "usbserial", "slab")):
        score += 6
    if "bluetooth-incoming-port" in haystack or "bluetooth incoming port" in haystack:
        score -= 20
    if _looks_like_bluetooth_serial(haystack, device):
        score += 5
        if prefer_bluetooth:
            score += 8
    return score


def _looks_like_bluetooth_serial(haystack: str, device: str) -> bool:
    """Return ``True`` when a serial device name resembles a BT adapter."""
    probe = f"{haystack} {device}".lower()
    return any(
        token in probe
        for token in (
            "bluetooth",
            "ble",
            "hc-05",
            "hc05",
            "hc-06",
            "hc06",
            "hm-10",
            "hm10",
            "spp",
            "devb",
            "hiwonder",
        )
    )


def _detect_arm_path_compat(prefer_bluetooth: bool) -> str | None:
    """Call ``detect_arm_path`` with compatibility for test monkeypatches."""
    try:
        return detect_arm_path(prefer_bluetooth=prefer_bluetooth)
    except TypeError:
        return detect_arm_path()


def _serial_candidates_from_pyserial_compat(prefer_bluetooth: bool) -> list[tuple[int, str]]:
    """Call pyserial candidate collector with compatibility for patched tests."""
    try:
        return _serial_candidates_from_pyserial(prefer_bluetooth=prefer_bluetooth)
    except TypeError:
        return _serial_candidates_from_pyserial()


def _serial_candidates_from_dev_compat(prefer_bluetooth: bool) -> list[tuple[int, str]]:
    """Call /dev candidate collector with compatibility for patched tests."""
    try:
        return _serial_candidates_from_dev(prefer_bluetooth=prefer_bluetooth)
    except TypeError:
        return _serial_candidates_from_dev()


def _camera_path_for_index(index: int) -> str:
    # Linux-style video device path, with fallback for macOS/Windows.
    candidate = Path(f"/dev/video{index}")
    if candidate.exists():
        return str(candidate)
    return f"index:{index}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect arm/camera and write hardware config")
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--camera-max-index", type=int, default=8)
    parser.add_argument("--prefer-bluetooth", action="store_true")
    args = parser.parse_args()

    if args.camera_max_index < 0:
        raise ValueError("--camera-max-index must be >= 0")

    result = detect_and_save_hardware_paths(
        config_path=Path(args.config_file),
        camera_max_index=args.camera_max_index,
        prefer_bluetooth=args.prefer_bluetooth,
    )
    print(f"Hardware config updated: {args.config_file}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
