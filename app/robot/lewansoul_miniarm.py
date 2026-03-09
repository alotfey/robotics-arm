from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from app.robot.driver import RobotArmDriver


@dataclass
class LewanSoulConfig:
    port: str
    baud_rate: int = 115200
    dry_run: bool = True


class LewanSoulMiniArmDriver(RobotArmDriver):
    """LewanSoul miniArm serial adapter.

    This wrapper uses a simple line protocol placeholder that can be replaced with the
    official command format from the LewanSoul SDK while keeping the app interfaces stable.
    """

    def __init__(self, config: LewanSoulConfig, console: Console | None = None) -> None:
        self._config = config
        self._console = console or Console()
        self._serial: Any | None = None

    def connect(self) -> None:
        if self._config.dry_run:
            self._console.log("[yellow]DRY-RUN[/yellow] connect robot")
            return
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("pyserial is required for live robot control.") from exc
        self._serial = serial.Serial(self._config.port, self._config.baud_rate, timeout=0.2)
        self._console.log(f"Connected to {self._config.port} @ {self._config.baud_rate}")
        time.sleep(0.3)

    def home(self) -> None:
        self._send_command("HOME")

    def move_axis(self, axis: str, delta: float) -> None:
        self._send_command(f"MOVE {axis} {delta:.2f}")

    def set_gripper(self, open: bool) -> None:
        self._send_command("GRIP OPEN" if open else "GRIP CLOSE")

    def stop_all(self) -> None:
        self._send_command("STOP")

    def disconnect(self) -> None:
        if self._config.dry_run:
            self._console.log("[yellow]DRY-RUN[/yellow] disconnect robot")
            return
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._console.log("Disconnected robot")

    def _send_command(self, command: str) -> None:
        if self._config.dry_run:
            self._console.log(f"[yellow]DRY-RUN[/yellow] {command}")
            return
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Robot serial connection is not open")
        payload = f"{command}\n".encode("ascii")
        self._serial.write(payload)
