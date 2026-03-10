from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GestureName(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    GRIP_TOGGLE = "GRIP_TOGGLE"


class AxisName(str, Enum):
    BASE = "base"
    SHOULDER = "shoulder"


class GestureEvent(BaseModel):
    name: GestureName
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp_ms: int = Field(ge=0)
    stable_frames: int = Field(ge=0)


class CameraConfig(BaseModel):
    camera_index: int = 0
    width: int = Field(default=640, ge=160, le=3840)
    height: int = Field(default=480, ge=120, le=2160)
    fps: int = Field(default=30, ge=1, le=120)


class RobotConfig(BaseModel):
    port: str = "/dev/tty.usbmodem"
    baud_rate: int = Field(default=115200, ge=9600, le=2000000)
    dry_run: bool = True
    home_on_startup: bool = True


class SafetyConfig(BaseModel):
    no_hand_timeout_ms: int = Field(default=800, ge=100, le=5000)
    debounce_frames: int = Field(default=5, ge=1, le=60)
    max_command_hz: float = Field(default=10.0, gt=0.5, le=60.0)
    max_joint_step_deg: float = Field(default=5.0, gt=0.1, le=20.0)
    base_min_deg: float = -90.0
    base_max_deg: float = 90.0
    shoulder_min_deg: float = -20.0
    shoulder_max_deg: float = 90.0

    @field_validator("base_max_deg")
    @classmethod
    def validate_base_bounds(cls, value: float, info: object) -> float:
        """Validate that base joint max bound is above min bound.

        Args:
            value: Proposed maximum base angle.
            info: Pydantic validation context carrying sibling fields.

        Returns:
            float: Unmodified validated value.

        Raises:
            ValueError: If ``base_max_deg`` is not greater than ``base_min_deg``.
        """
        min_value = info.data.get("base_min_deg", -90.0)  # type: ignore[attr-defined]
        if value <= min_value:
            raise ValueError("base_max_deg must be greater than base_min_deg")
        return value

    @field_validator("shoulder_max_deg")
    @classmethod
    def validate_shoulder_bounds(cls, value: float, info: object) -> float:
        """Validate that shoulder joint max bound is above min bound.

        Args:
            value: Proposed maximum shoulder angle.
            info: Pydantic validation context carrying sibling fields.

        Returns:
            float: Unmodified validated value.

        Raises:
            ValueError: If ``shoulder_max_deg`` is not greater than ``shoulder_min_deg``.
        """
        min_value = info.data.get("shoulder_min_deg", -20.0)  # type: ignore[attr-defined]
        if value <= min_value:
            raise ValueError("shoulder_max_deg must be greater than shoulder_min_deg")
        return value


class RuntimeConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    preview: bool = True


ControlMode = Literal["dry-run", "live"]
