from __future__ import annotations

from abc import ABC, abstractmethod


class RobotArmDriver(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Open any resources required to control the robot arm."""
        raise NotImplementedError

    @abstractmethod
    def home(self) -> None:
        """Move the robot arm to its safe home pose."""
        raise NotImplementedError

    @abstractmethod
    def move_axis(self, axis: str, delta: float) -> None:
        """Move a single robot axis by a relative delta.

        Args:
            axis: Driver-specific axis identifier.
            delta: Relative movement amount in degrees.
        """
        raise NotImplementedError

    @abstractmethod
    def set_gripper(self, open: bool) -> None:
        """Set the gripper open or closed state.

        Args:
            open: ``True`` to open, ``False`` to close.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_all(self) -> None:
        """Stop all active robot motion immediately."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Release driver resources and close any open connections."""
        raise NotImplementedError
