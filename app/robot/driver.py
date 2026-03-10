from __future__ import annotations

from abc import ABC, abstractmethod


class RobotArmDriver(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def home(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_axis(self, axis: str, delta: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_gripper(self, open: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop_all(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError
