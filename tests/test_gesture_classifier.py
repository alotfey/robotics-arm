from __future__ import annotations

from dataclasses import dataclass

from app.models import GestureName
from app.vision.gestures import GestureClassifier


@dataclass
class FakePoint:
    x: float
    y: float


def _make_points() -> list[FakePoint]:
    return [FakePoint(0.5, 0.5) for _ in range(21)]


def test_classify_grip_toggle_closed_fist() -> None:
    pts = _make_points()
    # all fingers folded: tips below PIP joints
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        pts[tip].y = 0.7
        pts[pip].y = 0.6

    label, confidence = GestureClassifier._classify(pts)

    assert label == GestureName.GRIP_TOGGLE
    assert confidence > 0.8


def test_classify_up_open_hand() -> None:
    pts = _make_points()
    # open fingers
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        pts[tip].y = 0.2
        pts[pip].y = 0.4
    # wrist lower than middle tip to indicate up
    pts[0].y = 0.6
    pts[12].y = 0.2

    label, confidence = GestureClassifier._classify(pts)

    assert label == GestureName.UP
    assert confidence >= 0.6


def test_classify_left_pointing() -> None:
    pts = _make_points()
    pts[0].x = 0.6
    pts[8].x = 0.4
    pts[8].y = 0.2
    pts[6].y = 0.4
    # Other fingers folded
    for tip, pip in [(12, 10), (16, 14), (20, 18)]:
        pts[tip].y = 0.8
        pts[pip].y = 0.6

    label, _ = GestureClassifier._classify(pts)
    assert label == GestureName.LEFT
