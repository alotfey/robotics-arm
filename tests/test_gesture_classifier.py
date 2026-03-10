from __future__ import annotations

from dataclasses import dataclass

from app.models import GestureName
from app.vision.gestures import GestureClassifier


@dataclass
class FakePoint:
    x: float
    y: float


def _make_points() -> list[FakePoint]:
    """Create a plausible baseline hand landmark set for tests.

    Returns:
        list[FakePoint]: Landmark list indexed like MediaPipe hand points.
    """
    pts = [FakePoint(0.5, 0.5) for _ in range(21)]

    # Wrist and knuckles.
    pts[0] = FakePoint(0.50, 0.78)
    pts[5] = FakePoint(0.42, 0.62)   # index_mcp
    pts[9] = FakePoint(0.50, 0.60)   # middle_mcp
    pts[13] = FakePoint(0.58, 0.62)  # ring_mcp
    pts[17] = FakePoint(0.64, 0.66)  # pinky_mcp

    # Thumb baseline.
    pts[1] = FakePoint(0.44, 0.72)
    pts[2] = FakePoint(0.40, 0.69)
    pts[3] = FakePoint(0.37, 0.65)
    pts[4] = FakePoint(0.34, 0.61)

    # Open-ish default finger pose.
    _set_open_vector(pts, (8, 7, 6, 5), -0.08, -0.30)
    _set_open_vector(pts, (12, 11, 10, 9), 0.00, -0.33)
    _set_open_vector(pts, (16, 15, 14, 13), 0.08, -0.30)
    _set_open_vector(pts, (20, 19, 18, 17), 0.12, -0.24)
    return pts


def _set_open_vector(
    pts: list[FakePoint],
    finger: tuple[int, int, int, int],
    dx: float,
    dy: float,
) -> None:
    """Set one finger joints along an open directional vector.

    Args:
        pts: Landmark array to mutate.
        finger: Landmark indices in ``(tip, dip, pip, mcp)`` order.
        dx: X displacement from MCP to tip.
        dy: Y displacement from MCP to tip.
    """
    tip, dip, pip, mcp = finger
    mcp_pt = pts[mcp]
    pts[pip] = FakePoint(mcp_pt.x + dx * 0.45, mcp_pt.y + dy * 0.45)
    pts[dip] = FakePoint(mcp_pt.x + dx * 0.70, mcp_pt.y + dy * 0.70)
    pts[tip] = FakePoint(mcp_pt.x + dx, mcp_pt.y + dy)


def _set_closed_fist(pts: list[FakePoint]) -> None:
    """Fold all fingers toward the wrist to simulate a fist.

    Args:
        pts: Landmark array to mutate.
    """
    wrist = pts[0]
    for tip, dip, pip, mcp in [(8, 7, 6, 5), (12, 11, 10, 9), (16, 15, 14, 13), (20, 19, 18, 17)]:
        mcp_pt = pts[mcp]
        vx = wrist.x - mcp_pt.x
        vy = wrist.y - mcp_pt.y
        pts[pip] = FakePoint(mcp_pt.x + vx * 0.82, mcp_pt.y + vy * 0.82)
        pts[dip] = FakePoint(mcp_pt.x + vx * 0.90, mcp_pt.y + vy * 0.90)
        pts[tip] = FakePoint(mcp_pt.x + vx * 0.97, mcp_pt.y + vy * 0.97)

    # Fold thumb toward palm center.
    pts[4] = FakePoint(0.46, 0.74)
    pts[3] = FakePoint(0.44, 0.72)
    pts[2] = FakePoint(0.42, 0.70)


def test_classify_grip_toggle_closed_fist() -> None:
    """Closed fist is classified as grip toggle."""
    pts = _make_points()
    _set_closed_fist(pts)

    label, confidence = GestureClassifier._classify(pts)

    assert label == GestureName.GRIP_TOGGLE
    assert confidence >= 0.75


def test_classify_up_open_hand() -> None:
    """Open hand facing up is classified as UP."""
    pts = _make_points()
    _set_open_vector(pts, (8, 7, 6, 5), -0.08, -0.32)
    _set_open_vector(pts, (12, 11, 10, 9), 0.00, -0.35)
    _set_open_vector(pts, (16, 15, 14, 13), 0.08, -0.32)
    _set_open_vector(pts, (20, 19, 18, 17), 0.12, -0.25)

    label, confidence = GestureClassifier._classify(pts)

    assert label == GestureName.UP
    assert confidence >= 0.6


def test_classify_down_open_hand() -> None:
    """Open hand facing down is classified as DOWN."""
    pts = _make_points()
    _set_open_vector(pts, (8, 7, 6, 5), -0.06, 0.28)
    _set_open_vector(pts, (12, 11, 10, 9), 0.00, 0.32)
    _set_open_vector(pts, (16, 15, 14, 13), 0.06, 0.28)
    _set_open_vector(pts, (20, 19, 18, 17), 0.10, 0.24)

    label, confidence = GestureClassifier._classify(pts)

    assert label == GestureName.DOWN
    assert confidence >= 0.6


def test_classify_right_open_hand() -> None:
    """Open hand facing right is classified as RIGHT."""
    pts = _make_points()
    _set_open_vector(pts, (8, 7, 6, 5), 0.26, -0.03)
    _set_open_vector(pts, (12, 11, 10, 9), 0.28, -0.02)
    _set_open_vector(pts, (16, 15, 14, 13), 0.26, 0.00)
    _set_open_vector(pts, (20, 19, 18, 17), 0.22, 0.02)

    label, confidence = GestureClassifier._classify(pts)

    assert label == GestureName.RIGHT
    assert confidence >= 0.6


def test_classify_left_pointing() -> None:
    """Index-only pointing left is classified as LEFT."""
    pts = _make_points()
    _set_closed_fist(pts)
    _set_open_vector(pts, (8, 7, 6, 5), -0.30, -0.03)

    label, _ = GestureClassifier._classify(pts)
    assert label == GestureName.LEFT


def test_classify_ambiguous_open_hand_returns_none() -> None:
    """Diagonal ambiguous open-hand pose does not force a command."""
    pts = _make_points()
    _set_open_vector(pts, (8, 7, 6, 5), 0.22, -0.22)
    _set_open_vector(pts, (12, 11, 10, 9), 0.23, -0.23)
    _set_open_vector(pts, (16, 15, 14, 13), 0.22, -0.22)
    _set_open_vector(pts, (20, 19, 18, 17), 0.20, -0.20)

    label, confidence = GestureClassifier._classify(pts)

    assert label is None
    assert confidence == 0.0


def test_classify_partial_fist_is_not_grip_toggle() -> None:
    """Partially open fist does not trigger grip toggle."""
    pts = _make_points()
    _set_closed_fist(pts)
    _set_open_vector(pts, (8, 7, 6, 5), 0.00, -0.30)

    label, _ = GestureClassifier._classify(pts)

    assert label != GestureName.GRIP_TOGGLE
