"""Tests for zone counters (line and ROI)."""

import pytest

from src.zones import LineCounter, RoiCounter


@pytest.fixture
def horizontal_line_counter() -> LineCounter:
    """Line across the X axis from (0, 0) to (100, 0)."""

    return LineCounter(point_a=(0, 0), point_b=(100, 0))


@pytest.fixture
def square_roi_counter() -> RoiCounter:
    """Square ROI with corners (0, 0) - (100, 100)."""

    return RoiCounter(polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])


def test_line_counter_counts_a_to_b(horizontal_line_counter, track_factory):
    """Track crossing from negative side to positive increments a_to_b."""

    counts = horizontal_line_counter.update([track_factory(1, (10, -20))])
    assert counts == {"a_to_b": 0, "b_to_a": 0}

    counts = horizontal_line_counter.update([track_factory(1, (12, 25))])
    assert counts == {"a_to_b": 1, "b_to_a": 0}


def test_line_counter_counts_b_to_a(horizontal_line_counter, track_factory):
    """Track crossing the other way increments b_to_a."""

    counts = horizontal_line_counter.update([track_factory(2, (50, 20))])
    assert counts == {"a_to_b": 0, "b_to_a": 0}

    counts = horizontal_line_counter.update([track_factory(2, (52, -30))])
    assert counts == {"a_to_b": 0, "b_to_a": 1}


def test_line_counter_no_false_positive(horizontal_line_counter, track_factory):
    """Multiple updates on same side do not create extra counts."""

    horizontal_line_counter.update([track_factory(3, (60, -30))])
    counts = horizontal_line_counter.update([track_factory(3, (65, -10))])
    assert counts == {"a_to_b": 0, "b_to_a": 0}


def test_roi_counter_inside_outside(square_roi_counter, track_factory):
    """ROI counter tracks presence and unique visitors."""

    # Enter ROI
    result = square_roi_counter.update([track_factory(7, (50, 50))])
    assert result == {"present": 1, "unique_ids": 1}

    # Same track stays inside; unique should remain 1
    result = square_roi_counter.update([track_factory(7, (55, 55))])
    assert result == {"present": 1, "unique_ids": 1}

    # Track leaves ROI; present drops to 0 but unique count persists
    result = square_roi_counter.update([track_factory(7, (150, 150))])
    assert result == {"present": 0, "unique_ids": 1}


def test_roi_counter_counts_edge_as_inside(square_roi_counter, track_factory):
    """Points on the polygon edge are inclusive."""

    result = square_roi_counter.update([track_factory(8, (0, 40))])
    assert result == {"present": 1, "unique_ids": 1}
