"""
zones.py - Defines counting zones (Line, ROI) for object tracking.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    from .track import Track
except ImportError:
    from track import Track

__all__ = ["LineCounter", "RoiCounter"]


class LineCounter:
    """
    Counts objects crossing a defined line segment.

    The line is oriented from `point_a` to `point_b`. For each track, the
    center of its bounding box is evaluated against the line using the signed
    perpendicular distance. Update returns per-frame counts (deltas):

    - a_to_b: transition from side -1 to +1
    - b_to_a: transition from side +1 to -1
    """

    def __init__(self, point_a: Tuple[int, int], point_b: Tuple[int, int]) -> None:
        """
        Initializes the line counter.

        Args:
            point_a (Tuple[int, int]): The starting point of the line (x, y).
            point_b (Tuple[int, int]): The ending point of the line (x, y).
        """
        self.point_a = point_a
        self.point_b = point_b
        self._side_by_id: Dict[int, int] = {}

        # Precompute geometry helpers
        ax, ay = float(point_a[0]), float(point_a[1])
        bx, by = float(point_b[0]), float(point_b[1])
        self._a = np.array([ax, ay], dtype=np.float32)
        self._b = np.array([bx, by], dtype=np.float32)
        self._ab = self._b - self._a
        self._ab_len = float(np.linalg.norm(self._ab))
        if self._ab_len == 0.0:
            raise ValueError("LineCounter requires two distinct points to define a line")

        # Pixel threshold: treat points closer than eps to the line as 'on the line'
        self._eps: float = 2.0

    def update(self, tracks: List[Track]) -> Dict[str, int]:
        """
        Updates the counter based on the latest track positions.

        Args:
            tracks (List[Track]): A list of tracked objects produced by the
                tracker, where each Track contains bounding box coordinates
                (`x1`, `y1`, `x2`, `y2`), a `conf` score, and an integer `id`.

        Returns:
            Dict[str, int]: A dictionary with counts for each direction using
                canonical keys: {"a_to_b": int, "b_to_a": int}.
        """
        counts = {"a_to_b": 0, "b_to_a": 0}
        for track in tracks:
            if isinstance(track, dict):
                track_id = int(track["id"])
                x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            else:
                track_id = int(track.id)
                x1, y1, x2, y2 = track.x1, track.y1, track.x2, track.y2

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Signed perpendicular distance from center to line AB
            apx, apy = float(cx) - float(self._a[0]), float(cy) - float(self._a[1])
            cross = self._ab[0] * apy - self._ab[1] * apx
            d = float(cross) / self._ab_len

            # Map to side with hysteresis epsilon
            side = 1 if d >= self._eps else (-1 if d <= -self._eps else 0)
            if side == 0:
                # Do not change last side when on the line to avoid jitter
                old_side = self._side_by_id.get(track_id)
                if old_side is None:
                    self._side_by_id[track_id] = 0
                continue

            old_side = self._side_by_id.get(track_id)

            if old_side is not None and old_side != side:
                if old_side == -1 and side == 1:
                    counts["a_to_b"] += 1
                elif old_side == 1 and side == -1:
                    counts["b_to_a"] += 1

            self._side_by_id[track_id] = side

        return counts


class RoiCounter:
    """
    Counts objects currently inside a defined Region of Interest (ROI).

    A point on any polygon edge is considered inside (inclusive edges).
    """

    def __init__(self, polygon: List[Tuple[int, int]]) -> None:
        """
        Initializes the ROI counter.

        Args:
            polygon (List[Tuple[int, int]]): A list of (x, y) vertices
                                             defining the polygon ROI.
        """
        if not polygon or len(polygon) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        self.polygon = np.array(polygon, dtype=np.int32)
        self._seen: set[int] = set()

    def update(self, tracks: List[Track]) -> Dict[str, int]:
        """
        Updates the counter based on the latest track positions.

        Args:
            tracks (List[Track]): A list of tracked objects produced by the
                tracker, where each Track contains bounding box coordinates
                (`x1`, `y1`, `x2`, `y2`), a `conf` score, and an integer `id`.

        Returns:
            Dict[str, int]: A dictionary with the count of objects currently
                present inside the ROI and the total number of unique track IDs
                that have ever been inside the ROI, using canonical keys:
                {"present": int, "unique_ids": int}.
        """
        present_ids: set[int] = set()
        for track in tracks:
            if isinstance(track, dict):
                track_id = int(track["id"])
                x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            else:
                track_id = int(track.id)
                x1, y1, x2, y2 = track.x1, track.y1, track.x2, track.y2

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # pointPolygonTest returns +1 (inside), -1 (outside), or 0 (on edge)
            is_inside = cv2.pointPolygonTest(self.polygon, (cx, cy), False) >= 0

            if is_inside:
                present_ids.add(track_id)
                self._seen.add(track_id)

        unique_ids = len(self._seen)
        return {"present": len(present_ids), "unique_ids": unique_ids}
