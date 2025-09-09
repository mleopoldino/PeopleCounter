"""
track.py - Module for multi-object tracking.

This module provides the MultiObjectTracker class, which is a wrapper
around the supervision.ByteTrack algorithm to assign and maintain
stable IDs for detected objects across video frames.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict

import numpy as np
import supervision as sv

__all__ = ["MultiObjectTracker", "Track"]


class Track(TypedDict):
    """A dictionary representing a single tracked object."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    id: int


class MultiObjectTracker:
    """
    A tracker for assigning and managing stable IDs for multiple objects.

    This class wraps the supervision.ByteTrack algorithm to process detections
    from consecutive frames and maintain a consistent identity for each
    detected object.
    """

    def __init__(
        self, max_age: int = 30, min_hits: int = 1, iou_thresh: float = 0.2
    ) -> None:
        """
        Initializes the MultiObjectTracker.

        Note: The arguments are stored for potential future tuning but are not
        directly mapped to the current `sv.ByteTrack` implementation, which
        uses its own default parameters.

        Args:
            max_age (int): The maximum number of consecutive frames an object
                can be undetected before its ID is discarded. Defaults to 30.
            min_hits (int): The minimum number of consecutive frames an object
                must be detected to be considered a stable track. Defaults to 1.
            iou_thresh (float): The Intersection over Union (IoU) threshold
                used to associate detections with existing tracks. Defaults to 0.2.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self._tracker = sv.ByteTrack()

    def update(
        self, frame: np.ndarray, detections: List[Tuple[float, float, float, float, float]]
    ) -> List[Track]:
        """
        Updates the tracker with a new frame and its detections.

        Args:
            frame (np.ndarray): The current video frame. Although unused by this
                specific tracker, it's kept for API compatibility.
            detections (List[Tuple[float, float, float, float, float]]): A list
                of detections for the current frame, where each detection is a
                tuple of (x1, y1, x2, y2, confidence).

        Returns:
            List[Track]: A list of dictionaries, where each dictionary
            represents a tracked object and contains its bounding box
            coordinates, confidence score, and a unique tracking ID.
        """
        if frame is None or frame.size == 0:
            return []

        empty_detections = sv.Detections.empty()
        if not detections:
            try:
                self._tracker.update_with_detections(empty_detections)
            except AttributeError:
                self._tracker.update(with_detections=empty_detections)
            return []

        # Convert to sv.Detections, now including class_id (0=person)
        n_detections = len(detections)
        xyxy = np.array([d[:4] for d in detections], dtype=np.float32)
        confidence = np.array([d[4] for d in detections], dtype=np.float32)
        class_id = np.zeros((n_detections,), dtype=np.int32)
        detections_sv = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=class_id
        )

        # Update tracker, with fallback for older supervision versions
        try:
            tracked_detections = self._tracker.update_with_detections(detections_sv)
        except AttributeError:
            tracked_detections = self._tracker.update(with_detections=detections_sv)

        # Format output, filtering out tracks without an ID
        output: List[Track] = []
        if tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                if tracked_detections.tracker_id[i] is not None:
                    x1, y1, x2, y2 = tracked_detections.xyxy[i]
                    conf = tracked_detections.confidence[i]
                    track_id = tracked_detections.tracker_id[i]

                    output.append(
                        {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "conf": float(conf),
                            "id": int(track_id),
                        }
                    )
        return output
