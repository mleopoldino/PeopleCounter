"""
draw.py - Manages all frame annotations for visualization.

This module centralizes drawing logic using the 'supervision' library's
annotators for consistency and clarity.
"""

from __future__ import annotations

from typing import List

import numpy as np
import supervision as sv

# Custom classes
from .track import Track
from .zones import LineCounter, RoiCounter


class Annotator:
    """A wrapper for drawing all visual elements on a frame."""

    def __init__(self):
        """Initializes all necessary annotators."""
        # Zone annotators
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, color=sv.Color.RED)
        self.roi_annotator = sv.PolygonZoneAnnotator(
            zone=sv.PolygonZone(
                polygon=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # Placeholder
            ),
            color=sv.Color.GREEN,
            thickness=2,
            text_thickness=1,
            text_scale=0.5,
        )
        # Track annotators
        self.box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.BLUE,
            text_color=sv.Color.WHITE,
            text_position=sv.Position.TOP_LEFT,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=sv.Color.BLUE,
            position=sv.Position.CENTER,
            trace_length=30,
            thickness=2,
        )

    def annotate(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        line_counter: LineCounter,
        roi_counter: RoiCounter,
    ) -> np.ndarray:
        """
        Draws all annotations (zones, tracks, boxes, labels) on the frame.

        Args:
            frame (np.ndarray): The input frame to draw on.
            tracks (List[Track]): A list of tracked objects.
            line_counter (LineCounter): The line counter instance.
            roi_counter (RoiCounter): The ROI counter instance.

        Returns:
            np.ndarray: The frame with all elements annotated.
        """
        # Annotate zones first, so they are in the background
        line_zone = sv.LineZone(
            start=sv.Point(x=line_counter.point_a[0], y=line_counter.point_a[1]),
            end=sv.Point(x=line_counter.point_b[0], y=line_counter.point_b[1]),
        )
        self.line_annotator.annotate(frame=frame, line_counter=line_zone)

        self.roi_annotator.zone.polygon = roi_counter.polygon
        frame = self.roi_annotator.annotate(scene=frame)

        # Convert our custom Track list to a supervision.Detections object
        if tracks:
            xyxy = np.array([[t["x1"], t["y1"], t["x2"], t["y2"]] for t in tracks])
            tracker_ids = np.array([t["id"] for t in tracks])
            # Since all tracks are people, we can assign class_id=0
            class_ids = np.zeros(len(tracks), dtype=int)
            detections = sv.Detections(
                xyxy=xyxy, tracker_id=tracker_ids, class_id=class_ids
            )

            # Create labels for each track
            labels = [f"ID:{tid}" for tid in detections.tracker_id]

            # Annotate traces, then boxes and labels on top
            frame = self.trace_annotator.annotate(scene=frame, detections=detections)
            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            frame = self.label_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

        return frame
