"""
detect.py - Module for person detection.

This module provides the PersonDetector class, which is a wrapper around a
computer vision model (e.g., YOLOv8) to detect people in an image frame.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

__all__ = ["PersonDetector"]


class PersonDetector:
    """
    A detector for identifying people in video frames.

    This class wraps a detection model, processes input frames, and returns
    bounding boxes for detected individuals of the 'person' class.
    """

    def __init__(
        self,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the PersonDetector.

        Args:
            conf (float): The confidence threshold for a detection to be
                considered valid. Defaults to 0.35.
            iou (float): The Intersection over Union (IoU) threshold for
                Non-Maximum Suppression (NMS). Defaults to 0.45.
            imgsz (int): The image size for model inference. Defaults to 640.
            device (Optional[str]): The device to run the model on,
                e.g., 'cpu' or 'cuda:0'. If None, the model's default
                is used. Defaults to None.
        """
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO("yolov8n.pt")
        self.model.fuse()
        if self.device:
            try:
                self.model.to(self.device)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to move model to device %s: %s", self.device, e
                )

    def predict(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Detects people in a single video frame.

        Args:
            frame (np.ndarray): The input video frame as a NumPy array
                (in BGR or RGB format).

        Returns:
            List[Tuple[float, float, float, float, float]]: A list of
            detections. Each detection is a tuple containing the bounding
            box coordinates (x1, y1, x2, y2) and the confidence score.
            Example: [(x1, y1, x2, y2, conf), ...].
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model(
            frame,
            imgsz=self.imgsz,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        detections = sv.Detections.from_ultralytics(results[0])

        # Redundant filter for safety, as 'classes=[0]' should already handle this
        person_detections = detections[detections.class_id == 0]

        output: List[Tuple[float, float, float, float, float]] = []
        for xyxy, confidence in zip(
            person_detections.xyxy, person_detections.confidence
        ):
            x1, y1, x2, y2 = xyxy
            output.append(
                (float(x1), float(y1), float(x2), float(y2), float(confidence))
            )

        return output