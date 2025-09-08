"""
test_detect_smoke.py - Smoke tests for the PersonDetector class.

This test file uses mocking to isolate the PersonDetector from actual
model loading (ultralytics) and data conversion (supervision), ensuring
that no network access or file downloads occur.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# This import works if pytest is run from the project root directory.
from src.detect import PersonDetector


@pytest.fixture
def mock_ultralytics():
    """Mocks the ultralytics.YOLO model to prevent model downloads."""
    with patch("src.detect.YOLO") as mock_yolo_class:
        mock_model_instance = MagicMock()
        # The result of calling the model instance is a list of result objects
        mock_model_instance.return_value = [MagicMock()]
        mock_yolo_class.return_value = mock_model_instance
        yield mock_model_instance


@pytest.fixture
def mock_supervision():
    """Mocks supervision.Detections.from_ultralytics."""
    with patch("src.detect.sv.Detections.from_ultralytics") as mock_from_ultralytics:
        yield mock_from_ultralytics


def test_predict_with_empty_or_none_frame(mock_ultralytics, mock_supervision):
    """
    Tests that predict returns an empty list for None or empty frames
    and that the underlying model is not called.
    """
    detector = PersonDetector()
    assert detector.predict(None) == []
    assert detector.predict(np.array([])) == []
    mock_ultralytics.assert_not_called()


def test_predict_filters_person_class(mock_ultralytics, mock_supervision):
    """
    Tests that predict correctly filters for the 'person' class (id 0)
    and returns data in the expected float tuple format.
    """
    # 1. Setup Mock Data for supervision.Detections.from_ultralytics
    mock_detections = MagicMock()
    mock_detections.xyxy = np.array([[10, 20, 30, 40], [1, 2, 3, 4]])
    mock_detections.confidence = np.array([0.95, 0.5])
    mock_detections.class_id = np.array([0, 1])  # class 0 (person) and class 1

    # 2. Simulate the filtering behavior: detections[detections.class_id == 0]
    def getitem_side_effect(mask):
        # This is a simplified simulation of boolean array indexing on the mock
        if isinstance(mask, np.ndarray) and mask.dtype == bool and mask.sum() == 1:
            filtered_detections = MagicMock()
            filtered_detections.xyxy = np.array([mock_detections.xyxy[0]])
            filtered_detections.confidence = np.array([mock_detections.confidence[0]])
            return filtered_detections
        return MagicMock()  # Return an empty mock for any other case

    mock_detections.__getitem__.side_effect = getitem_side_effect
    mock_supervision.return_value = mock_detections

    # 3. Run Test
    detector = PersonDetector()
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    predictions = detector.predict(dummy_frame)

    # 4. Assert
    expected = [(10.0, 20.0, 30.0, 40.0, 0.95)]
    assert predictions == expected

    # Verify that the underlying model and sv converter were called once
    mock_ultralytics.assert_called_once()
    mock_supervision.assert_called_once()
