"""Tests for app.py validation functions."""

from src.app import validate_model_exists


def test_validate_model_exists_with_standard_models():
    """Standard Ultralytics model names should be valid."""
    assert validate_model_exists("yolov8n.pt") is True
    assert validate_model_exists("yolov8s.pt") is True
    assert validate_model_exists("yolov8m.pt") is True
    assert validate_model_exists("yolov8l.pt") is True
    assert validate_model_exists("yolov8x.pt") is True


def test_validate_model_exists_with_nonexistent_model():
    """Non-existent model names should be invalid."""
    assert validate_model_exists("nonexistent_model.pt") is False
    assert validate_model_exists("yolov9z.pt") is False


def test_validate_model_exists_with_local_file(tmp_path):
    """Local model files should be valid if they exist."""
    model_file = tmp_path / "custom_model.pt"
    model_file.touch()

    assert validate_model_exists(str(model_file)) is True
    assert validate_model_exists(str(tmp_path / "missing.pt")) is False
