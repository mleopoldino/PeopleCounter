"""Smoke test for the main application pipeline."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def app_module(monkeypatch):
    """Provides a lightweight version of src.app with stubbed dependencies."""

    detect_stub = types.ModuleType("src.detect")

    class DummyPersonDetector:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, frame):
            return []

    detect_stub.PersonDetector = DummyPersonDetector
    monkeypatch.setitem(sys.modules, "src.detect", detect_stub)

    track_stub = types.ModuleType("src.track")

    class DummyMultiObjectTracker:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, frame, detections):
            return []

    track_stub.MultiObjectTracker = DummyMultiObjectTracker
    monkeypatch.setitem(sys.modules, "src.track", track_stub)

    draw_stub = types.ModuleType("src.draw")

    class DummyAnnotator:
        def annotate(self, frame, tracks, line_counter, roi_counter):
            return frame

    draw_stub.Annotator = DummyAnnotator
    monkeypatch.setitem(sys.modules, "src.draw", draw_stub)

    original_app = sys.modules.pop("src.app", None)
    import src.app as app

    yield app

    sys.modules.pop("src.app", None)
    if original_app is not None:
        sys.modules["src.app"] = original_app


def test_app_main_smoke(app_module, monkeypatch, capsys):
    import cv2

    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    cv2.set_video_capture_frames(frames)

    args = SimpleNamespace(
        source="dummy",
        conf=0.5,
        iou=0.4,
        device=None,
        imgsz=640,
        csv=None,
        out=None,
    )
    monkeypatch.setattr(app_module, "parse_args", lambda: args)

    app_module.main()

    captured = capsys.readouterr()
    assert "Configuração" in captured.out
    assert any("line_in" in line for line in captured.out.splitlines())
