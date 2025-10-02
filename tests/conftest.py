import sys
import types
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest


def _point_on_segment(
    point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]
) -> bool:
    """Returns True if a point lies on a segment (inclusive)."""

    px, py = point
    x1, y1 = start
    x2, y2 = end

    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross) > 1e-6:
        return False

    dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
    return dot <= 0


def _point_polygon_test(
    polygon: np.ndarray, point: Tuple[float, float], _: bool
) -> int:
    """Simplistic substitute for cv2.pointPolygonTest."""

    coords: List[Tuple[float, float]] = [tuple(map(float, vert)) for vert in polygon]
    px, py = point

    for i in range(len(coords)):
        start = coords[i]
        end = coords[(i + 1) % len(coords)]
        if _point_on_segment((px, py), start, end):
            return 0

    inside = False
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]

        intersects = ((y1 > py) != (y2 > py)) and (
            px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-9) + x1
        )
        if intersects:
            inside = not inside

    return 1 if inside else -1


_CV2_STATE = {"frames": []}


def _set_video_capture_frames(frames: List[np.ndarray]) -> None:
    """Stores frames for the fake VideoCapture to emit."""

    _CV2_STATE["frames"] = [np.array(frame, copy=True) for frame in frames]


class _FakeVideoCapture:
    def __init__(self, source: object) -> None:
        self._frames = [frame.copy() for frame in _CV2_STATE["frames"]]
        self._index = 0
        self._opened = True
        self.source = source

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._index < len(self._frames):
            frame = self._frames[self._index]
            self._index += 1
            return True, frame.copy()
        return False, None

    def release(self) -> None:
        self._opened = False

    def get(self, prop: int) -> float:
        return 30.0 if prop == 5 else 0.0  # CAP_PROP_FPS = 5


class _FakeVideoWriter:
    def __init__(self, filename, fourcc, fps, frameSize):
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.frameSize = frameSize
        self._opened = True

    def write(self, frame: np.ndarray) -> None:
        pass

    def release(self) -> None:
        self._opened = False


cv2_stub = types.SimpleNamespace(
    VideoCapture=_FakeVideoCapture,
    VideoWriter=_FakeVideoWriter,
    VideoWriter_fourcc=lambda *args: 0x00000021,
    imshow=lambda *_args, **_kwargs: None,
    waitKey=lambda _timeout: -1,
    destroyAllWindows=lambda: None,
    pointPolygonTest=_point_polygon_test,
    putText=lambda img, *args, **kwargs: img,
    setNumThreads=lambda _n: None,
    imread=lambda *_args, **_kwargs: np.zeros((1, 1, 3), dtype=np.uint8),
    imwrite=lambda *_args, **_kwargs: True,
    FONT_HERSHEY_SIMPLEX=0,
    LINE_AA=16,
    IMREAD_COLOR=1,
    IMREAD_UNCHANGED=-1,
    CAP_PROP_FPS=5,
    set_video_capture_frames=_set_video_capture_frames,
)


sys.modules.setdefault("cv2", cv2_stub)

# Adiciona a raiz do projeto ao sys.path (…/PeopleCounter)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def track_factory() -> "TrackFactory":
    """Helper fixture to build dict-based tracks with a given center point."""

    def _build(
        track_id: int,
        center: Tuple[float, float],
        box_size: Tuple[float, float] = (4.0, 4.0),
    ) -> Dict[str, float]:
        half_w = box_size[0] / 2
        half_h = box_size[1] / 2
        cx, cy = center
        return {
            "id": track_id,
            "x1": cx - half_w,
            "y1": cy - half_h,
            "x2": cx + half_w,
            "y2": cy + half_h,
            "conf": 0.9,
        }

    return _build


TrackFactory = Callable[
    [int, Tuple[float, float], Tuple[float, float]], Dict[str, float]
]
