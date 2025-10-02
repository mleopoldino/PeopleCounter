# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
python -m src.app --source 0  # webcam
python -m src.app --source data/samples/video.mp4  # video file
```

Common flags:
- `--model yolov8n.pt` - model weights (yolov8n.pt is fastest, yolov8s.pt more accurate)
- `--conf 0.35` - detection confidence threshold (0.0-1.0)
- `--iou 0.45` - IoU threshold for NMS
- `--device cpu` or `--device cuda:0` - inference device
- `--csv metrics.csv` - log metrics to CSV
- `--out summary.json` - save final summary

### Testing
```bash
pytest -q                    # run all tests quietly
pytest tests/test_zones.py   # run specific test file
pytest -v                    # verbose output
```

### Linting & Formatting
```bash
make lint                    # check with ruff and black
ruff check src tests         # linting only
black --check src tests      # format check only
black src tests              # auto-format
```

### Makefile Shortcuts
```bash
make run    # runs src/app.py with webcam
make test   # runs pytest -q
make lint   # runs ruff and black checks
```

## Architecture

### Core Pipeline Flow
The application runs a video processing loop:
1. **app.py** - Main entry point. Orchestrates the frame loop:
   - Reads frames from video source (webcam/file/stream)
   - Calls detector → tracker → zone counters → annotator
   - Displays annotated frames with OpenCV
   - Emits metrics (JSON to stdout, optional CSV)
   - Saves final summary on exit

2. **detect.py (PersonDetector)** - Wraps YOLOv8 for person detection:
   - Uses Ultralytics YOLO model (class 0 = person)
   - Returns raw detections: list of `(x1, y1, x2, y2, conf)`

3. **track.py (MultiObjectTracker)** - Assigns stable IDs to detections:
   - Wraps `supervision.ByteTrack`
   - Input: detections from detector
   - Output: list of `Track` dicts with `{x1, y1, x2, y2, conf, id}`

4. **zones.py (LineCounter & RoiCounter)** - Implements counting logic:
   - **LineCounter**: detects when object centers cross a line (A→B or B→A)
     - Uses signed perpendicular distance to track side transitions
     - Hysteresis threshold (epsilon=2px) to prevent jitter
   - **RoiCounter**: counts objects inside a polygon ROI
     - Uses `cv2.pointPolygonTest` for inclusion test
     - Tracks both current occupancy and total unique IDs seen

5. **draw.py (Annotator)** - Centralized visualization:
   - Uses `supervision` annotators for consistency
   - Draws zones (line in red, ROI in green)
   - Draws tracks: bounding boxes (blue), ID labels, traces

6. **metrics.py** - Telemetry utilities:
   - `SecondBasedEmitter`: triggers callback at time intervals
   - CSV and JSON export functions

### Key Data Structures
- **Track**: `TypedDict` with `{x1, y1, x2, y2, conf, id}` - represents a tracked person
- **Detections**: List of tuples `(x1, y1, x2, y2, conf)` - raw YOLOv8 output
- Zone update methods return dicts:
  - LineCounter: `{"a_to_b": int, "b_to_a": int}`
  - RoiCounter: `{"present": int, "unique_ids": int}`

### Dependencies
- **ultralytics** (YOLOv8) - AGPL-3 licensed
- **supervision** - ByteTrack tracker + annotators
- **opencv-python** - video I/O and display
- **pytest** - testing framework
- **ruff** - fast Python linter
- **black** - opinionated code formatter

### Testing Structure
- `tests/test_zones.py` - unit tests for LineCounter and RoiCounter
- `tests/test_metrics.py` - unit tests for metrics utilities
- `tests/test_detect_smoke.py` - smoke test for PersonDetector
- `tests/test_pipeline_smoke.py` - end-to-end integration test
- `tests/conftest.py` - shared pytest fixtures

### Line Crossing Logic
The LineCounter uses signed perpendicular distance from the line AB:
- Side -1: below/left of line (relative to A→B direction)
- Side +1: above/right of line
- Side 0: on the line (within epsilon=2px)
- Counts increment when a track's side changes from -1 to +1 or vice versa
- Hysteresis prevents double-counting near the line

### ROI Inclusion Logic
RoiCounter uses the center of each bounding box:
- `cv2.pointPolygonTest` returns +1 (inside), -1 (outside), or 0 (on edge)
- Points on the polygon edge are considered inside (>= 0 test)
- `_seen` set accumulates all unique IDs ever inside ROI

## Important Notes
- Application requires GUI for `cv2.imshow` - not suitable for headless servers without modification
- YOLO models are downloaded automatically by Ultralytics on first use
- YOLOv8 models are AGPL-3 licensed, which affects derivative work licensing
- The tracker keeps IDs stable across frames but may reassign after occlusions
- FPS is averaged over 30 frames for stability
- Press 'q' (with video window focused) to quit gracefully
