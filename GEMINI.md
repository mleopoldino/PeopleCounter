## Project Overview

This is a Python project for real-time people counting using YOLOv8 and ByteTrack. It can process video from a webcam, video files, or streams. The main features are:

*   **Person Detection:** Uses YOLOv8 for detecting people in each frame.
*   **Object Tracking:** Employs ByteTrack via the `supervision` library for tracking detected individuals.
*   **Counting Modes:** 
    *   **Line Crossing:** Counts entries and exits across a defined line.
    *   **Region of Interest (ROI):** Monitors the current occupancy and unique IDs within a designated area.
*   **Real-time Visualization:** Uses OpenCV to display the video feed with tracking information.
*   **Data Logging:** Periodically logs metrics in JSON format with an option to export to CSV.

The project is structured to be extensible, with plans for features like annotated video export, a REST API for real-time metrics, and dashboards.

## Building and Running

The project uses a `Makefile` to simplify common tasks.

*   **Run the application (webcam):**
    ```bash
    make run
    ```
    or
    ```bash
    python src/app.py --source 0
    ```
*   **Run tests:**
    ```bash
    make test
    ```
*   **Run linter:**
    ```bash
    make lint
    ```

## Development Conventions

*   **Code Formatting:** The project uses `black` for code formatting and `isort` for import sorting. Configuration is in `pyproject.toml`.
*   **Linting:** `ruff` is used for linting, with rules defined in `pyproject.toml`.
*   **Dependencies:** Project dependencies are managed in `requirements.txt`.
*   **Testing:** `pytest` is the testing framework.
