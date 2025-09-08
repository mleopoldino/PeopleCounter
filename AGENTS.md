# Repository Guidelines

## Project Structure & Module Organization
- Root: Make targets, configs, and docs.
- `src/app.py`: entry point (webcam/video loop). Place new code under `src/` as packages (e.g., `src/people_counter/…`).
- `tests/`: pytest tests (start with `test_*.py`).
- `data/samples/`: small demo videos; do not commit large binaries.
- `pyproject.toml`: formatter/linter configuration.

## Build, Test, and Development Commands
- `make run`: runs the demo (`python src/app.py --source 0`).
- `python src/app.py --source <0|path|url>`: run with webcam, file, or stream.
- `make test`: execute unit tests via pytest.
- `make lint`: run Ruff checks and Black in check mode.

## Coding Style & Naming Conventions
- Python 3.10; line length 88 (Black/Ruff).
- Use packages/modules under `src/people_counter/` (e.g., `detector.py`, `tracker.py`, `counters/line.py`).
- Filenames: `snake_case.py`; classes: `PascalCase`; functions/vars: `snake_case`.
- Keep functions small; prefer pure helpers where possible.
- Type hints encouraged for new/changed code; add concise docstrings.

## Testing Guidelines
- Framework: pytest. Place tests in `tests/` named `test_*.py`.
- Write tests for new behavior and bug fixes; include edge cases (empty frames, invalid sources).
- Run locally with `make test`. Aim for fast tests; skip GPU-only paths.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (e.g., "add line counter"), optional body for rationale.
- Prefer focused commits with passing tests/lints.
- PRs: include summary, rationale, and how to test (commands, sample inputs). Attach screenshots/GIFs for UI windows when relevant.
- Link related issues and note any follow-ups or known limitations.

## Security & Configuration Tips
- Dependencies: see `requirements.txt`. On Linux, install `ffmpeg` and `libgl1` if OpenCV/YOLO needs them.
- Do not commit secrets or large media files. Use `.env` locally if needed (not tracked).
- First YOLO model download occurs at runtime; avoid bundling weights in the repo.
- On macOS, grant camera permissions if using `--source 0`.

## Notes
- Keep the demo runnable at all times. If adding heavier features, guard them behind flags and provide sensible defaults.
