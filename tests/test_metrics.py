"""Tests for metrics utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import pytest

from src.metrics import SecondBasedEmitter, append_to_csv, write_summary_json


@pytest.fixture
def time_sequence(monkeypatch) -> Iterator[float]:
    """Produces sequential values for time.time when iterated."""

    sequence = iter([0.0, 0.5, 1.2, 1.8, 2.3])

    def fake_time() -> float:
        return next(sequence)

    monkeypatch.setattr("src.metrics.time.time", fake_time)
    return sequence


def test_second_based_emitter_triggers_once_per_second(time_sequence) -> None:
    calls: list[int] = []
    emitter = SecondBasedEmitter(interval_seconds=1)

    def callback() -> None:
        calls.append(1)

    for _ in range(5):
        try:
            emitter.tick(callback)
        except StopIteration:
            break

    assert calls == [1, 1]


def test_append_to_csv_creates_header_and_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    header = ["timestamp", "line_in", "line_out"]

    append_to_csv(csv_path, {"timestamp": 1, "line_in": 2, "line_out": 3}, header)
    append_to_csv(csv_path, {"timestamp": 4, "line_in": 5, "line_out": 6}, header)

    with csv_path.open("r", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))

    assert rows == [header, ["1", "2", "3"], ["4", "5", "6"]]


def test_write_summary_json(tmp_path: Path) -> None:
    json_path = tmp_path / "summary.json"
    payload = {"total_frames": 10, "line_in": 3, "line_out": 1}

    write_summary_json(json_path, payload)

    with json_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    assert data == payload

    contents = json_path.read_text(encoding="utf-8")
    assert "\n    " in contents  # indent=4 produces spaced lines
