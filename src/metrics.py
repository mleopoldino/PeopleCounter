"""
This module provides utilities for handling metrics, including timed emission and
logging to files.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


class SecondBasedEmitter:
    """
    A utility to trigger a callback function at a specified second-based interval.
    """

    def __init__(self, interval_seconds: int = 1):
        """
        Initializes the emitter.

        Args:
            interval_seconds (int): The interval in seconds to wait between emissions.
        """
        if interval_seconds < 1:
            raise ValueError("Interval must be at least 1 second.")
        self.interval_seconds = interval_seconds
        self.last_emission_time = 0

    def tick(self, callback: Callable[[], None]) -> None:
        """
        Checks if the interval has passed and triggers the callback if so.

        This method should be called in a loop (e.g., once per frame).

        Args:
            callback (Callable[[], None]): The function to call when the interval
                is reached.
        """
        current_time = time.time()
        if current_time - self.last_emission_time >= self.interval_seconds:
            callback()
            self.last_emission_time = current_time


def append_to_csv(
    file_path: Path, data_row: Dict[str, Any], header_columns: List[str]
) -> None:
    """
    Appends a data row to a CSV file.

    If the file does not exist, it is created and the header is written first.

    Args:
        file_path (Path): The path to the CSV file.
        data_row (Dict[str, Any]): The dictionary containing the data to append.
        header_columns (List[str]): The list of column names for the header.
    """
    file_exists = file_path.is_file()

    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_row)


def write_summary_json(file_path: Path, summary_data: Dict[str, Any]) -> None:
    """
    Writes a summary dictionary to a JSON file.

    Args:
        file_path (Path): The path to the output JSON file.
        summary_data (Dict[str, Any]): The data to write.
    """
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
