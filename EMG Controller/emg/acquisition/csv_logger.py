"""Simple thread-safe append-only CSV logger."""

import csv
import os
import threading
from typing import Iterable, Sequence


class CSVLogger:
    def __init__(self, filepath: str, header: Sequence[str]):
        self.filepath = filepath
        self._lock = threading.Lock()
        is_new = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        self._file = open(filepath, "a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(header)
            self._file.flush()

    def write_rows(self, rows: Iterable[Sequence]):
        if not rows:
            return
        with self._lock:
            self._writer.writerows(rows)
            self._file.flush()

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.close()