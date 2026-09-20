"""Thread-safe fixed-capacity ring buffer for multi-channel sample data."""

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    """Fixed-capacity circular buffer holding samples as columns.

    Internal storage shape is (n_channels, capacity). Safe to push from
    one (acquisition) thread while reading from another (e.g. GUI/plot
    thread) - all operations are protected by a lock.
    """

    def __init__(self, n_channels: int, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.n_channels = n_channels
        self.capacity = capacity
        self._data = np.zeros((n_channels, capacity), dtype=np.float64)
        self._write_idx = 0
        self._count = 0
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray):
        """Append new samples. chunk shape: (n_channels, n_samples)."""
        if chunk.size == 0:
            return
        if chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {chunk.shape[0]}"
            )
        n = chunk.shape[1]
        with self._lock:
            if n >= self.capacity:
                # the chunk alone is bigger than the whole buffer - just keep the tail
                self._data[:] = chunk[:, -self.capacity:]
                self._write_idx = 0
                self._count = self.capacity
                return
            end = self._write_idx + n
            if end <= self.capacity:
                self._data[:, self._write_idx:end] = chunk
            else:
                first = self.capacity - self._write_idx
                self._data[:, self._write_idx:] = chunk[:, :first]
                self._data[:, : end - self.capacity] = chunk[:, first:]
            self._write_idx = end % self.capacity
            self._count = min(self.capacity, self._count + n)


    def get_latest(self, n: Optional[int] = None) -> np.ndarray:
        """Return the most recent `n` samples (or everything buffered if
        n is None), oldest-first, shape (n_channels, n)."""
        with self._lock:
            count = self._count if n is None else min(n, self._count)
            if count == 0:
                return np.zeros((self.n_channels, 0))
            start = (self._write_idx - count) % self.capacity
            if start + count <= self.capacity:
                return self._data[:, start:start + count].copy()
            first = self.capacity - start
            return np.concatenate(
                [self._data[:, start:], self._data[:, : count - first]], axis=1
            )
        

    def clear(self):
        with self._lock:
            self._data[:] = 0.0
            self._write_idx = 0
            self._count = 0
            

    def __len__(self):
        with self._lock:
            return self._count