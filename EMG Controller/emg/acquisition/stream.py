"""EMGStream: continuously pulls data from a MindRove device, buffers it,
filters it, and logs it to CSV - all in a background thread."""

import threading
import time
from typing import Optional, Tuple

import numpy as np

from emg.acquisition.device import MindRove
from emg.acquisition.ring_buffer import RingBuffer
from emg.acquisition.filters import EMGFilter
from emg.acquisition.csv_logger import CSVLogger


class EMGStream:
    """
    Owns the acquisition loop. On start(), spins up a background thread that:
      1. calls device.pull_chunk() (drains the SDK's internal ring buffer)
      2. pushes the raw EMG samples into `self.RawBuffer`
      3. filters them and pushes the result into `self.FilteredBuffer`
      4. appends every sample (timestamp, raw EMG channels, marker) to CSV

    Events / markers:
      Call `stream.mark_event(code)` (from any thread, e.g. your GUI) when
      something happens. That forwards straight to the board's own
      insert_marker(), so the marker is embedded in the board's own
      data stream on the board's own clock - it will show up in the
      `marker` column of the CSV on the correct sample, no manual
      timestamp alignment needed. Note: the marker attaches to the *next*
      sample the board captures after the call, so expect it to land
      within roughly one sample period of when you called mark_event.
    """

    def __init__(
        self,
        device: MindRove,
        csv_path: str,
        raw_buffer_seconds: float = 10.0,
        filtered_buffer_seconds: float = 10.0,
        poll_interval_s: float = 0.02,
        bandpass: Tuple[float, float] = (20.0, 200.0),
        notch_freq: Optional[float] = 60.0,
    ):
        self.device = device
        self.poll_interval_s = poll_interval_s

        n_channels = len(device.emg_channels)
        fs = device.sampling_rate

        self.RawBuffer = RingBuffer(n_channels, max(1, int(raw_buffer_seconds * fs)))
        self.FilteredBuffer = RingBuffer(n_channels, max(1, int(filtered_buffer_seconds * fs)))
        self._filter = EMGFilter(n_channels, fs, bandpass=bandpass, notch_freq=notch_freq)

        header = ["timestamp"] + [f"emg_{i}" for i in range(n_channels)] + ["marker"]
        self._csv = CSVLogger(csv_path, header)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._n_samples_written = 0

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        if self._running:
            return
        if not self.device.is_connected():
            raise RuntimeError("Connect the MindRove device before starting the stream")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="EMGStream")
        self._running = True
        self._thread.start()


    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._running = False
        self._csv.close()


    def is_running(self) -> bool:
        return self._running
    

    # -- events ----------------------------------------------------------------
    def mark_event(self, code: float):
        """See class docstring. Safe to call from a different thread (e.g.
        your GUI's main thread) than the acquisition loop."""
        self.device.insert_marker(code)


    # -- internals --------------------------------------------------------------
    def _run(self):
        while not self._stop_event.is_set():
            chunk = self.device.pull_chunk()
            if chunk.size > 0 and chunk.shape[1] > 0:
                self._process_chunk(chunk)
            time.sleep(self.poll_interval_s)
            

    def _process_chunk(self, chunk: np.ndarray):
        emg = chunk[self.device.emg_channels, :]
        ts = chunk[self.device.timestamp_channel, :]
        markers = chunk[self.device.marker_channel, :]

        self.RawBuffer.push(emg)
        self.FilteredBuffer.push(self._filter.apply(emg))

        rows = np.vstack([ts, emg, markers]).T  # (n_samples, 1 + n_channels + 1)
        self._csv.write_rows(rows.tolist())
        self._n_samples_written += rows.shape[0]