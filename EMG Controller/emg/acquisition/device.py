from typing import Optional

import numpy as np
import time as time
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds


class MindRove:
    """Connects to an EMG board and exposes pull_chunk()/insert_marker()/disconnect()."""

    def __init__(
        self,
        board_id: int = BoardIds.MINDROVE_WIFI_BOARD.value,
        params: Optional[MindRoveInputParams] = None,
    ):
        self.board_id = board_id
        self.params = params or MindRoveInputParams()
        self._board: Optional[BoardShim] = None

        # Row indices in the 2D array returned by get_board_data(), and the
        # sampling rate - all board-agnostic, queried from the SDK itself.
        self.emg_channels = BoardShim.get_emg_channels(board_id)
        self.timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        self.marker_channel = BoardShim.get_marker_channel(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)

        print(f"EMG Channels {self.emg_channels}")
        print(f"Timestamp Channels {self.emg_channels}")


    def connect(self, buffer_size: int = 450000):
        """Prepare the session and start streaming into the SDK's internal
        ring buffer. buffer_size is that internal buffer's capacity (in
        samples) - make it generous; our own pull loop drains it often."""
        board = BoardShim(self.board_id, self.params)

        # Try to connect
        try:
            print("Preparing...")
            board.prepare_session()
        except Exception as exc:
            raise ConnectionError(
                f"Could not find/connect to a MindRove board "
                f"(board_id={self.board_id}): {exc}"
            ) from exc

        # Try to start stream
        try:
            board.start_stream(buffer_size)
        except Exception as exc:
            # prepare_session() succeeded but streaming didn't - close the
            # half-open session before giving up, then report the failure.
            try:
                board.release_session()
            except Exception:
                pass
            raise ConnectionError(
                f"MindRove board connected but failed to start streaming: {exc}"
            ) from exc

        if not self._wait_for_data(board, 5):
            self._cleanup_after_failure(board)
            raise ConnectionError(
                f"No data received from the MindRove board within "
                f"{5:.1f}s of starting the stream - the board is "
            )

        print("Device successfully connected")
        self._board = board


    def pull_chunk(self) -> np.ndarray:
        """Pull everything currently buffered and clear the library's ring
        buffer (get_board_data semantics). Returns shape
        (n_all_channels, n_samples); n_samples may be 0 if nothing is new."""
        if self._board is None:
            raise RuntimeError("MindRove.connect() must be called first")
        return self._board.get_board_data()
    

    def insert_marker(self, value: float):
        """Write a marker onto the board's own data stream. It will appear
        in the marker row of the next chunk pulled via pull_chunk(),
        aligned to the board's own sample clock."""
        if self._board is None:
            raise RuntimeError("MindRove.connect() must be called first")
        self._board.insert_marker(float(value))


    def disconnect(self):
        if self._board is not None:
            try:
                self._board.stop_stream()
            finally:
                self._board.release_session()
                self._board = None


    def is_connected(self) -> bool:
        return self._board is not None
    

    def __enter__(self):
        self.connect()
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _wait_for_data(self, board: BoardShim, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(poll_interval_s)
            chunk = board.get_board_data()
            if chunk.size > 0 and chunk.shape[1] > 0:
                return True
        return False
 
    def _cleanup_after_failure(self, board: BoardShim):
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass