from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


class EMGFilter:
    """Bandpass + notch filter applied independently per channel, with
    persistent state so it can be called repeatedly on small chunks."""

    def __init__(
        self,
        n_channels: int,
        sampling_rate: float,
        bandpass: Tuple[float, float] = (20.0, 450.0),
        notch_freq: Optional[float] = 60.0,  # set to None to skip notch filtering
        notch_q: float = 30.0,
        order: int = 4,
    ):
        self.n_channels = n_channels
        self.fs = sampling_rate

        low, high = bandpass
        high = min(high, sampling_rate / 2.0 * 0.99)  # stay just under Nyquist
        self.bp_sos = butter(order, [low, high], btype="bandpass", output="sos", fs=sampling_rate)
        self._zi_bp_template = sosfilt_zi(self.bp_sos)  # shape (n_sections, 2)
        self._zi_bp = None  # per-channel state, set on first apply()

        self.notch_sos = None
        self._zi_notch_template = None
        self._zi_notch = None

        if notch_freq is not None:
            b, a = iirnotch(notch_freq, notch_q, sampling_rate)
            self.notch_sos = tf2sos(b, a)
            self._zi_notch_template = sosfilt_zi(self.notch_sos)


    def apply(self, chunk: np.ndarray) -> np.ndarray:
        """chunk shape (n_channels, n_samples) -> filtered, same shape."""
        if chunk.shape[1] == 0:
            return chunk

        if self._zi_bp is None:
            # Scale the template state by each channel's first sample so the
            # filter starts "warmed up" instead of producing a startup
            # transient (ramping from zero).
            self._zi_bp = np.stack(
                [self._zi_bp_template * chunk[ch, 0] for ch in range(self.n_channels)],
                axis=-1,
            )
            if self.notch_sos is not None:
                self._zi_notch = np.stack(
                    [self._zi_notch_template * chunk[ch, 0] for ch in range(self.n_channels)],
                    axis=-1,
                )

        out = np.empty_like(chunk)
        for ch in range(self.n_channels):
            y, self._zi_bp[:, :, ch] = sosfilt(self.bp_sos, chunk[ch], zi=self._zi_bp[:, :, ch])
            if self.notch_sos is not None:
                y, self._zi_notch[:, :, ch] = sosfilt(self.notch_sos, y, zi=self._zi_notch[:, :, ch])
            out[ch] = y
        return out


    def reset(self):
        self._zi_bp = None
        self._zi_notch = None