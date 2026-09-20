"""
EMG Data Collection GUI
========================

A PySide6 application that guides a participant through a cued-action
protocol for EMG data collection:

    - A customizable list of actions (2-8 items, each with an editable label).
    - Customizable number of trial blocks and trials-per-action-per-block.
    - Each trial shows a brief "Get ready" cue, then the action label for a
      randomized duration (default 2-10 s), then a rest period.
    - A live EMG plot panel on the side (multi-channel, scrolling).
    - Connect / Disconnect buttons for the MindRove EMG device (the connection
      runs in a background thread; failures are reported in a dialog).
    - A timestamped event log (trial start/end, action, block/trial index)
      that can be exported to CSV, so you can sync it against your raw EMG
      recording afterwards.

Dependencies:
    pip install PySide6 pyqtgraph numpy scipy mindrove

Run:
    python emg_gui.py

--------------------------------------------------------------------------
HARDWARE INTEGRATION (emg.acquisition package):
  1. Connect        -> MindRove.connect() runs in a QThread, then an EMGStream
                       is started. It logs timestamp + raw EMG + marker for
                       every sample to ./recordings/emg_session_<time>.csv.
  2. Live plot      -> EMGPlotWidget.attach_source() polls the stream's
                       FilteredBuffer ring buffer on a QTimer.
  3. Event markers  -> SessionController.trialEvent is forwarded to
                       EMGStream.mark_event(). Marker code =
                       (action_index + 1) * 10 + (1 start | 2 end), e.g.
                       11 = action[0] started, 12 = action[0] ended.
--------------------------------------------------------------------------
"""

import os
import sys
import csv
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QCheckBox, QProgressBar,
    QSplitter, QTabWidget, QGroupBox, QMessageBox, QFileDialog,
    QPlainTextEdit, QSizePolicy, QFrame, QAbstractItemView
)

from emg.acquisition import MindRove, EMGStream

RECORDINGS_DIR = "recordings"

# Values written to the CSV's `marker` column (on the board's own clock):
#   code = (action_index + 1) * 10 + suffix
MARKER_SUFFIX = {"action_start": 1, "action_end": 2}


# ==========================================================================
# Data model
# ==========================================================================

@dataclass
class SessionSettings:
    actions: List[str] = field(default_factory=lambda: ["Rest", "Wrist Flexion", "Wrist Extension"])
    blocks: int = 3
    trials_per_action: int = 5
    action_min_s: float = 2.0
    action_max_s: float = 10.0
    rest_min_s: float = 2.0
    rest_max_s: float = 4.0
    cue_s: float = 1.0                # "Get ready" time shown before the action label
    inter_block_rest_s: float = 10.0  # longer break between blocks
    initial_delay_s: float = 10.0     # countdown before the first trial so the subject can get ready

    def validate(self) -> Optional[str]:
        if not (2 <= len(self.actions) <= 8):
            return "You need between 2 and 8 actions."
        if len(set(a.strip() for a in self.actions)) != len(self.actions):
            return "Action names must be unique."
        if any(not a.strip() for a in self.actions):
            return "Action names cannot be empty."
        if self.blocks < 1 or self.trials_per_action < 1:
            return "Blocks and trials-per-action must be at least 1."
        if self.action_min_s <= 0 or self.action_max_s < self.action_min_s:
            return "Action duration range is invalid."
        if self.rest_min_s <= 0 or self.rest_max_s < self.rest_min_s:
            return "Rest duration range is invalid."
        if self.initial_delay_s < 0:
            return "Initial countdown cannot be negative."
        return None


def build_trial_sequence(settings: SessionSettings):
    """Returns list of blocks; each block is a shuffled list of action names,
    with each action appearing `trials_per_action` times."""
    session = []
    for _ in range(settings.blocks):
        block = []
        for action in settings.actions:
            block.extend([action] * settings.trials_per_action)
        random.shuffle(block)
        session.append(block)
    return session


# ==========================================================================
# Settings panel (actions + trial/timing configuration)
# ==========================================================================

class SettingsPanel(QWidget):
    settingsApplied = Signal(SessionSettings)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._load_defaults()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Actions table -------------------------------------------------
        actions_box = QGroupBox("Actions (2-8)")
        actions_layout = QVBoxLayout(actions_box)

        self.actions_table = QTableWidget(0, 1)
        self.actions_table.setHorizontalHeaderLabels(["Action label"])
        self.actions_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.actions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        actions_layout.addWidget(self.actions_table)

        actions_btn_row = QHBoxLayout()
        self.add_action_btn = QPushButton("Add action")
        self.remove_action_btn = QPushButton("Remove selected")
        self.add_action_btn.clicked.connect(self.add_action_row)
        self.remove_action_btn.clicked.connect(self.remove_selected_action)
        actions_btn_row.addWidget(self.add_action_btn)
        actions_btn_row.addWidget(self.remove_action_btn)
        actions_layout.addLayout(actions_btn_row)

        layout.addWidget(actions_box)

        # --- Trial / block configuration -----------------------------------
        trial_box = QGroupBox("Trial structure")
        trial_form = QFormLayout(trial_box)

        self.blocks_spin = QSpinBox()
        self.blocks_spin.setRange(1, 50)

        self.trials_per_action_spin = QSpinBox()
        self.trials_per_action_spin.setRange(1, 50)

        trial_form.addRow("Trial blocks:", self.blocks_spin)
        trial_form.addRow("Trials per action / block:", self.trials_per_action_spin)

        layout.addWidget(trial_box)

        # --- Timing configuration -------------------------------------------
        timing_box = QGroupBox("Timing (seconds)")
        timing_form = QFormLayout(timing_box)

        self.cue_spin = self._make_dspin(0.0, 30.0, 1.0)
        self.action_min_spin = self._make_dspin(0.1, 120.0, 2.0)
        self.action_max_spin = self._make_dspin(0.1, 120.0, 10.0)
        self.rest_min_spin = self._make_dspin(0.1, 120.0, 2.0)
        self.rest_max_spin = self._make_dspin(0.1, 120.0, 4.0)
        self.inter_block_spin = self._make_dspin(0.0, 300.0, 10.0)
        self.initial_delay_spin = self._make_dspin(0.0, 300.0, 10.0)

        timing_form.addRow('"Get ready" cue:', self.cue_spin)
        timing_form.addRow("Action duration min:", self.action_min_spin)
        timing_form.addRow("Action duration max:", self.action_max_spin)
        timing_form.addRow("Rest duration min:", self.rest_min_spin)
        timing_form.addRow("Rest duration max:", self.rest_max_spin)
        timing_form.addRow("Inter-block break:", self.inter_block_spin)
        timing_form.addRow("Initial countdown:", self.initial_delay_spin)

        layout.addWidget(timing_box)

        # --- Apply / Save / Load --------------------------------------------
        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply settings")
        self.apply_btn.setStyleSheet("font-weight: bold;")
        self.save_btn = QPushButton("Save config…")
        self.load_btn = QPushButton("Load config…")
        self.apply_btn.clicked.connect(self._on_apply)
        self.save_btn.clicked.connect(self.save_config)
        self.load_btn.clicked.connect(self.load_config)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)

        layout.addStretch(1)

    @staticmethod
    def _make_dspin(lo, hi, default):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(0.5)
        s.setDecimals(1)
        s.setValue(default)
        return s

    def _load_defaults(self):
        defaults = SessionSettings()
        for name in defaults.actions:
            self.add_action_row(name)
        self.blocks_spin.setValue(defaults.blocks)
        self.trials_per_action_spin.setValue(defaults.trials_per_action)
        self.cue_spin.setValue(defaults.cue_s)
        self.action_min_spin.setValue(defaults.action_min_s)
        self.action_max_spin.setValue(defaults.action_max_s)
        self.rest_min_spin.setValue(defaults.rest_min_s)
        self.rest_max_spin.setValue(defaults.rest_max_s)
        self.inter_block_spin.setValue(defaults.inter_block_rest_s)
        self.initial_delay_spin.setValue(defaults.initial_delay_s)

    def add_action_row(self, name: str = ""):
        if self.actions_table.rowCount() >= 8:
            QMessageBox.warning(self, "Limit reached", "You can have at most 8 actions.")
            return
        row = self.actions_table.rowCount()
        self.actions_table.insertRow(row)
        item = QTableWidgetItem(name if name else f"Action {row + 1}")
        self.actions_table.setItem(row, 0, item)

    def remove_selected_action(self):
        rows = sorted({idx.row() for idx in self.actions_table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        if self.actions_table.rowCount() - len(rows) < 2:
            QMessageBox.warning(self, "Limit reached", "You need at least 2 actions.")
            return
        for row in rows:
            self.actions_table.removeRow(row)

    def collect_settings(self) -> SessionSettings:
        actions = []
        for row in range(self.actions_table.rowCount()):
            item = self.actions_table.item(row, 0)
            actions.append(item.text().strip() if item else "")
        return SessionSettings(
            actions=actions,
            blocks=self.blocks_spin.value(),
            trials_per_action=self.trials_per_action_spin.value(),
            action_min_s=self.action_min_spin.value(),
            action_max_s=self.action_max_spin.value(),
            rest_min_s=self.rest_min_spin.value(),
            rest_max_s=self.rest_max_spin.value(),
            cue_s=self.cue_spin.value(),
            inter_block_rest_s=self.inter_block_spin.value(),
            initial_delay_s=self.initial_delay_spin.value(),
        )

    def _on_apply(self):
        settings = self.collect_settings()
        error = settings.validate()
        if error:
            QMessageBox.critical(self, "Invalid settings", error)
            return
        self.settingsApplied.emit(settings)

    def save_config(self):
        settings = self.collect_settings()
        error = settings.validate()
        if error:
            QMessageBox.critical(self, "Invalid settings", error)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "emg_session_config.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w") as f:
            json.dump(settings.__dict__, f, indent=2)

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.actions_table.setRowCount(0)
        for name in data.get("actions", []):
            self.add_action_row(name)
        self.blocks_spin.setValue(data.get("blocks", 3))
        self.trials_per_action_spin.setValue(data.get("trials_per_action", 5))
        self.cue_spin.setValue(data.get("cue_s", 1.0))
        self.action_min_spin.setValue(data.get("action_min_s", 2.0))
        self.action_max_spin.setValue(data.get("action_max_s", 10.0))
        self.rest_min_spin.setValue(data.get("rest_min_s", 2.0))
        self.rest_max_spin.setValue(data.get("rest_max_s", 4.0))
        self.inter_block_spin.setValue(data.get("inter_block_rest_s", 10.0))
        self.initial_delay_spin.setValue(data.get("initial_delay_s", 10.0))


# ==========================================================================
# EMG connection + live plot panel
# ==========================================================================

class ConnectWorker(QThread):
    """Runs MindRove.connect() off the GUI thread - it can block for several
    seconds (session setup + waiting for the first samples)."""
    connected = Signal(object)   # the connected MindRove
    failed = Signal(str)         # human-readable error

    def run(self):
        try:
            device = MindRove()
            device.connect()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.connected.emit(device)


class EMGConnectionPanel(QWidget):
    connectRequested = Signal(str)     # port/device string
    disconnectRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)

        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.addItems(["COM3", "COM4", "/dev/ttyUSB0", "simulated"])

        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.disconnect_btn.clicked.connect(self.on_disconnect_clicked)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.connect_btn)
        btn_row.addWidget(self.disconnect_btn)
        layout.addRow(btn_row)

        self.status_label = QLabel("● Disconnected")
        self.status_label.setStyleSheet("color: #b00020; font-weight: bold;")
        layout.addRow("Status:", self.status_label)

    def on_connect_clicked(self):
        # The MindRove is a Wi-Fi board found by the SDK, so `port` is unused.
        port = self.port_combo.currentText()
        self.connectRequested.emit(port)

    def on_disconnect_clicked(self):
        self.disconnectRequested.emit()

    def set_connecting(self):
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(False)
        self.status_label.setText("● Connecting…")
        self.status_label.setStyleSheet("color: #ef6c00; font-weight: bold;")

    def set_connected(self, connected: bool):
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.port_combo.setEnabled(not connected)
        if connected:
            self.status_label.setText("● Connected")
            self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        else:
            self.status_label.setText("● Disconnected")
            self.status_label.setStyleSheet("color: #b00020; font-weight: bold;")


class EMGPlotWidget(QWidget):
    """Scrolling multi-channel EMG plot. Feed it data via push_sample()."""

    def __init__(self, num_channels: int = 4, window_seconds: float = 5.0,
                 assumed_rate_hz: float = 1000.0, parent=None):
        super().__init__(parent)
        self.window_seconds = window_seconds
        self.assumed_rate_hz = assumed_rate_hz
        self.num_channels = num_channels
        self.buffer_len = int(window_seconds * assumed_rate_hz)

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Channels:"))
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, 16)
        self.channel_spin.setValue(num_channels)
        self.channel_spin.valueChanged.connect(self.set_num_channels)
        controls.addWidget(self.channel_spin)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        controls.addWidget(self.clear_btn)

        self.simulate_check = QCheckBox("Simulate signal (test only)")
        self.simulate_check.toggled.connect(self._toggle_simulation)
        controls.addWidget(self.simulate_check)
        controls.addStretch(1)
        layout.addLayout(controls)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=False, alpha=0.2)
        self.plot_widget.setLabel("bottom", "Samples")
        self.plot_widget.getPlotItem().hideAxis("left")
        layout.addWidget(self.plot_widget)

        self._curves = []
        self._buffers = []
        self._offsets = []
        self._scales = []
        self._source = None            # RingBuffer while showing live hardware data
        self._fs = assumed_rate_hz
        self._init_channels(num_channels)

        self._sim_timer = QTimer(self)
        self._sim_timer.setInterval(20)  # ~50 Hz demo updates
        self._sim_timer.timeout.connect(self._simulate_tick)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(50)  # ~20 Hz redraw
        self._live_timer.timeout.connect(self._live_tick)

    def _init_channels(self, n):
        self.plot_widget.clear()
        self._curves = []
        self._buffers = []
        self._offsets = []
        self._scales = [0.0] * n
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                  "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
        spacing = 4.0
        for i in range(n):
            buf = deque([0.0] * self.buffer_len, maxlen=self.buffer_len)
            curve = self.plot_widget.plot(pen=pg.mkPen(colors[i % len(colors)], width=1))
            self._curves.append(curve)
            self._buffers.append(buf)
            self._offsets.append(i * spacing)
        self.num_channels = n

    def set_num_channels(self, n: int):
        self._init_channels(n)

    def clear(self):
        if self._source is not None:
            self._source.clear()
        for buf in self._buffers:
            buf.clear()
            buf.extend([0.0] * self.buffer_len)
        self._redraw()

    def push_sample(self, values):
        """Call this from your acquisition code with one new sample per
        channel, e.g. push_sample([ch1_value, ch2_value, ch3_value])."""
        for i, v in enumerate(values[:self.num_channels]):
            self._buffers[i].append(float(v))
        self._redraw()

    def attach_source(self, ring_buffer, sampling_rate: float):
        """Plot live data from a RingBuffer (e.g. EMGStream.FilteredBuffer)
        instead of pushed/simulated samples."""
        self.detach_source()
        self._source = ring_buffer
        self._fs = float(sampling_rate)
        self.buffer_len = max(1, int(self.window_seconds * self._fs))

        self.simulate_check.setChecked(False)
        self.simulate_check.setEnabled(False)

        n = ring_buffer.n_channels
        self.channel_spin.blockSignals(True)
        self.channel_spin.setRange(1, max(16, n))
        self.channel_spin.setValue(n)
        self.channel_spin.blockSignals(False)
        self.channel_spin.setEnabled(False)   # channel count is set by the device
        self._init_channels(n)

        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setXRange(-self.window_seconds, 0, padding=0)
        self._live_timer.start()

    def detach_source(self):
        if self._source is None:
            return
        self._live_timer.stop()
        self._source = None
        self.simulate_check.setEnabled(True)
        self.channel_spin.setEnabled(True)
        self.plot_widget.setLabel("bottom", "Samples")
        self.plot_widget.enableAutoRange(axis="x")
        self._init_channels(self.num_channels)

    def _live_tick(self):
        if self._source is None:
            return
        data = self._source.get_latest(self.buffer_len)   # (n_channels, n), oldest first
        n = data.shape[1]
        if n == 0:
            for curve in self._curves:
                curve.setData([], [])
            return
        t = (np.arange(n) - n) / self._fs                 # seconds, 0 = now
        for i, curve in enumerate(self._curves):
            y = data[i]
            peak = float(np.max(np.abs(y)))
            # Auto-gain per channel: jump up instantly, relax slowly.
            if peak > self._scales[i]:
                scale = peak
            else:
                scale = 0.98 * self._scales[i] + 0.02 * peak
            scale = max(scale, 1e-9)
            self._scales[i] = scale
            curve.setData(t, y / scale * 1.8 + self._offsets[i])

    def _redraw(self):
        for i, (curve, buf) in enumerate(zip(self._curves, self._buffers)):
            y = np.asarray(buf) + self._offsets[i]
            curve.setData(y)

    def _toggle_simulation(self, checked):
        if checked:
            self._sim_timer.start()
        else:
            self._sim_timer.stop()

    def _simulate_tick(self):
        # Demo-only random data so you can see the plot working without hardware.
        vals = [random.uniform(-1, 1) + random.gauss(0, 0.3) for _ in range(self.num_channels)]
        self.push_sample(vals)


# ==========================================================================
# Session controller (drives the cue -> action -> rest state machine)
# ==========================================================================

class SessionController(QObject):
    phaseChanged = Signal(str, str, float)      # phase, display_text, duration_s
    progressUpdated = Signal(float, float)       # fraction_done, seconds_remaining
    trialEvent = Signal(str, str, int, int, float)  # event_type, action, block_idx, trial_idx, timestamp
    sessionFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings: Optional[SessionSettings] = None
        self.sequence = []
        self.block_idx = 0
        self.trial_idx = 0
        self.phase = "IDLE"
        self.phase_start = 0.0
        self.phase_duration = 0.0
        self.paused = False
        self.pause_remaining = 0.0

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(50)
        self.tick_timer.timeout.connect(self._tick)

    def start(self, settings: SessionSettings):
        self.settings = settings
        self.sequence = build_trial_sequence(settings)
        self.block_idx = 0
        self.trial_idx = 0
        self.paused = False
        if settings.initial_delay_s > 0:
            self._start_prep()
        else:
            self._start_cue()
        self.tick_timer.start()

    def stop(self):
        if self.phase == "ACTION":
            # Close the open action so start/end events (and markers) stay paired.
            self._emit_event("action_end")
        self.tick_timer.stop()
        self.phase = "IDLE"

    def toggle_pause(self):
        if not self.tick_timer.isActive() and self.phase == "IDLE":
            return
        self.paused = not self.paused
        if self.paused:
            elapsed = time.time() - self.phase_start
            self.pause_remaining = max(0.0, self.phase_duration - elapsed)
        else:
            self.phase_duration = self.pause_remaining
            self.phase_start = time.time()

    def current_action(self):
        return self.sequence[self.block_idx][self.trial_idx]

    def _emit_event(self, event_type):
        self.trialEvent.emit(
            event_type, self.current_action(), self.block_idx, self.trial_idx, time.time()
        )

    def _start_prep(self):
        """Countdown before the first trial. No trial events/markers are emitted."""
        self.phase = "PREP"
        self.phase_duration = self.settings.initial_delay_s
        self.phase_start = time.time()
        self.phaseChanged.emit("PREP", "Get ready…\nSession starts soon", self.phase_duration)

    def _start_cue(self):
        self.phase = "CUE"
        self.phase_duration = self.settings.cue_s
        self.phase_start = time.time()
        action = self.current_action()
        self.phaseChanged.emit("CUE", f"Get ready:\n{action}", self.phase_duration)

    def _start_action(self):
        self.phase = "ACTION"
        self.phase_duration = random.uniform(self.settings.action_min_s, self.settings.action_max_s)
        self.phase_start = time.time()
        action = self.current_action()
        self._emit_event("action_start")
        self.phaseChanged.emit("ACTION", action, self.phase_duration)

    def _start_rest(self):
        self.phase = "REST"
        self.phase_duration = random.uniform(self.settings.rest_min_s, self.settings.rest_max_s)
        self.phase_start = time.time()
        self._emit_event("action_end")
        self.phaseChanged.emit("REST", "Rest", self.phase_duration)

    def _start_block_break(self):
        self.phase = "BLOCK_BREAK"
        self.phase_duration = self.settings.inter_block_rest_s
        self.phase_start = time.time()
        self.phaseChanged.emit(
            "BLOCK_BREAK",
            f"Block {self.block_idx + 1} complete.\nTake a break.",
            self.phase_duration,
        )

    def _finish(self):
        self.tick_timer.stop()
        self.phase = "DONE"
        self.phaseChanged.emit("DONE", "Session complete!", 0.0)
        self.sessionFinished.emit()

    def _advance(self):
        """Called when the current phase's timer has elapsed."""
        if self.phase == "PREP":
            self._start_cue()
        elif self.phase == "CUE":
            self._start_action()
        elif self.phase == "ACTION":
            self._start_rest()
        elif self.phase == "REST":
            self._advance_trial()
        elif self.phase == "BLOCK_BREAK":
            self._start_cue()

    def _advance_trial(self):
        self.trial_idx += 1
        if self.trial_idx >= len(self.sequence[self.block_idx]):
            self.trial_idx = 0
            self.block_idx += 1
            if self.block_idx >= len(self.sequence):
                self._finish()
                return
            self._start_block_break()
        else:
            self._start_cue()

    def _tick(self):
        if self.paused or self.phase in ("IDLE", "DONE"):
            return
        elapsed = time.time() - self.phase_start
        remaining = max(0.0, self.phase_duration - elapsed)
        fraction = 0.0 if self.phase_duration <= 0 else min(1.0, elapsed / self.phase_duration)
        self.progressUpdated.emit(fraction, remaining)
        if elapsed >= self.phase_duration:
            self._advance()

    def progress_text(self):
        if not self.sequence:
            return ""
        total_trials_in_block = len(self.sequence[self.block_idx])
        return (f"Block {self.block_idx + 1}/{len(self.sequence)}"
                f"   Trial {self.trial_idx + 1}/{total_trials_in_block}")


# ==========================================================================
# Trial display (center panel)
# ==========================================================================

class TrialDisplay(QWidget):
    startRequested = Signal()
    pauseRequested = Signal()
    stopRequested = Signal()

    PHASE_COLORS = {
        "PREP": "#00838f",
        "CUE": "#f9a825",
        "ACTION": "#2e7d32",
        "REST": "#1565c0",
        "BLOCK_BREAK": "#6a1b9a",
        "DONE": "#37474f",
        "IDLE": "#37474f",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        self.action_label = QLabel("Configure and start a session")
        self.action_label.setAlignment(Qt.AlignCenter)
        self.action_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(28)
        font.setBold(True)
        self.action_label.setFont(font)
        self.action_label.setMinimumHeight(160)
        self.action_label.setFrameShape(QFrame.StyledPanel)
        self.action_label.setStyleSheet("background-color: #37474f; color: white; border-radius: 8px;")
        layout.addWidget(self.action_label, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start session")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.startRequested)
        self.pause_btn.clicked.connect(self.pauseRequested)
        self.stop_btn.clicked.connect(self.stopRequested)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        log_box = QGroupBox("Event log")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_view)
        self.export_log_btn = QPushButton("Export log to CSV…")
        log_layout.addWidget(self.export_log_btn)
        layout.addWidget(log_box)

        self._log_rows = []  # (timestamp, event_type, action, block, trial)

    def show_phase(self, phase: str, text: str, duration: float):
        self.action_label.setText(text)
        color = self.PHASE_COLORS.get(phase, "#37474f")
        self.action_label.setStyleSheet(
            f"background-color: {color}; color: white; border-radius: 8px;"
        )

    def update_progress(self, fraction: float, remaining: float):
        self.progress_bar.setValue(int(fraction * 1000))
        self.time_label.setText(f"{remaining:0.1f} s remaining")

    def set_progress_text(self, text: str):
        self.progress_label.setText(text)

    def append_log(self, event_type, action, block_idx, trial_idx, timestamp):
        self._log_rows.append((timestamp, event_type, action, block_idx + 1, trial_idx + 1))
        t_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        self.log_view.appendPlainText(
            f"[{t_str}] block={block_idx + 1} trial={trial_idx + 1} "
            f"{event_type:<12} action={action}"
        )

    def set_running_state(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)

    def export_log(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event_type", "action", "block", "trial"])
            for row in self._log_rows:
                writer.writerow(row)


# ==========================================================================
# Main window
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Data Collection")
        self.resize(1280, 800)

        self.settings_panel = SettingsPanel()
        self.emg_conn_panel = EMGConnectionPanel()
        self.emg_plot = EMGPlotWidget(num_channels=4)
        self.trial_display = TrialDisplay()
        self.controller = SessionController()

        self._current_settings: Optional[SessionSettings] = None

        self.device: Optional[MindRove] = None
        self.stream: Optional[EMGStream] = None
        self._connect_worker: Optional[ConnectWorker] = None

        # --- left tabs: setup + device ---
        left_tabs = QTabWidget()
        left_tabs.addTab(self.settings_panel, "Actions && Trials")
        left_tabs.addTab(self.emg_conn_panel, "EMG Device")

        # --- right panel: EMG plot ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("<b>Live EMG</b>"))
        right_layout.addWidget(self.emg_plot)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_tabs)
        splitter.addWidget(self.trial_display)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 3)
        self.setCentralWidget(splitter)

        self._build_menu()
        self._wire_signals()
        self.statusBar().showMessage("Configure actions and trial settings, then Apply.")

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        save_action = QAction("Save config…", self)
        load_action = QAction("Load config…", self)
        save_action.triggered.connect(self.settings_panel.save_config)
        load_action.triggered.connect(self.settings_panel.load_config)
        file_menu.addAction(save_action)
        file_menu.addAction(load_action)

    def _wire_signals(self):
        self.settings_panel.settingsApplied.connect(self._on_settings_applied)

        self.emg_conn_panel.connectRequested.connect(self._on_connect_requested)
        self.emg_conn_panel.disconnectRequested.connect(self._on_disconnect_requested)

        self.trial_display.startRequested.connect(self._on_start)
        self.trial_display.pauseRequested.connect(self._on_pause)
        self.trial_display.stopRequested.connect(self._on_stop)
        self.trial_display.export_log_btn.clicked.connect(self._on_export_log)

        # Connected before append_log so the marker goes out before any UI work.
        self.controller.trialEvent.connect(self._on_trial_event)
        self.controller.phaseChanged.connect(self._on_phase_changed)
        self.controller.progressUpdated.connect(self.trial_display.update_progress)
        self.controller.trialEvent.connect(self.trial_display.append_log)
        self.controller.sessionFinished.connect(self._on_session_finished)

    # -- settings -----------------------------------------------------------
    def _on_settings_applied(self, settings: SessionSettings):
        self._current_settings = settings
        n_trials = settings.blocks * len(settings.actions) * settings.trials_per_action
        self.statusBar().showMessage(
            f"Settings applied: {len(settings.actions)} actions, "
            f"{settings.blocks} blocks, {n_trials} total trials. Ready to start."
        )

    # -- EMG connection ------------------------------------------------------
    def _on_connect_requested(self, port: str):
        # `port` is unused: the MindRove is a Wi-Fi board the SDK finds itself.
        if self._connect_worker is not None:
            return
        self.emg_conn_panel.set_connecting()
        self.statusBar().showMessage("Connecting to MindRove board…")

        worker = ConnectWorker(self)
        worker.connected.connect(self._on_device_connected)
        worker.failed.connect(self._on_connect_failed)
        worker.finished.connect(self._on_connect_worker_finished)
        self._connect_worker = worker
        worker.start()

    def _on_connect_worker_finished(self):
        if self._connect_worker is not None:
            self._connect_worker.deleteLater()
            self._connect_worker = None

    def _on_device_connected(self, device: MindRove):
        stream = None
        try:
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            csv_path = os.path.join(
                RECORDINGS_DIR, time.strftime("emg_session_%Y%m%d_%H%M%S.csv")
            )
            stream = EMGStream(device, csv_path=csv_path)
            stream.start()
        except Exception as exc:
            if stream is not None:
                stream.stop()
            self._safe_disconnect(device)
            self._on_connect_failed(
                f"Connected to the board, but could not start recording: {exc}"
            )
            return

        self.device = device
        self.stream = stream
        self.emg_plot.attach_source(stream.FilteredBuffer, device.sampling_rate)
        self.emg_conn_panel.set_connected(True)
        self.statusBar().showMessage(f"Connected. Recording to {csv_path}")

    def _on_connect_failed(self, message: str):
        self.emg_conn_panel.set_connected(False)
        self.statusBar().showMessage("Connection failed.")
        QMessageBox.critical(self, "EMG connection failed", message)

    def _on_disconnect_requested(self):
        if self.controller.phase not in ("IDLE", "DONE"):
            QMessageBox.warning(
                self, "Session running",
                "Stop the session before disconnecting the EMG device.",
            )
            return
        self._teardown_device()
        self.emg_conn_panel.set_connected(False)
        self.statusBar().showMessage("Disconnected. Recording saved.")

    def _teardown_device(self):
        self.emg_plot.detach_source()
        if self.stream is not None:
            try:
                self.stream.stop()          # joins the thread, closes the CSV
            except Exception as exc:
                print(f"Error stopping stream: {exc}")
            self.stream = None
        if self.device is not None:
            self._safe_disconnect(self.device)
            self.device = None

    @staticmethod
    def _safe_disconnect(device):
        try:
            device.disconnect()
        except Exception as exc:
            print(f"Error disconnecting device: {exc}")

    # -- event markers ----------------------------------------------------------
    def _on_trial_event(self, event_type, action_name, block_idx, trial_idx, timestamp):
        """Write each trial event into the EMG stream as a marker."""
        suffix = MARKER_SUFFIX.get(event_type)
        if suffix is None or self.stream is None or not self.stream.is_running():
            return
        try:
            action_index = self.controller.settings.actions.index(action_name)
            self.stream.mark_event((action_index + 1) * 10 + suffix)
        except Exception as exc:
            self.statusBar().showMessage(f"Could not write marker: {exc}")

    # -- session control ------------------------------------------------------
    def _on_start(self):
        if self._current_settings is None:
            QMessageBox.warning(self, "No settings", "Please Apply settings before starting.")
            return
        if self.stream is None or not self.stream.is_running():
            reply = QMessageBox.question(
                self, "No EMG device",
                "No EMG device is connected, so nothing will be recorded and no "
                "markers will be written.\n\nStart the session anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self.trial_display.set_running_state(True)
        self.controller.start(self._current_settings)

    def _on_pause(self):
        self.controller.toggle_pause()
        self.trial_display.pause_btn.setText(
            "Resume" if self.controller.paused else "Pause"
        )

    def _on_stop(self):
        self.controller.stop()
        self.trial_display.set_running_state(False)
        self.trial_display.pause_btn.setText("Pause")
        self.statusBar().showMessage("Session stopped.")

    def _on_phase_changed(self, phase, text, duration):
        self.trial_display.show_phase(phase, text, duration)
        self.trial_display.set_progress_text(self.controller.progress_text())

    def _on_session_finished(self):
        self.trial_display.set_running_state(False)
        self.trial_display.pause_btn.setText("Pause")
        self.statusBar().showMessage("Session complete.")

    def _on_export_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export log", "emg_session_log.csv", "CSV (*.csv)")
        if path:
            self.trial_display.export_log(path)

    def closeEvent(self, event):
        if self._connect_worker is not None:
            QMessageBox.information(
                self, "Please wait",
                "Still connecting to the EMG device - try closing again in a moment.",
            )
            event.ignore()
            return
        self.controller.stop()          # emits a closing marker if an action is open
        if self.stream is not None:
            time.sleep(0.15)            # let the acquisition thread write that marker
        self._teardown_device()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()