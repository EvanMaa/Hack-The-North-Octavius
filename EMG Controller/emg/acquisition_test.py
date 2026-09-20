"""Example usage of emg_pipeline.

1) Standalone acquisition
2) How to wire it into the trial GUI (emg_gui.py) so each action's
   start/end is written into the CSV as a marker, using the board's own
   clock via insert_marker().
"""

import time

from emg.acquisition import MindRove, EMGStream


def standalone_example():
    device = MindRove()  # uses BoardIds.MINDROVE_WIFI_BOARD by default
    try:
        device.connect()
    except ConnectionError:
        print("Error")
        raise("Error")

    stream = EMGStream(device, csv_path="session_001.csv")
    stream.start()

    time.sleep(20)
    print("HOLD")
    stream.mark_event(1)   # e.g. "action A started"
    time.sleep(3)
    print("STOP")
    stream.mark_event(-1)  # e.g. "action A ended"
    time.sleep(2)

    stream.stop()
    device.disconnect()


# --------------------------------------------------------------------------
# GUI integration
# --------------------------------------------------------------------------
# In emg_gui.py's MainWindow, after constructing the device + stream:
#
#     self.device = MindRove()
#     self.stream = None  # created after Connect is pressed
#
# Encode each event as a single float so you can decode it later, e.g.
#     code = (action_index + 1) * 10 + (1 if event_type == "action_start" else 2)
# so 11 = action[0] started, 12 = action[0] ended, 21 = action[1] started, etc.
#
# Then in MainWindow (replacing the emg_gui.py stubs):
#
#     def _on_connect_requested(self, port: str):
#         self.device.connect()
#         self.stream = EMGStream(self.device, csv_path="session.csv")
#         self.stream.start()
#         self.emg_conn_panel.set_connected(True)
#
#     def _on_disconnect_requested(self):
#         self.stream.stop()
#         self.device.disconnect()
#         self.emg_conn_panel.set_connected(False)
#
# And connect the SessionController's existing `trialEvent` signal
# (event_type, action_name, block_idx, trial_idx, timestamp) straight to
# a marker call - this is the one line that ties the two packages together:
#
#     self.controller.trialEvent.connect(self._on_trial_event)
#
#     def _on_trial_event(self, event_type, action_name, block_idx, trial_idx, timestamp):
#         action_index = self._current_settings.actions.index(action_name)
#         suffix = 1 if event_type == "action_start" else 2
#         code = (action_index + 1) * 10 + suffix
#         self.stream.mark_event(code)
#
# Also feed the live plot from the filtered buffer instead of random data -
# e.g. a QTimer every ~50ms calling:
#
#     latest = self.stream.FilteredBuffer.get_latest(200)  # last 200 samples
#     self.emg_plot.push_sample(latest[:, -1])              # most recent sample per channel


if __name__ == "__main__":
    standalone_example()