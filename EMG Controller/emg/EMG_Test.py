import logging

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from mindrove.board_shim import (
    BoardShim,
    MindRoveInputParams,
    BoardIds,
)
from mindrove.data_filter import (
    DataFilter,
    FilterTypes,
    WindowOperations,
    DetrendOperations,
)


class Graph:
    def __init__(self, board_shim):
        # ------------------------------------------------------------------
        # PyQtGraph configuration
        # ------------------------------------------------------------------
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOption("antialias", True)

        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()

        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)

        if not self.exg_channels:
            raise RuntimeError("No EXG channels were found.")

        if self.sampling_rate <= 0:
            raise RuntimeError(
                f"Invalid sampling rate: {self.sampling_rate}"
            )

        # ------------------------------------------------------------------
        # Display settings
        # ------------------------------------------------------------------
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = int(self.window_size * self.sampling_rate)

        # ------------------------------------------------------------------
        # Qt application
        # ------------------------------------------------------------------
        # QApplication must be created only once.
        self.app = QtWidgets.QApplication.instance()

        if self.app is None:
            self.app = QtWidgets.QApplication([])

        # GraphicsLayoutWidget replaces the old GraphicsWindow API.
        self.win = pg.GraphicsLayoutWidget(
            title="MindRove Plot",
            size=(1200, 800),
        )

        self.win.resize(1200, 800)
        self.win.show()

        # ------------------------------------------------------------------
        # Initialize plots
        # ------------------------------------------------------------------
        self._init_pens()
        self._init_timeseries()
        self._init_psd()
        self._init_band_plot()

        # ------------------------------------------------------------------
        # Update timer
        # ------------------------------------------------------------------
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        # Keep references to Qt objects alive.
        self.app.aboutToQuit.connect(self._on_quit)

        # ------------------------------------------------------------------
        # Start Qt event loop
        # ------------------------------------------------------------------
        self._exec_qt()

    # ----------------------------------------------------------------------
    # Qt compatibility
    # ----------------------------------------------------------------------
    def _exec_qt(self):
        """
        Qt6 uses exec().
        Older Qt5 bindings use exec_().

        This makes the application work with either API.
        """
        exec_method = getattr(self.app, "exec", None)

        if exec_method is not None:
            exec_method()
        else:
            self.app.exec_()

    def _on_quit(self):
        """Stop the update timer when the window/application closes."""
        if hasattr(self, "timer"):
            self.timer.stop()

    # ----------------------------------------------------------------------
    # Plot appearance
    # ----------------------------------------------------------------------
    def _init_pens(self):
        colors = [
            "#A54E4E",
            "#A473B6",
            "#5B45A4",
            "#2079D2",
            "#32B798",
            "#2FA537",
            "#9DA52F",
            "#A57E2F",
            "#A53B2F",
        ]

        self.pens = [
            pg.mkPen(color=color, width=2)
            for color in colors
        ]

        self.brushes = [
            pg.mkBrush(color)
            for color in colors
        ]

    # ----------------------------------------------------------------------
    # Time-series plots
    # ----------------------------------------------------------------------
    def _init_timeseries(self):
        self.plots = []
        self.curves = []

        for i, _channel in enumerate(self.exg_channels):
            plot = self.win.addPlot(row=i, col=0)

            plot.showAxis("left", False)
            plot.showAxis("bottom", False)

            plot.setMenuEnabled("left", False)
            plot.setMenuEnabled("bottom", False)

            if i == 0:
                plot.setTitle("Time Series")

            curve = plot.plot(
                pen=self.pens[i % len(self.pens)]
            )

            self.plots.append(plot)
            self.curves.append(curve)

        # Give the left column a reasonable amount of space.
        self.win.ci.layout.setColumnStretchFactor(0, 2)

    # ----------------------------------------------------------------------
    # PSD plots
    # ----------------------------------------------------------------------
    def _init_psd(self):
        num_channels = len(self.exg_channels)

        # Avoid rowspan=0 for unusual boards.
        rowspan = max(1, num_channels // 2)

        self.psd_plot = self.win.addPlot(
            row=0,
            col=1,
            rowspan=rowspan,
        )

        self.psd_plot.showAxis("left", False)
        self.psd_plot.setMenuEnabled("left", False)
        self.psd_plot.setTitle("PSD")

        # Keep the original behavior: logarithmic Y axis.
        self.psd_plot.setLogMode(False, True)

        self.psd_curves = []

        self.psd_size = DataFilter.get_nearest_power_of_two(
            self.sampling_rate
        )

        for i, _channel in enumerate(self.exg_channels):
            curve = self.psd_plot.plot(
                pen=self.pens[i % len(self.pens)]
            )

            # Downsampling is supported by current PyQtGraph versions.
            curve.setDownsampling(
                auto=True,
                method="mean",
                ds=3,
            )

            self.psd_curves.append(curve)

    # ----------------------------------------------------------------------
    # Band-power plot
    # ----------------------------------------------------------------------
    def _init_band_plot(self):
        num_channels = len(self.exg_channels)
        row = max(1, num_channels // 2)

        self.band_plot = self.win.addPlot(
            row=row,
            col=1,
            rowspan=max(1, num_channels // 2),
        )

        self.band_plot.showAxis("left", False)
        self.band_plot.showAxis("bottom", False)

        self.band_plot.setMenuEnabled("left", False)
        self.band_plot.setMenuEnabled("bottom", False)

        self.band_plot.setTitle("Band Power")

        self.band_names = [
            "Delta",
            "Theta",
            "Alpha",
            "Beta",
            "Gamma",
        ]

        x = [1, 2, 3, 4, 5]
        y = [0, 0, 0, 0, 0]

        self.band_bar = pg.BarGraphItem(
            x=x,
            height=y,
            width=0.8,
            pen=self.pens[0],
            brush=self.brushes[0],
        )

        self.band_plot.addItem(self.band_bar)

        # Set sensible X-axis labels.
        axis = self.band_plot.getAxis("bottom")

        if axis is not None:
            axis.setTicks([
                list(zip(
                    x,
                    self.band_names,
                ))
            ])

        self.band_plot.setYRange(0, 100)

    # ----------------------------------------------------------------------
    # Signal processing
    # ----------------------------------------------------------------------
    def update(self):
        try:
            data = self.board_shim.get_current_board_data(
                self.num_points
            )
        except Exception:
            logging.exception("Failed to retrieve board data.")
            return

        if data is None or data.size == 0:
            return

        if data.ndim != 2:
            logging.warning(
                "Unexpected board data shape: %s",
                getattr(data, "shape", None),
            )
            return

        if data.shape[1] == 0:
            return

        avg_bands = [0.0] * 5
        valid_channels = 0

        for count, channel in enumerate(self.exg_channels):

            # Make sure the channel exists in the returned data.
            if channel >= data.shape[0]:
                logging.warning(
                    "Channel index %s is outside data shape %s",
                    channel,
                    data.shape,
                )
                continue

            channel_data = data[channel]

            if channel_data.size == 0:
                continue

            # --------------------------------------------------------------
            # Filter signal
            # --------------------------------------------------------------
            try:
                DataFilter.detrend(
                    channel_data,
                    DetrendOperations.CONSTANT.value,
                )

                DataFilter.perform_bandpass(
                    channel_data,
                    self.sampling_rate,
                    30.0,
                    56.0,
                    2,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )

                DataFilter.perform_bandstop(
                    channel_data,
                    self.sampling_rate,
                    4.0,
                    50.0,
                    2,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )

                DataFilter.perform_bandstop(
                    channel_data,
                    self.sampling_rate,
                    4.0,
                    60.0,
                    2,
                    FilterTypes.BUTTERWORTH.value,
                    0,
                )

            except Exception:
                logging.exception(
                    "Signal filtering failed for channel %s",
                    channel,
                )
                continue

            # --------------------------------------------------------------
            # Time-series plot
            # --------------------------------------------------------------
            self.curves[count].setData(channel_data)

            # --------------------------------------------------------------
            # PSD + band power
            # --------------------------------------------------------------
            if data.shape[1] <= self.psd_size:
                continue

            try:
                psd_data = DataFilter.get_psd_welch(
                    channel_data,
                    self.psd_size,
                    self.psd_size // 2,
                    self.sampling_rate,
                    WindowOperations.BLACKMAN_HARRIS.value,
                )

                frequencies = psd_data[1]
                power = psd_data[0]

                # Plot up to 70 Hz.
                lim = min(70, len(frequencies))

                self.psd_curves[count].setData(
                    frequencies[:lim],
                    power[:lim],
                )

                # ----------------------------------------------------------
                # Calculate frequency bands
                # ----------------------------------------------------------
                avg_bands[0] += DataFilter.get_band_power(
                    psd_data, 1.0, 4.0
                )

                avg_bands[1] += DataFilter.get_band_power(
                    psd_data, 4.0, 8.0
                )

                avg_bands[2] += DataFilter.get_band_power(
                    psd_data, 8.0, 13.0
                )

                avg_bands[3] += DataFilter.get_band_power(
                    psd_data, 13.0, 30.0
                )

                avg_bands[4] += DataFilter.get_band_power(
                    psd_data, 30.0, 50.0
                )

                valid_channels += 1

            except Exception:
                logging.exception(
                    "PSD calculation failed for channel %s",
                    channel,
                )

        # ------------------------------------------------------------------
        # Average band powers
        # ------------------------------------------------------------------
        if valid_channels > 0:
            avg_bands = [
                int(value * 100 / valid_channels)
                for value in avg_bands
            ]
        else:
            avg_bands = [0, 0, 0, 0, 0]

        self.band_bar.setOpts(height=avg_bands)

        # Usually unnecessary because QTimer already runs inside the Qt
        # event loop, but harmless when called here.
        self.app.processEvents()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    BoardShim.enable_dev_board_logger()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    params = MindRoveInputParams()
    board_shim = None

    try:
        logging.info("Creating MindRove board connection...")

        board_shim = BoardShim(
            BoardIds.MINDROVE_WIFI_BOARD,
            params,
        )

        logging.info("Preparing session...")
        board_shim.prepare_session()

        logging.info("Starting stream...")
        board_shim.start_stream()

        logging.info("Starting graphical interface...")
        Graph(board_shim)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    except Exception:
        logging.exception("Application error.")

    finally:
        if board_shim is not None:
            try:
                logging.info("Stopping stream...")
                board_shim.stop_stream()
            except Exception:
                logging.exception("Error stopping stream.")

            try:
                if board_shim.is_prepared():
                    logging.info("Releasing session...")
                    board_shim.release_session()
            except Exception:
                logging.exception("Error releasing board session.")


if __name__ == "__main__":
    main()