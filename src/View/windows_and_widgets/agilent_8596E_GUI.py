# Created by Jannet Trabelsi on 2025-09-02
# Please note: the controller class raises errors. However, the GUI sets default values to solve for invalid inputs

import numpy as np
from PyQt5.QtWidgets import  QMessageBox, QWidget
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from src.Controller.Agilent8596E import Agilent8596E
from .agilent_8596E_design import Ui_Form
import pyqtgraph as pg
from PyQt5.QtCore import QTimer
# Assuming the .ui file is converted to design.py
#To convert agilent_8596E_design.ui to .py, paste this into the terminal:
# pyuic5 -x agilent_8596E_design.ui -o agilent_8596E_design.py
# if running from Pittqlabsys single NV the agilent_analyzer will be the agilent class talking to the hardware directly otherwise,
# it will be the device client class which talks to the server

# constants:
_default_smooth_UI = 1
_max_freq_UI = 12800
_min_freq_UI = 0.009
_MAX_SET_AVG_PTS_UI = 16384
_default_sweep_time = 1000000 # 1 second
_max_sweep_time = 100000000
_min_sweep_time_zero_span = 20
_min_sweep_time_nonzero_span = 20000
_max_span = 2943
_min_resolution_BW = 0.00003
_max_BW = 3
_min_video_BW = 0.00003
_MAX_SET_AVG_PTS = 16384

class SpectrumAnalyzerView(QWidget, Ui_Form):
    """
    This is the main window of the application. It allows us to control the spectrum analyzer using buttons and LineEdits
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.agilent_analyzer = None
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.acquire_and_plot_data)
        self.update_fft = QTimer(self)
        self.update_fft.timeout.connect(self.fftacquire_and_plot_data)

        # Amplitude vs Frequency plot
        self.plot_widget = pg.PlotWidget(title="Amplitude vs Frequency")
        self.plot_widget.setLabel('left', 'Amplitude (dBm)')
        self.plot_widget.setLabel('bottom', 'Frequency (MHz)')
        self.plot_container.layout().addWidget(self.plot_widget)

        # Initialize data arrays for plotting
        self.frequency_data = []
        self.amplitude_data = []

        # Initialize plot
        self.sa_plot = self.plot_widget.plot(pen='r', name='Amplitude (dBm)')

        # Connect buttons to functions
        self.connectButton.clicked.connect(self.connect_to_instrument)
        self.startButton.clicked.connect(self.start_process)
        self.stopButton.clicked.connect(self.stop_process)
        self.clearButton.clicked.connect(self.clear_data)
        self.centerfreqButton.clicked.connect(self.centerfreq)
        self.spanButton.clicked.connect(self.span)
        self.saveButton.clicked.connect(self.save_data)
        self.snapshotButton.clicked.connect(self.snapshot_data)
        self.lastspanButton.clicked.connect(self.set_last_span_UI)
        self.setsweeptimeButton.clicked.connect(self.set_sweep_time_UI)
        self.resolutionbwButton.clicked.connect(self.set_resolution_bw_UI)
        self.videobwButton.clicked.connect(self.set_video_bw_UI)
        self.markerminButton.clicked.connect(self.set_marker_min_UI)
        self.markermaxButton.clicked.connect(self.set_marker_max_UI)
        self.markerdeltaButton.clicked.connect(self.set_marker_delta_UI)
        self.markerspanButton.clicked.connect(self.set_marker_span_UI)
        self.videoavgButton.clicked.connect(self.video_avg_UI)
        self.clravgButton.clicked.connect(self.clr_avg_UI)
        self.initializeButton.clicked.connect(self.initialize_device)
        self.fftButton.clicked.connect(self.fft_UI)
        self.fft_sweep_timepushButton.clicked.connect(self.set_fft_sweep_time)
        self.startfftButton.clicked.connect(self.start_fft)
        self.stopfftButton.clicked.connect(self.stop_fft)

    def connect_to_instrument(self):
        try:
            self.agilent_analyzer = Agilent8596E()
            QMessageBox.information(self, 'Success', f'Connected to Agilent 8596E on GPIB 18')
            self.cfLineEdit.setText(str(self.agilent_analyzer.read_probes("center frequency")/ 1000000))
            self.spanLineEdit.setText(str(self.agilent_analyzer.read_probes("span")/ 1000000))
            self.sweeptimeLineEdit.setText(str(self.agilent_analyzer.read_probes("sweep time")* 1000000))
            self.resolutionbwLineEdit.setText(str(self.agilent_analyzer.read_probes("resolution band width")/ 1000000))
            self.videobwLineEdit.setText(str(self.agilent_analyzer.read_probes("video band width")/ 1000000))
            self.fft_sweep_timeLineEdit.setText(str(self.agilent_analyzer.read_probes("sweep time")* 1000000))


        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def start_process(self):
        try:
            print("Starting process...")
            self.startButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.clearButton.setEnabled(False)
            self.spanLineEdit.setEnabled(False)
            self.cfLineEdit.setEnabled(False)
            self.spanButton.setEnabled(False)
            self.centerfreqButton.setEnabled(False)
            self.snapshotButton.setEnabled(False)
            self.setsweeptimeButton.setEnabled(False)
            self.saveButton.setEnabled(False)
            self.lastspanButton.setEnabled(False)
            self.resolutionbwButton.setEnabled(False)
            self.videobwButton.setEnabled(False)
            self.markerminButton.setEnabled(False)
            self.markermaxButton.setEnabled(False)
            self.markerdeltaButton.setEnabled(False)
            self.markerspanButton.setEnabled(False)
            self.videoavgButton.setEnabled(False)
            self.clravgButton.setEnabled(False)
            self.update_timer.start(500)
        except ValueError as e:
            print(f"ValueError: {e}")
            QMessageBox.warning(self, 'Warning', 'Invalid numeric input')
        except Exception as e:
            print(f"Exception: {e}")
            QMessageBox.warning(self, 'Warning', f'Unexpected error: {e}')

    def stop_process(self):
        self.update_timer.stop()
        self.spanLineEdit.setEnabled(True)
        self.cfLineEdit.setEnabled(True)
        self.spanButton.setEnabled(True)
        self.centerfreqButton.setEnabled(True)
        self.snapshotButton.setEnabled(True)
        self.setsweeptimeButton.setEnabled(True)
        self.saveButton.setEnabled(True)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.clearButton.setEnabled(True)
        self.lastspanButton.setEnabled(True)
        self.resolutionbwButton.setEnabled(True)
        self.videobwButton.setEnabled(True)
        self.markerminButton.setEnabled(True)
        self.markermaxButton.setEnabled(True)
        self.markerdeltaButton.setEnabled(True)
        self.markerspanButton.setEnabled(True)
        self.videoavgButton.setEnabled(True)
        self.clravgButton.setEnabled(True)

    def clear_data(self):
        self.frequency_data.clear()
        self.amplitude_data.clear()
        self.sa_plot.setData([], [])

    def update_plot(self):
        if self.frequency_data is None or self.amplitude_data is None:
            return
        if len(self.frequency_data) == 0 or len(self.amplitude_data) == 0:
            return
        self.sa_plot.setData(self.frequency_data, self.amplitude_data)

    def acquire_and_plot_data(self):
        try:
            self.agilent_analyzer.write("TDF P;")
            self.agilent_analyzer.write("SNGLS;")
            self.agilent_analyzer.write("TS;")
            self.agilent_analyzer.write("MKPK HI;")
            center_freq = self.agilent_analyzer.read_probes("center frequency")/ 1000000
            mkfreq = self.agilent_analyzer.read_probes("marker frequency") / 1000000
            amplitude = self.agilent_analyzer.read_probes("marker amplitude")
            print(f"Status: cf={center_freq}")
            print(f"Status: markerfreq={mkfreq}")
            print(f"Status: amplitude={amplitude}")
            self.agilent_analyzer.write("TS;")
            trace_data = np.array(self.agilent_analyzer.read_probes("trace AP"))
            num_points = len(trace_data)
            center_freq = self.agilent_analyzer.read_probes("center frequency")/ 1000000
            print(f"Status: cf={center_freq}")
            span = self.agilent_analyzer.read_probes("span")/ 1000000
            start_freq = center_freq - (span / 2)
            stop_freq = center_freq + (span / 2)
            freqs = np.linspace(start_freq, stop_freq, num_points)
            self.agilent_analyzer.write("CONTS;")
            self.frequency_data = list(freqs / 1e9)
            self.amplitude_data = list(trace_data)
            self.update_plot()

        except Exception as e:
            print(f"Error acquiring data: {e}")
            self.update_timer.stop()
            QMessageBox.critical(self, "Acquisition Error", str(e))

    def centerfreq(self):
        print("inside centerfreq...")
        center_freq = self.agilent_analyzer.read_probes("center frequency")/ 1000000
        cf = self.cfLineEdit.text()
        if isinstance(cf, str) or isinstance(cf, int) or isinstance(cf, float):
            try:
                cf = float(cf)
                if cf > _max_freq_UI or cf < _min_freq_UI:
                    print("wrong center frequency input")
                    cf = center_freq
            except ValueError:
                cf = center_freq
        else:
            print("wrong center frequency input")
            cf = center_freq
        self.agilent_analyzer.update({"center frequency": cf})
        self.cfLineEdit.setText(str((self.agilent_analyzer.read_probes("center frequency")/ 1000000)))

    def span(self):
        print("Inside span...")
        span = self.spanLineEdit.text()
        center_freq = self.agilent_analyzer.read_probes("center frequency")/ 1000000
        sp = self.agilent_analyzer.read_probes("span")/ 1000000
        if isinstance(span, str) or isinstance(span, int) or isinstance(span, float):
            try:
                span = float(span)
                if (center_freq - (span / 2)) < 0 or (center_freq + (span / 2)) > _max_freq_UI:
                    print("wrong span input")
                    span = sp
            except ValueError:
                span = sp
        else:
            print("wrong span input")
            span = sp
        self.agilent_analyzer.update({"span": span})
        self.spanLineEdit.setText(str(self.agilent_analyzer.read_probes("span")/ 1000000))

    def save_data(self):
        if not self.frequency_data or not self.amplitude_data:
            QMessageBox.warning(self, 'Warning', 'No data to save')
            return
        # Open file dialog to choose file save location
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Spectrum Data As"
        )
        if not file_path:
            return  # User cancelled the save dialog
        try:
            # Save frequency and amplitude data to CSV
            df = pd.DataFrame({
                'Frequency_MHz': self.frequency_data,
                'Amplitude_dBm': self.amplitude_data
            })
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, 'Success', f'Data saved to {file_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save data: {e}')

    def snapshot_data(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("PNG files", "*.PNG")],
            title="Save Snapshot Data As"
        )
        pixmap = self.plot_widget.grab()
        pixmap.save(file_path, "PNG")
        if not file_path:
            return

    def initialize_device(self):
        self.agilent_analyzer.write("IP;")
        self.startButton.setEnabled(True)
        self.cfLineEdit.setText(str(self.agilent_analyzer.read_probes("center frequency")/ 1000000))
        self.spanLineEdit.setText(str(self.agilent_analyzer.read_probes("span")/ 1000000))
        self.sweeptimeLineEdit.setText(str(self.agilent_analyzer.read_probes("sweep time")* 1000000))
        self.fft_sweep_timeLineEdit.setText(str(self.set_fft_sweep_time()))
        self.resolutionbwLineEdit.setText(str(self.agilent_analyzer.read_probes("resolution band width")/ 1000000))
        self.videobwLineEdit.setText(str(self.agilent_analyzer.read_probes("video band width")/ 1000000))

    def set_last_span_UI(self):
        self.agilent_analyzer.write("LSPAN;")
        span = self.agilent_analyzer.read_probes("span")/ 1000000
        self.spanLineEdit.setText(str(span))

    def set_fft_sweep_time(self):
        time = self.fft_sweep_timeLineEdit.text()
        if isinstance(time, str) or isinstance(time, int) or isinstance(time, float):
            try:
                time = float(time)
                if time < _min_sweep_time_zero_span or time > _max_sweep_time:
                    print("wrong sweep time input")
                    time = _default_sweep_time
                    print("Set sweep time: " + str(time))
                else:
                    print("Set sweep time: " + str(time))
            except ValueError:
                print("wrong sweep time input")
                time = _default_sweep_time
                print("Set sweep time: " + str(time))
        else:
            print("wrong sweep time input")
            time = _default_sweep_time
            print("Set sweep time: " + str(time))
        self.fft_sweep_timeLineEdit.setText(str(time))

    def set_sweep_time_UI(self):
        time = self.sweeptimeLineEdit.text()
        span = self.agilent_analyzer.read_probes("span")/ 1000000
        if isinstance(time, str) or isinstance(time, int) or isinstance(time, float):
            try:
                time = float(time)
                if span == 0:
                    if time < _min_sweep_time_zero_span or time > _max_sweep_time:
                        print ("wrong sweep time input")
                        time = _default_sweep_time
                        print("Set sweep time: " + str(time))
                    else:
                        print("Set sweep time: " + str(time))
                else:
                    if time < _min_sweep_time_nonzero_span or time > _max_sweep_time:
                        print("wrong sweep time input")
                        time = _default_sweep_time
                        print("Set sweep time: " + str(time))
                    else:
                        print("Set sweep time: " + str(time))
            except ValueError:
                print("wrong sweep time input")
                time = _default_sweep_time
                print("Set sweep time: " + str(time))
        else:
            print("wrong sweep time input")
            time = _default_sweep_time
            print("Set sweep time: " + str(time))
        self.agilent_analyzer.update({"sweep time": time})
        self.sweeptimeLineEdit.setText(str(time))

    def set_resolution_bw_UI(self):
        RBWMHZ = self.resolutionbwLineEdit.text()
        rbw = self.agilent_analyzer.read_probes("resolution band width")/ 1000000
        if isinstance(RBWMHZ, str) or isinstance(RBWMHZ, int) or isinstance(RBWMHZ, float):
            try:
                RBWMHZ = float(RBWMHZ)
                if RBWMHZ < _min_resolution_BW or RBWMHZ > _max_BW:
                    print("Wrong RBW input")
                    RBWMHZ = rbw
            except ValueError:
                print("Wrong RBW input")
                RBWMHZ = rbw
        self.agilent_analyzer.update({"resolution band width": RBWMHZ})
        self.resolutionbwLineEdit.setText(str(RBWMHZ))

    def set_video_bw_UI(self):
        VBWMHZ = self.videobwLineEdit.text()
        vbw = self.agilent_analyzer.read_probes("video band width")/ 1000000
        if isinstance(VBWMHZ, str) or isinstance(VBWMHZ, int) or isinstance(VBWMHZ, float):
            try:
                VBWMHZ = float(VBWMHZ)
                if VBWMHZ < _min_video_BW or VBWMHZ > _max_BW:
                    print("Wrong VBW input")
                    VBWMHZ = vbw
            except ValueError:
                print("Wrong VBW input")
                VBWMHZ = vbw
        self.agilent_analyzer.update({"video band width": VBWMHZ})
        self.videobwLineEdit.setText(str(VBWMHZ))

    def set_marker_min_UI(self):
        self.agilent_analyzer.write("MKMIN;")

    def set_marker_max_UI(self):
        self.agilent_analyzer.write("MKPK HI;")

    def set_marker_delta_UI(self):
        self.agilent_analyzer.write("MKD;")

    def set_marker_span_UI(self):
        self.agilent_analyzer.write("MKSP;")
        span = self.agilent_analyzer.read_probes("span")/ 1000000
        self.spanLineEdit.setText(str(span))

    def video_avg_UI(self):
        num_points = self.videoavgLineEdit.text()
        if isinstance(num_points, str) or isinstance(num_points, int) or isinstance(num_points, float):
            try:
                num_points = int(num_points)
                if num_points < 1 or num_points > _MAX_SET_AVG_PTS:
                    num_points = _default_smooth_UI
            except ValueError:
                num_points = _default_smooth_UI
        self.agilent_analyzer.update({"video average": num_points})
        self.videoavgLineEdit.setText(str(num_points))

    def clr_avg_UI(self):
        self.agilent_analyzer.write("CLRAVG;")

    def fft_UI(self):
        center_frequencyMHZ = self.agilent_analyzer.read_probes("center frequency")/ 1000000
        spanMHZ=self.agilent_analyzer.read_probes("span")/ 1000000
        RBWMHZ=self.agilent_analyzer.read_probes("resolution band width")/ 1000000
        fft_sweep_time = float(self.fft_sweep_timeLineEdit.text())
        QMessageBox.information(self, 'Max Modulation Rate', f'Please note that the maximum modulation rate that our device can resolve is 400/(2*fft_sweep_time) = '+str(400/(2*fft_sweep_time))+' MHz')
        window = self.comboBox.currentText()
        self.single_FFT(center_frequencyMHZ, spanMHZ, RBWMHZ, fft_sweep_time, window)
        if self.agilent_analyzer.read_probes("FFT clip"):
            QMessageBox.information(self, 'FFT signal is clipped', f'FFT signal is clipped! Please change your parameters and run FFT again.')

    def single_FFT(self, center_frequencyMHZ=300, spanMHZ=0.2, RBWMHZ=3, sweep_time=3333, window = "FLATTOP"):
        """FFT weights the source trace with the function in the window trace. The transform is computed and the results are placed in the destination
        trace. Unlike FFTAUTO and FFTCONTS, FFT performs the FFT measurement only once. Use
        FFTAUTO or FFTCONTS if you want the FFT measurement to be performed at the end of every
        measurement sweep.
        The spectrum analyzer should be in linear mode when using the FFT command. The FFT
        results are displayed on the spectrum analyzer in logarithmic scale. For the horizontal
        dimension, the frequency on the left side of the graph is 0 Hz, and on the right side is Fmax.
        Fmax can be calculated using a few simple equations and the sweep time of the spectrum
        analyzer. The sweep time divided by the number of trace array elements containing amplitude
        information is equal to the sampling period. The reciprocal of the sampling period is the
        sampling rate. The sampling rate divided by two yields Fmax.
        For example, let the sweep time of the spectrum analyzer be 20 ms and the number of trace
        elements be 400. The sweep time (20 ms) divided by 400 equals 50 ps, the sampling period.
        The sample rate is l/50 ps. Fmax equals l/50 p.s divided by 2, or 10 kHz.
        FFT is designed to be used in transforming zero span information into the frequency domain.
        Performing FFT on a frequency sweep (when the frequency span is greater than zero) will not
        provide time-domain results.
        The windowing function stored in the window trace may be selected with the trace window
        (TWNDOW) command. You may also store your own values in that trace. The trace window
        function modifies the contents of a trace array according to one of three built-in algorithms:
        UNIFORM, HANNING, or FLATTOP
        """
        self.agilent_analyzer.write("IP;")
        self.agilent_analyzer.write("SNGLS;")
        self.agilent_analyzer.update({"center frequency": center_frequencyMHZ})
        print("center frequency:")
        print(self.agilent_analyzer.read_probes("center frequency")/ 1000000)
        self.agilent_analyzer.update({"span": spanMHZ})
        print("span:")
        print(self.agilent_analyzer.read_probes("span")/ 1000000)
        self.agilent_analyzer.write("TS;")
        self.agilent_analyzer.write("MKPK HI;")
        print("Marker frequency at peak:")
        print(self.agilent_analyzer.self.read_probes("marker frequency") / 1000000)
        self.agilent_analyzer.write("MKTRACK ON;")
        self.agilent_analyzer.write("CONTS;")
        self.agilent_analyzer.update({"resolution band width": RBWMHZ})
        print("resolution bw:")
        print(self.agilent_analyzer.read_probes("resolution band width")/ 1000000)
        self.agilent_analyzer.write("MKTRACK OFF;")
        self.agilent_analyzer.update({"span": 0})
        print("span:")
        print(self.agilent_analyzer.read_probes("span")/ 1000000)
        self.agilent_analyzer.write("MKPK HI;")
        print("marker freq at peak:")
        print(self.agilent_analyzer.read_probes("marker frequency") / 1000000)
        self.agilent_analyzer.write("MKRL")
        self.agilent_analyzer.write("LN")
        self.agilent_analyzer.write("SNGLS;")
        self.agilent_analyzer.update({"sweep time": sweep_time})
        print("sweep time:")
        print(self.agilent_analyzer.read_probes("sweep time")* 1000000)
        self.agilent_analyzer.write("TS;")
        self.agilent_analyzer.write("TWNDOW TRB," + window + ";")
        self.agilent_analyzer.write("FFT TRA,TRA,TRB;")
        self.agilent_analyzer.write("VIEW TRA;")
        self.agilent_analyzer.write("MKPK HI;")
        self.agilent_analyzer.write("MKD;")
        self.agilent_analyzer.write("MKPK NH;")
        print("marker freq at NH peak:")
        print(self.agilent_analyzer.read_probes("marker frequency") / 1000000)
        self.agilent_analyzer.write("MKREAD FFT;")
        print(self.agilent_analyzer.read_probes("marker frequency") / 1000000 )

    def start_fft(self):
        try:
            print("Starting fft...")
            self.startfftButton.setEnabled(False)
            self.stopfftButton.setEnabled(True)
            self.clearButton.setEnabled(False)
            self.spanLineEdit.setEnabled(False)
            self.cfLineEdit.setEnabled(False)
            self.spanButton.setEnabled(False)
            self.centerfreqButton.setEnabled(False)
            self.snapshotButton.setEnabled(False)
            self.setsweeptimeButton.setEnabled(False)
            self.saveButton.setEnabled(False)
            self.lastspanButton.setEnabled(False)
            self.resolutionbwButton.setEnabled(False)
            self.videobwButton.setEnabled(False)
            self.markerminButton.setEnabled(False)
            self.markermaxButton.setEnabled(False)
            self.markerdeltaButton.setEnabled(False)
            self.markerspanButton.setEnabled(False)
            self.videoavgButton.setEnabled(False)
            self.clravgButton.setEnabled(False)
            self.update_fft.start(500)
        except ValueError as e:
            print(f"ValueError: {e}")
            QMessageBox.warning(self, 'Warning', 'Invalid numeric input')
        except Exception as e:
            print(f"Exception: {e}")
            QMessageBox.warning(self, 'Warning', f'Unexpected error: {e}')

    def stop_fft(self):
        self.update_fft.stop()
        self.spanLineEdit.setEnabled(True)
        self.cfLineEdit.setEnabled(True)
        self.spanButton.setEnabled(True)
        self.centerfreqButton.setEnabled(True)
        self.snapshotButton.setEnabled(True)
        self.setsweeptimeButton.setEnabled(True)
        self.saveButton.setEnabled(True)
        self.startfftButton.setEnabled(True)
        self.stopfftButton.setEnabled(False)
        self.clearButton.setEnabled(True)
        self.lastspanButton.setEnabled(True)
        self.resolutionbwButton.setEnabled(True)
        self.videobwButton.setEnabled(True)
        self.markerminButton.setEnabled(True)
        self.markermaxButton.setEnabled(True)
        self.markerdeltaButton.setEnabled(True)
        self.markerspanButton.setEnabled(True)
        self.videoavgButton.setEnabled(True)
        self.clravgButton.setEnabled(True)

    def fftupdate_plot(self):
        if self.fftfrequency_data is None or self.fftamplitude_data is None:
            return
        if len(self.fftfrequency_data) == 0 or len(self.fftamplitude_data) == 0:
            return
        self.sa_plot.setData(self.fftfrequency_data, self.fftamplitude_data)

    def fftacquire_and_plot_data(self):
        try:
            self.agilent_analyzer.write("TDF P;")
            self.agilent_analyzer.write("SNGLS;")
            self.agilent_analyzer.write("TS;")
            self.agilent_analyzer.write("MKPK HI;")
            center_freq = self.agilent_analyzer.read_probes("center frequency")/ 1000000
            mkfreq = self.agilent_analyzer.read_probes("marker frequency") / 1000000
            amplitude = self.agilent_analyzer.read_probes("marker amplitude")
            print(f"Status: cf={center_freq}")
            print(f"Status: markerfreq={mkfreq}")
            print(f"Status: amplitude={amplitude}")
            self.agilent_analyzer.write("TS;")
            ffttrace_data = np.array(self.agilent_analyzer.read_probes("trace AP"))
            num_points = len(ffttrace_data)
            span = self.agilent_analyzer.read_probes("span")/ 1000000
            start_freq = 0
            fft_sweep_time = float(self.fft_sweep_timeLineEdit.text())
            stop_freq = 400/(2*fft_sweep_time)
            freqs = np.linspace(start_freq, stop_freq, num_points)
            self.agilent_analyzer.write("CONTS;")
            self.fftfrequency_data = list(freqs / 1e9)
            self.fftamplitude_data = list(ffttrace_data)
            self.fftupdate_plot()

        except Exception as e:
            print(f"Error acquiring data: {e}")
            self.update_fft.stop()
            QMessageBox.critical(self, "Acquisition Error", str(e))