# Created by Jannet Trabelsi on 2025-10-10
# UI rebuilt in code (no .ui / Ui_Form) with responsive layouts; plot + camera
# sizes are driven by the actual frame dimensions instead of hardcoded 680x510.
from __future__ import annotations
import sys
from src.View.windows_and_widgets.camera_widget import Amscope_Camera_View
from src.View.windows_and_widgets.camera_widget import ROPER_CASCADE_CCD_View
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QMessageBox,
    QSlider,
    QLineEdit,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
)


class Display_View(QWidget):
    """
    Display widget: live/snapshot camera view with z-vs-x / z-vs-y line plots
    and crosshair controls. All widgets are created in code (no .ui file).
    """
    x_crosshair = pyqtSignal(int)
    y_crosshair = pyqtSignal(int)
    # ---- AutoTune config (Roper only). EDIT the sequences to match your hardware. ----
    AUTOTUNE_TARGET_MIN = 15000
    AUTOTUNE_TARGET_MAX = 39000
    AUTOTUNE_SATURATION = 40000
    # microns per pixel (from calibration)
    MICRONS_PER_PIXEL_H = 0.3064457122  # horizontal -> z_vs_x_widget (x axis)
    MICRONS_PER_PIXEL_V = 0.3045589354  # vertical   -> z_vs_y_widget (y axis)
    # both arrays ordered from LEAST to MOST exposure
    ROPER_NOGAIN_SEQUENCE = [10, 50, 100, 150, 200, 300, 500]  # inttime (us)
    ROPER_GAIN_SEQUENCE = [(10, 1), (50, 1), (100, 1), (100, 100),
                           (100, 500), (100, 1000), (100, 1500),
                           (100, 2000), (100, 2500), (100, 3000), (150, 3000),
                           (200, 3000), (300, 3000), (400, 3000), (500, 3000)]
    update_get_img = pyqtSignal(int)

    # default plot "thickness" (the short dimension of each strip plot)
    _ZX_HEIGHT = 150   # z_vs_x is wide and short
    _ZY_WIDTH = 165    # z_vs_y is tall and narrow

    def __init__(self, display_choice="MU300", snapshot_or_live=1, parent=None):
        super().__init__(parent)
        self._build_ui()                      # <-- replaces self.setupUi(self)
        self.widget = None
        self.cascade_controls = None          # container for the Roper line-edits/buttons
        self._suppress_crosshair_signal = False
        self._autotune_mode = None
        self._autotune_seq = None
        self._autotune_index = 0
        self.parent_widget.setStyleSheet("background-color: white;")

        self.display_choice = display_choice
        self.snapshot_or_live = snapshot_or_live
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.acquire_and_plot_data)

        # plots
        self.z_vs_x_plot = pg.PlotWidget(title="z vs x")
        self.z_vs_x_plot.setLabel('left', 'z')
        self.z_vs_x_plot.setLabel('bottom', 'x')
        self.z_vs_x_widget.setLayout(QVBoxLayout())
        self.z_vs_x_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.z_vs_x_widget.layout().addWidget(self.z_vs_x_plot)

        self.z_vs_y_plot = pg.PlotWidget(title="z vs y")
        self.z_vs_y_plot.setLabel('left', 'y')   # y on vertical axis
        self.z_vs_y_plot.setLabel('top', 'z')    # z on horizontal axis
        self.z_vs_y_plot.invertY(True)           # top to bottom
        self.z_vs_y_widget.setLayout(QVBoxLayout())
        self.z_vs_y_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.z_vs_y_widget.layout().addWidget(self.z_vs_y_plot)

        self.crosshair_y.setValue(0)   # THIS WILL LATER LOAD WITH CONFIG FILE
        self.crosshair_x.setValue(0)   # THIS WILL LATER LOAD WITH CONFIG FILE
        self.crosshair_width.setValue(1)

        # Initialize data arrays for plotting
        self.x = []
        self.y = []
        self.z_x = []
        self.z_y = []

        # Initialize plot curves
        self.zx_plot = self.z_vs_x_plot.plot(pen='r', name='zx')
        self.zy_plot = self.z_vs_y_plot.plot(pen='r', name='zy')

        # image dimensions: start at 0, filled in from the real camera frame
        self.w = 0
        self.h = 0

        # Connect buttons to functions
        self.crosshairButton.clicked.connect(self.crosshair)
        self.center_Button.clicked.connect(self.center)
        self.clear_crosshair_Button.clicked.connect(self.clear_crosshair)
        self.AutoTune_checkBox.toggled.connect(self.on_autotune_toggled)
        self.connect_to_display()
        self.start()
        self._apply_crosshair_ranges()
        self.widget.mouseMoved.connect(self.on_widget_hover)
        self.widget.mouseClicked.connect(self.on_widget_click)
        self.crosshair_frozen = False  # Default: move with hover
        self.crosshair_x.valueChanged.connect(self.on_crosshair_changed)
        self.crosshair_y.valueChanged.connect(self.on_crosshair_changed)
        self.crosshair_width.valueChanged.connect(self.on_crosshair_changed)
        self.x_selected = 0
        self.y_selected = 0
        self.Rotate_Button.clicked.connect(self.rotate_image)
        self.axis_choice.currentTextChanged.connect(self.on_axis_choice_changed)
        self._apply_axis_labels()
        self._match_plot_sizes()

    # ============================================================
    # UI construction (was previously the generated Ui_Form)
    # ============================================================
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.parent_widget = QWidget()
        outer.addWidget(self.parent_widget)

        grid = QGridLayout(self.parent_widget)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)

        # plot containers
        self.z_vs_x_widget = QWidget()
        self.z_vs_x_widget.setFixedHeight(self._ZX_HEIGHT)
        self.z_vs_y_widget = QWidget()
        self.z_vs_y_widget.setFixedWidth(self._ZY_WIDTH)

        # crosshair control panel (top-right)
        self._build_crosshair_panel()

        # camera container: verticalLayout holds the camera view + cascade controls
        self.camera_container = QWidget()
        self.verticalLayout = QVBoxLayout(self.camera_container)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(4)
        self.verticalLayout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # MU300 sliders panel (bottom, full width)
        self._build_sliders_panel()

        # layout:
        #   row0: [ z vs x ]      [ crosshair panel ]
        #   row1: [ camera   ]    [ z vs y          ]
        #   row2: [ MU300 sliders (spanning)        ]
        grid.addWidget(self.z_vs_x_widget,   0, 0)
        grid.addWidget(self.crosshair_panel, 0, 1)
        grid.addWidget(self.camera_container, 1, 0, Qt.AlignTop | Qt.AlignLeft)
        grid.addWidget(self.z_vs_y_widget,   1, 1, Qt.AlignTop)
        grid.addWidget(self.sliders_panel,   2, 0, 1, 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)
        grid.setRowStretch(1, 1)

    def _build_crosshair_panel(self):
        self.crosshair_panel = QWidget()
        g = QGridLayout(self.crosshair_panel)
        g.setContentsMargins(4, 4, 4, 4)
        g.setSpacing(4)

        self.crosshairButton = QPushButton("+")
        self.clear_crosshair_Button = QPushButton("Clear")
        self.axis_choice = QComboBox()
        self.axis_choice.addItem("Pixels")
        self.axis_choice.addItem("Microns")

        self.crosshair_width = QSpinBox()
        self.crosshair_width.setMaximum(10000)
        self.step_label = QLabel("step")
        self.crosshair_step = QLineEdit()

        self.x_label = QLabel("x:")
        self.crosshair_x = QDoubleSpinBox()
        self.crosshair_x.setDecimals(2)
        self.crosshair_x.setMaximum(1e9)
        self.y_label = QLabel("y:")
        self.crosshair_y = QDoubleSpinBox()
        self.crosshair_y.setDecimals(2)
        self.crosshair_y.setMaximum(1e9)

        self.center_Button = QPushButton("center")
        self.Rotate_Button = QPushButton("Rot")
        self.AutoTune_checkBox = QCheckBox("AutoTune")

        g.addWidget(self.crosshairButton,        0, 0)
        g.addWidget(self.clear_crosshair_Button, 0, 1)
        g.addWidget(self.axis_choice,            0, 2)
        g.addWidget(self.crosshair_width,        1, 0)
        g.addWidget(self.step_label,             1, 1)
        g.addWidget(self.crosshair_step,         1, 2)
        g.addWidget(self.x_label,                2, 0)
        g.addWidget(self.crosshair_x,            2, 1, 1, 2)
        g.addWidget(self.y_label,                3, 0)
        g.addWidget(self.crosshair_y,            3, 1, 1, 2)
        g.addWidget(self.center_Button,          4, 0)
        g.addWidget(self.Rotate_Button,          4, 1)
        g.addWidget(self.AutoTune_checkBox,      4, 2)
        g.setRowStretch(5, 1)

    def _build_sliders_panel(self):
        # the 9 MU300 sliders, named horizontalSlider_10 .. _18 (used by build_sliders)
        self.sliders_panel = QWidget()
        g = QGridLayout(self.sliders_panel)
        g.setContentsMargins(4, 4, 4, 4)
        names = ["exposure gain", "exposure time", "brightness", "saturation",
                 "contrast", "Gamma", "Temp", "Tint", "Hue"]
        for k, name in enumerate(names):
            idx = 10 + k
            slider = QSlider(Qt.Horizontal)
            setattr(self, f"horizontalSlider_{idx}", slider)
            col = (k // 3) * 2     # 3 columns of (label, slider)
            row = k % 3
            g.addWidget(QLabel(name), row, col)
            g.addWidget(slider, row, col + 1)

    def _match_plot_sizes(self):
        """Schedule an alignment pass (after layout settles) so the plot data
        areas line up with the camera image edges."""
        QTimer.singleShot(0, self._align_axes_to_camera)

    def _align_axes_to_camera(self):
        """Align the plot DATA areas (not the widgets) with the camera image:
        x=0 at the image's left edge, y=0 at the image's top edge, and the data
        span equal to the image width/height."""
        if not self.w or not self.h or self.widget is None:
            return
        try:
            # data-area rectangles inside each plot widget (in widget pixels)
            rx = self.z_vs_x_plot.getPlotItem().getViewBox().geometry()
            ry = self.z_vs_y_plot.getPlotItem().getViewBox().geometry()
            if rx.width() <= 0 or ry.height() <= 0:
                return  # not rendered yet; a later pass will catch it

            # z vs x: make the data area exactly as wide as the image
            nondata_w = self.z_vs_x_widget.width() - rx.width()
            self.z_vs_x_widget.setFixedWidth(int(round(self.w + max(0, nondata_w))))

            # z vs y: make the data area exactly as tall as the image
            nondata_h = self.z_vs_y_widget.height() - ry.height()
            self.z_vs_y_widget.setFixedHeight(int(round(self.h + max(0, nondata_h))))

            # shift the camera so its top-left corner meets the two data origins:
            #   left margin  = where z-vs-x data starts (past its z value-axis)
            #   top margin   = where z-vs-y data starts (past its title/top axis)
            left_off = int(round(rx.left()))
            top_off = int(round(ry.top()))
            self.verticalLayout.setContentsMargins(left_off, top_off, 0, 0)
        except Exception:
            pass

    # ============================================================
    # Units / scaling helpers
    # ============================================================
    def _x_scale(self):  # microns per pixel, or 1.0 in pixel mode (horizontal)
        return self.MICRONS_PER_PIXEL_H if self._using_microns() else 1.0

    def _y_scale(self):  # vertical
        return self.MICRONS_PER_PIXEL_V if self._using_microns() else 1.0

    def _set_crosshair_px(self, px, py):
        """Put PIXEL coords into the spinboxes, shown in the current unit (no re-trigger)."""
        self._suppress_crosshair_signal = True
        self.crosshair_x.setValue(px * self._x_scale())
        self.crosshair_y.setValue(py * self._y_scale())
        self._suppress_crosshair_signal = False

    def _get_crosshair_px(self):
        """Read the spinboxes (current unit) and return PIXEL coords."""
        px = int(round(self.crosshair_x.value() / self._x_scale()))
        py = int(round(self.crosshair_y.value() / self._y_scale()))
        return px, py

    def _apply_crosshair_ranges(self):
        self._suppress_crosshair_signal = True
        self.crosshair_x.setMinimum(0.0)
        self.crosshair_y.setMinimum(0.0)
        self.crosshair_x.setMaximum(self.w * self._x_scale())
        self.crosshair_y.setMaximum(self.h * self._y_scale())
        self._suppress_crosshair_signal = False

    def rotate_image(self):
        view = self.widget
        if view is None or not hasattr(view, "rotate_90"):
            return  # only the Roper view supports rotation
        view.rotate_90()
        # refresh immediately with a rotated frame
        frame = view.get_latest_frame()
        if frame is not None:
            view.show_frame(frame)
            self.img_gray = frame
            self.h, self.w = frame.shape
            self._apply_crosshair_ranges()
            self._match_plot_sizes()
            self.draw_crosshair(self.x_selected, self.y_selected)
            self._update_intensity_readout()

    def update_choices(self, display_choice, snapshot_or_live):
        # gets the signals from main and only updates the display if the choices differ
        if display_choice != self.display_choice:
            print("update_choices called display choice changed")
            self.update_timer.stop()
            self.display_choice = display_choice
            self.connect_to_display()
        if snapshot_or_live != self.snapshot_or_live:
            print("update_choices called snapshot_or_live changed to:" + str(snapshot_or_live))
            self.update_timer.stop()
            self.snapshot_or_live = snapshot_or_live
        self.start()

    def update_color_scale_option(self, new_color_scale_choice):
        if self.display_choice == "CASCADE CCD":
            self.widget.color_scale_choice = new_color_scale_choice

    def connect_to_display(self):
        """Connect the camera device; remove any previously-loaded view first."""
        # remove any previously-loaded camera view so they don't stack/overlap
        if self.widget is not None:
            try:
                self.widget.stop_live_view()
            except Exception:
                pass
            self.verticalLayout.removeWidget(self.widget)
            self.widget.setParent(None)
            self.widget.deleteLater()
            self.widget = None
        if self.display_choice == 'MU300':
            self.crosshairButton.setEnabled(True)
            try:
                self.widget = Amscope_Camera_View()
                self.verticalLayout.addWidget(self.widget)
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        elif self.display_choice == 'CASCADE CCD':
            self.crosshairButton.setEnabled(True)
            try:
                self.widget = ROPER_CASCADE_CCD_View()
                self.verticalLayout.addWidget(self.widget)
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            return

    def _update_get_image_button(self):
        is_roper = (self.display_choice == 'CASCADE CCD')
        enabled = (is_roper and self.snapshot_or_live == 0)
        self.update_get_img.emit(enabled)
        self.AutoTune_checkBox.setEnabled(is_roper)

    def on_autotune_toggled(self, checked):
        if not checked:  # ignore the un-check
            return
        if self.display_choice != 'CASCADE CCD':
            self.AutoTune_checkBox.setChecked(False)
            return

        # 1) Gain / No gain / Cancel
        box = QMessageBox(self)
        box.setWindowTitle("AutoTune")
        box.setText("Choose AutoTune mode:")
        gain_btn = box.addButton("Gain", QMessageBox.AcceptRole)
        nogain_btn = box.addButton("No gain", QMessageBox.AcceptRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()

        if clicked is cancel_btn or clicked is None:
            self.AutoTune_checkBox.setChecked(False)  # deselect
            return

        if clicked is gain_btn:
            self._autotune_mode = "gain"
            seq = self.ROPER_GAIN_SEQUENCE
            labels = [f"{i}: inttime={it} us, gain={g}" for i, (it, g) in enumerate(seq)]
        else:
            self._autotune_mode = "nogain"
            seq = self.ROPER_NOGAIN_SEQUENCE
            labels = [f"{i}: inttime={it} us" for i, it in enumerate(seq)]

        # 2) pick the starting point in the chosen array
        from PyQt5.QtWidgets import QInputDialog
        label, ok = QInputDialog.getItem(
            self, "AutoTune start", "Select the starting point:", labels, 0, False)
        if not ok:
            self.AutoTune_checkBox.setChecked(False)
            return

        self._autotune_seq = seq
        self._autotune_index = labels.index(label)

        # 3) run the tuning loop
        self._autotune_run()

    def _apply_autotune_setting(self, setting):
        if self._autotune_mode == "gain":
            inttime, gain = setting
            self.widget.hcam.update({'inttime': float(inttime), 'gain': float(gain)})
        else:
            self.widget.hcam.update({'inttime': float(setting)})

    def _autotune_run(self):
        seq = self._autotune_seq
        idx = self._autotune_index
        frame = None
        max_pixel = None
        for _ in range(2 * len(seq)):  # bounded; can't spin forever
            self._apply_autotune_setting(seq[idx])
            QApplication.processEvents()  # keep UI alive during exposures
            frame = self.widget.get_latest_frame()
            if frame is None:
                break
            max_pixel = float(np.max(frame))
            if max_pixel >= self.AUTOTUNE_SATURATION or max_pixel > self.AUTOTUNE_TARGET_MAX:
                if idx > 0:
                    idx -= 1  # too bright -> less exposure
                    continue
                break  # already at the lowest setting
            elif max_pixel < self.AUTOTUNE_TARGET_MIN:
                if idx < len(seq) - 1:
                    idx += 1  # too dim -> more exposure
                    continue
                break  # already at the highest setting
            else:
                break  # in the ideal 15k-39k window
        self._autotune_index = idx

        # reflect the chosen values in the line-edits (if the cascade controls exist)
        try:
            if self._autotune_mode == "gain":
                it, g = seq[idx]
                self.inttime_edit.setText(str(it))
                self.gain_edit.setText(str(g))
            else:
                self.inttime_edit.setText(str(seq[idx]))
        except Exception:
            pass

        # show the tuned frame and refresh the plots
        if frame is not None:
            self.widget.show_frame(frame)
            self.img_gray = frame
            self.h, self.w = frame.shape
            self._match_plot_sizes()
            self.draw_crosshair(self.x_selected, self.y_selected)
            self._update_intensity_readout()

        if max_pixel is not None:
            QMessageBox.information(
                self, "AutoTune",
                f"Final max pixel: {int(max_pixel)} "
                f"(target {self.AUTOTUNE_TARGET_MIN}-{self.AUTOTUNE_TARGET_MAX}).")

    def get_image_snapshot(self):
        """Roper snapshot: grab a full image with read_probes('image') and overwrite the display."""
        try:
            raw = self.widget.hcam.read_probes("image")
        except Exception as e:
            QMessageBox.warning(self, "Get image", f"Could not collect image: {e}")
            return

        # MATLAB double -> numpy (column-major), same as get_latest_frame
        try:
            frame = np.array(raw._data, dtype=np.float64).reshape(raw.size, order="F")
        except Exception:
            frame = np.asarray(raw, dtype=np.float64)

        # overwrite whatever was showing
        self.widget.show_frame(frame)

        # refresh the z-vs-x / z-vs-y plots from the new image
        self.img_gray = frame
        self.h, self.w = frame.shape
        self._apply_crosshair_ranges()
        self._match_plot_sizes()
        self.draw_crosshair(self.x_selected, self.y_selected)
        self._update_intensity_readout()

    # 0 means snapshot, 1 means live
    def start(self):
        self._update_get_image_button()
        if self.snapshot_or_live == 0:
            self.update_timer.stop()
            if self.widget is not None and self.display_choice == 'CASCADE CCD':
                try:
                    self.widget.stop_live_view()
                except Exception:
                    pass
                # --- NEW: show one live-style frame the instant we connect ---
                frame = self.widget.snapshot_one_frame()
                if frame is not None:
                    self.img_gray = frame
                    self.h, self.w = frame.shape
                    self._apply_crosshair_ranges()
                    self._match_plot_sizes()
                    self.build_cascade_controls()  # inttime/gain panel + intensity readout
                    self.draw_crosshair(getattr(self, "x_selected", 0),
                                        getattr(self, "y_selected", 0))
        else:
            if self.widget is None:
                return
            self.widget.start_live_view()
            # if the hardware didn't actually open, stop here (no crash, no sliders/plots)
            if getattr(self.widget, 'hcam', None) is None:
                return
            # pick up the real sensor size from the camera view and size everything to it
            if getattr(self.widget, 'w', 0) and getattr(self.widget, 'h', 0):
                self.w, self.h = self.widget.w, self.widget.h
                self._apply_crosshair_ranges()
                self._match_plot_sizes()
            if self.display_choice == 'MU300':
                self._set_sliders_visible(True)
                self._remove_cascade_controls()
                self.build_sliders()
            elif self.display_choice == 'CASCADE CCD':
                self._set_sliders_visible(False)
                self.build_cascade_controls()
            try:
                self.update_timer.start(500)
            except ValueError as e:
                print(f"ValueError: {e}")
                QMessageBox.warning(self, 'Warning', 'Invalid numeric input')
            except Exception as e:
                print(f"Exception: {e}")
                QMessageBox.warning(self, 'Warning', f'Unexpected error: {e}')

    def build_sliders(self):
        if self.widget is None or getattr(self.widget, "hcam", None) is None:
            return
        params = [
            "exposure gain", "exposure time", "brightness", "saturation",
            "contrast", "Gamma", "Temp", "Tint", "Hue"]
        i = 9
        for name in params:
            i += 1
            current_val = self.widget.hcam.read_probes(name)
            min_value, max_value = self.widget.hcam.return_min_max(name)
            slider = getattr(self, f"horizontalSlider_{i}")
            self._add_slider(name, slider, min_value, max_value, current_val)

    def _set_sliders_visible(self, visible):
        # hide/show the whole MU300 slider panel (and the individual sliders)
        self.sliders_panel.setVisible(visible)
        for i in range(10, 19):
            slider = getattr(self, f"horizontalSlider_{i}", None)
            if slider is not None:
                slider.setVisible(visible)

    def build_cascade_controls(self):
        # rebuild from scratch so they never stack when you switch/restart
        self._remove_cascade_controls()

        # prefill the fields with the camera's current values
        try:
            cur_inttime = self.widget.hcam.read_probes("inttime")
        except Exception:
            cur_inttime = ""
        try:
            cur_gain = self.widget.hcam.read_probes("gain")
        except Exception:
            cur_gain = ""

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        # --- integration time row ---
        self.inttime_edit = QLineEdit(str(cur_inttime))
        inttime_btn = QPushButton("Set integration time")
        inttime_btn.clicked.connect(self._apply_cascade_inttime)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Integration time (us):"))
        row1.addWidget(self.inttime_edit)
        row1.addWidget(inttime_btn)
        vbox.addLayout(row1)

        # --- gain row ---
        self.gain_edit = QLineEdit(str(cur_gain))
        gain_btn = QPushButton("Set gain")
        gain_btn.clicked.connect(self._apply_cascade_gain)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Gain:"))
        row2.addWidget(self.gain_edit)
        row2.addWidget(gain_btn)
        vbox.addLayout(row2)

        # --- live min/max intensity readouts ---
        self.min_intensity_label = QLabel("min intensity: --")
        self.max_intensity_label = QLabel("max intensity: --")
        vbox.addWidget(self.min_intensity_label)
        vbox.addWidget(self.max_intensity_label)

        self.cascade_controls = container
        self.verticalLayout.addWidget(container)
        self._update_intensity_readout()

    def _remove_cascade_controls(self):
        container = getattr(self, "cascade_controls", None)
        if container is not None:
            self.verticalLayout.removeWidget(container)
            container.setParent(None)
            container.deleteLater()
            self.cascade_controls = None
        self.min_intensity_label = None
        self.max_intensity_label = None

    def _update_intensity_readout(self):
        """Refresh the min/max intensity labels from the current frame (Roper panel only)."""
        minlbl = getattr(self, "min_intensity_label", None)
        maxlbl = getattr(self, "max_intensity_label", None)
        if minlbl is None or maxlbl is None:
            return
        img = getattr(self, "img_gray", None)
        if img is None:
            return
        try:
            vmin = float(np.min(img))
            vmax = float(np.max(img))
            minlbl.setText(f"min intensity: {vmin:.0f}")
            maxlbl.setText(f"max intensity: {vmax:.0f}")
        except Exception:
            pass

    def _apply_cascade_inttime(self):
        text = self.inttime_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", f"'{text}' is not a valid number.")
            return
        try:
            self.widget.hcam.update({'inttime': value})
        except Exception as e:
            QMessageBox.warning(self, "Camera error", f"Could not set integration time: {e}")

    def _apply_cascade_gain(self):
        text = self.gain_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", f"'{text}' is not a valid number.")
            return
        try:
            self.widget.hcam.update({'gain': value})
        except Exception as e:
            QMessageBox.warning(self, "Camera error", f"Could not set gain: {e}")

    def _add_slider(self, name, slider, min_value, max_value, current_val):
        value_label = QLabel(str(current_val))
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(current_val)

        def on_change(value):
            value_label.setText(str(value))
            self.widget.hcam.update({name: value})
        slider.valueChanged.connect(on_change)

    def acquire_and_plot_data(self):
        try:
            img_rgb = self.widget.get_latest_frame()
            if img_rgb is None:
                return
            if img_rgb.ndim == 2:
                # Roper Cascade: already 2-D grayscale (raw CCD counts)
                self.h, self.w = img_rgb.shape
                self.img_gray = img_rgb
            else:
                # Amscope: RGB -> grayscale (unchanged)
                self.h, self.w, _ = img_rgb.shape
                self.img_gray = np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140])
            # Crosshair center coordinates (pixels)
            x = int(self.x_selected)
            y = int(self.y_selected)
            # Crosshair thickness (averaging width)
            width = int(self.crosshair_width.value())
            half_w = max(1, width // 2)
            # Ensure bounds don't exceed image size
            y_min = max(0, y - half_w)
            y_max = min(self.h, y + half_w)
            x_min = max(0, x - half_w)
            x_max = min(self.w, x + half_w)
            # z vs x = average intensity across horizontal stripe
            self.z_x = np.mean(self.img_gray[y_min:y_max, :], axis=0)
            # z vs y = average intensity across vertical stripe
            self.z_y = np.mean(self.img_gray[:, x_min:x_max], axis=1)
            # Axes (always stored in pixels)
            self.x = np.arange(self.w)
            self.y = np.arange(self.h)
            self.update_plot()
            self._match_plot_sizes()
            self._update_intensity_readout()

        except Exception as e:
            print(f"Error acquiring data: {e}")
            self.update_timer.stop()
            QMessageBox.critical(self, "Acquisition Error", str(e))

    def update_plot(self):
        if self.x is None or self.y is None or self.z_x is None or self.z_y is None:
            return
        if len(self.x) == 0 or len(self.y) == 0 or len(self.z_x) == 0 or len(self.z_y) == 0:
            return
        x_axis, y_axis = self._scaled_axes()
        self.zx_plot.setData(x_axis, self.z_x)
        self.zy_plot.setData(self.z_y, y_axis)

    def _using_microns(self):
        return self.axis_choice.currentText().lower().startswith("micron")

    def _scaled_axes(self):
        # self.x / self.y are always stored in pixels; convert only for display
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if self._using_microns():
            x = x * self.MICRONS_PER_PIXEL_H  # horizontal
            y = y * self.MICRONS_PER_PIXEL_V  # vertical
        return x, y

    def _apply_axis_labels(self):
        unit = "microns" if self._using_microns() else "pixels"
        self.z_vs_x_plot.setLabel('bottom', f'x ({unit})')  # horizontal axis
        self.z_vs_y_plot.setLabel('left', f'y ({unit})')    # vertical axis

    def on_axis_choice_changed(self, _text=None):
        self._apply_axis_labels()
        self._apply_crosshair_ranges()  # ranges first (so the value isn't clamped)
        self._set_crosshair_px(self.x_selected, self.y_selected)  # re-show current point in new unit
        self.update_plot()

    def close(self):
        if self.snapshot_or_live == 1:
            self.widget.stop_live_view()
            self.widget.stop()

    def plot_clicked(self, mouse_event):
        if mouse_event.button() == Qt.LeftButton:
            viewbox = self.plot_widget.getViewBox()
            mouse_point = viewbox.mapSceneToView(mouse_event.scenePos())
            self.x_selected = mouse_point.x()
            self.y_selected = mouse_point.y()

    def draw_crosshair(self, x, y):
        x = int(x)
        y = int(y)
        thickness = int(self.crosshair_width.value())
        self.widget.label.enable_crosshair(x, y, thickness)  # overlay is always in pixels
        if self.snapshot_or_live == 0:
            half_w = max(1, thickness // 2)
            y_min = max(0, y - half_w)
            y_max = min(self.h, y + half_w)
            x_min = max(0, x - half_w)
            x_max = min(self.w, x + half_w)
            self.z_x = np.mean(self.img_gray[y_min:y_max, :], axis=0)
            self.z_y = np.mean(self.img_gray[:, x_min:x_max], axis=1)
            self.x = np.arange(self.w)
            self.y = np.arange(self.h)
            self.update_plot()

    def on_crosshair_changed(self):
        if self._suppress_crosshair_signal:  # ignore programmatic setValue
            return
        px, py = self._get_crosshair_px()  # user typed/stepped in current unit
        self.x_selected = px
        self.y_selected = py
        self.draw_crosshair(px, py)
        self.x_crosshair.emit(px)
        self.y_crosshair.emit(py)

    def on_widget_hover(self, x, y):  # x, y are pixels from the label
        if not self.crosshair_frozen:
            self.x_selected = x
            self.y_selected = y
            self._set_crosshair_px(x, y)
            self.draw_crosshair(x, y)

    def on_widget_click(self, x, y):  # x, y are pixels from the label
        self.crosshair_frozen = True
        self.x_selected = x
        self.y_selected = y
        self._set_crosshair_px(x, y)
        self.draw_crosshair(x, y)
        self.widget.label.waiting_for_crosshair_click(False)

    def center(self):
        x_center = self.w // 2
        y_center = self.h // 2
        self.x_selected = x_center
        self.y_selected = y_center
        self._set_crosshair_px(x_center, y_center)
        self.draw_crosshair(x_center, y_center)

    def crosshair(self):
        self.widget.label.waiting_for_crosshair_click(True)
        self.crosshair_frozen = False
        self._set_crosshair_px(self.x_selected, self.y_selected)
        self.draw_crosshair(self.x_selected, self.y_selected)

    def clear_crosshair(self):
        self.widget.label.disable_crosshair()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Display_View()
    w.show()
    sys.exit(app.exec())
