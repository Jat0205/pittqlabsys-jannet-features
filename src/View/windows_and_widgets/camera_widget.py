# original amscope only: Kai's code
# modified by: Jannet Trabelsi: 10_2025: fixed amscope bugs and added roper cascade along with the parameter adjustments
from __future__ import annotations
import sys
import ctypes
import time
from typing import Optional
from src.Controller import toupcam
from src.Controller import Amscope_MU_Camera
from src.Controller import Roper_Cascade_Camera
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QCheckBox,
    QVBoxLayout,
    QHBoxLayout,
    QDesktopWidget,
    QMessageBox,
    QSlider
)
import numpy as np
import weakref
from src.core.struct_hdf5 import save_parameters_hdf5, load_data

class SnapWin(QWidget):
    """Separate window that shows still‑image captures."""

    def __init__(self, w: int, h: int):
        super().__init__()
        self.setWindowTitle("Snapshot")
        self.setFixedSize(w, h)
        self.label = QLabel(self)
        self.label.resize(w, h)
        self.label.setScaledContents(False)

    def show_frame(self, qimg: QImage):
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.show()

class Amscope_Camera_View(QWidget):
    """Live‑view window. Compatible with the legacy *app.py* launcher."""

    eventImage = pyqtSignal(int)
    mouseMoved = pyqtSignal(int, int)
    mouseClicked = pyqtSignal(int, int)

    def __init__(
        self,
        gain: int = 100,
        integration_time_us: int = 10_000,
        res: str = "low",
    ) -> None:
        super().__init__()

        self.hcam: Optional[Amscope_MU_Camera.Amscope_MU_Camera] = None
        self.buf: Optional[ctypes.Array] = None
        self.w = self.h = 0
        self.gain = gain
        self.integration = integration_time_us  # already in µs
        self.res = "low"

        # frame counter for FPS display
        self._frame_accum = 0
        self._last_tick = time.perf_counter()

        self._init_ui()
        self.crosshair_enabled = False
        self.crosshair_x = None
        self.crosshair_y = None
        self.crosshair_thickness = 1

    # ── UI ────────────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        # center the window on whatever display we’re on
        #self.setFixedSize(820, 640)  # temp; corrected once cam opens
        geo = self.frameGeometry()
        geo.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(geo.topLeft())

        # widgets
        self.label = QLabel(self)
        self.label = CrosshairLabel(self)
        self.label.mouseMoved.connect(self.mouseMoved)
        self.label.mouseClicked.connect(self.mouseClicked)

        self.label.setScaledContents(False)  # don’t resample!

        self.cb_auto = QCheckBox("Auto Exposure", self)
        self.cb_auto.stateChanged.connect(self._on_auto_exp_toggled)

        self.cb_fps = QCheckBox("Show FPS", self)

        # layout
        cols = QVBoxLayout(self)
        cols.addWidget(self.label, stretch=1)
        row = QHBoxLayout()
        row.addWidget(self.cb_auto)
        row.addWidget(self.cb_fps)
        row.addStretch(1)
        cols.addLayout(row)

    # ── Camera setup ──────────────────────────────────────────────────────────

    def _init_camera(self) -> None:
        cams = toupcam.Toupcam.EnumV2()
        if not cams:
            self.setWindowTitle("No camera found")
            self.cb_auto.setEnabled(False)
            return

        self.camname = cams[0].displayname
        self.setWindowTitle(self.camname)
        self.eventImage.connect(self._on_event_image)

        try:
            self.hcam = Amscope_MU_Camera.Amscope_MU_Camera()
        except toupcam.HRESULTException as ex:
            QMessageBox.warning(self, "", f"Failed to open camera (hr=0x{ex.hr:x})")
            return

        # basic settings
        self.hcam.set_ExpoAGain(self.gain)
        self._clamp_and_set_exposure(self.integration)
        self._apply_resolution(self.res)

        # negotiate RGB/BGR for zero‑copy into QImage
        if sys.platform != "win32":
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BYTEORDER, 1)  # BGR on Linux/mac

        # internal buffer (mutable)
        stride = ((self.w * 24 + 31) // 32) * 4
        self.buf = ctypes.create_string_buffer(stride * self.h)

        # resize widget exactly to sensor size (no scaling cost)
        self.setFixedSize(self.w, self.h + 40)  # + controls bar
        self.label.setFixedSize(self.w, self.h)

        # reflect current auto‑exposure state
        self.cb_auto.setChecked(self.hcam.get_AutoExpoEnable())


        # start stream
        try:
            self_ref = weakref.ref(self)  # weak reference
            self.hcam.StartPullModeWithCallback(self._camera_cb, self_ref)
        except toupcam.HRESULTException as ex:
            QMessageBox.warning(self, "", f"Stream start failed (hr=0x{ex.hr:x})")
            return

    def _clamp_and_set_exposure(self, target_us: int) -> None:
        lo, hi, _ = self.hcam.get_ExpTimeRange()
        self.hcam.put_ExpoTime(max(lo, min(target_us, hi)))

    def _apply_resolution(self, res: str) -> None:
        match res:
            case "high":
                self.hcam.put_eSize(0)  # 2048, 1536
            case "mid":
                self.hcam.put_eSize(1)
            case _:
                self.hcam.put_eSize(2)
        self.w, self.h = self.hcam.get_Size()

    # ── Toupcam callback (runs in SDK thread) ─────────────────────────────––

    @staticmethod
    def _camera_cb(event: int, ctx: "Amscope_Camera_View") -> None:
        if event == toupcam.TOUPCAM_EVENT_IMAGE:
            try:
                ctx.hcam.PullImageV2(ctx.buf, 24, None)
            except toupcam.HRESULTException:
                return  # drop frame
            ctx.eventImage.emit(event)
        elif event == toupcam.TOUPCAM_EVENT_STILLIMAGE:
            try:
                ctx.hcam.PullStillImageV2(ctx.buf, 24, None)
            except toupcam.HRESULTException:
                return
            ctx.eventImage.emit(event)

    # ── Qt slot (runs in GUI thread) ─────────────────────────────────────────

    @pyqtSlot(int)
    def _on_event_image(self, event: int) -> None:
        stride = ((self.w * 24 + 31) // 32) * 4
        qimg = QImage(self.buf, self.w, self.h, stride, QImage.Format_RGB888)
        # qimg.save("frame.png")
        if event == toupcam.TOUPCAM_EVENT_IMAGE:
            pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)
            self._update_fps()
        else:
            pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)
            if not hasattr(self, "_snap_win"):
                self._snap_win = SnapWin(self.w, self.h)
            self._snap_win.show_frame(qimg)

    # ── Misc callbacks ──────────────────────────────────────────────────────

    def _on_auto_exp_toggled(self, state: int) -> None:
        if self.hcam:
            self.hcam.put_AutoExpoEnable(state == Qt.Checked)

    def _update_fps(self) -> None:
        if not self.cb_fps.isChecked():
            return
        self._frame_accum += 1
        now = time.perf_counter()
        if now - self._last_tick >= 1.0:
            fps = self._frame_accum / (now - self._last_tick)
            self.setWindowTitle(f"{self.camname} – {fps:.1f} fps")
            self._frame_accum = 0
            self._last_tick = now

    # ── API for *app.py* ────────────────────────────────────────────────────

    def snap(self):
        if self.hcam:
            self.hcam.Snap(0)

    def stop(self):
        if self.hcam is not None:
            self.hcam.close()
            self.hcam = None

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def closeEvent(self, evt):  # noqa: N802 (Qt override)
        self.stop()
        super().closeEvent(evt)

    def stop_live_view(self):
        self.hcam.pause(0)
        self.hcam.stop()

    def start_live_view(self):
        if self.hcam is None:                 # open hardware only once
            self._init_camera()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns the latest frame as a NumPy array."""
        if not self.buf:
            return None
        try:
            arr = np.frombuffer(self.buf, dtype=np.uint8).reshape((self.h, self.w, 3))
            return arr
        except Exception as e:
            print(f"Frame conversion error: {e}")
            return None

class ROPER_CASCADE_CCD_View(QWidget):
    """Live‑view window. Compatible with the legacy *app.py* launcher."""

    eventImage = pyqtSignal(int)
    mouseMoved = pyqtSignal(int, int)
    mouseClicked = pyqtSignal(int, int)

    def __init__(
            self,
            gain: float = 1.0,
            integration_time_us: float = 100.0,
    ) -> None:
        super().__init__()

        self.hcam = None
        self.buf = None
        self.w = self.h = 0
        self.gain = gain
        self.integration = integration_time_us  # µs

        # frame counter for FPS display
        self._frame_accum = 0
        self._last_tick = time.perf_counter()

        self._init_ui()
        self.crosshair_enabled = False
        self.crosshair_x = None
        self.crosshair_y = None
        self.crosshair_thickness = 1

        self.color_scale_choice = "Grey"
        self.rotation = 0  # degrees, multiples of 90; applied to every frame
        self._lut_cache = {}  # caches 256x3 uint8 colormaps built from MATLAB

    # ── UI ────────────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        # center the window on whatever display we’re on
        # self.setFixedSize(820, 640)  # temp; corrected once cam opens
        geo = self.frameGeometry()
        geo.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(geo.topLeft())

        # widgets
        self.label = QLabel(self)
        self.label = CrosshairLabel(self)
        self.label.mouseMoved.connect(self.mouseMoved)
        self.label.mouseClicked.connect(self.mouseClicked)

        self.label.setScaledContents(False)  # don’t resample!
        self.cb_fps = QCheckBox("Show FPS", self)
        self.inttime = 100.0
        self.gain = 1.0

        # layout
        cols = QVBoxLayout(self)
        cols.addWidget(self.label, stretch=1)
        row = QHBoxLayout()
        row.addWidget(self.cb_fps)
        row.addStretch(1)
        cols.addLayout(row)

    # ── Camera setup ──────────────────────────────────────────────────────────

    def _init_camera(self) -> None:

        self.camname = "Roper Cascade"
        self.setWindowTitle(self.camname)
        self.eventImage.connect(self._on_event_image)

        try:
            self.hcam = Roper_Cascade_Camera.Roper_Cascade_Camera()
        except Exception as ex:
            QMessageBox.warning(self, "", f"Failed to open camera (hr=0x{ex})")
            return

        # basic settings
        self.hcam.update({'gain': self.gain})
        self.hcam.update({'inttime': self.inttime})

        # internal buffer (mutable)
        stride = ((self.w * 24 + 31) // 32) * 4
        self.buf = ctypes.create_string_buffer(stride * self.h)

        # resize widget exactly to sensor size (no scaling cost)
        self.setFixedSize(self.w, self.h + 40)  # + controls bar
        self.label.setFixedSize(self.w, self.h)

        # start stream
        try:
            self.hcam.read_probes("imagefast_int")
        except Exception as ex:
            QMessageBox.warning(self, "", f"Stream start failed {ex}")
            return

    def rotate_90(self):
        """Rotate all subsequent frames by another 90 degrees (cumulative)."""
        self.rotation = (self.rotation + 90) % 360

    def get_latest_frame(self):
        """Grab one frame from the Roper via getimagefast -> 2-D float ndarray (raw counts)."""
        if self.hcam is None:
            return None
        try:
            raw = self.hcam.read_probes("imagefast_int")  # -> matlab.double, 2-D
            arr = np.array(raw._data, dtype=np.float64).reshape(raw.size, order="F")
            if self.rotation:
                arr = np.rot90(arr, k=self.rotation // 90)
            return np.ascontiguousarray(arr)
        except Exception as e:
            print(f"Frame conversion error: {e}")
            return None

    def frame_to_display_rgb(self, frame):
        """Convert a raw 2-D frame to on-screen RGB (autoscale + colormap), (h, w, 3) uint8."""
        if frame is None:
            return None
        fmin, fmax = float(np.min(frame)), float(np.max(frame))
        if fmax > fmin:
            idx = ((frame - fmin) * (255.0 / (fmax - fmin))).astype(np.uint8)
        else:
            idx = np.zeros(frame.shape, dtype=np.uint8)
        lut = self._get_lut(self.color_scale_choice)
        return np.ascontiguousarray(lut[idx])

    def get_display_rgb(self):
        """Grab a fresh frame and return it as on-screen RGB."""
        return self.frame_to_display_rgb(self.get_latest_frame())

    def show_frame(self, frame):
        """Render a given raw 2-D frame to the label (same autoscale + colormap as live)."""
        rgb = self.frame_to_display_rgb(frame)
        if rgb is None:
            return
        self._last_disp = rgb  # keep ref alive for QImage
        h, w = frame.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def get_image(self):
        if self.hcam is None:
            return None
        try:
            raw = self.hcam.read_probes("image")  # -> matlab.double, 2-D
            # MATLAB stores data column-major, so reshape with order='F'
            arr = np.array(raw._data, dtype=np.float64).reshape(raw.size, order="F")
            return arr
        except Exception as e:
            print(f"Error: {e}")
            return None

    @pyqtSlot(int)
    def _on_event_image(self, event: int) -> None:
        frame = self.get_latest_frame()
        if frame is None:
            return

        # autoscale CCD counts to 8-bit just for display
        fmin, fmax = float(np.min(frame)), float(np.max(frame))
        if fmax > fmin:
            idx = ((frame - fmin) * (255.0 / (fmax - fmin))).astype(np.uint8)
        else:
            idx = np.zeros(frame.shape, dtype=np.uint8)

        # map the 8-bit index through the selected colormap -> RGB
        lut = self._get_lut(self.color_scale_choice)
        rgb = np.ascontiguousarray(lut[idx])  # (h, w, 3) uint8
        self._last_disp = rgb  # keep ref alive for QImage
        h, w = idx.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

        # optional FPS in title (inlined; Roper class has no _update_fps)
        if self.cb_fps.isChecked():
            self._frame_accum += 1
            now = time.perf_counter()
            if now - self._last_tick >= 1.0:
                fps = self._frame_accum / (now - self._last_tick)
                self.setWindowTitle(f"{self.camname} – {fps:.1f} fps")
                self._frame_accum = 0
                self._last_tick = now

    def start_live_view(self):
        if self.hcam is None:                 # open hardware only once
            self._init_camera()
            if self.hcam is None:
                return
            # size the window from the first real frame (no fixed SDK buffer here)
            frame = self.get_latest_frame()
            if frame is not None:
                self.h, self.w = frame.shape
                self.setFixedSize(self.w, self.h + 40)
                self.label.setFixedSize(self.w, self.h)
        # (re)start polling; reuse the timer so resuming doesn't reopen the camera
        if getattr(self, "_poll_timer", None) is None:
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(lambda: self.eventImage.emit(0))
        self._poll_timer.start(50)  # ms; increase if CCD readout can't keep up

    def stop_live_view(self):
        timer = getattr(self, "_poll_timer", None)
        if timer is not None:
            timer.stop()
            self._poll_timer = None

    def snapshot_one_frame(self):
        """Snapshot mode: open the camera if needed, grab ONE frame, size the
        view to it, and display it. Does NOT start the live poll timer.
        Returns the raw 2-D frame (or None if the camera didn't open / no frame)."""
        if self.hcam is None:  # open hardware only once
            self._init_camera()
            if self.hcam is None:
                return None
        frame = self.get_latest_frame()
        if frame is None:
            return None
        self.h, self.w = frame.shape
        self.setFixedSize(self.w, self.h + 40)
        self.label.setFixedSize(self.w, self.h)
        self.show_frame(frame)  # autoscale + colormap -> label
        return frame

    def closeEvent(self, evt):  # noqa: N802 (Qt override)
        self.stop_live_view()
        if self.hcam is not None:
            try:
                self.hcam.close()  # sends closeinstrument + quits MATLAB engine
            except Exception as e:
                print(f"Camera close error: {e}")
            self.hcam = None
        super().closeEvent(evt)

    def stop(self):
        if self.hcam is not None:
            self.hcam.close()
            self.hcam = None

    def _get_lut(self, name):
        """Return a 256x3 uint8 RGB lookup table for the chosen colormap.
        jet/hot/cool come straight from MATLAB; built once then cached."""
        if name in self._lut_cache:
            return self._lut_cache[name]

        n = 256
        if name == "Grey":
            # plain grayscale ramp (no MATLAB needed)
            ramp = np.arange(n, dtype=np.uint8)
            lut = np.repeat(ramp[:, None], 3, axis=1)
        else:
            eng = self.hcam.eng  # MATLAB engine owned by the device class
            if name == "Jet_Plus_White":
                jet = np.array(eng.jet(float(n))._data,
                               dtype=np.float64).reshape((n, 3), order="F")
                n_white = 32  # rows fading white -> jet's blue (tweak to taste)
                fade = np.linspace(1.0, 0.0, n_white)[:, None]
                bottom = fade * np.ones((n_white, 3)) + (1.0 - fade) * jet[0]
                body = jet[np.linspace(0, n - 1, n - n_white).astype(int)]
                cmap = np.vstack([bottom, body])
            else:
                fn = {"Jet": eng.jet, "Hot": eng.hot, "Cool": eng.cool}.get(name)
                if fn is None:  # unknown name -> fall back to grayscale
                    ramp = np.arange(n, dtype=np.uint8)
                    lut = np.repeat(ramp[:, None], 3, axis=1)
                    self._lut_cache[name] = lut
                    return lut
                cmap = np.array(fn(float(n))._data,
                                dtype=np.float64).reshape((n, 3), order="F")
            lut = np.clip(cmap * 255.0, 0, 255).astype(np.uint8)

        self._lut_cache[name] = lut
        return lut

class CrosshairLabel(QLabel):
    mouseMoved = pyqtSignal(int, int)
    mouseClicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._crosshair_enabled = False
        self._crosshair_pos = None  # (x, y)
        self._crosshair_thickness = 1
        self._waiting_for_crosshair_click = False

    def enable_crosshair(self, x, y, thickness=1):
        self._crosshair_pos = (x, y)
        self._crosshair_thickness = thickness
        self._crosshair_enabled = True
        self.update()

    def disable_crosshair(self):
        self._crosshair_enabled = False
        self._crosshair_pos = None
        self.update()

    def paintEvent(self, event):
        # Paint the base image
        super().paintEvent(event)
        if self._crosshair_enabled and self._crosshair_pos:
            painter = QPainter(self)
            pen = QPen(Qt.red, 1)
            painter.setPen(pen)
            x, y = self._crosshair_pos
            # span the full label (image) instead of a hardcoded 680x510
            x_line_len = self.width()
            y_line_len = self.height()
            half_thickness = self._crosshair_thickness//2

            painter.drawLine(x - x_line_len, y - half_thickness, x + x_line_len, y - half_thickness)
            painter.drawLine(x - x_line_len, y + half_thickness, x + x_line_len, y + half_thickness)
            painter.drawLine(x - half_thickness, y - y_line_len, x - half_thickness, y + y_line_len)
            painter.drawLine(x + half_thickness, y - y_line_len, x + half_thickness, y + y_line_len)
            painter.end()

    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.x(), event.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._waiting_for_crosshair_click:
            self.mouseClicked.emit(event.x(), event.y())

    def waiting_for_crosshair_click(self, enable):
        self._waiting_for_crosshair_click = enable
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Amscope_Camera_View()
    w.show()
    sys.exit(app.exec())