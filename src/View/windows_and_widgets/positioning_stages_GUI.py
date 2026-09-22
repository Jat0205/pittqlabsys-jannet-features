# Created by Jannet Trabelsi on 2025-09-02
# Please note: the controller class raises errors. However, the GUI sets default values to solve for invalid inputs
import time
from PyQt5.QtWidgets import QMessageBox, QWidget
from src.Controller.newport_conex_cc import Newport_CONEX_CC_xy_stage
from src.Controller.nanodrive import MCLNanoDrive
from src.Controller.MCL_z_microdrive import MCLZMicroDrive
from .positioning_stages_design import Ui_Form
from datetime import datetime
import os
from PyQt5.QtWidgets import QFileDialog
from src.core.struct_hdf5 import StructArray, MyStruct, save_data, load_data
import numpy as np
import cv2
from typing import List, Tuple
from PyQt5.QtWidgets import QMessageBox, QPushButton
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QApplication, QProgressDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel   # or add to your top imports
# Assuming the .ui file is converted to design.py
#To convert positioning_stages_design.ui to .py, paste this into the terminal:
# pyuic5 -x positioning_stages_design.ui -o positioning_stages_design.py

# constants:
_MAX_X_1 = 100
_MIN_X_1 = 0
_MAX_Y_1 = 100
_MIN_Y_1 = 0
_MAX_Z_1 = 100
_MIN_Z_1 = 0
_MAX_X_2 = 100
_MIN_X_2 = 0
_MAX_Y_2 = 100
_MIN_Y_2 = 0
_MAX_Z_2 = 100
_MIN_Z_2 = 0
_MAX_Z_3 = 25000
_MIN_Z_3 = -25000
_MAX_MCL_nanodrive_X = 100
_MIN_MCL_nanodrive_X = 0
_MAX_MCL_nanodrive_Y = 100
_MIN_MCL_nanodrive_Y = 0
_MAX_MCL_nanodrive_Z =100
_MIN_MCL_nanodrive_Z = 0
_MAX_MCL_microdrive_Z = 50000
_MIN_MCL_microdrive_Z = 0
_MAX_X_CONEX = 24000.0
_MIN_X_CONEX = 0.0
_MAX_Y_CONEX = 24000.0
_MIN_Y_CONEX = 0.0
from PyQt5.QtCore import pyqtSignal

class positioning_stages_view(QWidget, Ui_Form):
    """
    This is the widget of the positioning stages. It allows us to control positioning devices using buttons and LineEdits
    """
    display_choice_changed = pyqtSignal(str)
    snapshot_mode_changed = pyqtSignal(int)
    snapButtonclicked = pyqtSignal(int)
    save_or_find_nv_button_clicked = pyqtSignal(int)
    get_img = pyqtSignal(float, float)   # (integration_time_us, gain)
    take_img_signal = pyqtSignal(int)
    server_off_button_clicked = pyqtSignal(int)
    color_scale_changed = pyqtSignal(str)
    Get_Image_Button_Clicked = pyqtSignal(int)
    connect_to_display_clicked = pyqtSignal(int)

    # --- AutoTune config: MUST stay in sync with Display_View. Roper only. ---
    AUTOTUNE_TARGET_MIN = 15000
    AUTOTUNE_TARGET_MAX = 39000
    AUTOTUNE_SATURATION = 40000
    ROPER_NOGAIN_SEQUENCE = [10, 50, 100, 150, 200, 300, 500]  # inttime (us)
    ROPER_GAIN_SEQUENCE = [(10, 1), (50, 1), (100, 1), (100, 100),
                           (100, 500), (100, 1000), (100, 1500),
                           (100, 2000), (100, 2500), (100, 3000), (150, 3000),
                           (200, 3000), (300, 3000), (400, 3000), (500, 3000)]

    def __init__(self, parent = None):
        super().__init__(parent)
        self.setupUi(self)
        self.stage_1 = None
        self.stage_2 = None
        self.stage_3 = None

        self.xlineEdit_1.setEnabled(False)
        self.ylineEdit_1.setEnabled(False)
        self.zlineEdit_1.setEnabled(False)
        self.x_y_inc_lineEdit_1.setEnabled(False)
        self.z_inc_lineEdit_1.setEnabled(False)
        self.x_y_inc_lineEdit_2.setEnabled(False)
        self.z_inc_lineEdit_2.setEnabled(False)
        self.x_inc_1.setEnabled(False)
        self.y_inc_1.setEnabled(False)
        self.z_inc_1.setEnabled(False)
        self.x_dec_1.setEnabled(False)
        self.y_dec_1.setEnabled(False)
        self.z_dec_1.setEnabled(False)

        self.xlineEdit_2.setEnabled(False)
        self.ylineEdit_2.setEnabled(False)
        self.zlineEdit_2.setEnabled(False)
        self.x_inc_2.setEnabled(False)
        self.y_inc_2.setEnabled(False)
        self.z_inc_2.setEnabled(False)
        self.x_dec_2.setEnabled(False)
        self.y_dec_2.setEnabled(False)
        self.z_dec_2.setEnabled(False)
        self.confirm_x_button_1.setEnabled(False)
        self.confirm_y_button_1.setEnabled(False)
        self.confirm_z_button_1.setEnabled(False)
        self.confirm_x_button_2.setEnabled(False)
        self.confirm_y_button_2.setEnabled(False)
        self.confirm_z_button_2.setEnabled(False)
        self.home_button_3.setEnabled(False)

        self.zlineEdit_3.setEnabled(False)
        self.z_inc_3.setEnabled(False)
        self.z_dec_3.setEnabled(False)
        self.comfirm_z_3.setEnabled(False)

        # Connect buttons to functions
        self.connectButton_1.clicked.connect(self.connect_to_instrument_1)
        self.connectButton_2.clicked.connect(self.connect_to_instrument_2)
        self.connectButton_3.clicked.connect(self.connect_to_instrument_3)
        self.confirm_x_button_1.clicked.connect(lambda: self.set_position("x", 1))
        self.confirm_y_button_1.clicked.connect(lambda: self.set_position("y", 1))
        self.confirm_z_button_1.clicked.connect(lambda: self.set_position("z", 1))
        self.confirm_x_button_2.clicked.connect(lambda: self.set_position("x", 2))
        self.confirm_y_button_2.clicked.connect(lambda: self.set_position("y", 2))
        self.confirm_z_button_2.clicked.connect(lambda: self.set_position("z", 2))
        self.comfirm_z_3.clicked.connect(lambda: self.set_position("z", 3))
        self.home_button_3.clicked.connect(lambda: self.set_position("home", 3))
        self.x_inc_1.clicked.connect(lambda: self.change_position("x", 1, 1)) # 1 for inc 0 for dec
        self.x_inc_2.clicked.connect(lambda: self.change_position("x", 2, 1))
        self.x_dec_1.clicked.connect(lambda: self.change_position("x", 1, 0))
        self.x_dec_2.clicked.connect(lambda: self.change_position("x", 2, 0))
        self.y_inc_1.clicked.connect(lambda: self.change_position("y", 1, 1))
        self.y_inc_2.clicked.connect(lambda: self.change_position("y", 2, 1))
        self.y_dec_1.clicked.connect(lambda: self.change_position("y", 1, 0))
        self.y_dec_2.clicked.connect(lambda: self.change_position("y", 2, 0))
        self.z_inc_1.clicked.connect(lambda: self.change_position("z", 1, 1))
        self.z_inc_2.clicked.connect(lambda: self.change_position("z", 2, 1))
        self.z_dec_1.clicked.connect(lambda: self.change_position("z", 1, 0))
        self.z_dec_2.clicked.connect(lambda: self.change_position("z", 2, 0))
        self.z_inc_3.clicked.connect(lambda: self.change_position("z", 3, 1))
        self.z_dec_3.clicked.connect(lambda: self.change_position("z", 3, 0))
        self.server_off_button.clicked.connect(self.close_server)
        self.save_button.clicked.connect(self.save)
        self.Find_NV_Button.clicked.connect(self.find_NV)
        self.snapButton.clicked.connect(self.send_snapshotButtonclicked_signal)
        # Connect combobox signals to emitters
        self.display_option.currentTextChanged.connect(self.on_display_choice_changed)
        self.color_scale_option.currentTextChanged.connect(self.on_color_scale_changed)
        self.snapshot_live_comboBox.currentTextChanged.connect(self.on_snapshot_or_live_changed)
        self.Get_Image_Button.clicked.connect(self.get_image_snapshot)
        self.ConnectToDisplayButton.clicked.connect(self.on_connect_to_display_clicked)
        self.data_saving_path = None
        self.data_reader = None
        self.frame = None
        self.x_crosshair = None
        self.y_crosshair = None
        self.start_camera_scan.clicked.connect(self._start_camera_scan)
        self.go_to_nv.clicked.connect(self._go_to_nv)
        self.scan_preview = QLabel(self)

    def connect_to_instrument_1(self):
        """This function connects the devices: please make sure that your stage has the function get_position(self, axis)"""
        stage_name = self.comboBox_1.currentText()
        if stage_name == 'MCL_nanodrive':
            _MAX_X_1 = _MAX_MCL_nanodrive_X
            _MIN_X_1 = _MIN_MCL_nanodrive_X
            _MAX_Y_1 = _MAX_MCL_nanodrive_Y
            _MIN_Y_1 = _MIN_MCL_nanodrive_Y
            _MAX_Z_1 = _MAX_MCL_nanodrive_Z
            _MIN_Z_1 = _MIN_MCL_nanodrive_Z
            try:
                self.stage_1 = MCLNanoDrive()
                self.zlineEdit_1.setEnabled(True)
                self.zlineEdit_1.setText(str(self.stage_1.get_position('z')))
                self.confirm_z_button_1.setEnabled(True)
                self.z_inc_1.setEnabled(True)
                self.z_dec_1.setEnabled(True)
                self.z_inc_lineEdit_1.setEnabled(True)
                QMessageBox.information(self, 'Success', f'Connected to MCL_nanodrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        elif stage_name == 'Newport_Conex_microdrive':
            try:
                self.stage_1 = Newport_CONEX_CC_xy_stage()
                _MAX_X_1  = _MAX_X_CONEX
                _MIN_X_1 = _MIN_X_CONEX
                _MAX_Y_1 = _MAX_Y_CONEX
                _MIN_Y_1 = _MIN_Y_CONEX
                QMessageBox.information(self, 'Success', f'Connected to Newport_Conex_microdrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            return
        self.xlineEdit_1.setText(str(self.stage_1.get_position('x')))
        self.ylineEdit_1.setText(str(self.stage_1.get_position('y')))
        self.xlineEdit_1.setEnabled(True)
        self.ylineEdit_1.setEnabled(True)
        self.confirm_x_button_1.setEnabled(True)
        self.confirm_y_button_1.setEnabled(True)
        self.x_y_inc_lineEdit_1.setEnabled(True)
        self.x_inc_1.setEnabled(True)
        self.y_inc_1.setEnabled(True)
        self.x_dec_1.setEnabled(True)
        self.y_dec_1.setEnabled(True)

    def connect_to_instrument_2(self):
        """This function connects the devices: please make sure that your stage has the function get_position(self, axis)"""
        stage_name = self.comboBox_2.currentText()
        if stage_name == 'MCL_nanodrive':
            _MAX_X_2 = _MAX_MCL_nanodrive_X
            _MIN_X_2 = _MIN_MCL_nanodrive_X
            _MAX_Y_2 = _MAX_MCL_nanodrive_Y
            _MIN_Y_2 = _MIN_MCL_nanodrive_Y
            _MAX_Z_2 = _MAX_MCL_nanodrive_Z
            _MIN_Z_2 = _MIN_MCL_nanodrive_Z
            try:
                self.stage_2 = MCLNanoDrive()
                self.zlineEdit_2.setEnabled(True)
                self.confirm_z_button_2.setEnabled(True)
                self.zlineEdit_2.setText(str(self.stage_2.get_position('z')))
                self.z_inc_2.setEnabled(True)
                self.z_dec_2.setEnabled(True)
                self.z_inc_lineEdit_2.setEnabled(True)
                QMessageBox.information(self, 'Success', f'Connected to MCL_nanodrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        elif stage_name == 'Newport_Conex_microdrive':
            try:
                self.stage_2 = Newport_CONEX_CC_xy_stage()
                _MAX_X_2 = _MAX_X_CONEX
                _MIN_X_2 = _MIN_X_CONEX
                _MAX_Y_2 = _MAX_Y_CONEX
                _MIN_Y_2 = _MIN_Y_CONEX
                QMessageBox.information(self, 'Success', f'Connected to Newport_Conex_microdrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            return
        self.xlineEdit_2.setText(str(self.stage_2.get_position('x')))
        self.ylineEdit_2.setText(str(self.stage_2.get_position('y')))
        self.xlineEdit_2.setEnabled(True)
        self.ylineEdit_2.setEnabled(True)
        self.confirm_x_button_2.setEnabled(True)
        self.confirm_y_button_2.setEnabled(True)
        self.x_y_inc_lineEdit_2.setEnabled(True)
        self.x_inc_2.setEnabled(True)
        self.y_inc_2.setEnabled(True)
        self.x_dec_2.setEnabled(True)
        self.y_dec_2.setEnabled(True)

    def connect_to_instrument_3(self):
        stage_name = self.comboBox_3.currentText()
        if stage_name == 'MCL_z_microdrive':
            _MAX_Z_3 = _MAX_MCL_microdrive_Z
            _MIN_Z_3 = _MIN_MCL_microdrive_Z
            try:
                self.stage_3 = MCLZMicroDrive()
                self.zlineEdit_3.setEnabled(True)
                self.comfirm_z_3.setEnabled(True)
                self.zlineEdit_3.setText(str(self.stage_3.get_position('z')))
                self.z_inc_3.setEnabled(True)
                self.z_dec_3.setEnabled(True)
                self.z_inc_lineEdit_3.setEnabled(True)
                self.home_button_3.setEnabled(True)
                QMessageBox.information(self, 'Success', f'Connected to MCL_z_microdrive')

            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            return

    def close(self):
        if self.stage_1 is not None:
            print("closing stage_1")
            self.stage_1.close()
        if self.stage_2 is not None:
            print("closing stage_2")
            self.stage_2.close()
        if self.stage_3 is not None:
            print("closing stage_3")
            self.stage_3.close()

    def set_position(self, axis, instrument_id):
        if axis == "home" and instrument_id == 3:
            self.stage_3.home_axis() # please note this is specific for the z microdrive as it doesn't have encoders
            #time.sleep(15)
            self.stage_3.homed = True
            self.zlineEdit_3.setText(str(self.stage_3.get_position()))
        else:
            stage, pos, max, min, line_edit, inc = self.selector(axis, instrument_id)
            if isinstance(pos, str) or isinstance(pos, int) or isinstance(pos, float):
                try:
                    if pos:
                        pos = float(pos)
                        if min<=pos<=max:
                            stage.set_position(axis, pos)
                            #time.sleep(15)
                            line_edit.setText(str(stage.get_position(axis)))
                        else:
                            self.error_box(
                                "OUT OF RANGE!",
                                "Please provide a position within the range"
                            )
                            return
                    else:
                        self.error_box(
                            "INVALID POSITION!",
                            "Please provide a valid position"
                        )
                        return
                except ValueError:
                    line_edit.setText(str(stage.get_position(axis)))
                    return
            else:
                self.error_box(
                    "INVALID POSITION!",
                    "Please provide a valid position"
                )
                return

    def selector(self, axis, instrument_id):
        if instrument_id == 1:
            stage = self.stage_1
        elif instrument_id == 2:
            stage = self.stage_2
        elif instrument_id == 3:
            stage = self.stage_3
        else:
            raise Exception
        if axis == 'x':
            if instrument_id == 1:
                pos = self.xlineEdit_1.text()
                max = _MAX_X_1
                min = _MIN_X_1
                line_edit = self.xlineEdit_1
                inc_line_edit = self.x_y_inc_lineEdit_1

            elif instrument_id == 2:
                pos = self.xlineEdit_2.text()
                max = _MAX_X_2
                min = _MIN_X_2
                line_edit = self.xlineEdit_2
                inc_line_edit = self.x_y_inc_lineEdit_2
            else:
                raise Exception
        elif axis == 'y':
            if instrument_id == 1:
                pos = self.ylineEdit_1.text()
                max = _MAX_Y_1
                min = _MIN_Y_1
                line_edit = self.ylineEdit_1
                inc_line_edit = self.x_y_inc_lineEdit_1
            elif instrument_id == 2:
                pos = self.ylineEdit_2.text()
                max = _MAX_Y_2
                min = _MIN_Y_2
                line_edit = self.ylineEdit_2
                inc_line_edit = self.x_y_inc_lineEdit_2
            else:
                raise Exception
        elif axis == 'z':
            if instrument_id == 1:
                pos = self.zlineEdit_1.text()
                max = _MAX_Z_1
                min = _MIN_Z_1
                line_edit = self.zlineEdit_1
                inc_line_edit = self.z_inc_lineEdit_1
            elif instrument_id == 2:
                pos = self.zlineEdit_2.text()
                max = _MAX_Z_2
                min = _MIN_Z_2
                line_edit = self.zlineEdit_2
                inc_line_edit = self.z_inc_lineEdit_2
            elif instrument_id == 3:
                pos = self.zlineEdit_3.text()
                max = _MAX_Z_3
                min = _MIN_Z_3
                line_edit = self.zlineEdit_3
                inc_line_edit = self.z_inc_lineEdit_3
            else:
                raise Exception
        else:
            raise Exception
        return stage, pos, max, min, line_edit, inc_line_edit

    def change_position(self, axis, instrument_id, increase):
        # 1 for inc 0 for dec
        stage, pos, max, min, line_edit, inc_line_edit = self.selector(axis, instrument_id)
        inc_step = inc_line_edit.text()
        pos = float(stage.get_position(axis))
        if isinstance(inc_step, str) or isinstance(inc_step, int) or isinstance(inc_step, float):
            try:
                inc_step = float(inc_step)
                if increase:
                    new_pos = pos + inc_step
                else:
                    new_pos = pos - inc_step
                if new_pos < max and new_pos > min:
                    stage.set_position(axis, new_pos)
                    time.sleep(1)
                    line_edit.setText(str(stage.get_position(axis)))
                else:
                    self.error_box(
                        "OUT OF RANGE!",
                        "Please provide a position within the range"
                    )
                    return
            except ValueError:
                line_edit.setText(str(stage.get_position(axis)))
                return
        else:
            line_edit.setText(str(stage.get_position(axis)))
            return

    def on_display_choice_changed(self, text):
        # This one should be handled in main window by the display
        disp = self.display_option.currentText()
        if disp == "MU300":
            self.snapshot_live_comboBox.setEnabled(True)
        self.display_choice_changed.emit(text)

    def on_color_scale_changed(self, text):
        option = self.color_scale_option.currentText()
        if option == "Grey":
            self.color_scale_changed.emit(text)
        elif option == "Jet_Plus_White":
            self.color_scale_changed.emit(text)
        elif option == "Jet":
            self.color_scale_changed.emit(text)
        elif option == "Hot":
            self.color_scale_changed.emit(text)
        elif option == "Cool":
            self.color_scale_changed.emit(text)

    def get_image_snapshot(self):
        self.Get_Image_Button_Clicked.emit(1)

    def on_connect_to_display_clicked(self):
        self.connect_to_display_clicked.emit(1)

    def Update_Get_Image_Button(self, enabled):
        self.Get_Image_Button.setEnabled(enabled)

    def on_snapshot_or_live_changed(self, text):
        # snapshot mode = 0, live mode = 1
        mode = 0 if text.lower() == "snapshot" else 1
        self.snapshot_mode_changed.emit(mode)

    def display_choice(self):
        disp = self.display_option.currentText()
        if disp == "MU300":
            self.snapshot_live_comboBox.setEnabled(True)
        return disp

    def snapshot_or_live(self):
        return self.snapshot_live_comboBox.currentText()

    def send_snapshotButtonclicked_signal(self):
        self.snapButtonclicked.emit(1)

    def save(self):
        self.save_or_find_nv_button_clicked.emit(1)

        # --- UI → keys ---
        sample_selection = self.Sample_Selector_comboBox.currentText()
        point_selection = self.Point_Selector_comboBox.currentText()
        point_status = self.point_status_comboBox.currentText()

        point_key = point_selection.lower().replace(" ", "_")
        if point_status == "FINAL" and point_selection == "NV":
            self.error_box(
                "YOU CANNOT SELECT FINAL NV POINT",
                "To find NV, click find NV button!"
            )
            return
        if not hasattr(self, 'stage_1') or not hasattr(self, 'stage_2') or not hasattr(self, 'stage_3') or self.stage_1 == None or self.stage_2 == None or self.stage_3 == None:
            self.error_box(
                "No stage connected",
                "Please connect to the stages first"
            )
            return

        # --------------------------------------------------
        # File handling
        # --------------------------------------------------
        if sample_selection == "New Sample" and point_status == "INITIAL":
            directory, filename = self.open_directory_dialog(self.data_saving_path)
            if filename is None:
                return

            self.data_saving_path = directory
            full_path = os.path.join(directory, filename)

            if os.path.exists(full_path):
                if not self.confirm_overwrite(filename):
                    return

            mode = "w"

        else:
            full_path = self.open_file_dialog(self.data_saving_path)
            if full_path is None:
                return
            mode = "r+"
        # new root
        root = MyStruct()

        # INITIAL / FINAL
        if not hasattr(root, point_status):
            setattr(root, point_status, MyStruct())

        point_status_object = getattr(root, point_status)

        # bottom_left / nv / etc
        if not hasattr(point_status_object, point_key):
            setattr(point_status_object, point_key, MyStruct())

        point = getattr(point_status_object, point_key)


        # --------------------------------------------------
        # Identify stages
        # --------------------------------------------------
        stage_1_name = self.comboBox_1.currentText()
        # --------------------------------------------------
        # Snapshot metadata
        # --------------------------------------------------
        if "nanodrive" in stage_1_name.lower():
            point.nano_x = self.stage_1.get_position("x")
            point.nano_y = self.stage_1.get_position("y")
            point.nano_z = self.stage_1.get_position("z")
            point.micro_x = self.stage_2.get_position("x")
            point.micro_y = self.stage_2.get_position("y")
        else:
            point.nano_x = self.stage_2.get_position("x")
            point.nano_y = self.stage_2.get_position("y")
            point.nano_z = self.stage_2.get_position("z")
            point.micro_x = self.stage_1.get_position("x")
            point.micro_y = self.stage_1.get_position("y")

        point.micro_z = self.stage_3.get_position()
        point.camera_x = self.x_crosshair
        point.camera_y = self.y_crosshair

        point.timestamp = datetime.utcnow().isoformat()

        # --------------------------------------------------
        # Capture camera image
        # --------------------------------------------------

        self.take_img_signal.emit(1)
        point.camera_image = self.frame
        # --------------------------------------------------
        # SAVE (single call)
        # --------------------------------------------------
        save_data(
            filename=full_path,
            obj=root,
            mode=mode,
            swmr=False  # snapshot, not live
        )

    def error_box(self, text, info, title="Error"):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setInformativeText(info)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)

        return msg.exec() == QMessageBox.Ok

    def open_file_dialog(self, start_path):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open sample file",
            start_path,
            "HDF5 files (*.h5);;All files (*)"
        )

        if not filename:
            raise RuntimeError("No file selected")

        return filename

    def open_directory_dialog(self, start_path):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select file location",
            start_path,
            "HDF5 Files (*.h5);;All Files (*)"
        )

        # User pressed Cancel
        if not file_path:
            return None, None

        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        return directory, filename

    def confirm_overwrite(self, point_key):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Overwrite point?")
        msg.setText(f"Point '{point_key}' already exists.")
        msg.setInformativeText("Do you want to overwrite it?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)

        return msg.exec() == QMessageBox.Yes

    def compute_homography_from_corners(self,
            old_corners: List[np.ndarray],
            new_corners: List[np.ndarray], method) -> np.ndarray:

        if len(old_corners) != 4 or len(new_corners) != 4:
            raise ValueError("Expected 4 corners for old and new quads")

        src_pts = np.array(old_corners, dtype=np.float64)
        dst_pts = np.array(new_corners, dtype=np.float64)

        # ensure points are numpy arrays and float32
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # check number of points
        if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
            raise ValueError("Need at least 4 points to compute homography")

        if method == "LMEDS":
            H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.LMEDS)
        elif method == "RANSAC":
            H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        elif method == "RHOMBUS":
            H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.RHO)
        else:
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC)

        if status is not None and not np.all(status):
            print("Warning: some corners treated as outliers")

        if H is None:
            raise ValueError("Homography computation failed")

        # Optional: verify fourth corner
        tr_hom = np.array([src_pts[1, 0], src_pts[1, 1], 1.0])
        tr_pred = H @ tr_hom
        tr_pred /= tr_pred[2]
        expected = np.array([dst_pts[1, 0], dst_pts[1, 1], 1.0])
        if np.linalg.norm(tr_pred - expected) > 1e-6:
            print(f"Warning: Fourth corner verification failed. "
                  f"Expected ({expected[0]:.3f},{expected[1]:.3f}), got ({tr_pred[0]:.3f},{tr_pred[1]:.3f})")

        return H

    def map_point_with_homography(self, point: np.ndarray, H: np.ndarray) -> np.ndarray:

        if len(point) == 2:
            pt_hom = np.array([point[0], point[1], 1.0])
        else:
            pt_hom = np.array(point)

        mapped = H @ pt_hom
        mapped /= mapped[2]
        return mapped[:2]

    # affine

    def from_four_corners_to_DMT_or_DMNT(self,
            corners_microdrive: List[np.ndarray],
            reference_order: Tuple[str] = ("top_left", "top_right", "bottom_right", "bottom_left")
    ) -> np.ndarray:
        

        if len(corners_microdrive) != 4:
            raise ValueError(f"Expected 4 corners, got {len(corners_microdrive)}")

        # Extract points in the specified order
        # Assuming input order matches reference_order
        bl_idx = reference_order.index("bottom_left")
        br_idx = reference_order.index("bottom_right")
        tl_idx = reference_order.index("top_left")

        bottom_left = np.array(corners_microdrive[bl_idx], dtype=float)
        bottom_right = np.array(corners_microdrive[br_idx], dtype=float)
        top_left = np.array(corners_microdrive[tl_idx], dtype=float)

        # Diamond coordinate system definition:
        # In diamond coords:
        # bottom_left = (0, 0)
        # bottom_right = (1, 0)
        # top_left = (0, 1)
        # top_right = (1, 1)

        # Source points in microdrive coordinates (homogeneous)
        src_points = np.array([
            [bottom_left[0], bottom_left[1], 1],
            [bottom_right[0], bottom_right[1], 1],
            [top_left[0], top_left[1], 1]
        ]).T  # Shape: (3, 3)

        # Destination points in diamond coordinates (homogeneous)
        dst_points = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1]
        ]).T  # Shape: (3, 3)

        # Solve for transformation matrix T such that: dst = T @ src
        # T is 3x3, we need T @ src = dst
        # T = dst @ inv(src)

        try:
            T = dst_points @ np.linalg.inv(src_points)
        except np.linalg.LinAlgError:
            raise ValueError("Corners are colinear or form a degenerate shape")
        return T

    def from_DMT_and_MNV_old_get_DNV_old(self, M_point: np.ndarray, T_matrix: np.ndarray) -> np.ndarray:

        if len(M_point) == 2:
            M_hom = np.array([M_point[0], M_point[1], 1.0])
        else:
            M_hom = np.array(M_point)

        D_hom = T_matrix @ M_hom
        D_hom = D_hom / D_hom[2]  # Normalize

        return D_hom[:2]

    def choose_method(self) -> str:
        """
        Ask the user which NV mapping method to use: Affine or Homography.
        Returns:
            "affine" or "homography"
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Select NV Mapping Method")
        msg.setText("Which method would you like to use for NV relocation?")
        msg.setInformativeText("Choose Affine or Homography.")

        msg.setStandardButtons(QMessageBox.NoButton)

        # Add custom buttons
        btn_affine = QPushButton("Affine")
        btn_homography = QPushButton("Homography")
        msg.addButton(btn_affine, QMessageBox.AcceptRole)
        msg.addButton(btn_homography, QMessageBox.AcceptRole)

        # Show dialog and wait for response
        ret = msg.exec()

        clicked_button = msg.clickedButton()
        if clicked_button == btn_affine:
            return "affine"
        else:
            return "homography"

    def choose_homography_method(self) -> str:
        """
        Ask the user which NV mapping method to use: LMEDS, RANSAC, RHOMBUS in Homography.
        Returns:
            "LMEDS", "RANSAC", "RHOMBUS", or "USAC_MAGSAC
        """
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Select NV Mapping Method")
        msg.setText("Which method would you like to use for NV relocation?")
        msg.setInformativeText("Choose Homography.")

        msg.setStandardButtons(QMessageBox.NoButton)

        # Add custom buttons
        btn_LMEDS = QPushButton("LMEDS")
        btn_RANSAC = QPushButton("RANSAC")
        btn_RHOMBUS = QPushButton("RHOMBUS")
        btn_USAC_MAGSAC = QPushButton("USAC_MAGSAC")
        msg.addButton(btn_LMEDS, QMessageBox.AcceptRole)
        msg.addButton(btn_RANSAC, QMessageBox.AcceptRole)
        msg.addButton(btn_RHOMBUS, QMessageBox.AcceptRole)
        msg.addButton(btn_USAC_MAGSAC, QMessageBox.AcceptRole)
        # Show dialog and wait for response
        ret = msg.exec()

        clicked_button = msg.clickedButton()
        if clicked_button == btn_LMEDS:
            return "LMEDS"
        elif clicked_button == btn_RANSAC:
            return "RANSAC"
        elif clicked_button == btn_RHOMBUS:
            return "RHOMBUS"
        else:
            return "USAC_MAGSAC"

    def _choose_scan_type(self):
        """Laser vs white-light. Returns 'laser', 'white_light', or None."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Scan Type")
        msg.setText("Which light source is this scan?")
        msg.setInformativeText("Laser -> stored as 'camera_image'.\n"
                               "White light -> stored as 'white_light_camera_image'.")
        msg.setStandardButtons(QMessageBox.Cancel)
        btn_laser = QPushButton("Laser")
        btn_white = QPushButton("White light")
        msg.addButton(btn_laser, QMessageBox.AcceptRole)
        msg.addButton(btn_white, QMessageBox.AcceptRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_laser:
            return "laser"
        if clicked == btn_white:
            return "white_light"
        return None

    def _choose_new_or_append(self):
        """New scan file vs append into an existing one. Returns 'new', 'append', or None."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Scan File")
        msg.setText("Start a new scan file, or add this scan to an existing one?")
        msg.setInformativeText("Appending stores this scan's images under the SAME "
                               "points (same coordinates) as the existing scan.")
        msg.setStandardButtons(QMessageBox.Cancel)
        btn_new = QPushButton("New scan file")
        btn_append = QPushButton("Append to existing")
        msg.addButton(btn_new, QMessageBox.AcceptRole)
        msg.addButton(btn_append, QMessageBox.AcceptRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_new:
            return "new"
        if clicked == btn_append:
            return "append"
        return None

    def _append_camera_scan(self, edges_path, image_attr):
        """Add a second-modality image (e.g. white light) to an EXISTING scan file,
        under the SAME points/coordinates as the original scan. Positions are
        replayed from the coords already saved in the file. The scan file must have
        been created with the SAME wire-edges file (verified against the stored path)."""
        scan_full_path = self.open_file_dialog(self.data_saving_path)
        if not scan_full_path:
            return
        self.data_saving_path = os.path.dirname(scan_full_path)

        structure = load_data(scan_full_path)

        # --- verify the wire-edges file matches the one stored in the scan ---
        meta = getattr(structure, "scan_meta", None)
        stored_edges = getattr(meta, "wire_edges_path", None) if meta is not None else None
        if stored_edges is None:
            self.error_box(
                "No stored wire-edges path",
                "This scan file has no wire-edges path saved, so it can't be safely "
                "appended to. Re-create it as a new scan (which now stores the path).")
            return
        if os.path.normcase(os.path.abspath(stored_edges)) != \
                os.path.normcase(os.path.abspath(edges_path)):
            self.error_box(
                "Wire-edges file mismatch",
                f"This scan was created with:\n{stored_edges}\n\n"
                f"but you selected:\n{edges_path}\n\n"
                "Select the same wire-edges file and try again.")
            return

        # --- existing points in numeric order: image_1, image_2, ... image_10 ---
        def _img_index(nm):
            try:
                return int(nm.split("_")[1])
            except (IndexError, ValueError):
                return 1 << 30

        image_names = sorted((n for n in vars(structure) if n.startswith("image_")),
                             key=_img_index)
        if not image_names:
            self.error_box("No scan points", "This file has no image_N points to append to.")
            return

        # optional safety: warn if this modality already exists on the points
        if getattr(getattr(structure, image_names[0]), image_attr, None) is not None:
            if not self.confirm_overwrite(f"{image_attr} (all points)"):
                return

        def _scalar(v):
            if v is None:
                return float("nan")
            a = np.asarray(v).ravel()
            return float(a[0]) if a.size else float("nan")

        if self.error_box("Warning", "Please put filters in the path. Click Ok when ready.",
                          "Camera_Safety") != True:
            return
        at = self._setup_scan_autotune()
        if at is None:
            return
        autotune_on, at_mode, at_seq, at_idx, fixed_it, fixed_ga = at

        settle_time = 100  # ms
        capture_timeout_s = 30  # tune above your longest exposure
        number_of_scans = len(image_names)
        # connect the stages append drives, BEFORE the progress window exists
        if getattr(self, "stage_2", None) is None:
            self.connect_to_instrument_2()
        if getattr(self, "stage_1", None) is None:
            self.connect_to_instrument_1()
        if getattr(self, "stage_3", None) is None:
            self.connect_to_instrument_3()
        if (getattr(self, "stage_1", None) is None or
                getattr(self, "stage_2", None) is None or
                getattr(self, "stage_3", None) is None):
            self.error_box("Stages not connected",
                           "Connect stage_1 (nano z), stage_2 (xy) and stage_3 "
                           "(micro z) before appending.")
            return
        move_errors = []

        self.error_box("Starting Scan", "Append scan has started", "Scan Status")
        scan_start = time.monotonic()
        progress = QProgressDialog("Time remaining: estimating…", "Cancel",
                                   0, number_of_scans, self)
        progress.setWindowTitle("Append Scan")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        done = 0
        for name in image_names:
            point = getattr(structure, name)
            micro_x = _scalar(getattr(point, "micro_x", None))
            micro_y = _scalar(getattr(point, "micro_y", None))
            micro_z = _scalar(getattr(point, "micro_z", None))
            nano_z = _scalar(getattr(point, "nano_z", None))

            # drive each stage in NATIVE units. Record failures but DON'T abort and
            # DON'T pop a modal box here (that box gets trapped behind the progress
            # window; closing that window to dismiss it cancels the whole append —
            # which is why you were only getting one image).
            try:
                if not np.isnan(micro_x):
                    self.stage_2.set_position("x", float(micro_x))
                if not np.isnan(micro_y):
                    self.stage_2.set_position("y", float(micro_y))
            except Exception as e:
                move_errors.append(f"{name}: XY -> {e}")
            try:
                if not np.isnan(nano_z):
                    self.stage_1.set_position("z", float(nano_z))
            except Exception as e:
                move_errors.append(f"{name}: nano_z -> {e}")
            try:
                if not np.isnan(micro_z):
                    self.stage_3.set_position("z", float(micro_z))
            except Exception as e:
                move_errors.append(f"{name}: micro_z -> {e}")
            try:  # best-effort position readback into the line-edits
                self.xlineEdit_2.setText(str(self.stage_2.get_position("x")))
                self.ylineEdit_2.setText(str(self.stage_2.get_position("y")))
                self.zlineEdit_3.setText(str(self.stage_3.get_position("z")))
                self.zlineEdit_1.setText(str(self.stage_1.get_position("z")))
            except Exception:
                pass
            time.sleep(settle_time / 1000)

            # request a frame and wait (same bounded-wait pattern as the main scan)
            # capture (autotune re-tunes exposure per point; index carries forward)
            if autotune_on:
                frame, at_idx, _it, _ga = self._autotune_capture(at_mode, at_seq, at_idx,
                                                                 capture_timeout_s)
            else:
                frame = self._capture_frame(fixed_it, fixed_ga, capture_timeout_s)

            # store the new image UNDER THE SAME point -> same coords as the original
            setattr(point, image_attr, frame)
            setattr(point, f"{image_attr}_timestamp", datetime.utcnow().isoformat())
            if hasattr(self, "show_scan_frame"):
                self.show_scan_frame(frame)

            done += 1
            elapsed = time.monotonic() - scan_start
            remaining = (elapsed / done) * (number_of_scans - done)
            progress.setValue(done)
            progress.setLabelText(f"Image {done}/{number_of_scans}\n"
                                  f"Time remaining: {remaining:0.0f} s")
            if progress.wasCanceled():
                break

        # re-save the WHOLE structure so the new field sits alongside existing data
        # under each point (we loaded it first, so nothing is lost)
        save_data(filename=scan_full_path, obj=structure, mode="w", swmr=False)
        progress.close()
        if move_errors:
            preview = "\n".join(move_errors[:10])
            more = "" if len(move_errors) <= 10 else f"\n…and {len(move_errors) - 10} more."
            self.error_box(
                "Some stage moves reported errors",
                "Every point was still captured, but these moves raised errors, so "
                "those images may not be at their recorded positions:\n\n" + preview + more)
        self.error_box("Finished Scanning", "Append scan is finished.", "Scan Status")

    def _start_camera_scan(self):
        # --- choose light source; decides which attribute the image is stored under ---
        scan_type = self._choose_scan_type()  # "laser" | "white_light" | None
        if scan_type is None:
            return
        image_attr = "camera_image" if scan_type == "laser" else "white_light_camera_image"

        # --- new scan file, or append into an existing scan? ---
        scan_mode = self._choose_new_or_append()  # "new" | "append" | None
        if scan_mode is None:
            return

        # --- edges file needed for BOTH paths (append uses it only to verify it
        #     matches the edges path stored in the scan) ---
        edges_path = self.open_file_dialog(self.data_saving_path)
        if not edges_path:
            return

        if scan_mode == "append":
            self._append_camera_scan(edges_path, image_attr)
            return
        number_of_z_points = 1
        z_range = 0
        if self.z_scan_checkBox.isChecked():
            z_min, ok = QInputDialog.getDouble(self, "Z Scan", "Minimum Z (µm):", 30.0)
            if not ok:
                return
            z_max, ok = QInputDialog.getDouble(self, "Z Scan", "Maximum Z (µm):", 80.0)
            if not ok:
                return
            z_step, ok = QInputDialog.getDouble(self, "Z Scan", "Z Step (µm):", 1.0)
            if not ok:
                return
            z_range = z_max - z_min
            if z_max <= z_min or z_range > 50:
                QMessageBox.warning(self, "Invalid Range", "Please select a range smaller than 50 microns and larger than zero")
                return
            number_of_z_points = int(round(z_range / z_step)) + 1
            if number_of_z_points < 1:
                QMessageBox.warning(self, "Invalid Step", "Please select a valid Z step.")
                return
        # Continue with the scan...
        scan_directory, scan_filename = self.open_directory_dialog(self.data_saving_path)
        if scan_filename is None:
            return

        self.data_saving_path = scan_directory
        scan_full_path = os.path.join(scan_directory, scan_filename)

        if os.path.exists(scan_full_path):
            if not self.confirm_overwrite(scan_filename):
                return

        mode = "w"
        structure = load_data(edges_path)
        initial_coords = self.get_coords(structure.INITIAL, "wire_edge", "microxy")
        final_coords = self.get_coords(structure.FINAL, "wire_edge", "microxy")
        initialx = initial_coords[0]*1000 # conex is in mm so we multiply by 1000 to go to microns
        initialy = initial_coords[1]*1000 # conex is in mm so we multiply by 1000 to go to microns
        finalx = final_coords[0]*1000 # conex is in mm so we multiply by 1000 to go to microns
        finaly = final_coords[1]*1000 # conex is in mm so we multiply by 1000 to go to microns
        x_span = finalx - initialx
        y_span = finaly - initialy
        x_range = abs(x_span)
        y_range = abs(y_span)

        if x_range > y_range:
            longer_range = x_range
            shorter_range = y_range
            longer_axis = "x"
            shorter_axis = "y"
            shorter_axis_initial = initialy
            longer_axis_initial = initialx
            longer_span = x_span
            shorter_span = y_span
        else:
            longer_range = y_range
            shorter_range = x_range
            longer_axis = "y"
            shorter_axis = "x"
            shorter_axis_initial = initialx
            longer_axis_initial = initialy
            longer_span = y_span
            shorter_span = x_span
        longer_step_mag = 90  # 90 microns is the biggest scan range using the nanodrive
        number_of_2D_points = int(longer_range / longer_step_mag)
        if number_of_2D_points < 1:  # was `== 0`; `< 1` also guards negatives
            number_of_2D_points = 1
        number_of_scans = number_of_z_points * number_of_2D_points
        settle_time = 100  # ms
        # signed steps so the scan walks INITIAL -> FINAL in either direction
        longer_axis_steps = longer_step_mag if longer_span >= 0 else -longer_step_mag
        shorter_axis_steps = shorter_span / number_of_2D_points
        focused_nano_z_location = float(self.get_coords(structure.INITIAL, "wire_edge", "nanoz")[0])
        focused_micro_z_location = float(self.get_coords(structure.INITIAL, "wire_edge", "microz")[0])
        returned = self.error_box("Warning", "Please put filters in the path. Click Ok when ready.", "Camera_Safety")
        image_number = 0
        if returned == True:
            at = self._setup_scan_autotune()
            if at is None:
                return
            autotune_on, at_mode, at_seq, at_idx, fixed_it, fixed_ga = at
            if self.z_scan_checkBox.isChecked():
                z_pos = z_min
            else:
                z_pos = focused_nano_z_location
            micro_z_pos = focused_micro_z_location
            shorter_axis_pos = shorter_axis_initial
            longer_axis_pos = longer_axis_initial
            root = MyStruct()
            root.scan_meta = MyStruct()
            root.scan_meta.wire_edges_path = os.path.abspath(edges_path)
            # connect the stages append drives, BEFORE the progress window exists
            if getattr(self, "stage_2", None) is None:
                self.connect_to_instrument_2()
            if getattr(self, "stage_1", None) is None:
                self.connect_to_instrument_1()
            if getattr(self, "stage_3", None) is None:
                self.connect_to_instrument_3()
            if (getattr(self, "stage_1", None) is None or
                    getattr(self, "stage_2", None) is None or
                    getattr(self, "stage_3", None) is None):
                self.error_box("Stages not connected",
                               "Connect stage_1 (nano z), stage_2 (xy) and stage_3 "
                               "(micro z) before appending.")
                return
            move_errors = []
            self.error_box("Starting Scan", "Camera Scan has started", "Scan Status")
            scan_start = time.monotonic()
            progress = QProgressDialog("Time remaining: estimating…", "Cancel",
                                       0, number_of_scans, self)
            progress.setWindowTitle("Camera Scan")
            progress.setWindowModality(Qt.WindowModal)  # also blocks re-clicking the scan button
            progress.setMinimumDuration(0)
            progress.setValue(0)
            while image_number < number_of_scans:
                # take data focused_z_location => max; take data min => focused_z_location such as focused_z_location is the z location of the focused laser
                image_number += 1
                if (micro_z_pos != self.stage_3.get_position("z")):
                    self.stage_3.set_position("z", micro_z_pos)
                if (z_pos != self.stage_1.get_position("z")):
                    self.stage_1.set_position("z", int(z_pos))
                self.stage_2.set_position(shorter_axis, shorter_axis_pos/ 1000)
                self.stage_2.set_position(longer_axis, longer_axis_pos / 1000)
                time.sleep(settle_time/1000)
                point_status = f"image_{image_number}"
                setattr(root, point_status, MyStruct())
                point = getattr(root, point_status)
                point.micro_x = self.stage_2.get_position("x")
                point.micro_y = self.stage_2.get_position("y")
                point.micro_z = self.stage_3.get_position("z")
                point.nano_z = self.stage_1.get_position("z")
                point.timestamp = datetime.utcnow().isoformat()
                # capture (autotune re-tunes exposure per point; index carries forward)
                if autotune_on:
                    frame, at_idx, _it, _ga = self._autotune_capture(at_mode, at_seq, at_idx)
                else:
                    frame = self._capture_frame(fixed_it, fixed_ga)
                setattr(point, image_attr, frame)
                # --- progress + time remaining ---
                elapsed = time.monotonic() - scan_start
                avg = elapsed / image_number  # image_number already incremented at top
                remaining = avg * (number_of_scans - image_number)
                progress.setValue(image_number)
                progress.setLabelText(
                    f"Image {image_number}/{number_of_scans}\n"
                    f"Time remaining: {remaining:0.0f} s")
                if progress.wasCanceled():
                    break

                if self.z_scan_checkBox.isChecked():
                    if z_pos < z_max:
                        z_pos = z_pos + z_step
                    else:
                        z_pos = z_min
                        shorter_axis_pos += shorter_axis_steps
                        longer_axis_pos += longer_axis_steps
                else:
                    shorter_axis_pos += shorter_axis_steps
                    longer_axis_pos += longer_axis_steps
            save_data(
                filename=scan_full_path,
                obj=root,
                mode=mode,
                swmr=False  # snapshot, not live
            )
            self.error_box("Finished Scanning", "Camera Scan is finished.", "Scan Status")
            progress.close()
        else:
            return

    def _resolve_setting(self, mode, setting):
        """One autotune-sequence entry -> (inttime, gain) floats.
        'nogain' uses unity gain (1.0), since get_img always sets both."""
        if mode == "gain":
            inttime, gain = setting
            return float(inttime), float(gain)
        return float(setting), 1.0

    def _capture_frame(self, inttime, gain, capture_timeout_s=30):
        """Ask main_window for one frame at (inttime, gain) via get_img, then wait
        for it to set self.frame_ready / self.frame. Returns the frame (or None)."""
        self.frame_ready = False
        self.get_img.emit(float(inttime), float(gain))
        wait_start = time.monotonic()
        while not self.frame_ready:
            QApplication.processEvents()
            time.sleep(0.02)
            if time.monotonic() - wait_start > capture_timeout_s:
                break
        return self.frame

    def _autotune_capture(self, mode, seq, index, capture_timeout_s=30):
        """Capture at the current point, adjusting exposure from seq[index] until the
        brightest pixel lands in the target window (same logic as Display_View, but
        capture goes through get_img). Returns (frame, new_index, inttime, gain).
        Pass new_index back in next time so the search starts warm."""
        idx = index
        frame = None
        it = ga = None
        for _ in range(2 * len(seq)):
            it, ga = self._resolve_setting(mode, seq[idx])
            frame = self._capture_frame(it, ga, capture_timeout_s)
            if frame is None:
                break
            max_pixel = float(np.max(frame))
            if max_pixel >= self.AUTOTUNE_SATURATION or max_pixel > self.AUTOTUNE_TARGET_MAX:
                if idx > 0:
                    idx -= 1
                    continue
                break
            elif max_pixel < self.AUTOTUNE_TARGET_MIN:
                if idx < len(seq) - 1:
                    idx += 1
                    continue
                break
            else:
                break
        return frame, idx, it, ga

    def _setup_scan_autotune(self):
        """Ask whether to autotune during the scan. Returns None if cancelled, else:
          (True,  mode, seq, start_index, None, None)   # autotune on
          (False, None, None, None,       it,   ga)     # fixed exposure
        """
        box = QMessageBox(self)
        box.setWindowTitle("AutoTune")
        box.setText("AutoTune the exposure during the scan?")
        yes_btn = box.addButton("Yes", QMessageBox.AcceptRole)
        no_btn = box.addButton("No", QMessageBox.AcceptRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel_btn or clicked is None:
            return None

        if clicked is no_btn:
            # fixed exposure: ask once, reuse for every point
            it, ok = QInputDialog.getInt(self, "Exposure", "Integration time (us):",
                                         100, 1, 10_000_000, 1)
            if not ok:
                return None
            ga, ok = QInputDialog.getInt(self, "Exposure", "Gain (1 = no gain):",
                                         1, 1, 1_000_000, 1)
            if not ok:
                return None
            return (False, None, None, None, it, ga)

        # yes -> gain / no gain
        box2 = QMessageBox(self)
        box2.setWindowTitle("AutoTune")
        box2.setText("Choose AutoTune mode:")
        gain_btn = box2.addButton("Gain", QMessageBox.AcceptRole)
        nogain_btn = box2.addButton("No gain", QMessageBox.AcceptRole)
        cancel_btn2 = box2.addButton("Cancel", QMessageBox.RejectRole)
        box2.exec()
        clicked2 = box2.clickedButton()
        if clicked2 is cancel_btn2 or clicked2 is None:
            return None

        if clicked2 is gain_btn:
            mode = "gain"
            seq = self.ROPER_GAIN_SEQUENCE
            labels = [f"{i}: inttime={it} us, gain={g}" for i, (it, g) in enumerate(seq)]
        else:
            mode = "nogain"
            seq = self.ROPER_NOGAIN_SEQUENCE
            labels = [f"{i}: inttime={it} us" for i, it in enumerate(seq)]

        label, ok = QInputDialog.getItem(self, "AutoTune start",
                                         "Select the starting point:", labels, 0, False)
        if not ok:
            return None
        return (True, mode, seq, labels.index(label), None, None)

    def _go_to_nv(self):
        path = self.open_file_dialog(self.data_saving_path)
        if not path:
            return
        structure = load_data(path)

        def _scalar(v):
            if v is None:
                return float("nan")
            a = np.asarray(v).ravel()
            return float(a[0]) if a.size else float("nan")

        results = []
        for name in vars(structure):  # image_1, image_2, ...
            if not name.startswith("image_"):
                continue
            point = getattr(structure, name)
            img = getattr(point, "camera_image", None)
            if img is None:
                continue
            arr = np.asarray(img)
            if arr.size == 0:
                continue
            results.append((
                float(arr.max()),  # brightest single pixel
                name,
                _scalar(getattr(point, "micro_x", None)),
                _scalar(getattr(point, "micro_y", None)),
                _scalar(getattr(point, "micro_z", None)),
                _scalar(getattr(point, "nano_z", None)),
            ))

        if not results:
            QMessageBox.warning(self, "No Data", "No images with camera_image found.")
            return

        results.sort(key=lambda r: r[0], reverse=True)
        top3 = results[:3]

        # let the user pick which candidate to drive to
        choice = self._choose_nv_candidate(top3)
        if choice is None:
            return  # cancelled
        _peak, _name, x, y, z, zn = choice

        self._go_to_nv_position(x, y, z, zn)
        return choice

    def _choose_nv_candidate(self, top3):
        """Show the top-3 candidates as radio buttons; return the chosen tuple or None."""
        from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QRadioButton,
                                     QDialogButtonBox, QButtonGroup, QLabel)
        dlg = QDialog(self)
        dlg.setWindowTitle("Top 3 NV Candidates")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Select the point to move to:"))

        group = QButtonGroup(dlg)
        for i, (peak, name, x, y, z, zn) in enumerate(top3):
            rb = QRadioButton(
                f"{name}:   peak={peak:.1f}    x={x:.4f}  y={y:.4f}  z={z:.4f} znano = {zn:.4f}",)
            if i == 0:
                rb.setChecked(True)  # default to the brightest
            group.addButton(rb, i)
            layout.addWidget(rb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec_() != QDialog.Accepted:
            return None
        idx = group.checkedId()
        if idx < 0:
            return None
        return top3[idx]

    def _go_to_nv_position(self, micro_x, micro_y, micro_z, nano_z):
        """Drive the XY stage (stage_2) to (micro_x, micro_y), then the Z stage
        (stage_3) to micro_z. Connect a stage first if it isn't already.
        Values are used in the stages' NATIVE units (same units they were saved
        in), so no unit conversion is applied here."""
        # --- XY: stage_2 ---
        if getattr(self, "stage_2", None) is None:
            self.connect_to_instrument_2()  # reads comboBox_2 and connects
        if getattr(self, "stage_2", None) is None:
            self.error_box("No XY stage",
                           "Could not connect the XY stage. Select it in its "
                           "dropdown and try again.")
            return
        if getattr(self, "stage_1", None) is None:
            self.connect_to_instrument_1()  # reads comboBox_2 and connects
        if getattr(self, "stage_1", None) is None:
            self.error_box("No XYZ stage",
                           "Could not connect the XYZ stage. Select it in its "
                           "dropdown and try again.")
            return

        if np.isnan(micro_x) or np.isnan(micro_y):
            self.error_box("Missing XY", "This point has no micro_x / micro_y saved.")
        else:
            try:
                self.stage_2.set_position("x", float(micro_x))
                self.stage_2.set_position("y", float(micro_y))
            except Exception as e:
                self.error_box("XY move failed", str(e))
                return
        if np.isnan(nano_z):
            self.error_box("Missing XYZ", "This point has no nano_z saved.")
        else:
            try:
                self.stage_1.set_position("z", float(nano_z))
            except Exception as e:
                self.error_box("Z move failed", str(e))
                return

        # --- Z: stage_3 ---
        if getattr(self, "stage_3", None) is None:
            self.connect_to_instrument_3()  # reads comboBox_3 and connects
        if getattr(self, "stage_3", None) is None:
            self.error_box("No Z stage",
                           "Moved XY, but could not connect the Z stage.")
            return

        if np.isnan(micro_z):
            self.error_box("Missing Z", "This point has no micro_z saved.")
        else:
            try:
                self.stage_3.set_position("z", float(micro_z))
            except Exception as e:
                self.error_box("Z move failed", str(e))
                return

        # reflect the new positions in the line-edits (best effort)
        try:
            self.xlineEdit_2.setText(str(self.stage_2.get_position("x")))
            self.ylineEdit_2.setText(str(self.stage_2.get_position("y")))
            self.zlineEdit_3.setText(str(self.stage_3.get_position("z")))
            self.zlineEdit_1.setText(str(self.stage_1.get_position("z")))
        except Exception:
            pass

    def find_NV(self) -> np.ndarray:

        self.save_or_find_nv_button_clicked.emit(1)
        path = self.open_file_dialog(self.data_saving_path)
        if not path:
            return
        method = self.choose_method()
        structure = load_data(path)
        if method == "affine":
            old_corners, new_corners, MNV_old = self.extract_corners(structure)
            # Compute transformations
            DMT = self.from_four_corners_to_DMT_or_DMNT(old_corners)
            DMNT = self.from_four_corners_to_DMT_or_DMNT(new_corners)

            MNV_new_direct = self.from_DMT_and_MNV_old_get_DNV_old(MNV_old, np.linalg.inv(DMNT) @ DMT)
            return MNV_new_direct
        elif method == "homography":
            method = self.choose_homography_method()

            old_corners, new_corners, MNV_old = self.extract_corners(structure)
            H_direct = self.compute_homography_from_corners(old_corners, new_corners, method)
            nv_new = self.map_point_with_homography(MNV_old, H_direct)
            return nv_new
        else:
            raise ValueError(f"Method {method} not implemented")

    def extract_corners(self, structure):
        order = ["top_left", "top_right", "bottom_right", "bottom_left"]
        old_corners = [self.get_coords(structure.INITIAL, n, "microxy") for n in order]
        new_corners = [self.get_coords(structure.FINAL, n, "microxy") for n in order]
        nv_position = self.get_coords(structure.INITIAL, "nv", "microxy")

        return old_corners, new_corners, nv_position

    def get_coords(self, block, name, coords):
        pt = getattr(block, name)  # MyStruct directly

        if pt is None:
            raise ValueError(f"No data for {name}")
        if coords == "microxy":
            return np.array([float(pt.micro_x), float(pt.micro_y)])
        elif coords == "microz":
            return np.array([float(pt.micro_z)])
        elif coords == "nanoz":
            return np.array([float(pt.nano_z)])
        elif coords == "nanox":
            return np.array([float(pt.nano_x)])
        elif coords == "nanoy":
            return np.array([float(pt.nano_y)])
        else:
            raise ValueError(f"Unknown coordinates {coords}")

    def close_server(self):
        self.server_off_button_clicked.emit(1)