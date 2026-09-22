'''
Nanodrive ADwin Confocal Scan Slow Module

This module implements slow, high-resolution scanning for confocal microscopy using:
- MCL NanoDrive for sample stage positioning
- ADwin Gold II for photon counting and timing
- Point-by-point scanning for maximum precision

The slow method goes point by point to ensure the scan is precise and accurate 
at the cost of execution time, but provides the highest quality images.
'''

import numpy as np
from pyqtgraph.exporters import ImageExporter
from pathlib import Path

from src.core import Parameter, Experiment
from src.core.helper_functions import get_configured_confocal_scans_folder
from src.core.adwin_helpers import get_adwin_binary_path
from time import sleep
import pyqtgraph as pg
import datetime
from src.core.struct_hdf5 import MyStruct

class NanodriveAdwinConfocalScanSlow(Experiment):
    '''
    Slow, high-precision confocal microscope scan using MCL NanoDrive and ADwin Gold II.
    
    This class runs a confocal microscope scan using the MCL NanoDrive to move 
    the sample stage and the ADwin Gold II to get count data. The slow method 
    goes point by point to ensure the scan is precise and accurate at the cost 
    of execution time.

    Hardware Dependencies:
    - MCL NanoDrive: For precise sample stage positioning
    - ADwin Gold II: For photon counting and timing control
    - ADbasic Binary: Trial_Counter.TB1 for counter operations
    '''

    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x',35,float,'x-coordinate start in microns'),
                   Parameter('y',35,float,'y-coordinate start in microns')
                   ]),
        Parameter('point_b',
                  [Parameter('x',95,float,'x-coordinate end in microns'),
                   Parameter('y', 95, float, 'y-coordinate end in microns')
                   ]),
        Parameter('z_pos', 50.0, float, 'z position of nanodrive; useful for z-axis sweeps to find NVs'),
        Parameter('resolution', 1, float, 'Resolution of each pixel in microns'),
        Parameter('time_per_pt', 5.0, float, 'Time in ms at each point to get counts'),
        Parameter('settle_time',0.2,float,'Time in seconds to allow NanoDrive to settle to correct position'),
        Parameter('ending_behavior', 'return_to_origin', ['return_to_inital_pos', 'return_to_origin', 'leave_at_corner'],'Nanodrive position after scan'),
        Parameter('3D_scan',
                  # steps z from z_min to z_max (inclusive) and takes a full x-y raster at each z. Useful for finding where NVs are in the focal plane
                  [Parameter('enable', False, bool, 'T/F to enable 3D scan'),
                   Parameter('folderpath', str(get_configured_confocal_scans_folder()), str,
                             'folder location to save images at each z-value'),
                   Parameter('z_min', 45.0, float, 'start z-position in microns for the 3D scan'),
                   Parameter('z_max', 55.0, float, 'end z-position in microns for the 3D scan'),
                   Parameter('z_step', 1.0, float, 'z step size in microns for the 3D scan')]),
        Parameter('MICROWAVE',
                  [Parameter('enable', False, bool,
                             'T/F to enable MW while MW is on: DO NOT DO IT IF THE AMP IS NOT POWERED!'),
                   Parameter('frequency', 2.0e9, float, 'MW Frequency'),
                   Parameter('power', -10.0, float, 'MW Power in dBm'),
                   Parameter('MW off after experiment?', True, bool,
                             "Choose if you want to turn the MW off after the experiment finishes"),
                   ]),
        Parameter('LASER',
                  [
                      Parameter('Filter Wheel OD', 0,
                                [0, 0.5, 1, 2, 3, 4], 'Filter Wheel OD'),
                      Parameter('Laser Control', 0.8, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                "Laser Control"),
                      Parameter('Laser off after experiment?', True, bool,
                                "Choose if you want to turn the laser off after the experiment finishes"), ]),
        # !!! If you see horizontial lines in the confocal image, the adwin arrays likely are corrupted. The fix is to reboot the adwin. You will nuke all
        # other process, variables, and arrays in the adwin. This parameter is added to make that easy to do in the GUI.
        Parameter('reboot_adwin', False, bool,'Will reboot adwin when experiment is executed. Useful is data looks fishy'),
        # clocks currently not implemented
        Parameter('laser_clock', 'Pixel', ['Pixel', 'Line', 'Frame', 'Aux'],
                  'Nanodrive clock used for turning laser on and off'),
        Parameter('save', False, bool, 'T/F to save each confocal image to an hdf5 file'),
        Parameter('filename', "nanodriveadwinconfocalscanslow", str,
                  'filename to save each confocal image to an hdf5 file'),
        Parameter('sample', "", str, "Sample Name to be saved with data"),
    ]

    #For actual experiment use LP100 [MCL_NanoDrive({'serial':2849})]. For testing using HS3 ['serial':2850]
    #_DEVICES = {'nanodrive': MCLNanoDrive(settings={'serial':2849}), 'adwin':AdwinGoldDevice()}  # Removed - devices now passed via constructor
    _DEVICES = {
        'nanodrive': 'nanodrive',
        'adwin': 'adwin',
        'proteus': 'proteus',
        'sg384': 'sg384',
        'filter_wheel': 'filter_wheel',
        'microdrive': 'microdrive',
    }
    _EXPERIMENTS = {}

    def __init__(self, devices, experiments=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Initializes and connects to devices
        Args:
            name (optional): name of experiment, if empty same as class name
            settings (optional): settings for this experiment, if empty same as default settings
        """
        super().__init__(name, settings=settings, sub_experiments=experiments, devices=devices, log_function=log_function, data_path=data_path)
        #get instances of devices
        self.nd = self.devices['nanodrive']['instance']
        self.adw = self.devices['adwin']['instance']
        self.sg384 = self.devices['sg384']['instance']
        self.proteus = self.devices['proteus']['instance']
        self.filter_wheel = self.devices['filter_wheel']['instance']
        self.microdrive = self.devices['microdrive']['instance']
        self.micro_x = None
        self.micro_y = None

    def save_hdf5(self):
        """this function defines its custom data and metadata to be saved and then calls the
        save_hdf_data function that is in the parent Experiment class, which adds the external
        devices in case you ever check the Get Basic Data checkbox in the GUI

        ONE hdf5 file is written per run of _function. Each z-slice is stored as its own
        confocal image (image_1, image_2, ...). If the 3D scan is disabled there is just
        image_1. The companion arrays (raw images, positions, counts) are saved stacked
        along the z-axis (axis 0 = slice index, ordered like image_1..image_N), and
        z_values holds the z-position of each image."""
        structure_to_save = MyStruct()

        # one confocal (count) image per z-slice -> image_1, image_2, ...
        data_dict = {}
        for n, count_img in enumerate(self.count_img_all, start=1):
            data_dict[f'image_{n}'] = count_img.T

        # companion data, stacked along z so it stays aligned with image_1..image_N
        data_dict['z_values'] = np.array(self.z_values)
        data_dict['raw_img'] = np.array([img.T for img in self.raw_img_all])
        data_dict['x_pos'] = np.array(self.x_pos_all)
        data_dict['y_pos'] = np.array(self.y_pos_all)
        data_dict['raw_counts'] = np.array(self.raw_counts_all)
        data_dict['count_rate'] = np.array(self.count_rate_all)
        data_dict['micro_xy'] = [self.micro_x, self.micro_y]

        # settings/z stay OUT of self.data (see _function) so the base Experiment.save_data()
        # pandas call does not choke on a nested dict. Metadata lives in meta instead.
        structure_to_save.data = MyStruct(**data_dict)
        structure_to_save.meta = MyStruct(
            settings=self.settings,
            end_time=self.e_t,
            start_time=self.s_t
        )
        structure_to_save.devices = self.devices
        self.save_hdf_data(structure_to_save)

    def setup_scan(self):
        '''
        Gets paths for adbasic file and loads them onto ADwin.
        '''
        self.adw.stop_process(1)
        sleep(0.1)
        self.adw.clear_process(1)
        
        # Use the helper function to find the binary file
        trial_counter_path = get_adwin_binary_path('Trial_Counter.TB1')
        self.adw.update({'process_1': {'load': str(trial_counter_path)}})
        #trial counter simply reads the counter value
        self.nd.clock_functions('Frame', reset=True)  # reset ALL clocks to default settings

        z_pos = self.settings['z_pos']
        if self.settings['z_pos'] < 0.0:
            z_pos = 0.0
        elif z_pos > 100.0:
            z_pos = 100.0
        self.nd.update({'z_pos': z_pos})

        # tracker to only save 3D image slice once
        self.data_collected = False

    def after_scan(self):
        '''
        Cleans up adwin and moves nanodrive to specified position
        '''
        # clearing process to aviod memory fragmentation when running different experiments in GUI
        self.adw.stop_process(1)    #neccesary if process is does not stop for some reason
        sleep(0.1)
        self.adw.clear_process(1)
        if self.settings['ending_behavior'] == 'return_to_inital_pos':
            self.nd.update({'x_pos': self.x_inital, 'y_pos': self.y_inital})
        elif self.settings['ending_behavior'] == 'return_to_origin':
            self.nd.update({'x_pos': 0.0, 'y_pos': 0.0})

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__

        3D scan: when 3D_scan.enable is True the nanodrive z is stepped from z_min to
        z_max (inclusive) in z_step increments and a full x-y raster is taken at every z.
        Each z-slice becomes image_1, image_2, ... inside ONE hdf5 file written at the end
        of the run. When 3D_scan is disabled a single image is taken at the current z and
        z is NOT moved (so a manually-set focus is not disturbed).
        """
        # resolve the 3D scan folder at runtime so the path is correct on this machine
        if not self.settings['3D_scan']['folderpath']:
            self.settings['3D_scan']['folderpath'] = str(get_configured_confocal_scans_folder())

        if self.settings['reboot_adwin'] == True:
            self.adw.reboot_adwin()
        self.filter_wheel.update({'OD': self.settings['LASER']['Filter Wheel OD']})
        if self.settings['MICROWAVE']['enable'] == True:
            if not self.sg384.is_connected:
                self.sg384.connect()
            # Set parameters
            frequency = self.settings['MICROWAVE']['frequency']
            power = self.settings['MICROWAVE']['power']
            # Set power
            self.sg384.set_power(power)
            # Set center frequency
            self.sg384.set_frequency(frequency)
            self.sg384._send('ENBR 1')
            self.proteus.set_channel_voltage_high(1, "MAX")
        self.proteus.set_channel_voltage_high(4, self.settings['LASER']["Laser Control"])
        self.setup_scan()
        sleep(0.1)

        x_min = self.settings['point_a']['x']
        x_max = self.settings['point_b']['x']
        y_min = self.settings['point_a']['y']
        y_max = self.settings['point_b']['y']
        step = self.settings['resolution']
        # array from point_a x,y to point_b x,y with step of resolution
        x_array = np.arange(x_min, x_max + step, step)
        y_array = np.arange(y_min, y_max + step, step)
        reversed_y_array = y_array[::-1]

        self.x_inital = self.nd.read_probes('x_pos')
        self.y_inital = self.nd.read_probes('y_pos')
        self.z_inital = self.nd.read_probes('z_pos')

        # ---------------------------------------------------------------------
        # Build the z-sweep array.
        #   * 3D scan ENABLED : step z from z_min -> z_max (inclusive) and run a
        #     full x-y raster at every z. z IS moved.
        #   * 3D scan DISABLED: original behaviour -> a single image at the
        #     current z. z is NOT moved (so a manually-set focus is not disturbed).
        # ---------------------------------------------------------------------
        if self.settings['3D_scan']['enable']:
            z_min = self.settings['3D_scan']['z_min']
            z_max = self.settings['3D_scan']['z_max']
            z_step = self.settings['3D_scan']['z_step']
            z_array = np.arange(z_min, z_max + z_step, z_step)
        else:
            self.settings['z_pos'] = self.z_inital
            z_array = np.array([self.z_inital])
        # ---------------------------------------------------------------------

        Nx = len(x_array)
        Ny = len(y_array)

        # progress tracking now spans the WHOLE 3D scan (all z-slices)
        interation_num = 0  # number to track progress
        total_interations = len(z_array) * ((x_max - x_min) / step + 1) * ((y_max - y_min) / step + 1)  # plus 1 because range is inclusive ie. [0,10]

        # formula to set adwin to count for correct time frame. The event section is run every delay*3.3ns so the counter increments for that time then is read and clear
        # time_per_pt is in millisecond and the adwin delay time is delay_value*3.3ns
        adwin_delay = round((self.settings['time_per_pt'] * 1e6) / (3.3))
        self.adw.update({'process_1': {'delay': adwin_delay, 'running': True}})

        # collect every finished z-slice so they can all go into ONE hdf5 file at the end of the run.
        # reset here so re-running _function starts a fresh file (no accumulation across runs)
        self.count_img_all = []
        self.raw_img_all = []
        self.x_pos_all = []
        self.y_pos_all = []
        self.raw_counts_all = []
        self.count_rate_all = []
        self.z_values = []

        # single start time for the whole run (one file per run)
        self.s_t = datetime.datetime.now()

        # ================= outer loop over z-slices =================
        for z_index, z in enumerate(z_array):
            if self._abort == True:
                break
            z = float(z)

            # move to this z-slice (only for a real 3D scan)
            if self.settings['3D_scan']['enable']:
                self.settings['z_pos'] = z
                self.nd.update({'z_pos': z})
                sleep(0.1)  # let the z stage settle / refocus

            # makes sure data is getting recorded. If still equal none after running experiment data is not being stored or not measured
            self.data['x_pos'] = None
            self.data['y_pos'] = None
            self.data['raw_counts'] = None
            self.data['counts'] = None
            self.data['count_img'] = None
            # local lists to store data and append to global self.data lists
            x_data = []
            y_data = []
            raw_counts_data = []
            count_rate_data = []

            self.data['count_img'] = np.zeros((Nx, Ny))

            # set inital x and y and set nanodrive stage to that position
            self.nd.update({'x_pos': x_min, 'y_pos': y_min})
            sleep(0.1)  # time for stage to move and adwin process to initilize

            forward = True  # used to rasterize more efficently going forward then back
            for i, x in enumerate(x_array):
                if self._abort:  # halts loop (and experiment) if stop button is pressed
                    break  # need to put break in x for loop which takes some time to stop but if stopped in y loop array sizes may mismatch and require a GUI restart
                x = float(x)
                img_row = []  # used for tracking image rows and adding to count_img; list not saved
                self.nd.update({'x_pos': x})

                if forward == True:
                    for y in y_array:
                        y = float(y)
                        self.nd.update({'y_pos': y})
                        sleep(self.settings['settle_time'])

                        x_pos = self.nd.read_probes('x_pos')
                        x_data.append(x_pos)
                        self.data['x_pos'] = x_data  # adds x postion to data
                        y_pos = self.nd.read_probes('y_pos')
                        y_data.append(y_pos)
                        self.data['y_pos'] = y_data  # adds y postion to data

                        raw_counts = self.adw.read_probes('int_var', id=1)  # raw number of counter triggers
                        count_rate = raw_counts * 1e3 / self.settings['time_per_pt']  # in units of counts/second

                        img_row.append(count_rate)
                        raw_counts_data.append(raw_counts)
                        count_rate_data.append(count_rate)
                        self.data['raw_counts'] = raw_counts_data
                        self.data['counts'] = count_rate_data

                else:
                    for y in reversed_y_array:
                        y = float(y)
                        self.nd.update({'y_pos': y})
                        sleep(self.settings['settle_time'])

                        x_pos = self.nd.read_probes('x_pos')
                        x_data.append(x_pos)
                        self.data['x_pos'] = x_data  # adds x postion to data
                        y_pos = self.nd.read_probes('y_pos')
                        y_data.append(y_pos)
                        self.data['y_pos'] = y_data  # adds y postion to data

                        raw_counts = self.adw.read_probes('int_var', id=1)  # raw number of counter triggers
                        count_rate = raw_counts * 1e3 / self.settings['time_per_pt']  # in units of counts/second

                        img_row.append(count_rate)
                        raw_counts_data.append(raw_counts)
                        count_rate_data.append(count_rate)
                        self.data['raw_counts'] = raw_counts_data
                        self.data['counts'] = count_rate_data
                    img_row.reverse()  # reversed since going from y_max --> y_min

                self.data['count_img'][i, :] = img_row
                forward = not forward

                interation_num = interation_num + len(y_array)
                self.progress = 100. * (interation_num + 1) / total_interations
                self.updateProgress.emit(int(round(self.progress)))

            # finalise this slice's data (this drives the live plot for the current slice)
            self.data['x_pos'] = x_data
            self.data['y_pos'] = y_data
            self.data['raw_counts'] = raw_counts_data
            self.data['counts'] = count_rate_data

            # tracker so _plot exports this z-slice's image
            self.data_collected = True

            # stash this finished slice; it becomes image_N in the single file written at the end.
            # copy the image so reusing self.data for the next slice can't mutate what we stored.
            # the slow scan has no separate 'raw' (uncropped) image, so raw_img == count_img here.
            self.count_img_all.append(self.data['count_img'].copy())
            self.raw_img_all.append(self.data['count_img'].copy())
            self.x_pos_all.append(np.array(x_data))
            self.y_pos_all.append(np.array(y_data))
            self.raw_counts_all.append(np.array(raw_counts_data))
            self.count_rate_all.append(np.array(count_rate_data))
            self.z_values.append(z)
        if self.settings['LASER']['Laser off after experiment?']:
            self.proteus.driver.off()
        if self.settings['MICROWAVE']['enable'] == True and self.settings['MICROWAVE']['MW off after experiment?']:
            self.sg384._send('ENBR 0')
        self.adw.update({'process_1': {'running': False}})
        # ONE end time and ONE save for the whole run -> image_1, image_2, ... in a single file.
        # skipped on abort so a partial run is not written.
        self.e_t = datetime.datetime.now()
        if self.settings['save'] and not self._abort:
            self.micro_x = self.microdrive.get_position("x")
            self.micro_y = self.microdrive.get_position("y")
            self.save_hdf5()

        self.after_scan()

    def _plot(self, axes_list, data=None):
        '''
        This function plots the data. It is triggered when the updateProgress signal is emited and when after the _function is executed.
        For the scan, image can only be plotted once all data is gathered so self.running prevents a plotting call for the updateProgress signal.
        '''
        def create_img(add_colobar=True):
            '''
            Creates a new image and ImageItem. Optionally create colorbar
            '''
            axes_list[0].clear()
            self.slow_count_image = pg.ImageItem(data['count_img'], interpolation='nearest')
            self.slow_count_image.setLevels(levels)
            self.slow_count_image.setRect(pg.QtCore.QRectF(extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2]))
            axes_list[0].addItem(self.slow_count_image)

            axes_list[0].setAspectLocked(True)
            axes_list[0].setLabel('left', 'y (µm)')
            axes_list[0].setLabel('bottom', 'x (µm)')
            axes_list[0].setTitle(f"Confocal Scan with z = {self.z_inital:.2f}")

            if add_colobar:
                self.colorbar = pg.ColorBarItem(values=(levels[0], levels[1]), label='counts/sec', colorMap='viridis')
                # layout is housing the PlotItem that houses the ImageItem. Add colorbar to layout so it is properly saved when saving dataset
                layout = axes_list[0].parentItem()
                layout.addItem(self.colorbar)
            self.colorbar.setImageItem(self.slow_count_image)

        if data is None:
            data = self.data
        if data is not None or data is not {}:

            # for colorbar to display graident without artificial zeros
            non_zero_values = data['count_img'][data['count_img'] > 0]
            if non_zero_values.size > 0:
                min = np.min(non_zero_values)
            else:  # if else to aviod ValueError
                min = 0

            levels = [min, np.max(data['count_img'])]
            extent = [self.settings['point_a']['x'], self.settings['point_b']['x'], self.settings['point_a']['y'],self.settings['point_b']['y']]
            # extent = [np.min(data['x_pos']), np.max(data['x_pos']), np.min(data['y_pos']), np.max(data['y_pos'])]

            if self._plot_refresh == True:
                # if plot refresh is true the ImageItem has been deleted and needs recreated
                create_img()
            else:
                try:
                    self.slow_count_image.setImage(data['count_img'], autoLevels=False)
                    self.slow_count_image.setLevels(levels)
                    self.colorbar.setLevels(levels)

                    if self.settings['3D_scan']['enable'] and self.data_collected:
                        axes_list[0].setTitle(f"Confocal Scan with z = {self.z_inital:.2f}")
                        scene = axes_list[0].scene()
                        exporter = ImageExporter(scene)
                        
                        # Use pathlib for cross-platform path handling
                        folder_path = Path(self.settings['3D_scan']['folderpath'])
                        try:
                            folder_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
                            filename = folder_path / f'confocal_scan_z_{self.z_inital:.2f}.png'
                            exporter.export(str(filename))
                        except Exception as e:
                            print(f"Warning: Failed to save 3D scan image: {e}")
                            print(f"Attempted to save to: {folder_path}")

                except RuntimeError:
                    # sometimes when clicking other experiments ImageItem is deleted but _plot_refresh is false. This ensures the image can be replotted
                    create_img(add_colobar=False)

    def _update(self,axes_list):
        self.slow_count_image.setImage(self.data['count_img'], autoLevels=False)
        self.slow_count_image.setLevels([np.min(self.data['count_img']),np.max(self.data['count_img'])])
        self.colorbar.setLevels([np.min(self.data['count_img']),np.max(self.data['count_img'])]) 