'''
This file has the experiment classes relevant to performing a scan with the confocal microscope. So far this includes:

- Confocal Scan Fast for larger images
- Confocal Scan Slow: a slow method that ensures image is accurate
- Confocal Point: Gets counts 1 point one time or continuously
'''

import numpy as np
from pyqtgraph.exporters import ImageExporter
from pathlib import Path
import warnings

from src.core import Parameter, Experiment
from src.core.helper_functions import get_configured_confocal_scans_folder
from src.core.adwin_helpers import get_adwin_binary_path
from time import sleep
import pyqtgraph as pg
import keyboard




class ConfocalScan_Fast(Experiment):
    '''
    DEPRECATED: This class has been replaced by NanodriveAdwinConfocalScanFast.
    
    This class runs a confocal microscope scan using the MCL NanoDrive to move the sample stage and the ADwin Gold II to get count data.
    The code loads a waveform on the nanodrive, starts the Adwin process, triggers a waveform aquisition, then reads the count data array from the Adwin.

    To get accurate counts, the loaded waveforms are extended to compensate for 'warm up' and 'cool down' movements. The data arrays are then
    manipulated to get the counts for the inputed region.
    
    WARNING: This class is deprecated and will be removed in a future version.
    Use NanodriveAdwinConfocalScanFast instead for better hardware-specific naming and organization.
    '''

    def __init__(self, devices, experiments=None, name=None, settings=None, log_function=None, data_path=None):
        warnings.warn(
            "ConfocalScan_Fast is deprecated and will be removed in a future version. "
            "Use NanodriveAdwinConfocalScanFast instead.",
            FutureWarning,
            stacklevel=2
        )
        super().__init__(name, settings=settings, sub_experiments=experiments, devices=devices, log_function=log_function, data_path=data_path)
        #get instances of devices
        self.nd = self.devices['nanodrive']['instance']
        self.adw = self.devices['adwin']['instance']

    _DEFAULT_SETTINGS = [
        Parameter('point_a',
                  [Parameter('x',5.0,float,'x-coordinate start in microns'),
                   Parameter('y',5.0,float,'y-coordinate start in microns')
                   ]),
        Parameter('point_b',
                  [Parameter('x',95.0,float,'x-coordinate end in microns'),
                   Parameter('y', 95.0, float, 'y-coordinate end in microns')
                   ]),
        Parameter('keyboard_control_nanodrive',
                  [Parameter('keyboard_control_on', False, bool, 'turn on for keyboard control of the nanodrive'),
                   Parameter('x_neg_key', 'a', str, 'negative x direction key'),
                   Parameter('x_pos_key', 'd', str, 'positive x direction key'),
                   Parameter('x_step', 0.1, float, 'step size in x direction'),
                   Parameter('y_neg_key', 's', str, 'negative y direction key'),
                   Parameter('y_pos_key', 'w', str, 'positive y direction key'),
                   Parameter('y_step', 0.1, float, 'step size in y direction'),
                   Parameter('z_pos_key', 'j', str, 'positive x direction key'),
                   Parameter('z_neg_key', 'k', str, 'negative x direction key'),
                   Parameter('z_step', 0.1, float, 'step size in x direction'),
                   Parameter('return_to_start_key', 'o', str, 'key to return to starting location'),
                   Parameter('settle_time', 0.2, float,
                             'Time in seconds to allow NanoDrive to settle to correct position'),
                   ]),
        Parameter('automated_optimization',
                  [Parameter('automated_optimization_on', False, bool,
                             'turn on for automated optimization of the counts'),
                   Parameter('xstep', 0.1, float, 'step size in x direction'),
                   Parameter('ystep', 0.1, float, 'step size in y direction'),
                   Parameter('zstep', 0.1, float, 'step size in x direction'),
                   Parameter('settle_time', 2.0, float,
                             'Time in seconds to allow NanoDrive to settle to correct position'),
                   Parameter('min_retention_ratio', 0.5, float,
                             'Minimum retention ratio: if optimized counts<min_retention_ratio*highest counts, go back to highest counts position'),
                   Parameter('damping_ratio', 0.9, float, 'damping ratio: go to x+damping_ratio*∆x instead of x+∆x'),
                   Parameter('Continuous_optimization_on', False, bool,
                             'turn on for continuous automated optimization of the counts'),
                   Parameter('acceptable_counts', 2500000, float,
                             'acceptable counts reference to make sure that we do not update our highest counts if they are too high (not reasonable)'),
                   # this is to make sure that we don't run optimization when it is not necessary especially that the nanodrive drifts the more you move it
                   Parameter('acceptable_counts_ratio', 1.1, float,
                             'acceptable counts ratio reference to make sure that we do not update our highest counts if they are too high (not reasonable)'),
                   Parameter('min_reoptimize_ratio', 0.5, float,
                             'Minimum reoptimize ratio: if optimized counts<min_reoptimize_ratio*highest_counts, optimize again (this is only if Continuous_optimization_on is False)')
                   ]),
        Parameter('z_pos',50.0,float,'z position of nanodrive; useful for z-axis sweeps to find NVs'),
        Parameter('resolution', 1.0, [2.0,1.0,0.5,0.25,0.1,0.05,0.025,0.001], 'Resolution of each pixel in microns. Limited to give '),
        Parameter('time_per_pt', 2.0, [2.0,5.0], 'Time in ms at each point to get counts; same as load_rate for nanodrive. Wroking values 2 or 5 ms'),
        Parameter('ending_behavior', 'return_to_origin', ['return_to_inital_pos', 'return_to_origin', 'leave_at_corner'],'Nanodrive position after scan'),
        Parameter('3D_scan',#using experiment iterator to sweep z-position can give an effective 3D scan as successive images. Useful for finding where NVs are in focal plane
                  [Parameter('enable',False,bool,'T/F to enable 3D scan'),
                         Parameter('folderpath',str(get_configured_confocal_scans_folder()),str,'folder location to save images at each z-value')]),
        #!!! If you see horizontial lines in the confocal image, the adwin arrays likely are corrupted. The fix is to reboot the adwin. You will nuke all
        #other process, variables, and arrays in the adwin. This parameter is added to make that easy to do in the GUI.
        Parameter('reboot_adwin',False,bool,'Will reboot adwin when experiment is executed. Useful is data looks fishy'),
        Parameter('cropping', #nested cause it does not need changed often
                  [Parameter('crop_data',True,bool,'Current logic scans over a larger area then crops data to requested size. Added for ease of seeing full image')]),
        #clocks currently not implemented
        Parameter('laser_clock', 'Pixel', ['Pixel','Line','Frame','Aux'], 'Nanodrive clocked used for turning laser on and off')
    ]

    #For actual experiment use LP100 [MCL_NanoDrive({'serial':2849})]. For testing using HS3 ['serial':2850]
    #_DEVICES = {'nanodrive': MCLNanoDrive(settings={'serial':2849}), 'adwin':AdwinGoldDevice()}  # Removed - devices now passed via constructor
    _DEVICES = {
        'nanodrive': 'nanodrive',
        'adwin': 'adwin'
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


    def setup_scan(self):
        '''
        Gets paths for adbasic file and loads them onto ADwin.
        '''
        self.adw.stop_process(2)
        sleep(0.1)
        self.adw.clear_process(2)
        
        # Use the helper function to find the binary file
        one_d_scan_path = get_adwin_binary_path('One_D_Scan.TB2')
        self.adw.update({'process_2': {'load': str(one_d_scan_path)}})
        # one_d_scan script increments an index then adds count values to an array in a constant time interval
        self.nd.clock_functions('Frame', reset=True)  # reset ALL clocks to default settings

        z_pos = self.settings['z_pos']
        #maz range is 0 to 100
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
        self.adw.stop_process(2)    #neccesary if process is does not stop for some reason
        sleep(0.1)
        self.adw.clear_process(2)
        if self.settings['ending_behavior'] == 'return_to_inital_pos':
            self.nd.update({'x_pos': self.x_inital, 'y_pos': self.y_inital})
        elif self.settings['ending_behavior'] == 'return_to_origin':
            self.nd.update({'x_pos': 0.0, 'y_pos': 0.0})

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        if self.settings['reboot_adwin'] == True:
            self.adw.reboot_adwin()
        self.setup_scan()
        sleep(0.1)

        #y scanning range is 5 to 95 to compensate for warm up time
        x_min = max(self.settings['point_a']['x'], 0.0)
        y_min = max(self.settings['point_a']['y'], 5.0)
        x_max = min(self.settings['point_b']['x'], 100.0)
        y_max = min(self.settings['point_b']['y'], 95.0)

        step = self.settings['resolution']
        num_points = (y_max - y_min) / step + 1
        print('num_points',num_points)
        if num_points < 91:
            new_step = self.correct_step(step)
            self.log(f'Works best with minimum 91 pixel resolution in y-direction. You are getting a free resolution upgrade to {new_step} um!')

        #array form point_a x,y to point_b x,y with step of resolution
        x_array = np.arange(x_min, x_max + step, step)
        y_array = np.arange(y_min, y_max+step, step)

        #adds point 5 um before and after
        y_before = np.arange(y_min-5.0,y_min,step)
        y_after = np.arange(y_max + step, y_max + 5.0 + step, step)
        y_array_adj = np.insert(y_array, 0, y_before)
        y_array_adj = np.append(y_array_adj, y_after)

        self.x_inital = self.nd.read_probes('x_pos')
        self.y_inital = self.nd.read_probes('y_pos')
        self.z_inital = self.nd.read_probes('z_pos')
        self.settings['z_pos'] = self.z_inital

        #makes sure data is getting recorded. If still equal none after running experiment data is not being stored or not measured
        self.data['x_pos'] = None
        self.data['y_pos'] = None
        self.data['raw_counts'] = None
        self.data['count_rate'] = None
        self.data['count_img'] = None
        self.data['raw_img'] = None
        #local lists to store data and append to global self.data lists
        x_data = []
        y_data = []
        raw_count_data = []
        count_rate_data = []
        index_list = []

        # set data to zero and update to plot while experiment runs
        Nx = len(x_array)
        Ny = len(y_array)
        self.data['count_img'] = np.zeros((Nx, Ny))
        self.data['raw_img'] = np.zeros((Nx, len(y_array_adj)+20))

        interation_num = 0 #number to track progress
        total_interations = ((x_max - x_min)/step + 1)*((y_max - y_min)/step + 1)       #plus 1 because in total_iterations because range is inclusive ie. [0,10]
        #print('total_interations=',total_interations)

        #formula to set adwin to count for correct time frame. The event section is run every delay*3.3ns so the counter increments for that time then is read and clear
        #time_per_pt is in millisecond and the adwin delay time is delay_value*3.3ns
        adwin_delay = round((self.settings['time_per_pt']*1e6) / (3.3))
        #print('adwin delay: ',delay)

        wf = list(y_array_adj)
        len_wf = len(y_array_adj)
        #print(len_wf,wf)
        load_read_ratio = self.settings['time_per_pt']/2.0 #used for scaling when rates are different
        num_points_read = int(load_read_ratio*len_wf + 20) #20 is added to compensate for start warm up producing ~15 points of unwanted values

        #set inital x and y and set nanodrive stage to that position
        self.nd.update({'x_pos':x_min,'y_pos':y_min-5.0,'num_datapoints':len_wf,'read_rate':2.0,'load_rate':self.settings['time_per_pt']})
        #load_rate is time_per_pt; 2.0ms = 5000Hz
        self.adw.update({'process_2':{'delay':adwin_delay}})
        sleep(0.1)  #time for stage to move to starting posiition and adwin process to initilize


        for i, x in enumerate(x_array):
            if self._abort == True:
                break
            img_row = []
            raw_img_row = []
            x = float(x)

            self.nd.update({'x_pos':x,'y_pos':y_min-5.0})     #goes to x position
            sleep(0.1)
            x_pos = self.nd.read_probes('x_pos')
            x_data.append(x_pos)
            self.data['x_pos'] = x_data     #adds x postion to data

            #The two different code lines to start counting seem to work for cropping. Honestly cant give a precise explaination, it seems to be related to
            #hardware delay. If the time_per_pt is 5.0 starting counting before waveform set up works to within 1 pixel with numpy cropping. If the
            #time_per_pt is 2.0 starting counting after waveform set up matches slow scan to a pixel. Sorry for a lack of explaination but this just seems to work.
            #See data/dylan_staples/confocal_scans_w_resolution_target for images and additional details
            if self.settings['time_per_pt'] == 5.0:
                self.adw.update({'process_2': {'running': True}})

            #trigger waveform on y-axis and record position data
            self.nd.setup(settings={'num_datapoints': len_wf, 'load_waveform': wf}, axis='y')
            self.nd.setup(settings={'num_datapoints': num_points_read, 'read_waveform': self.nd.empty_waveform},axis='y')

            #restricted load_rate and read_rate to ensure cropping works. 2ms and 5ms count times are good as smaller window for speed and a larger window if more counts are needed
            if  self.settings['time_per_pt'] == 2.0:
                self.adw.update({'process_2': {'running': True}})

            y_pos = self.nd.waveform_acquisition(axis='y')
            sleep(self.settings['time_per_pt']*len_wf/1000)

            #want to get data only in desired range not range±5um
            y_pos_array = np.array(y_pos)
            # index for the points of the read array when at y_min and y_max. Scale step by load_read_ratio to get points closest to y_min & y_max
            lower_index = np.where((y_pos_array > y_min - step / load_read_ratio) & (y_pos_array < y_min + step / load_read_ratio))[0]
            upper_index = np.where((y_pos_array > y_max - step / load_read_ratio) & (y_pos_array < y_max + step / load_read_ratio))[0]
            y_pos_cropped = list(y_pos_array[lower_index[0]:upper_index[0]])

            #y_data.extend(y_pos_cropped)
            y_data.append(list(y_pos))
            self.data['y_pos'] = y_data
            self.adw.update({'process_2':{'running':False}})

            #different index for count data if read and load rates are different
            counts_lower_index = int(lower_index[0] / load_read_ratio)
            counts_upper_index = int(upper_index[-1] / load_read_ratio)
            index_list.append(counts_upper_index)

            #get mode of index list and difference between mode and previous value
            index_mode = max(set(index_list), key=index_list.count)
            index_diff = abs(counts_upper_index - index_mode)
            # index starts at 0 so need to add 1 if there is an index difference
            if index_diff > 0:
                index_diff = index_diff + 1

            # get count data from adwin and record it
            raw_counts = np.array(list(self.adw.read_probes('int_array', id=1, length=len_wf+20)))
            # units of count/seconds
            count_rate = list(np.array(raw_counts) * 1e3 / self.settings['time_per_pt'])

            crop_index = -index_mode - 1 - index_diff
            if self.settings['time_per_pt'] == 5.0:
                crop_index = crop_index-2
            cropped_raw_counts = list(raw_counts[crop_index:crop_index + len(y_array)])
            cropped_count_rate = count_rate[crop_index:crop_index + len(y_array)]

            raw_count_data.append(cropped_raw_counts)
            self.data['raw_counts'] = raw_count_data

            count_rate_data.append(cropped_count_rate)
            self.data['count_rate'] = count_rate_data

            #adds count rate data to raw img and cropped count img
            raw_img_row.extend(count_rate)
            self.data['raw_img'][i, :] = raw_img_row
            img_row.extend(cropped_count_rate)
            self.data['count_img'][i, :] = img_row  # add previous scan data so image plots

            # updates process bar and plots count_img so far
            interation_num = interation_num + len(y_array)
            self.progress = 100. * (interation_num +1) / total_interations
            self.updateProgress.emit(self.progress)

        #tracker to only save test image once
        self.data_collected = True

        print('Data collected')
        self.data['x_pos'] = x_data
        self.data['y_pos'] = np.array(y_data)
        self.data['raw_counts'] = np.array(raw_count_data)
        self.data['count_rate'] = np.array(count_rate_data)
        #print('Position Data: ','\n',self.data['x_pos'],'\n',self.data['y_pos'],'\n','Max x: ',np.max(self.data['x_pos']),'Max y: ',np.max(self.data['y_pos']))
        #print('Counts: ','\n',self.count_data)
        #print('All data: ',self.data)

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
            self.count_image = pg.ImageItem(data['count_img'], interpolation='nearest')
            self.count_image.setLevels(levels)
            self.count_image.setRect(pg.QtCore.QRectF(extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2]))
            axes_list[0].addItem(self.count_image)

            axes_list[0].setAspectLocked(True)
            axes_list[0].setLabel('left', 'y (µm)')
            axes_list[0].setLabel('bottom', 'x (µm)')
            axes_list[0].setTitle(f"Confocal Scan with z = {self.settings['z_pos']:.2f}")

            if add_colobar:
                self.colorbar = pg.ColorBarItem(values=(levels[0], levels[1]), label='counts/sec', colorMap='viridis')
                # layout is housing the PlotItem that houses the ImageItem. Add colorbar to layout so it is properly saved when saving dataset
                layout = axes_list[0].parentItem()
                layout.addItem(self.colorbar)
            self.colorbar.setImageItem(self.count_image)

        if data is None:
            data = self.data
        if data is not None or data is not {}:
            #for colorbar to display graident without artificial zeros
            try: #sometimes when data is inputted as argument it does not have 'count_img' key; this try/except prevents error if that happens
                non_zero_values = data['count_img'][data['count_img'] > 0]
            except KeyError:
                data['count_img'] = self.data['count_img']
                non_zero_values = data['count_img'][data['count_img'] > 0]
            if non_zero_values.size > 0:
                min = np.min(non_zero_values)
            else: #if else to aviod ValueError
                min = 0

            levels = [min, np.max(data['count_img'])]
            extent = [self.settings['point_a']['x'], self.settings['point_b']['x'], self.settings['point_a']['y'],self.settings['point_b']['y']]

            if self._plot_refresh == True:
                # if plot refresh is true the ImageItem has been deleted and needs recreated
                create_img()
            else:
                try:
                    self.count_image.setImage(data['count_img'], autoLevels=False)
                    self.count_image.setLevels(levels)
                    self.colorbar.setLevels(levels)

                    if self.settings['3D_scan']['enable'] and self.data_collected:
                        print('z =', self.z_inital, 'max counts =', levels[1])
                        axes_list[0].setTitle(f"Confocal Scan with z = {self.z_inital:.2f}")
                        scene = axes_list[0].scene()
                        exporter = ImageExporter(scene)
                        
                        # Use pathlib for cross-platform path handling
                        folder_path = Path(self.settings['3D_scan']['folderpath'])
                        try:
                            folder_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
                            filename = folder_path / f'confocal_scan_z_{self.z_inital:.2f}.png'
                            exporter.export(str(filename))
                            print(f"Saved 3D scan image to: {filename}")
                        except Exception as e:
                            print(f"Warning: Failed to save 3D scan image: {e}")
                            print(f"Attempted to save to: {folder_path}")

                except RuntimeError:
                    # sometimes when clicking other experiments ImageItem is deleted but _plot_refresh is false. This ensures the image can be replotted
                    create_img(add_colobar=False)

    def _update(self,axes_list):
        self.count_image.setImage(self.data['count_img'], autoLevels=False)
        self.count_image.setLevels([np.min(self.data['count_img']), np.max(self.data['count_img'])])
        self.colorbar.setLevels([np.min(self.data['count_img']), np.max(self.data['count_img'])])

    def correct_step(self, old_step):
        '''
        Increases resolution by one threshold if the step size does not give enough points for a good y-array.
        For good y-array len() > 90
         '''
        if old_step == 1.0:
            return 0.5
        elif old_step > 1.0:
            return 1.0
        elif old_step == 0.5:
            return 0.25
        elif old_step == 0.25:
            return 0.1
        elif old_step == 0.1:
            return 0.05
        elif old_step == 0.05:
            return 0.025
        elif old_step == 0.025:
            return 0.001
        else:
            raise KeyError



class ConfocalScan_Slow(Experiment):
    '''
    DEPRECATED: This class has been replaced by NanodriveAdwinConfocalScanSlow.
    
    This class runs a confocal microscope scan using the MCL NanoDrive to move the sample stage and the ADwin Gold II to get count data.
    The slow method goes point by point to ensure the scan is precise and accurate at the cost of execution time
    
    WARNING: This class is deprecated and will be removed in a future version.
    Use NanodriveAdwinConfocalScanSlow instead for better hardware-specific naming and organization.
    '''

    def __init__(self, devices, experiments=None, name=None, settings=None, log_function=None, data_path=None):
        warnings.warn(
            "ConfocalScan_Slow is deprecated and will be removed in a future version. "
            "Use NanodriveAdwinConfocalScanSlow instead.",
            FutureWarning,
            stacklevel=2
        )
        super().__init__(name, settings=settings, sub_experiments=experiments, devices=devices, log_function=log_function, data_path=data_path)
        #get instances of devices
        self.nd = self.devices['nanodrive']['instance']
        self.adw = self.devices['adwin']['instance']

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
        Parameter('3D_scan',# using experiment iterator to sweep z-position can give an effective 3D scan as successive images. Useful for finding where NVs are in focal plane
                  [Parameter('enable', False, bool, 'T/F to enable 3D scan'),
                   Parameter('folderpath', str(get_configured_confocal_scans_folder()), str,'folder location to save images at each z-value')]),
        # !!! If you see horizontial lines in the confocal image, the adwin arrays likely are corrupted. The fix is to reboot the adwin. You will nuke all
        # other process, variables, and arrays in the adwin. This parameter is added to make that easy to do in the GUI.
        Parameter('reboot_adwin', False, bool,'Will reboot adwin when experiment is executed. Useful is data looks fishy'),
        # clocks currently not implemented
        Parameter('laser_clock', 'Pixel', ['Pixel','Line','Frame','Aux'], 'Nanodrive clock used for turning laser on and off')
    ]

    #For actual experiment use LP100 [MCL_NanoDrive({'serial':2849})]. For testing using HS3 ['serial':2850]
    #_DEVICES = {'nanodrive': MCLNanoDrive(settings={'serial':2849}), 'adwin':AdwinGoldDevice()}  # Removed - devices now passed via constructor
    _DEVICES = {
        'nanodrive': 'nanodrive',
        'adwin': 'adwin'
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
        """
        if self.settings['reboot_adwin'] == True:
            self.adw.reboot_adwin()
        self.setup_scan()
        sleep(0.1)

        x_min = self.settings['point_a']['x']
        x_max = self.settings['point_b']['x']
        y_min = self.settings['point_a']['y']
        y_max = self.settings['point_b']['y']
        step = self.settings['resolution']
        #array form point_a x,y to point_b x,y with step of resolution
        x_array = np.arange(x_min, x_max+step, step)
        y_array = np.arange(y_min, y_max + step, step)
        reversed_y_array = y_array[::-1]

        self.x_inital = self.nd.read_probes('x_pos')
        self.y_inital = self.nd.read_probes('y_pos')
        self.z_inital = self.nd.read_probes('z_pos')
        self.settings['z_pos'] = self.z_inital

        #makes sure data is getting recorded. If still equal none after running experiment data is not being stored or measured
        self.data['x_pos'] = None
        self.data['y_pos'] = None
        self.data['raw_counts'] = None
        self.data['counts'] = None
        self.data['count_img'] = None
        #local lists to store data and append to global self.data lists
        x_data = []
        y_data = []
        raw_counts_data = []
        count_rate_data = []

        Nx = len(x_array)
        Ny = len(y_array)
        self.data['count_img'] = np.zeros((Nx, Ny))

        interation_num = 0 #number to track progress
        total_interations = ((x_max - x_min)/step + 1)*((y_max - y_min)/step + 1)       #plus 1 because in total_iterations range is inclusive ie. [0,10]
        #print('total_interations=',total_interations)

        #formula to set adwin to count for correct time frame. The event section is run every delay*3.3ns so the counter increments for that time then is read and clear
        #time_per_pt is in millisecond and the adwin delay time is delay_value*3.3ns
        adwin_delay = round((self.settings['time_per_pt']*1e6) / (3.3))
        #print('adwin delay: ',adwin_delay)  606061 for 2ms and 606061*3.3 ns ~= 2 ms

        self.adw.update({'process_1': {'delay': adwin_delay, 'running': True}})
        # print(adwin_delay * 3.3 * 1e-9)
        # set inital x and y and set nanodrive stage to that position
        self.nd.update({'x_pos': x_min, 'y_pos': y_min})
        sleep(0.1)  # time for stage to move and adwin process to initilize

        forward = True #used to rasterize more efficently going forward then back
        for i, x in enumerate(x_array):
            if self._abort:  # halts loop (and experiment) if stop button is pressed
                break #need to put break in x for loop which takes some time to stop but if stopped in y loop array sizes may mismatch and require a GUI restart
            x = float(x)
            img_row = []  #used for tracking image rows and adding to count_img; list not saved
            self.nd.update({'x_pos':x})

            if forward == True:
                for y in y_array:
                    y = float(y)
                    print(x,y)
                    self.nd.update({'y_pos':y})
                    sleep(self.settings['settle_time'])

                    x_pos = self.nd.read_probes('x_pos')
                    x_data.append(x_pos)
                    self.data['x_pos'] = x_data  # adds x postion to data
                    y_pos = self.nd.read_probes('y_pos')
                    y_data.append(y_pos)
                    self.data['y_pos'] = y_data  # adds y postion to data

                    raw_counts = self.adw.read_probes('int_var',id=1)   #raw number of counter triggers
                    count_rate = raw_counts*1e3 / self.settings['time_per_pt'] # in units of counts/second

                    img_row.append(count_rate)
                    raw_counts_data.append(raw_counts)
                    count_rate_data.append(count_rate)
                    self.data['raw_counts'] = raw_counts_data
                    self.data['counts'] = count_rate_data

            else:
                for y in reversed_y_array:
                    y = float(y)
                    print(x,y)
                    self.nd.update({'y_pos':y})
                    sleep(self.settings['settle_time'])

                    x_pos = self.nd.read_probes('x_pos')
                    x_data.append(x_pos)
                    self.data['x_pos'] = x_data  # adds x postion to data
                    y_pos = self.nd.read_probes('y_pos')
                    y_data.append(y_pos)
                    self.data['y_pos'] = y_data  # adds y postion to data

                    raw_counts = self.adw.read_probes('int_var', id=1)  # raw number of counter triggers
                    count_rate = raw_counts*1e3 / self.settings['time_per_pt'] # in units of counts/second

                    img_row.append(count_rate)
                    raw_counts_data.append(raw_counts)
                    count_rate_data.append(count_rate)
                    self.data['raw_counts'] = raw_counts_data
                    self.data['counts'] = count_rate_data
                img_row.reverse() #reversed since going from y_max --> y_min

            self.data['count_img'][i, :] = img_row
            forward = not forward

            interation_num = interation_num + len(y_array)
            self.progress = 100. * (interation_num + 1) / total_interations
            self.updateProgress.emit(self.progress)

        # tracker to only save test image once
        self.data_collected = True

        print('Data collected')
        self.data['x_pos'] = x_data
        self.data['y_pos'] = y_data
        self.data['raw_counts'] = raw_counts_data
        self.data['counts'] = count_rate_data

        #print('Position Data: ', '\n', self.data['x_pos'], '\n', self.data['y_pos'], '\n', 'Max x: ',np.max(self.data['x_pos']), 'Max y: ', np.max(self.data['y_pos']))
        #print('All data: ',self.data)

        self.adw.update({'process_2': {'running': False}})
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
                        print('z =', self.z_inital, 'max counts =', levels[1])
                        axes_list[0].setTitle(f"Confocal Scan with z = {self.z_inital:.2f}")
                        scene = axes_list[0].scene()
                        exporter = ImageExporter(scene)
                        
                        # Use pathlib for cross-platform path handling
                        folder_path = Path(self.settings['3D_scan']['folderpath'])
                        try:
                            folder_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
                            filename = folder_path / f'confocal_scan_z_{self.z_inital:.2f}.png'
                            exporter.export(str(filename))
                            print(f"Saved 3D scan image to: {filename}")
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



class Confocal_Point(Experiment):
    '''
    DEPRECATED: This class has been replaced by NanodriveAdwinConfocalPoint.
    
    This class implements a confocal microscope to get the counts at a single point. It uses the MCL NanoDrive to move the sample stage and the ADwin Gold to get count data.
    The 'continuous' parameter if false will return 1 data point. If true it offers live counting that continues until the stop button is clicked.
    
    WARNING: This class is deprecated and will be removed in a future version.
    Use NanodriveAdwinConfocalPoint instead for better hardware-specific naming and organization.
    '''

    def __init__(self, devices, experiments=None, name=None, settings=None, log_function=None, data_path=None):
        warnings.warn(
            "Confocal_Point is deprecated and will be removed in a future version. "
            "Use NanodriveAdwinConfocalPoint instead.",
            FutureWarning,
            stacklevel=2
        )
        super().__init__(name, settings=settings, sub_experiments=experiments, devices=devices, log_function=log_function, data_path=data_path)
        #get instances of devices
        self.nd = self.devices['nanodrive']['instance']
        self.adw = self.devices['adwin']['instance']

    _DEFAULT_SETTINGS = [
        Parameter('point',
                  [Parameter('x',0.0,float,'x-coordinate in microns'),
                   Parameter('y',0.0,float,'y-coordinate in microns'),
                   Parameter('z',0.0,float,'z-coordinate in microns')
                   ]),
        Parameter('count_time', 2.0, float, 'Time in ms at  point to get count data'),
        Parameter('num_cycles', 10, int, 'Number of samples to average; set as Par_10 in adbasic scirpt'),
        Parameter('plot_avg', True, bool, 'T/F to plot average count data'),
        Parameter('continuous', True, bool,'If experiment should return 1 value or continuously plot for optics optimization'),
        Parameter('graph_params',
                  [Parameter('plot_raw_counts', False, bool,'Sometimes counts/sec is rounded to zero. Check this to plot raw counts'),
                   Parameter('refresh_rate', 0.1, float,'For continuous counting this is the refresh rate of the graph in seconds (= 1/frames per second)'),
                   Parameter('length_data',500,int,'After so many data points matplotlib freezes GUI. Data dic will be cleared after this many entries'),
                   Parameter('font_size',32,int,'font size to make it easier to see on the fly if needed'),
                   ]),
        # clocks currently not implemented
        Parameter('laser_clock', 'Pixel', ['Pixel', 'Line', 'Frame', 'Aux'],'Nanodrive clocked used for turning laser on and off'),
    ]

    #For actual experiment use LP100 [MCL_NanoDrive({'serial':2849})]. For testing cautiously using HS3 ['serial':2850]
    #_DEVICES = {'nanodrive': MCLNanoDrive(settings={'serial':2849}), 'adwin':AdwinGoldDevice()}  # Removed - devices now passed via constructor
    _DEVICES = {
        'nanodrive': 'nanodrive',
        'adwin': 'adwin'
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


    def setup(self):
        '''
        Gets paths for adbasic file and loads them onto ADwin.
        '''
        self.adw.stop_process(1)
        sleep(0.1)
        self.adw.clear_process(1)
        
        # Use the helper function to find the binary file
        trial_counter_path = get_adwin_binary_path('Averagable_Trial_Counter.TB1')
        self.adw.update({'process_1': {'load': str(trial_counter_path)}})
        self.nd.clock_functions('Frame', reset=True)  # reset ALL clocks to default settings

    def cleanup(self):
        '''
        Cleans up adwin after experiment
        '''
        self.adw.stop_process(1)
        sleep(0.1)
        self.adw.clear_process(1)

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        self.setup()

        self.data['counts'] = None
        self.data['raw_counts'] = None
        # set to zero initially for smoother plotting
        count_rate_data = [0] * self.settings['graph_params']['length_data']
        raw_counts_data = [0] * self.settings['graph_params']['length_data']

        x = self.settings['point']['x']
        y = self.settings['point']['y']
        z = self.settings['point']['z']

        num_cycles = self.settings['num_cycles']
        self.adw.set_int_var(10,num_cycles)
        #set adwin delay which determines the counting time
        adwin_delay = round((self.settings['count_time']*1e6) / (3.3))
        self.adw.update({'process_1':{'delay':adwin_delay,'running':True}})
        self.nd.update({'x_pos':x,'y_pos':y,'z_pos':z})
        sleep(0.1)  #time for stage to move and adwin process to initilize

        # get the current xyz positions
        original_x = self.nd.read_probes('x_pos')
        original_y = self.nd.read_probes('y_pos')
        original_z = self.nd.read_probes('z_pos')
        print(f'nanodrive at x = {original_x}, y = {original_y}, z = {original_z}')

        sleep(0.1)  # time for stage to move and adwin process to initilize

        # determine if keyboard controls are on
        # if it is break out the parameters for keyboard tuning of the nanodrive
        if self.settings['keyboard_control_nanodrive']['keyboard_control_on']:
            keyboard_control_on = True
            x_neg_key = self.settings['keyboard_control_nanodrive']['x_neg_key']
            x_pos_key = self.settings['keyboard_control_nanodrive']['x_pos_key']
            x_step = self.settings['keyboard_control_nanodrive']['x_step']
            y_neg_key = self.settings['keyboard_control_nanodrive']['y_neg_key']
            y_pos_key = self.settings['keyboard_control_nanodrive']['y_pos_key']
            y_step = self.settings['keyboard_control_nanodrive']['y_step']
            z_neg_key = self.settings['keyboard_control_nanodrive']['z_neg_key']
            z_pos_key = self.settings['keyboard_control_nanodrive']['z_pos_key']
            z_step = self.settings['keyboard_control_nanodrive']['z_step']
            return_to_start_key = self.settings['keyboard_control_nanodrive']['return_to_start_key']
            settle_time = self.settings['keyboard_control_nanodrive']['settle_time']
        else:
            keyboard_control_on = False

        if self.settings['automated_optimization']['automated_optimization_on']:
            automated_optimization_on = True
            xstep = self.settings['automated_optimization']['xstep']
            ystep = self.settings['automated_optimization']['ystep']
            zstep = self.settings['automated_optimization']['zstep']
            settle_time = self.settings['keyboard_control_nanodrive']['settle_time']
            min_retention_ratio = self.settings['automated_optimization'][
                'min_retention_ratio']  # Minimum retention ratio: if optimized counts<min_retention_ratio*highest counts, go back to highest counts position
            damping_ratio = self.settings['automated_optimization']['damping_ratio']
            acceptable_counts = self.settings['automated_optimization']['acceptable_counts']
            acceptable_counts_ratio = self.settings['automated_optimization']['acceptable_counts_ratio']
            x_highest = self.nd.read_probes('x_pos')
            y_highest = self.nd.read_probes('y_pos')
            z_highest = self.nd.read_probes('z_pos')
            countsoriginal = self.adw.read_probes('int_var', id=1)  # read variable from adwin
            highest_counts = countsoriginal * 1e3 / self.settings['count_time']
            min_reoptimize_ratio = self.settings['automated_optimization']['min_reoptimize_ratio']
            if self.settings['automated_optimization']['Continuous_optimization_on']:
                Continuous_optimization_on = True
            else:
                Continuous_optimization_on = False
                # Minimum reoptimize ratio: if optimized counts<min_reoptimize_ratio*highest_counts, optimize again (this is only if Continuous_optimization_on is False)
        else:
            automated_optimization_on = False

        if self.settings['continuous'] == False:
            if self.settings['plot_avg']:
                counting_time = self.settings['count_time']*self.settings['num_cycles']
            else:
                counting_time = self.settings['count_time']
            sleep((counting_time*1.5)/1000)    #sleep for 1.5 times the count time to ensure enough time for counts. Does not affect counting window

            if self.settings['plot_avg']:
                raw_counts = self.adw.read_probes('int_var', id=5) / self.settings['num_cycles']  # Par_5 stores the total counts over 'num_cycles'
                counts = raw_counts * 1e3 / self.settings['count_time']
            else:
                raw_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                counts = raw_counts * 1e3 / self.settings['count_time']

            for i in range(0,2):        #just want the single value to be viewable so will plot a straight line (with 2 points) of its value
                raw_counts_data.append(raw_counts)
                count_rate_data.append(counts)
            self.data['raw_counts'] = raw_counts_data
            self.data['counts'] = count_rate_data

        elif self.settings['continuous'] == True:
            while self._abort == False:     #self._abort is defined in experiment.py and is true false while running and set false when stop button is hit
                sleep(self.settings['graph_params']['refresh_rate'])    #effictivly this sleep is the time interval the graph is refreshed (1/fps) counting window
                # if keyboard tuning is on, look for keyboard presses, and move nanodrive accordingly
                if keyboard_control_on:
                    if keyboard.is_pressed(x_neg_key):
                        self.nd.update({'x_pos': self.nd.read_probes('x_pos') - x_step})
                        sleep(settle_time)
                    if keyboard.is_pressed(x_pos_key):
                        self.nd.update({'x_pos': self.nd.read_probes('x_pos') + x_step})
                        sleep(settle_time)
                    if keyboard.is_pressed(y_neg_key):
                        self.nd.update({'y_pos': self.nd.read_probes('y_pos') - y_step})
                        sleep(settle_time)
                    if keyboard.is_pressed(y_pos_key):
                        self.nd.update({'y_pos': self.nd.read_probes('y_pos') + y_step})
                        sleep(settle_time)
                    if keyboard.is_pressed(z_neg_key):
                        self.nd.update({'z_pos': self.nd.read_probes('z_pos') - z_step})
                        sleep(settle_time)
                    if keyboard.is_pressed(z_pos_key):
                        self.nd.update({'z_pos': self.nd.read_probes('z_pos') + z_step})
                        sleep(settle_time)
                    if keyboard.is_pressed(return_to_start_key):
                        self.nd.update({'x_pos': original_x, 'y_pos': original_y, 'z_pos': original_z})
                        sleep(10 * settle_time)

                # JANNET's part:
                # Plot solution to make sure that the code and theory match
                # Play with parameters: step size and settle time
                # Go back 2 steps and then move in one direction to solve backlash
                # Add damping: x+ damping_ratio*∆x instead of x+∆x
                # Use: Try except to avoid crashing if there is an out of range error 
                # Skip optimization for certain axis if we get an out of range error: no assumptions
                # Go back to highest counts point if optimized counts are less than min_retention_ratio of the highest counts one 
                # 2 cases to run optimization: continuous optimization is on OR optimized_counts < min_reoptimize ratio * (original counts or highest_counts)
                # (User has to optimize manually (or with keyboard) and then add coordinates before restarting confocal point experiment with auto optimization:
                # This also ensures that we are getting counts from the same nanodiamond that we started measuring)
                if automated_optimization_on:
                    current_counts = (self.adw.read_probes('int_var', id=1) * 1e3) / self.settings['count_time']
                    print('current counts:', current_counts)
                    print('cutoff counts:', (min_reoptimize_ratio * highest_counts))
                    if Continuous_optimization_on or (
                            ((self.adw.read_probes('int_var', id=1) * 1e3) / self.settings['count_time']) < (
                            min_reoptimize_ratio * highest_counts)):
                        xminus, countsxminus = None, None
                        xpos, countsxold = None, None
                        xplus, countsxplus = None, None
                        yminus, countsyminus = None, None
                        ypos, countsyold = None, None
                        yplus, countsyplus = None, None
                        zminus, countszminus = None, None
                        zpos, countszold = None, None
                        zplus, countszplus = None, None
                        # x optimization
                        # go to x minus 2 steps (this way it solves backlash)
                        try:
                            self.nd.update({'x_pos': self.nd.read_probes('x_pos') - 2 * xstep})
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        sleep(settle_time)
                        # increase x (this will be x minus)
                        try:
                            self.nd.update({'x_pos': self.nd.read_probes('x_pos') + xstep})
                            sleep(settle_time)
                            xminus = self.nd.read_probes('x_pos')
                            # read
                            x_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countsxminus = x_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # increase x (this will be x old)
                        try:
                            self.nd.update({'x_pos': self.nd.read_probes('x_pos') + xstep})
                            sleep(settle_time)
                            xpos = self.nd.read_probes('x_pos')
                            # read
                            x_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countsxold = x_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # increase x (this will be x plus)
                        try:
                            self.nd.update({'x_pos': self.nd.read_probes('x_pos') + xstep})
                            sleep(settle_time)
                            xplus = self.nd.read_probes('x_pos')
                            # read
                            x_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countsxplus = x_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # find max:
                        x_points = [(xminus, countsxminus), (xpos, countsxold), (xplus, countsxplus)]
                        x_points_valid = [(x, c) for (x, c) in x_points if x is not None and c is not None]

                        if len(x_points_valid) < 3:
                            print("Skipping x-optimization: insufficient valid points.")
                        else:
                            x_pos_coords = np.array([p[0] for p in x_points])
                            x_count_coords = np.array([p[1] for p in x_points])
                            x_coefficients = np.polyfit(x_pos_coords, x_count_coords, 2)
                            ax, bx, cx = x_coefficients
                            # a * x**2 + b * x + c
                            xmax = float((-bx) / (2 * ax))
                            # x_counts = self.adw.read_probes('int_var', id=1)  #remove comment if you need to plot xyz
                            # countsxmax = x_counts * 1e3 / self.settings['count_time'] #remove comment if you need to plot xyz
                            try:
                                self.nd.update({'x_pos': xmax})
                                sleep(settle_time)
                                xmax = self.nd.read_probes('x_pos')
                                x_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                                countsxmax = x_counts * 1e3 / self.settings['count_time']
                            except Exception as e:
                                if "ARGUMENT_ERROR" in str(e):
                                    print(f"Warning: Skipping update due to out-of-range error: {e}")
                                else:
                                    raise Exception
                            # last position measured is x max so we will move from x max
                            if (countsxminus > countsxmax):
                                try:
                                    self.nd.update({'x_pos': xmax + (damping_ratio * (xminus - xmax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            elif (countsxold > countsxmax):
                                try:
                                    self.nd.update({'x_pos': xmax + (damping_ratio * (xpos - xmax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            elif (countsxplus > countsxmax):
                                try:
                                    self.nd.update({'x_pos': xmax + (damping_ratio * (xplus - xmax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            sleep(settle_time)

                        # y optimization
                        # decrease y by 2
                        try:
                            self.nd.update({'y_pos': self.nd.read_probes('y_pos') - 2 * ystep})
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        sleep(settle_time)
                        # increase y (this will be y minus)
                        try:
                            self.nd.update({'y_pos': self.nd.read_probes('y_pos') + ystep})
                            sleep(settle_time)
                            yminus = self.nd.read_probes('y_pos')
                            # read
                            y_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countsyminus = y_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # increase y (this will be y old)
                        try:
                            self.nd.update({'y_pos': self.nd.read_probes('y_pos') + ystep})
                            sleep(settle_time)
                            ypos = self.nd.read_probes('y_pos')
                            # read
                            y_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countsyold = y_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # increase y (this will be y plus)
                        try:
                            self.nd.update({'y_pos': self.nd.read_probes('y_pos') + ystep})
                            sleep(settle_time)
                            yplus = self.nd.read_probes('y_pos')
                            # read
                            y_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countsyplus = y_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # find max:
                        y_points = [(yminus, countsyminus), (ypos, countsyold), (yplus, countsyplus)]
                        y_points_valid = [(y, c) for (y, c) in y_points if y is not None and c is not None]

                        if len(y_points_valid) < 3:
                            print("Skipping y-optimization: insufficient valid points.")
                        else:
                            y_pos_coords = np.array([p[0] for p in y_points])
                            y_count_coords = np.array([p[1] for p in y_points])
                            y_coefficients = np.polyfit(y_pos_coords, y_count_coords, 2)
                            ay, by, cy = y_coefficients
                            # a * x**2 + b * x + c
                            ymax = float((-by) / (2 * ay))
                            # y_counts = self.adw.read_probes('int_var', id=1)  #remove comment if you need to plot xyz
                            # countsymax = y_counts * 1e3 / self.settings['count_time'] #remove comment if you need to plot xyz
                            try:
                                self.nd.update({'y_pos': ymax})
                                sleep(settle_time)
                                ymax = self.nd.read_probes('y_pos')
                                y_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                                countsymax = y_counts * 1e3 / self.settings['count_time']
                            except Exception as e:
                                if "ARGUMENT_ERROR" in str(e):
                                    print(f"Warning: Skipping update due to out-of-range error: {e}")
                                else:
                                    raise Exception
                            # last position measured is y max so we will move from y max
                            if (countsyminus > countsymax):
                                try:
                                    self.nd.update({'y_pos': ymax + (damping_ratio * (yminus - ymax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            elif (countsyold > countsymax):
                                try:
                                    self.nd.update({'y_pos': ymax + (damping_ratio * (ypos - ymax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            elif (countsyplus > countsymax):
                                try:
                                    self.nd.update({'y_pos': ymax + (damping_ratio * (yplus - ymax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            sleep(settle_time)
                        # z optimization
                        # decrease z 2 steps
                        try:
                            self.nd.update({'z_pos': self.nd.read_probes('z_pos') - 2 * zstep})
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        sleep(settle_time)
                        # increase z (this will be z minus)
                        try:
                            self.nd.update({'z_pos': self.nd.read_probes('z_pos') + zstep})
                            sleep(settle_time)
                            zminus = self.nd.read_probes('z_pos')
                            # read
                            z_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countszminus = z_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # increase z (this will be z old)
                        try:
                            self.nd.update({'z_pos': self.nd.read_probes('z_pos') + zstep})
                            sleep(settle_time)
                            zpos = self.nd.read_probes('z_pos')
                            # read
                            z_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countszold = z_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # increase z (this will be z plus)
                        try:
                            self.nd.update({'z_pos': self.nd.read_probes('z_pos') + zstep})
                            sleep(settle_time)
                            zplus = self.nd.read_probes('z_pos')
                            # read
                            z_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            countszplus = z_counts * 1e3 / self.settings['count_time']
                        except Exception as e:
                            if "ARGUMENT_ERROR" in str(e):
                                print(f"Warning: Skipping update due to out-of-range error: {e}")
                            else:
                                raise Exception
                        # find max:
                        z_points = [(zminus, countszminus), (zpos, countszold), (zplus, countszplus)]
                        z_points_valid = [(z, c) for (z, c) in z_points if z is not None and c is not None]

                        if len(z_points_valid) < 3:
                            print("Skipping z-optimization: insufficient valid points.")
                        else:
                            z_pos_coords = np.array([p[0] for p in z_points])
                            z_count_coords = np.array([p[1] for p in z_points])
                            z_coefficients = np.polyfit(z_pos_coords, z_count_coords, 2)
                            az, bz, cz = z_coefficients
                            # a * z**2 + b * z + c
                            zmax = float((-bz) / (2 * az))
                            # z_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                            # countszmax = z_counts * 1e3 / self.settings['count_time']
                            try:
                                self.nd.update({'z_pos': zmax})
                                sleep(settle_time)
                                zmax = self.nd.read_probes('z_pos')
                                z_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                                countszmax = z_counts * 1e3 / self.settings['count_time']
                            except Exception as e:
                                if "ARGUMENT_ERROR" in str(e):
                                    print(f"Warning: Skipping update due to out-of-range error: {e}")
                                else:
                                    raise Exception
                            # last position measured is zmax so we will move from zmax
                            if (countszminus > countszmax):
                                try:
                                    self.nd.update({'z_pos': zmax + (damping_ratio * (zminus - zmax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            elif (countszold > countszmax):
                                try:
                                    self.nd.update({'z_pos': zmax + (damping_ratio * (zpos - zmax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            elif (countszplus > countszmax):
                                try:
                                    self.nd.update({'z_pos': zmax + (damping_ratio * (zplus - zmax))})
                                except Exception as e:
                                    if "ARGUMENT_ERROR" in str(e):
                                        print(f"Warning: Skipping update due to out-of-range error: {e}")
                                    else:
                                        raise Exception
                            sleep(settle_time)
                        # Check if optimized counts<min_retention_ratio*highest counts, go back to highest counts position
                        optimized_counts = (self.adw.read_probes('int_var', id=1) * 1e3) / self.settings['count_time']
                        if optimized_counts > highest_counts:
                            if optimized_counts < acceptable_counts * acceptable_counts_ratio:
                                highest_counts = optimized_counts
                                x_highest = self.nd.read_probes('x_pos')
                                y_highest = self.nd.read_probes('y_pos')
                                z_highest = self.nd.read_probes('z_pos')
                                # the above section makes sure that the highest_counts can only be updated if they are reasonable (if they are higher than expected, they are probably just a fluctuation and therefore, they won't be a good reference)
                        elif optimized_counts < (min_retention_ratio * highest_counts):
                            xpos = self.nd.read_probes('x_pos')
                            ypos = self.nd.read_probes('y_pos')
                            zpos = self.nd.read_probes('z_pos')
                            self.nd.update({'x_pos': xpos + damping_ratio * (x_highest - xpos),
                                            'y_pos': ypos + damping_ratio * (y_highest - ypos),
                                            'z_pos': zpos + damping_ratio * (z_highest - zpos)})
                            sleep(settle_time)

                        """
                        # PyQtGraph plot setup
                        app = QApplication(sys.argv)
                        win = pg.plot(title="XMAX")
                        win.setLabel('bottom', 'x')
                        win.setLabel('left', 'f(x)')
                        win.showGrid(x=True, y=True)

                        # plot polyfit curve
                        x_range = np.linspace(min(x_pos_coords) - 2, max(x_pos_coords) + 2, 200)
                        y_fit = np.polyval(x_coefficients, x_range)
                        win.plot(x_range, y_fit, pen=pg.mkPen('w', width=2), name="Polyfit")

                        # plot sampled red points
                        win.plot(x_pos_coords, x_count_coords, pen=None, symbol='o', symbolBrush='r', symbolSize=10, name="Sampled points")

                        # plot xmax yellow point
                        win.plot([xmax], [countsxmax], pen=None, symbol='o', symbolBrush='y', symbolPen='w', symbolSize=12, name="xmax")

                        # PyQtGraph plot setup
                        win2 = pg.plot(title="YMAX")
                        win2.setLabel('bottom', 'y')
                        win2.setLabel('left', 'f(y)')
                        win2.showGrid(x=True, y=True)

                        # plot polyfit curve
                        y_range = np.linspace(min(y_pos_coords) - 2, max(y_pos_coords) + 2, 200)
                        y_fit = np.polyval(y_coefficients, y_range)
                        win2.plot(y_range, y_fit, pen=pg.mkPen('w', width=2), name="Polyfit")

                        # plot sampled red points
                        win2.plot(y_pos_coords, y_count_coords, pen=None, symbol='o', symbolBrush='r', symbolSize=10,
                                 name="Sampled points")

                        # plot ymax yellow point
                        win2.plot([ymax], [countsymax], pen=None, symbol='o', symbolBrush='y', symbolPen='w', symbolSize=12,
                                 name="ymax")

                        # PyQtGraph plot setup
                        win3 = pg.plot(title="ZMAX")
                        win3.setLabel('bottom', 'z')
                        win3.setLabel('left', 'f(z)')
                        win3.showGrid(x=True, y=True)

                        # plot polyfit curve
                        z_range = np.linspace(min(z_pos_coords) - 2, max(z_pos_coords) + 2, 200)
                        z_fit = np.polyval(z_coefficients, z_range)
                        win3.plot(z_range, z_fit, pen=pg.mkPen('w', width=2), name="Polyfit")

                        # plot sampled red points
                        win3.plot(z_pos_coords, z_count_coords, pen=None, symbol='o', symbolBrush='r', symbolSize=10,
                                  name="Sampled points")

                        # plot zmax yellow point
                        win3.plot([zmax], [countszmax], pen=None, symbol='o', symbolBrush='y', symbolPen='w', symbolSize=12,
                                  name="zmax")
                        sys.exit(app.exec_())
                        return
                    """
                if self.settings['plot_avg']:
                    raw_counts = self.adw.read_probes('int_var',id=5) / self.settings['num_cycles'] #Par_5 stores the total counts over 'num_cycles'
                    counts = raw_counts * 1e3 / self.settings['count_time']
                else:
                    raw_counts = self.adw.read_probes('int_var', id=1)  # read variable from adwin
                    counts = raw_counts * 1e3 / self.settings['count_time']

                #append most recent value and remove oldest value
                raw_counts_data.append(raw_counts)
                raw_counts_data.pop(0)
                count_rate_data.append(counts)
                count_rate_data.pop(0)
                self.data['raw_counts'] = raw_counts_data
                self.data['counts'] = count_rate_data
                #print('Current count rate', self.data['counts'][-1])

                self.progress = 50   #this is a infinite loop till stop button is hit; progress & updateProgress is only here to update plot
                self.updateProgress.emit(self.progress)     #calling updateProgress.emit triggers _plot

        self.adw.update({'process_1': {'running': False}})
        self.cleanup()

    def _plot(self, axes_list, data=None):
        '''
        This function plots the data. It is triggered when the updateProgress signal is emited and when after the _function is executed.
        '''
        if data is None:
            data = self.data
        if data is not None and data is not {}:

            if self.settings['graph_params']['plot_raw_counts'] == True:
                # sometimes counts are so low it rounds to zero. Plotting raw counts can be useful
                plot_counts = self.data['raw_counts']
                axes_label = 'counts'
            else:
                plot_counts = self.data['counts']
                axes_label = 'counts/sec'

            axes_list[0].clear()
            axes_list[0].plot(plot_counts)
            axes_list[0].showGrid(x=True, y=True)
            axes_list[0].setLabel('left', axes_label)
            x_ax_length = int(self.settings['graph_params']['length_data']*1.1)
            axes_list[0].setXRange(0, x_ax_length)

            axes_list[1].setText(f'{plot_counts[-1]/1000:.3f} k{axes_label}')

            # todo: Might be useful to include a max count number display

    def get_axes_layout(self, figure_list):
        """
        Overwrites default get_axes_layout. Adds a plot to bottom graph and label that displays a number to top graph.
        Args:
            figure_list: a list of bottom and top PyQtgraphWidget objects
        Returns:
            axes_list: a list of item objects
            axes_list = [<Plot item>,<Label item>]
        """
        axes_list = []
        if self._plot_refresh is True:
            for graph in figure_list:
                graph.clear()
            axes_list.append(figure_list[0].addPlot(row=0,col=0))

            label = pg.LabelItem(text='',size=f'{self.settings["graph_params"]["font_size"]}pt',bold=True)
            figure_list[1].addItem(label, row=0,col=0)
            axes_list.append(label)
        else:
            for graph in figure_list:
                axes_list.append(graph.getItem(row=0,col=0))

        return axes_list

    def _update(self, axes_list):
        Experiment._update(self, axes_list)