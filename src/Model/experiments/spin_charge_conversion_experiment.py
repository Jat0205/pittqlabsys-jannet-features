'''
Spin charge Conversion experiment

This module implements laser and microwave control for spin charge conversion experiment:
- MCL NanoDrive for sample stage positioning
- ADwin Gold II for photon counting and timing
'''
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from src.Model.sequence_parser import SequenceTextParser
from src.Model.sequence_builder import SequenceBuilder
from src.Model.proteus_hardware_calibrator import ProteusHardwareCalibrator
from src.Controller.Proteus_device import ProteusDevice
from src.Model.sequence import Sequence
from src.core import Parameter, Experiment

class SpinChargeConversionExperiment(Experiment):
    _DEFAULT_SETTINGS = [
        Parameter('sequence', [
            Parameter('file_path', '', str, 'Path to sequence definition file'),
            Parameter('name', 'scc', str, 'Sequence name'),
            Parameter('sample_rate', 1e9, float, 'Sample rate in Hz', units='Hz'),
            Parameter('repeat_count', 50000, int, 'Number of repetitions per scan point')
        ]),
        Parameter('microwave', [
            Parameter('frequency', 2.87e9, float, 'Microwave frequency in Hz', units='Hz'),
            Parameter('power', -10.0, float, 'Microwave power in dBm', units='dBm'),
            Parameter('delay', 25.0, float, 'Microwave delay in ns', units='ns')
        ]),
        Parameter('green_laser', [
            Parameter('power', 1.0, float, 'Laser power in mW', units='mW'),
            Parameter('wavelength', 532.0, float, 'Laser wavelength in nm', units='nm')
        ]),
        Parameter('orange_laser', [
            Parameter('power', 1.0, float, 'Laser power in mW', units='mW'),
            Parameter('wavelength', 594.0, float, 'Laser wavelength in nm', units='nm')
        ]),
        Parameter('delays', [
            Parameter('mw_delay', 25.0, float, 'Microwave delay in ns', units='ns'),
            Parameter('aom_delay', 50.0, float, 'AOM delay in ns', units='ns'),
            Parameter('counter_delay', 15.0, float, 'Counter delay in ns', units='ns')
        ]),
        Parameter('adwin', [
            Parameter('count_time', 300, float, 'Photon counting time in ns', units='ns'),
            Parameter('reset_time', 2000, float, 'Reset time between counts in ns', units='ns'),
            Parameter('repetitions_per_point', 50000, int, 'Number of repetitions per scan point')
        ]),
        Parameter('scan', [
            Parameter('preview_points', 10, int, 'Number of scan points to preview'),
            Parameter('auto_generate_files', True, bool, 'Automatically generate AWG files'),
            Parameter('output_directory', 'scc_output', str, 'Output directory for AWG files')
        ]),
        Parameter('optimization', [
            Parameter('enable_compression', True, bool, 'Enable memory compression'),
            Parameter('dead_time_threshold', 100000, int, 'Dead time threshold for compression (samples)'),
            Parameter('high_resolution_threshold', 1000, int, 'High resolution threshold (samples)')
        ])
    ]

    # For actual experiment use LP100 [MCL_NanoDrive({'serial':2849})]. For testing using HS3 ['serial':2850]
    # _DEVICES = {'nanodrive': MCLNanoDrive(settings={'serial':2849}), 'adwin':AdwinGoldDevice()}  # Removed - devices now passed via constructor
    _DEVICES = {
        'nanodrive': 'nanodrive',
        'adwin': 'adwin',
        'proteus': 'proteus',
        'sg384': 'sg384',
        'coherent_899_dye_laser': 'coherent_899_dye_laser',
        'spex_spectrometer': 'spex_spectrometer'
    }
    _EXPERIMENTS = {}

    def __init__(self, devices=None, experiments=None, name=None, settings=None, log_function=None, data_path=None, config_path: Optional[Path] = None):
        """
        Initializes and connects to devices
        Args:
            name (optional): name of experiment, if empty same as class name
            settings (optional): settings for this experiment, if empty same as default settings
        """
        super().__init__(name, settings=settings, sub_experiments=experiments, devices=devices,
                         log_function=log_function, data_path=data_path)
        # Setup logging
        self.repeat_count = None
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config_path = config_path or self.get_config_path("config.json")
        self.config = self._load_config()

        # Sequence components
        self.sequence_parser = SequenceTextParser()
        self.sequence_builder = SequenceBuilder()

        # Initialize hardware calibrator with experiment-specific connection file
        connection_file = Path(__file__).parent / "spin_charge_conversion_connection.json"
        self.hardware_calibrator = ProteusHardwareCalibrator(connection_file=str(connection_file))
        # Experiment parameters (will be set from _DEFAULT_SETTINGS)
        self.number_of_iterations = 0
        self.microwave_frequency = 2.87e9  # 2.87 GHz (NV center)
        self.microwave_power = -10.0  # dBm
        self.mw_delay = 25.0  # ns
        self.aom_delay = 50.0  # ns
        self.counter_delay = 15.0  # ns

        self.green_laser_power = 1.0  # mW
        self.green_laser_wavelength = 532  # nm

        self.orange_laser_power = 500.0  # mW
        self.orange_laser_wavelength = 594  # nm

        # Sequence data
        self.sequence_description = None
        self.scan_sequences = []
        self.current_scan_point = 0

        # ADwin parameters (from your code)
        self.count_time = 300  # ns
        self.reset_time = 2000  # ns
        self.repetitions_per_point = 50000  # 50K reps for statistics

        # Output paths
        self.output_dir = self.get_output_dir("scc_output")

        self.logger.info("SCC Pulsed Experiment initialized")
        # get instances of devices
        """self.nanodrive = self.devices['nanodrive']['instance']
        self.adwin = self.devices['adwin']['instance']
        self.awg = self.devices['awg520']['instance']
        self.microwave = self.devices['sg384']['instance']
        self.dye_laser = self.devices['coherent_899_dye_laser']['instance']
        self.spectrometer = self.devices['spex_spectrometer']['instance']"""
        self.proteus = ProteusDevice()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}

    def load_sequence_from_text(self, sequence_text: str) -> bool:
        """
        Load sequence definition from text using the sequence language.

        Args:
            sequence_text: Sequence definition in the sequence language format

        Returns:
            True if sequence loaded successfully
        """
        try:
            # Parse sequence text using the sequence language parser
            self.sequence_description = self.sequence_parser.parse_text(sequence_text)
            self.repeat_count = self.sequence_description.repeat_count
            if self.sequence_description:
                self.logger.info(f"Sequence loaded: {self.sequence_description.name}")
                self.logger.info(f"Variables: {len(self.sequence_description.variables)}")
                self.logger.info(f"Pulses: {len(self.sequence_description.pulses)}")
                self.logger.info(f"Markers: {len(self.sequence_description.markers)}")
                return True
            else:
                self.logger.error("Failed to parse sequence text")
                return False

        except Exception as e:
            self.logger.error(f"Error loading sequence: {e}")
            return False

    def load_sequence_from_file(self, sequence_file: Path) -> bool:
        """
        Load sequence definition from text file.

        Args:
            sequence_file: Path to sequence text file

        Returns:
            True if sequence loaded successfully
        """
        try:
            if not sequence_file.exists():
                self.logger.error(f"Sequence file not found: {sequence_file}")
                return False

            # Read sequence text
            with open(sequence_file, 'r') as f:
                sequence_text = f.read()

            # Parse sequence
            self.sequence_description = self.sequence_parser.parse_text(sequence_text)

            if self.sequence_description:
                self.logger.info(f"Sequence loaded: {self.sequence_description.name}")
                self.logger.info(f"Variables: {len(self.sequence_description.variables)}")
                self.logger.info(f"Pulses: {len(self.sequence_description.pulses)}")
                return True
            else:
                self.logger.error("Failed to parse sequence text")
                return False

        except Exception as e:
            self.logger.error(f"Error loading sequence: {e}")
            return False

    def build_scan_sequences(self) -> bool:
        """
        Build scan sequences from the loaded sequence description.

        Returns:
            True if sequences built successfully
        """
        try:
            if not self.sequence_description:
                self.logger.error("No sequence description loaded")
                return False
            # Build scan sequences
            self.scan_sequences = self.sequence_builder.build_scan_sequences(
                self.sequence_description
            )
            if self.sequence_description.variables:
                for seq in self.scan_sequences:
                    self.number_of_iterations += 1
            else:
                self.number_of_iterations = 1

            # Apply hardware calibration
            for i, sequence in enumerate(self.scan_sequences):
                calibrated_sequence = self.hardware_calibrator.calibrate_sequence(
                    sequence,
                    self.sequence_description.sample_rate
                )
                self.scan_sequences[i] = calibrated_sequence
            self.sequence_duration = calibrated_sequence.length
            print(f"jano {self.sequence_duration}")
            self.logger.info(f"Built {len(self.scan_sequences)} scan sequences")
            return True

        except Exception as e:
            self.logger.error(f"Error building scan sequences: {e}")
            return False

    def generate_awg_task_sequences_adwin_triggering_awg_case(self) -> bool:
        """
        Generate Proteus AWG waveforms with maximum timing resolution.
        One continuous waveform per channel.
        one segment and one task entry per iteration per channel.
        64-sample alignment is applied at the end of each waveform.
        The advantage of this option is that adwin does the counts then it tells proteus
        to move to the next sequence. This ensures that counts are measured as expected.
        """

        def prepare_markers_for_tabor(markers_array: np.ndarray) -> np.ndarray:
            """
            Correct Proteus marker packing for 16-bit DAC mode.
            markers_array: 1 marker value per waveform sample (0 or 255)
            """

            # Convert to binary 0/1
            m = (markers_array > 0).astype(np.uint8)

            # Ensure multiple of 4 samples
            if len(m) % 4 != 0:
                m = np.pad(m, (4 - len(m) % 4, 0))

            # One marker byte per 4 waveform samples
            marker_bytes = np.zeros(len(m) // 4, dtype=np.uint8)

            for i in range(len(marker_bytes)):
                block = m[i * 4:(i + 1) * 4]

                if np.any(block):
                    # Marker 1 ON for all 4 samples → 0b00010001
                    marker_bytes[i] = 0x11
                else:
                    marker_bytes[i] = 0x00

            return marker_bytes

        try:
            if not self.scan_sequences:
                self.logger.error("No scan sequences available")
                return False
            # ------------------------------------------------------------
            # DAC configuration
            # ------------------------------------------------------------
            max_dac = 65535
            half_dac = max_dac // 2
            ALIGNMENT = 64  # Proteus requirement (segment length)

            # ------------------------------------------------------------
            # Determine all channels, initialize buffers
            # ------------------------------------------------------------
            all_channels = set()
            channels = set()
            marker_indices = set()
            marker_channels = set()
            for sequence in self.scan_sequences:
                for _, pulse_ in enumerate(sequence.pulses):
                    pulse = pulse_[1]
                    ch = int(pulse.name.split("_")[-1])
                    channels.add(ch)
                    all_channels.add(ch)

                for marker in sequence.markers:
                    mkr_index = int(marker.name.split('_')[-2])
                    marker_indices.add(mkr_index)
                    mk_ch = int(marker.name.split("_")[-1])
                    marker_channels.add(mk_ch)
                    all_channels.add(mk_ch)

            segment_for_channel = {ch: set() for ch in all_channels}
            padded_zeros = {
                ch: 0 for ch in all_channels
            }
            max_delay = {
                it: 0 for it in range(self.number_of_iterations)
            }

            # ------------------------------------------------------------
            # Build continuous waveform per channel (sample-accurate)
            # ------------------------------------------------------------
            SEGNUM = 1
            for sequence in self.scan_sequences:
                seq_waveforms = sequence.to_waveform()
                for ch, data in seq_waveforms.items():
                    self.proteus.driver.set_channel(ch)
                    #self.proteus.driver.delete_all_segment()
                    analog_voltage = 1
                    self.proteus.driver.set_voltage(analog_voltage)
                    sampleRateDAC = 1E9
                    self.proteus.driver.apply_sampling_configuration(sampleRateDAC)
                    # so far envelope is a sum of pulses in a single iteration
                    # but we need to add another for loop inside items to get pulses and their timing
                    envelope = data["envelope"]
                    markers = data["markers"]
                    """import matplotlib.pyplot as plt
                    import numpy as np

                    samples = np.arange(len(envelope))

                    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

                    # Envelope plot
                    ax[0].plot(samples, envelope, color="blue")
                    ax[0].set_ylabel("Envelope")
                    ax[0].set_title("Envelope vs Samples")
                    ax[0].grid(True)

                    # Markers plot
                    ax[1].step(samples, markers, where="post", color="red")
                    ax[1].set_ylabel("Markers")
                    ax[1].set_xlabel("Sample index")
                    ax[1].set_title("Markers vs Samples")
                    ax[1].grid(True)

                    plt.tight_layout()
                    plt.show()"""

                    segment_num = SEGNUM
                    segment_for_channel[ch].add(segment_num)
                    rem = len(envelope) % ALIGNMENT
                    padded_zeros[ch] = rem
                    if rem:
                        envelope = np.pad(envelope, (ALIGNMENT - rem, 0))
                        markers = np.pad(markers, (ALIGNMENT - rem, 0))
                        padded_zeros[ch] = ALIGNMENT - rem
                    # Scale to DAC
                    dac_wave = np.clip(envelope, -1.0, 1.0)
                    dac_wave = ((dac_wave + 1.0) * half_dac).astype(np.uint16)
                    self.proteus.driver.define_trace(segment_num, len(dac_wave))
                    self.proteus.driver.select_segment(segment_num)

                    # Upload waveform
                    self.proteus.driver.write_trace_data(dac_wave)
                    resp = self.proteus.driver.query_error()
                    print(f"loading trace data {resp}")

                    # Convert to uint8
                    markers = markers.astype(np.uint8)

                    # Now prepare for Tabor
                    tabor_markers = prepare_markers_for_tabor(markers)

                    self.proteus.driver.write_marker_data(tabor_markers)
                    resp = self.proteus.driver.query_error()
                    print(f'Marker upload result: {resp}')
                    # proteus P1284M only has 1 marker per channel, for other awgs, please implement code that handles that
                    # I have already coded the marker index here: mkr_index = int(marker.name.split('_')[-2]) so, for example, if the user
                    # gives the following command: "marker, laser_int_1 on channel 4 at 0ns, 500ns" then you know whatever is after "laser_init_" is the marker index
                    marker_index = 1
                    self.proteus.driver.set_marker(marker_index)
                    self.proteus.driver.set_marker_ptop_voltage(0.5)
                    self.proteus.driver.set_marker_voltage_offset(0.347)
                    # Check for errors
                    resp = self.proteus.driver.query_error()
                    print(f'Marker setup result: {resp}')
                    SEGNUM += 1
            i = 0
            for sequence in self.scan_sequences:
                seq_waveforms = sequence.to_waveform()
                for ch1, data1 in seq_waveforms.items():
                    for ch2, data2 in seq_waveforms.items():
                        if padded_zeros[ch1] > padded_zeros[ch2] and padded_zeros[ch1]>max_delay[i]:
                            max_delay[i]=padded_zeros[ch1]
                        elif padded_zeros[ch2] > padded_zeros[ch1] and padded_zeros[ch2]>max_delay[i]:
                            max_delay[i]=padded_zeros[ch2]

                i += 1

            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                if ch in marker_channels:
                    self.proteus.driver.set_marker(1)
                    self.proteus.driver.set_marker_state('ON')
                if ch in channels:
                    self.proteus.driver.set_output("ON")
                #--self.proteus.driver.set_continuous_run(0)
            self.proteus.driver.set_trigger_level(2)
            self.proteus.driver.activate_instrument(1)
            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.set_trigger_source('TRG1')
                self.proteus.driver.set_trigger("TRG1") # :TRIG:SEL
                self.proteus.driver.set_trigger_state("ON") # :TRIG:STAT

            for ch in sorted(all_channels):
                task_num = 1
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.set_task_table_length(self.number_of_iterations)
                print(f"channel {ch}: {ch}")
                print(f"number of iterations: {self.number_of_iterations}")
                for segment_number in segment_for_channel[ch]:
                    print(f"channel: {ch} Segment: {segment_number}")
                    self.proteus.driver.set_task_number(task_num)
                    if task_num == 1:
                        self.proteus.driver.set_task_type("STAR") #--
                    elif task_num == self.number_of_iterations:
                        self.proteus.driver.set_task_type("END") #--
                    else:
                        self.proteus.driver.set_task_type("SEQ") #--
                    self.proteus.driver.set_task_segment_number(segment_number)
                    self.proteus.driver.set_trigger_IDLE_state("DC")
                    self.proteus.driver.set_trigger_IDLE_level(0)
                    if task_num == 1:
                        self.proteus.driver.set_enabling_task_signal("TRG1")
                    else:
                        self.proteus.driver.set_enabling_task_signal("TRG1")

                    if task_num == self.number_of_iterations:
                        self.proteus.driver.set_next1_task(1)
                    else:
                        self.proteus.driver.set_next1_task(task_num+1)

                    task_num+=1
                    #self.proteus.driver.set_task_delay(max_delay[task_num-2] - padded_zeros[ch])
                    print(f"max_delay[task_num-2] - padded_zeros[ch] {max_delay[task_num-2]-padded_zeros[ch]}")

                    self.proteus.driver.write_composer_array_to_task_table()
                    self.proteus.driver.set_waveform_type('TASK')
                ###self.proteus.driver.set_task_loop(self.repeat_count)
            self.proteus.driver.set_task_sync()
            resp = self.proteus.driver.set_trigger_coupling('ON')
            print(resp)
            # for trigger source INT send:
            self.proteus.driver.set_channel(1)
            # trigger command:
            self.proteus.driver.trigger()
            #time.sleep(1)
            #self.proteus.driver._close()
            return True
        except Exception as e:
            self.logger.exception("Error generating AWG sequences")
            return False

    def generate_awg_sequences_awg_triggering_adwin_case(self) -> bool:
        """
        Generate Proteus AWG waveforms with maximum timing resolution.
        One continuous waveform per channel.
        one segment and one task entry per channel.
        64-sample alignment is applied at the end of each waveform.
        The disadvantage of this option is that proteus does not know if the adwin is ready to count and whether it finished counting
        """

        def prepare_markers_for_tabor(markers_array: np.ndarray) -> np.ndarray:
            """
            Correct Proteus marker packing for 16-bit DAC mode.
            markers_array: 1 marker value per waveform sample (0 or 255)
            """

            # Convert to binary 0/1
            m = (markers_array > 0).astype(np.uint8)

            # Ensure multiple of 4 samples
            if len(m) % 4 != 0:
                m = np.pad(m, (0, 4 - len(m) % 4))

            # One marker byte per 4 waveform samples
            marker_bytes = np.zeros(len(m) // 4, dtype=np.uint8)

            for i in range(len(marker_bytes)):
                block = m[i * 4:(i + 1) * 4]

                if np.any(block):
                    # Marker 1 ON for all 4 samples → 0b00010001
                    marker_bytes[i] = 0x11
                else:
                    marker_bytes[i] = 0x00

            return marker_bytes

        try:
            if not self.scan_sequences:
                self.logger.error("No scan sequences available")
                return False
            # ------------------------------------------------------------
            # DAC configuration
            # ------------------------------------------------------------
            max_dac = 65535
            half_dac = max_dac // 2
            ALIGNMENT = 64  # Proteus requirement (segment length)

            # ------------------------------------------------------------
            # Determine all channels, initialize buffers
            # ------------------------------------------------------------
            all_channels = set()
            channels = set()
            marker_indices = set()
            marker_channels = set()
            for sequence in self.scan_sequences:
                # ++++++++++++++++++++++
                print(f"to waveform output: {sequence.to_waveform()}")
                # ++++++++++++++++++++++
                for _, pulse_ in enumerate(sequence.pulses):
                    pulse = pulse_[1]
                    ch = int(pulse.name.split("_")[-1])
                    channels.add(ch)
                    all_channels.add(ch)

                for marker in sequence.markers:
                    mkr_index = int(marker.name.split('_')[-2])
                    marker_indices.add(mkr_index)
                    mk_ch = int(marker.name.split("_")[-1])
                    marker_channels.add(mk_ch)
                    all_channels.add(mk_ch)

            channel_waveforms = {
                ch: [] for ch in all_channels
            }
            channel_markers = {
                ch: [] for ch in all_channels
            }
            for ch in all_channels:
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.delete_all_segment()
                analog_voltage = 1
                self.proteus.driver.set_voltage(analog_voltage)
                sampleRateDAC = 1E9
                self.proteus.driver.apply_sampling_configuration(sampleRateDAC)

            # ------------------------------------------------------------
            # Build continuous waveform per channel (sample-accurate)
            # ------------------------------------------------------------
            for sequence in self.scan_sequences:
                seq_waveforms = sequence.to_waveform()
                for ch, data in seq_waveforms.items():
                    envelope = data["envelope"]
                    markers = data["markers"]
                    if not np.all(markers == 0):
                        print(f"channel {ch}: Found non-zero markers")
                        print(f"unique values {np.unique(markers)}")

                    channel_waveforms[ch].append(envelope)
                    channel_markers[ch].append(markers)
            # ------------------------------------------------------------
            # Upload one segment per channel
            # ------------------------------------------------------------
            segment_index = 1
            segment_for_channel = {}

            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)

                waveform = np.concatenate(channel_waveforms[ch])
                print(f"waveform: {waveform}")

                # Apply required alignment
                rem = len(waveform) % ALIGNMENT
                print(f"rem: {rem}")
                if rem:
                    waveform = np.pad(waveform, (0, ALIGNMENT - rem))
                    print(f"waveform: {waveform}")

                # Scale to DAC
                dac_wave = np.clip(waveform, -1.0, 1.0)
                dac_wave = ((dac_wave + 1.0) * half_dac).astype(np.uint16)

                # Define and select segment
                self.proteus.driver.define_trace(segment_index, len(dac_wave))
                self.proteus.driver.select_segment(segment_index)

                # Upload waveform
                self.proteus.driver.write_trace_data(dac_wave)  # write, and wait while *OPC completes
                resp = self.proteus.driver.query_error()
                print('analog upload result: "{0}" after writing trace binary values'.format(resp))

                segment_for_channel[ch] = segment_index
                # Handle markers for this channel
                if ch in channel_markers:
                    markers = np.concatenate(channel_markers[ch])
                    print(f"\n=== MARKER VALUE ANALYSIS ===")
                    print(f"Markers dtype: {markers.dtype}")
                    print(f"Min value: {np.min(markers)}")
                    print(f"Max value: {np.max(markers)}")
                    print(f"Unique values: {np.unique(markers)}")

                    rem = len(waveform) % ALIGNMENT
                    if rem:
                        waveform_padded_len = len(waveform) + (ALIGNMENT - rem)
                    else:
                        waveform_padded_len = len(waveform)

                    print(f"Waveform after padding: {waveform_padded_len} samples")
                    print(f"Markers before padding: {len(markers)} samples")

                    # Now pad/truncate markers to match waveform length
                    if len(markers) < waveform_padded_len:
                        # Pad markers with zeros
                        markers = np.pad(markers, (0, waveform_padded_len - len(markers)), mode='constant')
                        print(f"Markers padded to match waveform: {len(markers)} samples")
                    elif len(markers) > waveform_padded_len:
                        # Truncate markers
                        markers = markers[:waveform_padded_len]
                        print(f"Markers truncated to match waveform: {len(markers)} samples")
                    else:
                        print(f"Markers already match waveform length")

                    if not np.all(markers == 0):

                        # Convert to uint8
                        markers = markers.astype(np.uint8)

                        # Now prepare for Tabor
                        tabor_markers = prepare_markers_for_tabor(markers)

                        # For 16-bit mode, marker bytes should be waveform_bytes / 4
                        expected_marker_bytes = len(dac_wave) // 4
                        actual_marker_bytes = len(tabor_markers)

                        if actual_marker_bytes != expected_marker_bytes:
                            if actual_marker_bytes < expected_marker_bytes:
                                tabor_markers = np.pad(tabor_markers, (0, expected_marker_bytes - actual_marker_bytes),
                                                       mode='constant')
                            else:
                                tabor_markers = tabor_markers[:expected_marker_bytes]

                        self.proteus.driver.write_marker_data(tabor_markers)
                        resp = self.proteus.driver.query_error()
                        print(f'Marker upload result: {resp}')
                        # proteus P1284M only has 1 marker per channel, for other awgs, please implement code that handels that
                        # I have already coded the marker index here: mkr_index = int(marker.name.split('_')[-2]) so if the user
                        # gives marker, laser_int_1 on channel 4 at 0ns, 500ns then you know whatever is after laser_init_ is the index
                        marker_index = 1
                        self.proteus.driver.set_marker(marker_index)
                        # Verify Proteus marker settings
                        resp = self.proteus.driver.get_marker()
                        print(f"Selected marker: {resp}")
                        resp = self.proteus.driver.get_marker_state()
                        print(f"Marker state: {resp}")
                        resp = self.proteus.driver.query_error()
                        print(f"System error: {resp}")

                        resp = self.proteus.driver.get_marker_ptop_voltage()
                        print(f"Marker voltage PTOP setting: {resp}")

                        resp = self.proteus.driver.get_marker_voltage()
                        print(f"Marker voltage LEV setting: {resp}")
                        self.proteus.driver.set_marker_ptop_voltage(1)
                        resp = self.proteus.driver.get_marker_ptop_voltage()
                        print(f"Marker voltage PTOP setting: {resp}")
                        resp = self.proteus.driver.get_marker_voltage_offset()
                        print(f"Marker voltage offset setting: {resp}")
                        self.proteus.driver.set_marker_voltage_offset(0.347)
                        resp = self.proteus.driver.get_marker_voltage_offset()
                        print(f"Marker voltage offset setting: {resp}")

                        # Check for errors
                        resp = self.proteus.driver.query_error()
                        print(f'Marker upload result: {resp}')
                    else:
                        print(f"Channel {ch}: Markers are all zeros, skipping marker upload")

                segment_index += 1

            # ------------------------------------------------------------
            # Build ONE task entry per channel
            # ------------------------------------------------------------

            # Task must be set per channel
            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.set_continuous_run(0)
                task_table_length = 1
                self.proteus.driver.set_task_table_length(task_table_length)
                self.proteus.driver.set_task_number(1)
                self.proteus.driver.set_task_type("SING")
                # self.proteus.driver.set_next1_task(0)
                self.proteus.driver.set_task_segment_number(segment_for_channel[ch])
                self.proteus.driver.set_task_loop(self.repeat_count)
                self.proteus.driver.write_composer_array_to_task_table()

                # ------------------------------------------------------------
                # Enable TASK mode and output
                # ------------------------------------------------------------
            self.proteus.driver.set_task_sync()
            self.proteus.start_sequence()
            time.sleep(1)
            self.proteus.driver._close()
            return True
        except Exception as e:
            self.logger.exception("Error generating AWG files")
            return False

    def show_sequence_preview(self, num_points: int = 10) -> None:
        """
        Show sequence preview window with first N scan points.

        Args:
            num_points: Number of scan points to preview
        """
        if not self.scan_sequences:
            messagebox.showerror("Error", "No scan sequences available. Build sequences first.")
            return

        # Limit to available sequences
        preview_sequences = self.scan_sequences[:min(num_points, len(self.scan_sequences))]

        # Create preview window
        preview_window = SequencePreviewWindow(preview_sequences, self.sequence_description)
        preview_window.show()

    def set_microwave_parameters(self, frequency: float, power: float, delay: float) -> None:
        """
        Set microwave parameters.

        Args:
            frequency: Frequency in Hz
            power: Power in dBm
            delay: Delay in ns
        """
        self.microwave_frequency = frequency
        self.microwave_power = power
        self.mw_delay = delay
        self.logger.info(f"Microwave: {frequency / 1e9:.3f} GHz, {power} dBm, {delay} ns delay")

    def set_green_laser_parameters(self, power: float, wavelength: float) -> None:
        """
        Set laser parameters.

        Args:
            power: Power in mW
            wavelength: Wavelength in nm
        """
        self.green_laser_power = power
        self.green_laser_wavelength = wavelength
        self.logger.info(f"green Laser: {power} mW, {wavelength} nm")

    def set_orange_laser_parameters(self, power: float, wavelength: float) -> None:
        """
        Set laser parameters.

        Args:
            power: Power in mW
            wavelength: Wavelength in nm
        """
        self.orange_laser_power = power
        self.orange_laser_wavelength = wavelength
        self.logger.info(f"orange Laser: {power} mW, {wavelength} nm")

    def set_delay_parameters(self, mw_delay: float, aom_delay: float, counter_delay: float) -> None:
        """
        Set delay parameters.

        Args:
            mw_delay: Microwave delay in ns
            aom_delay: AOM delay in ns
            counter_delay: Counter delay in ns
        """
        self.mw_delay = mw_delay
        self.aom_delay = aom_delay
        self.counter_delay = counter_delay
        self.logger.info(f"Delays: MW={mw_delay}ns, AOM={aom_delay}ns, Counter={counter_delay}ns")

    def get_adwin_parameters(self) -> Dict[str, Any]:
        """
        Get ADwin parameters for the experiment.

        Returns:
            Dictionary of ADwin parameters
        """
        return {
            '                      ': self.count_time,
            'reset_time': self.reset_time,
            'repetitions_per_point': self.repetitions_per_point,
            'microwave_frequency': self.microwave_frequency,
            'microwave_power': self.microwave_power,
            'green_laser_power': self.green_laser_power,
            'green_laser_wavelength': self.green_laser_wavelength,
            'orange_laser_power': self.orange_laser_power,
            'orange_laser_wavelength': self.orange_laser_wavelength
        }

    def run_experiment(self, frequency_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Run the complete SCC pulsed experiment.

        Args:
            frequency_range: List of frequencies in Hz to scan. If None, uses single frequency.

        Returns:
            Dictionary containing:
            - 'success': bool - Whether experiment completed successfully
            - 'frequencies': List[float] - Frequencies scanned
            - 'signal_counts': List[List[float]] - Signal counts for each frequency and sequence point
            - 'reference_counts': List[List[float]] - Reference counts for each frequency and sequence point
            - 'total_counts': List[List[float]] - Total counts for each frequency and sequence point
        """
        try:
            self.logger.info("Starting SCC Pulsed Experiment")

            # Step 1: Load sequence
            if not self.sequence_description:
                self.logger.error("No sequence loaded")
                return {'success': False, 'error': 'No sequence loaded'}

            # Step 2: Build scan sequences
            if not self.build_scan_sequences():
                return {'success': False, 'error': 'Failed to build scan sequences'}

            # Step 3: Generate AWG files
            if not self.generate_awg_files():
                return {'success': False, 'error': 'Failed to generate AWG files'}

            # Step 4: Setup ADwin for photon counting
            if not self._setup_adwin_counting():
                return {'success': False, 'error': 'Failed to setup ADwin counting'}

            # Step 5: Determine frequency range
            if frequency_range is None:
                frequency_range = [self.microwave_frequency]

            # Step 6: Run experiment for each frequency
            results = {
                'success': True,
                'frequencies': frequency_range,
                'signal_counts': [],
                'reference_counts': [],
                'total_counts': []
            }

            for freq in frequency_range:
                self.logger.info(f"Running experiment at {freq / 1e9:.3f} GHz")

                # Set SG384 frequency
                if 'sg384' in self.devices:
                    self.devices['sg384'].set_frequency(freq)
                    self.devices['sg384'].set_power(self.microwave_power)
                    self.logger.info(f"Set SG384 to {freq / 1e9:.3f} GHz, {self.microwave_power} dBm")

                # Run sequence and collect data
                freq_results = self._run_sequence_and_collect_data()
                if not freq_results['success']:
                    return {'success': False,
                            'error': f'Failed at frequency {freq / 1e9:.3f} GHz: {freq_results["error"]}'}

                results['signal_counts'].append(freq_results['signal_counts'])
                results['reference_counts'].append(freq_results['reference_counts'])
                results['total_counts'].append(freq_results['total_counts'])

            self.logger.info("SCC Pulsed Experiment completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return {'success': False, 'error': str(e)}

    def _setup_adwin_counting(self) -> bool:
        """
        Setup ADwin for photon counting using the measure_protocol.bas process.

        Returns:
            True if setup successful
        """
        try:
            if 'adwin' not in self.devices:
                self.logger.error("ADwin device not available")
                return False

            adwin = self.devices['adwin']

            # Load the measure_protocol.bas process (Process 2)
            # This process handles dual-gate counting triggered by AWG520
            process_file = "measure_protocol.__2"
            adwin.load_process(process_file)

            # Set ADwin parameters for counting
            # Par_3: count_time (with calibration offset)
            # Par_4: reset_time (with calibration offset)
            # Par_5: repetitions_per_point
            count_time_calibrated = self.count_time + 10  # Add calibration offset
            reset_time_calibrated = self.reset_time + 30  # Add calibration offset

            adwin.set_parameter(3, count_time_calibrated)
            adwin.set_parameter(4, reset_time_calibrated)
            adwin.set_parameter(5, self.repetitions_per_point)

            # Start the counting process
            adwin.start_process(process_file)

            self.logger.info(
                f"ADwin counting setup: count_time={self.count_time}ns, reset_time={self.reset_time}ns, reps={self.repetitions_per_point}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup ADwin counting: {e}")
            return False

    def _run_sequence_and_collect_data(self) -> Dict[str, Any]:
        """
        Run a single sequence and collect photon counting data from ADwin.

        Returns:
            Dictionary with signal_counts, reference_counts, total_counts
        """
        try:
            if 'awg520' not in self.devices:
                return {'success': False, 'error': 'AWG520 device not available'}

            if 'adwin' not in self.devices:
                return {'success': False, 'error': 'ADwin device not available'}

            awg = self.devices['awg520']
            adwin = self.devices['adwin']

            # Start AWG520 sequence (this will trigger ADwin via markers)
            # The AWG520 should be configured to use external triggering
            # and the sequence should include proper marker outputs for ADwin triggering
            awg.start_sequence()

            # Wait for sequence to complete and collect data
            # The ADwin measure_protocol.bas process accumulates counts
            # and stores them in Par_1 (signal) and Par_2 (reference) after each scan point

            signal_counts = []
            reference_counts = []
            total_counts = []

            # Collect data for each sequence point
            for i in range(len(self.scan_sequences)):
                # Wait for ADwin to complete counting for this sequence point
                # The measure_protocol.bas process will increment Par_10 when done
                max_wait_time = 10.0  # seconds
                wait_time = 0.0
                while wait_time < max_wait_time:
                    scan_point = adwin.get_parameter(10)  # Current scan point
                    if scan_point > i:
                        break
                    time.sleep(0.1)
                    wait_time += 0.1

                if wait_time >= max_wait_time:
                    self.logger.warning(f"Timeout waiting for scan point {i}")

                # Read accumulated counts from ADwin
                signal_count = adwin.get_parameter(1)  # Par_1: signal counts
                reference_count = adwin.get_parameter(2)  # Par_2: reference counts
                total_count = signal_count + reference_count

                signal_counts.append(signal_count)
                reference_counts.append(reference_count)
                total_counts.append(total_count)

                self.logger.info(
                    f"Scan point {i}: signal={signal_count}, reference={reference_count}, total={total_count}")

            # Stop AWG520
            awg.stop_sequence()

            return {
                'success': True,
                'signal_counts': signal_counts,
                'reference_counts': reference_counts,
                'total_counts': total_counts
            }

        except Exception as e:
            self.logger.error(f"Failed to run sequence and collect data: {e}")
            return {'success': False, 'error': str(e)}

    def create_example_scc_sequence(self) -> str:
        """
        Create an example SCC sequence using the sequence language.

        Returns:
            Sequence text in the sequence language format
        """
        sequence_text = """
sequence: name=SCC, type=SCC, duration=1400ns, sample_rate=1GHz, repeat=50000
shelving pulse on channel 3 at 0ns, square, 300ns, 0.6
ionization pulse on channel 3 at 300ns, square, 500ns, 1.0
readout pulse on channel 3 at 800ns, square, 600ns, 0.3
# Define scan variables (single variable for simplicity)
#variable pulse_duration, start=50ns, stop=500ns, steps=20

# Define the SCC pulse sequence
#marker, laser_int_1 on channel 1 at 0ns, 500ns
#pi/2 pulse on channel 1 at 500ns, gaussian, pulse_duration, 1.0
#pi/2 pulse on channel 2 at 500ns, gaussian, pulse_duration, 1.0
#wait pulse on channel 1 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
#wait pulse on channel 2 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
#pi/2 pulse on channel 1 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
#pi/2 pulse on channel 2 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
#shelving pulse on channel 3 at 4*pulse_duration+0.000000500, square, pulse_duration, 0.6
#ionization pulse on channel 3 at 5*pulse_duration+0.000000500, square, pulse_duration, 1.0
#readout pulse on channel 3 at 6*pulse_duration+0.000000500, square, 10*pulse_duration, 0.3
#marker, laser_readout_1 on channel 2 at pulse_duration+0.000000500,300ns
"""
        return sequence_text.strip()

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of experiment configuration.

        Returns:
            Dictionary with experiment summary
        """
        return {
            'name': 'SCC Experiment',
            'sequence_name': self.sequence_description.name if self.sequence_description else 'None',
            'scan_points': len(self.scan_sequences),
            'microwave_frequency_ghz': self.microwave_frequency / 1e9,
            'microwave_power_dbm': self.microwave_power,
            'green_laser_power_mw': self.green_laser_power,
            'green_laser_wavelength_nm': self.green_laser_wavelength,
            'orange_laser_power_mw': self.orange_laser_power,
            'orange_laser_wavelength_nm': self.orange_laser_wavelength,
            'delays_ns': {
                'mw': self.mw_delay,
                'aom': self.aom_delay,
                'counter': self.counter_delay
            },
            'adwin_parameters': self.get_adwin_parameters(),
            'output_directory': str(self.output_dir)
        }

class SequencePreviewWindow:
    """Window for previewing sequence scan points."""

    def __init__(self, sequences: List[Sequence], description):
        """Initialize preview window."""
        self.sequences = sequences
        self.description = description
        self.window = None
        self.sequence_builder = SequenceBuilder()
        self.anim = None

    def show(self):
        """Show the preview window."""
        # Create main window
        self.window = tk.Tk()
        self.window.title("SCC Sequence Preview")
        self.window.geometry("800x600")

        # Create notebook for different views
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab 1: Sequence overview
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        self._create_overview_tab(overview_frame)

        # Tab 2: Sequence plots
        plots_frame = ttk.Frame(notebook)
        notebook.add(plots_frame, text="Plots")
        self._create_plots_tab(plots_frame)

        # Tab 3: Parameters
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")
        self._create_parameters_tab(params_frame)

        # Show window
        self.window.mainloop()

    def _create_overview_tab(self, parent):
        """Create overview tab."""
        # Sequence info
        info_frame = ttk.LabelFrame(parent, text="Sequence Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(info_frame, text=f"Name: {self.description.name}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Total scan points: {len(self.sequences)}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Variables: {len(self.description.variables)}").pack(anchor='w')
        ttk.Label(info_frame, text=f"Pulses per sequence: {len(self.description.pulses)}").pack(anchor='w')

        # Variables info
        if self.description.variables:
            var_frame = ttk.LabelFrame(parent, text="Scan Variables", padding=10)
            var_frame.pack(fill='x', padx=10, pady=5)

            for name, var in self.description.variables.items():
                var_text = f"{var.name}: {var.start_value} to {var.stop_value} ({var.steps} steps)"
                ttk.Label(var_frame, text=var_text).pack(anchor='w')

    def _create_plots_tab(self, parent):
        """Create plots tab."""
        # Create matplotlib figure
        """fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.draw()

        # Embed in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Plot first few sequences
        if self.sequences:
            self.sequence_builder.animate_scan_sequences(
                self.sequences[:min(5, len(self.sequences))],
                title="Sequence Preview (First 5 Points)"
            )"""
        fig, ax = plt.subplots(figsize=(8, 6))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # IMPORTANT: store animation on self
        self.anim = self.sequence_builder.animate_scan_sequences(
            self.sequences[:min(5, len(self.sequences))],
            fig=fig,
            ax=ax,
            title="Sequence Preview (First 5 Points)"
        )

        canvas.draw()

    def _create_parameters_tab(self, parent):
        """Create parameters tab."""
        # Parameters info
        params_frame = ttk.LabelFrame(parent, text="Experiment Parameters", padding=10)
        params_frame.pack(fill='x', padx=10, pady=5)

        # This would show the current experiment parameters
        ttk.Label(params_frame, text="Microwave frequency: 2.87 GHz").pack(anchor='w')
        ttk.Label(params_frame, text="Microwave power: -10 dBm").pack(anchor='w')
        ttk.Label(params_frame, text="Laser power: 1.0 mW").pack(anchor='w')
        ttk.Label(params_frame, text="Laser wavelength: 532 nm").pack(anchor='w')
        ttk.Label(params_frame, text="MW delay: 25 ns").pack(anchor='w')
        ttk.Label(params_frame, text="AOM delay: 50 ns").pack(anchor='w')
        ttk.Label(params_frame, text="Counter delay: 15 ns").pack(anchor='w')

    def _function(self):
        # Excitation: green laser

        # Microwave

        # ionization

        # readout

        # Plot Avg counts / ionization pulse width time

        # Plot Average signal photons / ionization pulse width

        # Plot NV- population % SNR / ionization pulse width

        # Plot PL contrast / Microwave frequency

        # Plot Average signal photons / readout pulse width

        # Plot PL contrast / SCC readout time

        # Plot SNR / readout pulse width

        # Plot sensitivity / SCC readout time

        pass

    def _plot(self, axes_list, data=None):
        pass

    def _update(self, axes_list):
        pass


# Example usage and testing
if __name__ == "__main__":
    # Create experiment with mock devices for testing
    experiment = SpinChargeConversionExperiment(name="test_scc")

    # Set parameters
    """experiment.set_microwave_parameters(2.87e9, -10.0, 25.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(25.0, 50.0, 15.0)"""

    # Create and load example sequence
    sequence_text = experiment.create_example_scc_sequence()
    print("Example SCC Sequence:")
    print(sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text(sequence_text):
        print("Sequence loaded successfully")

        # Build sequences
        if experiment.build_scan_sequences():
            print("Scan sequences built successfully")

            # Generate AWG files
            if experiment.generate_awg_sequences_awg_triggering_adwin_case():
                print("AWG files generated successfully")

            """# Run experiment (with mock devices)
            print("Running experiment with mock devices...")
            results = experiment.run_experiment(frequency_range=[2.87e9, 2.88e9, 2.89e9])

            if results['success']:
                print("Experiment completed successfully!")
                print(f"Frequencies scanned: {[f/1e9 for f in results['frequencies']]} GHz")
                print(f"Signal counts shape: {len(results['signal_counts'])} frequencies x {len(results['signal_counts'][0])} points")
            else:
                print(f"Experiment failed: {results['error']}")"""

            """else:
                print("Failed to generate AWG files")"""
        else:
            print("Failed to build scan sequences")
    else:
        print("Failed to load sequence")
    experiment.show_sequence_preview(10)
    print("\nSCC Pulsed Experiment ready!")