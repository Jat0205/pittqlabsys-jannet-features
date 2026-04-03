#!/usr/bin/env python3
"""
ODMR Pulsed Experiment

This experiment implements pulsed ODMR measurements using the Proteus for
sequence generation and ADwin for photon counting. Users can specify
sequences using a text-based language and preview scan results.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.core.experiment import Experiment
from src.core import Parameter
from src.Model.sequence_parser import SequenceTextParser
from src.Model.sequence_builder import SequenceBuilder
from src.Model.proteus_hardware_calibrator import ProteusHardwareCalibrator
from src.Model.awg_file import AWGFile
from src.Model.sequence import Sequence
from src.Model.pulses import Pulse
from src.Controller.Proteus_device import ProteusDevice
from src.Controller.adwin_gold import AdwinGoldDevice
from src.Controller.mux_control import MUXControlDevice
import numpy as np
from src.core.adwin_helpers import get_adwin_binary_path


class ODMRPulsedExperiment(Experiment):
    """
    Pulsed ODMR experiment using Proteus for sequence generation and ADwin for counting.

    Features:
    - Text-based sequence definition using the sequence language
    - Sequence preview with first 10 scan points
    - Microwave frequency, power, and delay parameters
    - Laser power and wavelength configuration
    - Proteus triggers ADwin for photon counting
    - Memory optimization for long sequences
    """

    _DEFAULT_SETTINGS = [
        Parameter('sequence', [
            Parameter('file_path', '', str, 'Path to sequence definition file'),
            Parameter('name', 'odmr_pulsed', str, 'Sequence name'),
            Parameter('sample_rate', 1e9, float, 'Sample rate in Hz', units='Hz'),
            Parameter('repeat_count', 50000, int, 'Number of repetitions per scan point')
        ]),
        Parameter('microwave', [
            Parameter('frequency', 2.87e9, float, 'Microwave frequency in Hz', units='Hz'),
            Parameter('power', -10.0, float, 'Microwave power in dBm', units='dBm'),
            Parameter('delay', 25.0, float, 'Microwave delay in ns', units='ns')
        ]),
        Parameter('laser', [
            Parameter('power', 1.0, float, 'Laser power in mW', units='mW'),
            Parameter('wavelength', 532.0, float, 'Laser wavelength in nm', units='nm')
        ]),
        Parameter('delays', [
            Parameter('mw_delay', 25.0, float, 'Microwave delay in ns', units='ns'),
            Parameter('aom_delay', 50.0, float, 'AOM delay in ns', units='ns'),
            Parameter('counter_delay', 15.0, float, 'Counter delay in ns', units='ns')
        ]),
        Parameter('adwin', [
            Parameter('count_time', 300, float, 'Photon counting time in ns', units='ns'),
            Parameter('reset_time', 1750000, float, 'Reset time between counts in ns', units='ns'),
            Parameter('repetitions_per_point', 50000, int, 'Number of repetitions per scan point')
        ]),
        Parameter('scan', [
            Parameter('preview_points', 10, int, 'Number of scan points to preview'),
            Parameter('auto_generate_files', True, bool, 'Automatically generate AWG files'),
            Parameter('output_directory', 'odmr_pulsed_output', str, 'Output directory for AWG files')
        ]),
        Parameter('optimization', [
            Parameter('enable_compression', True, bool, 'Enable memory compression'),
            Parameter('dead_time_threshold', 100000, int, 'Dead time threshold for compression (samples)'),
            Parameter('high_resolution_threshold', 1000, int, 'High resolution threshold (samples)')
        ])
    ]

    _DEVICES = {
        'proteus': 'proteus',
        'adwin': 'adwin',
        'sg384': 'sg384',  # String reference to device loaded from config
        'mux': 'mux'
    }

    _EXPERIMENTS = {}

    def __init__(self, devices=None, experiments=None, name=None, settings=None, log_function=None, data_path=None,
                 config_path: Optional[Path] = None):
        """Initialize the ODMR Pulsed experiment."""
        super().__init__(name=name, settings=settings, devices=devices, sub_experiments=experiments,
                         log_function=log_function, data_path=data_path)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.repeat_count = None
        self.number_of_iterations = 0
        # Configuration
        self.config_path = config_path or self.get_config_path(
            "D:\\Duttlab\\Experiments\\AQuISS_default_save_location\\experiments_auto_generated\\ODMRPulsedExperiment.json")
        self.config = self._load_config()

        self.adwin = AdwinGoldDevice()
        # self.adwin = self.devices['adwin']['instance']
        # self.proteus = self.devices['proteus']['instance']
        # self.ag384 = self.devices['sg384']['instance']
        self.proteus = ProteusDevice()
        # self.proteus = None
        # self.mux = self.devices['mux']['instance']
        self.mux = MUXControlDevice()
        self.mux.select_trigger('pulsed')

        # Sequence components
        self.sequence_parser = SequenceTextParser()
        self.sequence_builder = SequenceBuilder()

        # Initialize hardware calibrator with experiment-specific connection file
        connection_file = Path(__file__).parent / "odmr_pulsed_connection.json"
        self.hardware_calibrator = ProteusHardwareCalibrator(connection_file=str(connection_file))

        # Experiment parameters (will be set from _DEFAULT_SETTINGS)
        self.microwave_frequency = 2.87e9  # 2.87 GHz (NV center)
        self.microwave_power = -10.0  # dBm
        self.mw_delay = 25.0  # ns
        self.aom_delay = 50.0  # ns
        self.counter_delay = 15.0  # ns

        self.laser_power = 1.0  # mW
        self.laser_wavelength = 532  # nm

        # Sequence data
        self.sequence_description = None
        self.scan_sequences = []
        self.current_scan_point = 0

        # ADwin parameters
        self.count_time = 300  # ns
        self.reset_time = 999400  # ns
        self.sequence_duration = 2500  # for now we have 2500 ns duration

        # Output paths
        self.output_dir = self.get_output_dir("odmr_pulsed_output")

        self.logger.info("ODMR Pulsed Experiment initialized")

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
            print("self.sequence_description.total_duration")
            print(self.sequence_description.total_duration)
            if self.sequence_description:
                self.logger.info(f"Sequence loaded: {self.sequence_description.name}")
                self.logger.info(f"Variables: {len(self.sequence_description.variables)}")
                self.logger.info(f"Pulses: {len(self.sequence_description.pulses)}")
                self.logger.info(f"Markers: {len(self.sequence_description.markers)}")
                return True
            else:
                self.logger.error("Failed to parse sequence text")
                print("Failed to parse sequence text")
                return False

        except Exception as e:
            self.logger.error(f"Error loading sequence: {e}")
            print(f"Error loading sequence: {e}")
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
                self.logger.info(f"Markers: {len(self.sequence_description.markers)}")
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
                    # self.proteus.driver.delete_all_segment()
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
                        if padded_zeros[ch1] > padded_zeros[ch2] and padded_zeros[ch1] > max_delay[i]:
                            max_delay[i] = padded_zeros[ch1]
                        elif padded_zeros[ch2] > padded_zeros[ch1] and padded_zeros[ch2] > max_delay[i]:
                            max_delay[i] = padded_zeros[ch2]

                i += 1

            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                if ch in marker_channels:
                    self.proteus.driver.set_marker(1)
                    self.proteus.driver.set_marker_state('ON')
                if ch in channels:
                    self.proteus.driver.set_output("ON")
                # --self.proteus.driver.set_continuous_run(0)
            self.proteus.driver.set_trigger_level(1.8)  # :TRIG:LEV
            self.proteus.driver.activate_instrument(1)
            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.set_trigger_source('TRG1')
                self.proteus.driver.set_trigger("TRG1")  # :TRIG:SEL
                self.proteus.driver.set_trigger_state("ON")  # :TRIG:STAT

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
                        self.proteus.driver.set_task_type("STAR")  # --
                    elif task_num == self.number_of_iterations:
                        self.proteus.driver.set_task_type("END")  # --
                    else:
                        self.proteus.driver.set_task_type("SEQ")  # --
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
                        self.proteus.driver.set_next1_task(task_num + 1)

                    task_num += 1
                    # self.proteus.driver.set_task_delay(max_delay[task_num-2] - padded_zeros[ch])
                    print(f"max_delay[task_num-2] - padded_zeros[ch] {max_delay[task_num - 2] - padded_zeros[ch]}")

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
            # time.sleep(1)
            # self.proteus.driver._close()
            return True
        except Exception as e:
            self.logger.exception("Error generating AWG sequences")
            return False

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

    def set_laser_parameters(self, power: float, wavelength: float) -> None:
        """
        Set laser parameters.

        Args:
            power: Power in mW
            wavelength: Wavelength in nm
        """
        self.laser_power = power
        self.laser_wavelength = wavelength
        self.logger.info(f"Laser: {power} mW, {wavelength} nm")

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
            'count_time': self.count_time,
            'reset_time': self.reset_time,
            'repetitions_per_point': self.repeat_count,
            'microwave_frequency': self.microwave_frequency,
            'microwave_power': self.microwave_power,
            'laser_power': self.laser_power,
            'laser_wavelength': self.laser_wavelength
        }

    def run_experiment(self, frequency_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Run the complete ODMR pulsed experiment.

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
            self.logger.info("Starting ODMR Pulsed Experiment")

            # Step 1: Load sequence
            if not self.sequence_description:
                self.logger.error("No sequence loaded")
                return {'success': False, 'error': 'No sequence loaded'}

            # Step 2: Build scan sequences
            if not self.build_scan_sequences():
                return {'success': False, 'error': 'Failed to build scan sequences'}

            # Step 3: Generate AWG sequences
            if not self.generate_awg_task_sequences_adwin_triggering_awg_case():
                return {'success': False, 'error': 'Failed to generate AWG sequences'}

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
                    return {'success': False, 'error': f'Failed at frequency {freq / 1e9:.3f} GHz'}

                results['signal_counts'].append(freq_results['signal_counts'])
                results['reference_counts'].append(freq_results['reference_counts'])
                results['total_counts'].append(freq_results['total_counts'])

            self.logger.info("ODMR Pulsed Experiment completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return {'success': False, 'error': str(e)}

    def _setup_adwin_counting(self) -> bool:
        """
        Setup ADwin for photon counting using the odmr_pulsed_counter.bas process.

        Returns:
            True if setup successful
        """
        try:
            # Load the odmr_pulsed_counter.bas process (Process 2)
            # This process handles dual-gate counting triggered by Proteus
            # process_file = "odmr_pulsed_counter.__2"
            process_number = 1

            if not self.adwin.is_connected:
                self.adwin.connect()

            # Proper cleanup like debug script
            self.log("Cleaning up any existing ADwin process...")
            try:
                self.adwin.stop_process(process_number)
                time.sleep(0.1)
            except Exception:
                pass
            try:
                self.adwin.clear_process(process_number)
            except Exception:
                pass
            # the following were tests for option 3
            """# test 1
            odmr_pulsed_counter_path = get_adwin_binary_path('odmr_pulsed_counter.TB2')
            # test 2
            odmr_pulsed_counter_path = get_adwin_binary_path('adbasic_long_waveform_v1.TB2')
            # test 3
            odmr_pulsed_counter_path = get_adwin_binary_path('adbasic_long_waveform_v2.TB2')
            #tested digout
            odmr_pulsed_counter_path = get_adwin_binary_path('test_adwin_digout.TB1')"""
            # option 3 file
            #odmr_pulsed_counter_path = get_adwin_binary_path('adwin_triggering_proteus.TB1')

            odmr_pulsed_counter_path = get_adwin_binary_path('test_adwin_delays.TB1')
            self.adwin.update({'process_1': {'load': str(odmr_pulsed_counter_path)}})

            # Start the counting process
            self.adwin.start_process(process_number)
            """print("here")
            time.sleep(100)
            count = self.adwin.get_int_data(1, 1)
            for ct in count:
                print(f"count: {ct}")"""
            self.logger.info(
                f"ADwin counting setup: count_time={self.count_time}ns, reset_time={self.reset_time}ns, reps={self.repeat_count}")
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
            # Wait for sequence to complete and collect data
            # The ADwin adwin_triggering_proteus.bas process accumulates counts
            # and stores them in arrays Data_1[iteration] (signal) and Data_2[iteration] (reference)

            total_counts = []
            # Collect data
            wait_time = 0
            # Collect data
            for seq in self.scan_sequences:
                wait_time += seq.length + 100
            wait_time = wait_time * self.repeat_count  # remember: wait_time is in ns
            wait_time = wait_time * (10 ** (-9)) + 10  # added 10 secs just in case
            # wait for length of the entire experiment, then get the data
            time.sleep(wait_time)
            """while self.adwin.get_int_var(7) == 0:
                time.sleep(0.01)"""
            print(f"self.adwin.get_int_var(8): {self.adwin.get_int_var(8)}")
            signal_counts = self.adwin.get_int_data(1, self.number_of_iterations)
            ref_counts = self.adwin.get_int_data(2, self.number_of_iterations)
            for i in range(self.number_of_iterations):
                total_counts.append(float(f"{ref_counts[i]}") + float(f"{signal_counts[i]}"))
            for signal_counts_val in signal_counts:
                self.logger.info(f"signal_counts_val {signal_counts_val}")
                print(f"signal_counts_val {signal_counts_val}")
            for reference_counts_val in ref_counts:
                self.logger.info(f"reference_counts_val {reference_counts_val}")
                print(f"reference_counts_val {reference_counts_val}")
            for total_counts_val in total_counts:
                self.logger.info(f"total_counts_val {total_counts_val}")
                print(f"total_counts_val {total_counts_val}")
            # Stop Proteus
            # self.proteus.stop_sequence()
            self.proteus.driver._close()
            print(f"end of _run_sequence_and_collect_data()")

            return {
                'success': True,
                'signal_counts': signal_counts,
                'reference_counts': ref_counts,
                'total_counts': total_counts
            }

        except Exception as e:
            self.logger.error(f"Failed to run sequence and collect data: {e}")
            return {'success': False, 'error': str(e)}

    def create_example_odmr_sequence(self) -> str:
        """
        Create an example ODMR sequence using the sequence language.

        Returns:
            Sequence text in the sequence language format
        """
        sequence_text = """
sequence: name=odmr_pulsed, type=odmr, duration=1002500ns, sample_rate=1GHz, repeat_count=10
variable pulse_duration, start=50ns, stop=500ns, steps=20
marker, laser_int_1 on channel 1 at 0ns, 500ns
pi/2 pulse on channel 1 at 500ns, gaussian, pulse_duration, 1.0
pi/2 pulse on channel 2 at 500ns, gaussian, pulse_duration, 1.0
wait pulse on channel 1 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
wait pulse on channel 2 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
pi/2 pulse on channel 1 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
pi/2 pulse on channel 2 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
marker, laser_readout_1 on channel 1 at 2500ns, 1ms
"""
        return sequence_text.strip()

    def create_example_rabi_sequence(self) -> str:
        """
        Create an example Rabi sequence using the sequence language.

        Returns:
            Sequence text in the sequence language format
        """
        sequence_text = """
sequence: name=rabi_pulsed, type=rabi, duration=1μs, sample_rate=1GHz, repeat=50000

# Define scan variables
variable pulse_duration, start=10ns, stop=200ns, steps=20

# Define the Rabi pulse sequence
# Single microwave pulse with variable duration (will be varied by the scanner)
# Channel 1: IQ modulator I input (microwave pulses)
pi/2 pulse on channel 1 at 0ns, gaussian, 50ns, 1.0

# Channel 2: IQ modulator Q input (for complex microwave pulses)
# For simple Rabi, we can use channel 2 for additional microwave control
# or leave it empty if not needed

# Laser control via Proteus markers:
# ch1_marker1: laser_switch (triggers laser on/off)
# ch2_marker1: counter_trigger (triggers ADwin counting)
"""
        return sequence_text.strip()


# Example usage and testing
if __name__ == "__main__":
    # Create experiment for testing
    experiment = ODMRPulsedExperiment(name="test_odmr")

    # Set parameters
    experiment._setup_adwin_counting()
    """experiment = ODMRPulsedExperiment(name="test_odmr")

    # Set parameters
    experiment.set_microwave_parameters(2.87e9, -10.0, 25.0)
    experiment.set_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(25.0, 50.0, 15.0)
    # Create and load example sequence
    sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text(sequence_text):
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    # Run experiment
    print("Running experiment...")
    # results = experiment.run_experiment(frequency_range=[2.87e9, 2.88e9, 2.89e9])
    results = experiment.run_experiment(frequency_range=[2.87e9])
    if results['success']:
        print("Experiment completed successfully!")
        print(f"Frequencies scanned: {[f / 1e9 for f in results['frequencies']]} GHz")
        print(
            f"Signal counts shape: {len(results['signal_counts'])} frequencies x {len(results['signal_counts'][0])} points")
    else:
        print(f"Experiment failed: {results['error']}")"""