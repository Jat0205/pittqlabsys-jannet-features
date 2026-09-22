#!/usr/bin/env python3
# Written by <Gurudev Dutt> and <Jannet Trabelsi>
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
import os
import logging
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import datetime
from src.Controller import SG384Generator, MCLNanoDrive
from src.core.experiment import Experiment
from src.core import Parameter
from src.Model.sequence_parser import SequenceTextParser
from src.Model.sequence_builder import SequenceBuilder
from src.Model.proteus_hardware_calibrator import ProteusHardwareCalibrator
from src.Model.sequence import Sequence
from src.core.struct_hdf5 import MyStruct
from src.Controller.Proteus_device import ProteusDevice
from src.Controller.adwin_gold import AdwinGoldDevice
from src.Controller.mux_control import MUXControlDevice
import numpy as np
from src.core.adwin_helpers import get_adwin_binary_path

ADWIN_TICKS_PER_NS = 0.3  # 300 MHz assumption; 1 tick = 3.333 ns


def readout_length_ns(count_time_ns, reset_time_ns):
    """Your readout convention: X = Y + reset + Y (signal window, reset, reference window)."""
    return 2 * count_time_ns + reset_time_ns


class ODMRPulsedExperiment(Experiment):
    """
    Pulsed ODMR experiment using Proteus for sequence generation and ADwin for counting.

    Features:
    - Text-based sequence definition using the sequence language
    - Sequence preview with first 10 scan points
    - Microwave frequency and power
    - Laser power and wavelength configuration
    - Proteus triggers ADwin for photon counting
    """
    _DEFAULT_SETTINGS = [
        Parameter('sequence', [
            Parameter('file_path', "D:\Data\jannet_trabelsi\March2026\odmr_sequence.txt", str,
                      'Path to sequence definition file'),
            Parameter('load_from_file', False, bool, 'load the sequence from a file'),
            Parameter('text',
                      "sequence: name=odmr_pulsed, type=odmr, duration=1002500ns, sample_rate=1GHz, repeat_count=50000\nvariable pulse_duration, start=50ns, stop=500ns, steps=20\nmarker, laser_int_1 on channel 1 at 0ns, 500ns\npi/2 pulse on channel 1 at 500ns, gaussian, pulse_duration, 1.0\npi/2 pulse on channel 2 at 500ns, gaussian, pulse_duration, 1.0 \nwait pulse on channel 1 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0\nwait pulse on channel 2 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0\npi/2 pulse on channel 1 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0\npi/2 pulse on channel 2 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0\nmarker, laser_readout_1 on channel 1 at 2500ns, 1ms\nmarker, readout_counts_1 on channel 2 at 2500ns, 300ns\nmarker, reference_counts_1 on channel 2 at 1002200ns, 300ns",
                      str, 'sequence_text')
        ]),
        Parameter('microwave', [
            Parameter('enable', True, bool, 'enable MICROWAVE'),
            Parameter('frequency range', [2.73632e9], list, 'Microwave frequency in Hz', units='Hz'),
            Parameter('power', 12, float, 'Microwave power in dBm', units='dBm')
        ]),
        Parameter('Laser Control', 0.8, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  "Laser Control"),
        Parameter('green_laser', [
            Parameter('power', 1.0, float, 'Green Laser power in mW', units='mW'),
            Parameter('wavelength', 532.0, float, 'Green Laser wavelength in nm', units='nm')
        ]),
        Parameter('Filter Wheel OD', 3,
                  [0, 0.5, 1, 2, 3, 4], 'Filter Wheel OD'),
        Parameter('delays', [
            Parameter('green_laser_delay', 0.0, float, 'green_laser_delay in ns', units='ns'),
            Parameter('mw_delay', 0.0, float, 'Microwave delay in ns', units='ns'),
            Parameter('iq_delay', 0.0, float, 'iq delay in ns', units='ns'),
            Parameter('counter_delay', 0.0, float, 'AOM delay in ns', units='ns'),
            Parameter('trigger_delay', 0.0, float, 'Counter delay in ns', units='ns')
        ]),
        Parameter('adwin', [
            Parameter('count_time', 300, float, 'Photon counting time in ns', units='ns'),
            Parameter('reset_time', 0, float, 'Reset time between counts in ns', units='ns')]),
        Parameter('iq_calibration', [
            Parameter('calibration_file', '', str,
                      'Path to iq_calibration_results.json (empty = use default location)'),
            Parameter('apply_q_correction', True, bool,
                      'Scale Q channel amplitude by calibrated factor'),
            Parameter('q_amplitude_correction', 0.89, float,
                      'Q channel scale factor (overridden by calibration file if found)'),
        ]),
        Parameter('scan', [
            Parameter('preview_points', 1, int, 'Number of scan points to preview'),
            Parameter('auto_generate_files', True, bool, 'Automatically generate AWG files'),
            Parameter('output_directory', 'odmr_pulsed_output', str, 'Output directory for AWG files')
        ]),
        Parameter('averaging', [
            Parameter('averages', 10, int,
                      'Number of full runs to average term-by-term (1 = original behavior)'),
            Parameter('optimize_between_runs', True, bool,
                      'Re-optimize confocal position between runs (never in the middle of a run)'),
            Parameter('opt_xstep', 0.1, float, 'Optimization x step', units='um'),
            Parameter('opt_ystep', 0.1, float, 'Optimization y step', units='um'),
            Parameter('opt_zstep', 0.1, float, 'Optimization z step', units='um'),
            Parameter('opt_settle_time', 2.0, float, 'Settle time after each stage move', units='s'),
            Parameter('opt_count_time', 2.0, float, 'Counting time per point during optimization', units='ms'),
            Parameter('opt_num_cycles', 10, int, 'Par_10 cycles for the counter during optimization'),
            Parameter('opt_damping_ratio', 0.9, float,
                      'Move to x + damping*(best - x) instead of the raw parabola vertex'),
            # --- retention safety net (ported from the confocal-point optimizer) ---
            Parameter('min_retention_ratio', 0.5, float,
                      'If post-optimization counts < min_retention_ratio * best-known counts, '
                      'snap back toward the best-known position'),
            Parameter('acceptable_counts', 2500000.0, float,
                      'Do not update the best-known baseline to counts above '
                      'acceptable_counts * acceptable_counts_ratio (rejects noise spikes). '
                      'Set a little above your real max cts/s during optimization.'),
            Parameter('acceptable_counts_ratio', 1.1, float,
                      'See acceptable_counts'),
            Parameter('min_reoptimize_ratio', 0.5, float,
                      'Only used when reoptimize_only_if_needed is True: skip the x/y/z pass '
                      'when current counts >= min_reoptimize_ratio * best-known counts'),
            Parameter('reoptimize_only_if_needed', True, bool,
                      'If True, skip optimization before a run when counts are still healthy '
                      '(fewer stage moves = less drift). If False, optimize before every run 2..N '
                      'and rely on the retention net to prevent walking off.')
        ]),
        Parameter('path', "D:\Data"),
        Parameter('filename', "odmr_pulsed_output"),
        Parameter('tag', "odmrpulsedexperiment"),
        Parameter('save', False),
        Parameter('sample', ""),
        Parameter('Magnet ON', False),
    ]

    _DEVICES = {
        'proteus': 'proteus',
        'adwin': 'adwin',
        'sg384': 'sg384',
        'mux_control': 'mux_control',
        'filter_wheel': 'filter_wheel',
        'nanodrive': 'nanodrive'  # used only for between-run optimization
    }

    _EXPERIMENTS = {}

    def __init__(self, devices=None, experiments=None, name=None, settings=None, log_function=None, data_path=None,
                 config_path: Optional[Path] = None, mode=None):
        """Initialize the ODMR Pulsed experiment."""
        super().__init__(name=name, settings=settings, devices=devices, sub_experiments=experiments,
                         log_function=log_function, data_path=data_path)
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.test_mode = mode  # <-- None/"" = real, "testing" = MultiHarp
        self.timing_result = None  # <-- filled in testing mode
        self.repeat_count = None
        self.number_of_iterations = 0
        self.config_path = config_path or self.get_config_path(
            "D:\\Duttlab\\Experiments\\AQuISS_default_save_location\\experiments_auto_generated\\ODMRPulsedExperiment.json")
        # Configuration
        self.config = self._load_config()
        self.sequence_text = None

        # Sequence components
        self.sequence_parser = SequenceTextParser()
        self.sequence_builder = SequenceBuilder()

        # Initialize hardware calibrator with experiment-specific connection file
        connection_file = Path(__file__).parent / "odmr_pulsed_connection.json"
        self.hardware_calibrator = ProteusHardwareCalibrator(connection_file=str(connection_file))
        """self.adwin = self.devices['adwin']['instance']
        self.proteus = self.devices['proteus']['instance']
        self.sg384 = self.devices['sg384']['instance']
        self.mux = self.devices['mux_control']['instance']
        self.filter_wheel = self.devices['filter_wheel']['instance']"""
        self.proteus = ProteusDevice()
        self.adwin = AdwinGoldDevice()
        self.mux = MUXControlDevice()
        self.sg384 = SG384Generator()
        self.nd = MCLNanoDrive()
        self.mux.select_trigger('pulsed')
        # Sequence data
        self.sequence_description = None
        self.scan_sequences = []
        self.current_scan_point = 0

        # ADwin parameters
        self.count_time = self.settings["adwin"]["count_time"]  # ns
        self.reset_time = self.settings["adwin"]["reset_time"]  # ns
        self.sequence_duration = 1700  # just a placeholder

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

    def _load_iq_calibration(self) -> float:
        """
        Load Q amplitude correction from calibration file.
        Returns the correction factor b to apply to channel 2.
        Falls back to settings value if file not found.
        """
        default_b = self.settings['iq_calibration']['q_amplitude_correction']

        cal_path = self.settings['iq_calibration']['calibration_file'].strip()
        if not cal_path:
            # Look for the file next to the calibration experiment
            cal_path = str(
                Path(__file__).parent / 'iq_calibration_results.json'
            )

        try:
            with open(cal_path, 'r') as f:
                cal = json.load(f)

            # Find the entry closest to current MW frequency
            mw_freq = self.settings['microwave']['frequency range'][0]
            entries = cal.get('frequencies', [])

            if entries:
                closest = min(entries, key=lambda e: abs(e['frequency_hz'] - mw_freq))
                b = closest['q_amplitude_b']
                print(f"  IQ cal loaded: b={b:.4f} "
                      f"(from {closest['frequency_hz'] / 1e9:.4f} GHz entry, "
                      f"IRR={closest['best_irr_db']:.1f} dB)")
            else:
                b = cal.get('q_amplitude_correction_default', default_b)
                print(f"  IQ cal loaded: b={b:.4f} (default from file)")

            return b

        except FileNotFoundError:
            print(f"  IQ calibration file not found at {cal_path}, using b={default_b:.4f}")
            return default_b
        except Exception as e:
            print(f"  IQ calibration load error: {e}, using b={default_b:.4f}")
            return default_b

    def load_sequence_from_text(self) -> bool:
        """
        Load sequence definition from text using the sequence language.

        Args:
            sequence_text: Sequence definition in the sequence language format

        Returns:
            True if sequence loaded successfully
        """
        try:
            # Parse sequence text using the sequence language parser
            self.sequence_description = self.sequence_parser.parse_text(self.sequence_text)
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
                print("Sequence file not found")
                return False

            # Read sequence text
            with open(sequence_file, 'r') as f:
                self.sequence_text = f.read()
            # Parse sequence
            self.sequence_description = self.sequence_parser.parse_text(self.sequence_text)
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
            self.sampling_rate = self.sequence_builder.sample_rate
            if self.sequence_description.variables:
                self.number_of_iterations = len(self.scan_sequences)
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
            # pad:
            if len(m) % 4 != 0:
                m = np.pad(m, (0, 4 - len(m) % 4))  # pad at the end
                # m = np.pad(m, (4 - len(m) % 4, 0))# pad at the beginning

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
            # Load IQ calibration
            if self.settings['iq_calibration']['apply_q_correction']:
                q_correction_b = self._load_iq_calibration()
            else:
                q_correction_b = 1.0
            print(f"Q amplitude correction b = {q_correction_b:.4f}")
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
            #segment_for_channel = {ch: set() for ch in all_channels}
            segment_for_channel = {ch: [] for ch in all_channels}
            padded_zeros = {
                ch: 0 for ch in all_channels
            }
            max_delay = {
                it: 0 for it in range(self.number_of_iterations)
            }

            # ------------------------------------------------------------
            # Build one segment PER CHANNEL PER ITERATION.
            # Every channel gets a segment at every scan point; a channel that
            # is idle this iteration gets a flat/zero segment. This keeps every
            # channel's task table the same length (= number_of_iterations) so
            # they stay in lockstep under the shared TRG1.
            # ------------------------------------------------------------
            SEGNUM = 1
            for sequence in self.scan_sequences:
                seq_waveforms = sequence.to_waveform()
                seq_len = sequence.length
                for ch in sorted(all_channels):  # <-- ALL channels, fixed order
                    data = seq_waveforms.get(ch)
                    if data is not None:
                        envelope = data["envelope"]
                        markers = data["markers"]
                    else:
                        # idle channel this iteration -> flat 0 V, no markers
                        envelope = np.zeros(seq_len, dtype=float)
                        markers = np.zeros(seq_len, dtype=np.uint8)

                    self.proteus.driver.set_channel(ch)
                    analog_voltage = "MAX"
                    self.proteus.driver.set_voltage(analog_voltage)
                    self.proteus.driver.apply_sampling_configuration(self.sampling_rate)

                    # ── Apply Q amplitude correction ──────────────────────
                    if ch == 2:
                        envelope = envelope * q_correction_b
                    # ──────────────────────────────────────────────────────

                    segment_num = SEGNUM
                    segment_for_channel[ch].append(segment_num)  # ordered list, scan order

                    rem = len(envelope) % ALIGNMENT
                    padded_zeros[ch] = rem
                    if rem:
                        envelope = np.pad(envelope, (0, ALIGNMENT - rem))
                        markers = np.pad(markers, (0, ALIGNMENT - rem))
                        padded_zeros[ch] = ALIGNMENT - rem

                    # Scale to DAC
                    dac_wave = np.clip(envelope, -1.0, 1.0)
                    dac_wave = ((dac_wave + 1.0) * half_dac).astype(np.uint16)
                    self.proteus.driver.define_trace(segment_num, len(dac_wave))
                    self.proteus.driver.select_segment(segment_num)

                    # Upload waveform
                    self.proteus.driver.write_trace_data(dac_wave)
                    resp = self.proteus.driver.query_error()
                    self.proteus.driver.inst.send_scpi_cmd(":VOLT:OFFS 0.012")

                    # Convert to uint8 and upload markers (zeros for idle channels)
                    markers = markers.astype(np.uint8)
                    tabor_markers = prepare_markers_for_tabor(markers)
                    self.proteus.driver.write_marker_data(tabor_markers)
                    resp = self.proteus.driver.query_error()
                    marker_index = 1
                    self.proteus.driver.set_marker(marker_index)
                    self.proteus.driver.set_marker_ptop_voltage(1.2)
                    self.proteus.driver.set_marker_voltage_offset(0.5)
                    resp = self.proteus.driver.query_error()
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
            self.proteus.driver.set_trigger_level(2)
            self.proteus.driver.activate_instrument(1)
            for ch in sorted(all_channels):
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.set_trigger_source('TRG1')
                self.proteus.driver.set_trigger("TRG1")  # :TRIG:SEL
                self.proteus.driver.set_trigger_state("ON")  # :TRIG:STAT

            for ch in sorted(all_channels):
                task_num = 1
                self.proteus.driver.set_channel(ch)
                n_tasks = len(segment_for_channel[ch])  # tasks for THIS channel
                self.proteus.driver.set_task_table_length(n_tasks)
                for segment_number in segment_for_channel[ch]:
                    self.proteus.driver.set_task_number(task_num)
                    if task_num == 1:
                        self.proteus.driver.set_task_type("STAR")
                    elif task_num == n_tasks:  # last task on THIS channel
                        self.proteus.driver.set_task_type("END")
                    else:
                        self.proteus.driver.set_task_type("SEQ")
                    self.proteus.driver.set_task_segment_number(segment_number)
                    self.proteus.driver.set_trigger_IDLE_state("DC")
                    self.proteus.driver.set_trigger_IDLE_level(0)
                    self.proteus.driver.set_enabling_task_signal("TRG1")  # every task gated by TRG1

                    if task_num == n_tasks:
                        self.proteus.driver.set_next1_task(1)  # wrap to first task
                    else:
                        self.proteus.driver.set_next1_task(task_num + 1)

                    task_num += 1
                    self.proteus.driver.write_composer_array_to_task_table()
                    self.proteus.driver.set_waveform_type('TASK')
            ###self.proteus.driver.set_task_loop(self.repeat_count)
            self.proteus.driver.set_task_sync()
            resp = self.proteus.driver.set_trigger_coupling('ON')
            # for trigger source INT send:
            self.proteus.driver.set_channel(1)
            # time.sleep(1)
            # self.proteus.driver._close()
            return True
        except Exception as e:
            self.logger.exception("Error generating AWG sequences")
            return False

    def _function(self):
        self.filter_wheel.update({'OD': self.settings['Filter Wheel OD']})
        self.adwin = self.devices['adwin']['instance']
        self.proteus = self.devices['proteus']['instance']
        self.sg384 = self.devices['sg384']['instance']
        self.mux = self.devices['mux_control']['instance']
        # nanodrive is only needed for between-run optimization; fetch defensively
        # so the experiment still runs if it isn't wired up.
        self.nd = self.devices.get('nanodrive', {}).get('instance', None)

        # new work to be tested:
        def build_calibration_overrides():
            return {
                "green_laser_delay": self.settings['delays']['green_laser_delay'],
                "mw_delay": self.settings['delays']['mw_delay'],
                "iq_delay": self.settings['delays']['iq_delay'],
                "counter_delay": self.settings['delays']['counter_delay'],
                "trigger_delay": self.settings['delays']['trigger_delay'],
                "units": "ns"
            }

        # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
        overrides = build_calibration_overrides()
        self.hardware_calibrator.update_calibration_delays(overrides)
        # Experiment parameters: the next section updates the settings every start click
        self.microwave_power = self.settings['microwave']['power']
        self.mw_delay = self.settings['delays']['mw_delay']
        self.green_laser_delay = self.settings['delays']['green_laser_delay']
        self.counter_delay = self.settings['delays']['counter_delay']
        self.iq_delay = self.settings['delays']['iq_delay']
        self.trigger_delay = self.settings['delays']['trigger_delay']

        self.green_laser_power = self.settings['green_laser']['power']
        self.green_laser_wavelength = self.settings['green_laser']['wavelength']

        # Output paths
        self.output_dir = self.get_output_dir("odmr_pulsed_output")
        self.logger.info("ODMR Pulsed Experiment initialized")

        # Create and load sequence
        source = None
        sequence_loaded = False

        if self.settings['sequence']['load_from_file']:
            file_path = self.settings['sequence']['file_path']
            file_path = Path(file_path)

            if file_path and os.path.exists(file_path):
                sequence_loaded = self.load_sequence_from_file(file_path)
                print(f"file path: {file_path}")
                source = "file"
            else:
                self.logger.info("File path is invalid or does not exist")
        else:
            self.sequence_text = self.settings['sequence']['text']
            print("ODMR Sequence:")
            print(self.sequence_text)
            print("\n" + "=" * 50 + "\n")
            sequence_loaded = self.load_sequence_from_text()
            source = "text"

        if sequence_loaded:
            self.logger.info(f"Sequence loaded successfully from {source}")
        else:
            self.logger.info(f"Failed to load sequence from {source}")
            return
        # Run experiment
        self.logger.info("Running experiment...")
        results = self.run_experiment_averaged(self.settings['microwave']['frequency range'])
        self.data = results
        self.save_hdf5()
        if results['success']:
            self.logger.info("Experiment completed successfully!")
            self.logger.info(f"Frequencies scanned: {[f / 1e9 for f in results['frequencies']]} GHz")
            self.logger.info(
                f"Signal counts shape: {len(results['signal_counts'])} frequencies x {len(results['signal_counts'][0])} points")
        else:
            self.logger.info(f"Experiment failed: {results['error']}")
        self.cleanup()
        self.show_sequence_preview(10)
        self.logger.info("\nODMR Pulsed Experiment ready!")

    def save_hdf5(self):
        """this function defines its custom data and metadata to be saved and then calls the
        save_hdf_data function that is in the parent Experiment class, which adds the external
        devices in case you ever check the Get Basic Data checkbox in the GUI"""
        structure_to_save = MyStruct()
        # Save every per-run array (signal_1, reference_1, total_1, ...), the
        # term-by-term averages (signal_avg, reference_avg, total_avg), and the
        # legacy signal_counts/reference_counts/total_counts (= the averages).
        data_kwargs = {}
        for key, val in self.data.items():
            if key.startswith(('signal_', 'reference_', 'total_')) or key in (
                    'signal_counts', 'reference_counts', 'total_counts'):
                data_kwargs[key] = val
        structure_to_save.data = MyStruct(**data_kwargs)
        structure_to_save.meta = MyStruct(
            gate_align_offset_ns=self.gate_align_offset_ns,
            count_time=self.count_time,
            counter_delay=self.counter_delay,
            end_time=self.e_t,
            frequency_range=self.data['frequencies'],
            green_laser_delay=self.green_laser_delay,
            green_laser_power=self.green_laser_power,
            green_laser_wavelength=self.green_laser_wavelength,
            iq_delay=self.iq_delay,
            microwave_power=self.microwave_power,
            mw_delay=self.mw_delay,
            number_of_iterations=self.number_of_iterations,
            averages=self.data.get('averages', 1),
            repeat_count=self.repeat_count,
            reset_time=self.reset_time,
            sampling_rate=self.sampling_rate,
            sequence_text=self.sequence_text,
            sequence_duration=self.sequence_duration,
            start_time=self.s_t,
            success=self.data["success"],
            trigger_delay=self.trigger_delay,
            adwin_file=self.adwin_file,
            adwin_reached_the_end=self.adwin_reached_the_end,
            number_of_sequence_done_by_adwin=self.number_of_sequences_done_by_adwin
        )
        structure_to_save.devices = self.devices
        self.save_hdf_data(structure_to_save)

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
            self.proteus.driver.set_channel(3)
            self.proteus.driver.set_voltage('MAX')
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

            channel_waveforms = {
                ch: [] for ch in all_channels
            }
            channel_markers = {
                ch: [] for ch in all_channels
            }
            for ch in all_channels:
                self.proteus.driver.set_channel(ch)
                self.proteus.driver.delete_all_segment()
                self.proteus.driver.set_voltage("MAX")
                self.proteus.driver.apply_sampling_configuration(self.sampling_rate)

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

                # Apply required alignment
                rem = len(waveform) % ALIGNMENT
                if rem:
                    waveform = np.pad(waveform, (0, ALIGNMENT - rem))

                # Scale to DAC
                dac_wave = np.clip(waveform, -1.0, 1.0)
                dac_wave = ((dac_wave + 1.0) * half_dac).astype(np.uint16)

                # Define and select segment
                self.proteus.driver.define_trace(segment_index, len(dac_wave))
                self.proteus.driver.select_segment(segment_index)

                # Upload waveform
                self.proteus.driver.write_trace_data(dac_wave)  # write, and wait while *OPC completes
                resp = self.proteus.driver.query_error()

                segment_for_channel[ch] = segment_index
                # Handle markers for this channel
                if ch in channel_markers:
                    markers = np.concatenate(channel_markers[ch])


                    rem = len(waveform) % ALIGNMENT
                    if rem:
                        waveform_padded_len = len(waveform) + (ALIGNMENT - rem)
                    else:
                        waveform_padded_len = len(waveform)

                    # Now pad/truncate markers to match waveform length
                    if len(markers) < waveform_padded_len:
                        # Pad markers with zeros
                        markers = np.pad(markers, (0, waveform_padded_len - len(markers)), mode='constant')
                    elif len(markers) > waveform_padded_len:
                        # Truncate markers
                        markers = markers[:waveform_padded_len]

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
                        # proteus P1284M only has 1 marker per channel, for other awgs, please implement code that handels that
                        # I have already coded the marker index here: mkr_index = int(marker.name.split('_')[-2]) so if the user
                        # gives marker, laser_int_1 on channel 4 at 0ns, 500ns then you know whatever is after laser_init_ is the index
                        marker_index = 1

                        self.proteus.driver.set_marker(marker_index)
                        # Verify Proteus marker settings
                        resp = self.proteus.driver.get_marker()
                        resp = self.proteus.driver.get_marker_state()
                        resp = self.proteus.driver.query_error()
                        resp = self.proteus.driver.get_marker_ptop_voltage()
                        resp = self.proteus.driver.get_marker_voltage()
                        self.proteus.driver.set_marker_ptop_voltage(1)
                        resp = self.proteus.driver.get_marker_ptop_voltage()
                        resp = self.proteus.driver.get_marker_voltage_offset()
                        self.proteus.driver.set_marker_voltage_offset(0.5)
                        resp = self.proteus.driver.get_marker_voltage_offset()
                        # Check for errors
                        resp = self.proteus.driver.query_error()
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
            # self.proteus.driver._close()
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
        this is just for testing: we need to only use json or gui for this
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
        this is just for testing: we need to only use json or gui for this
        Set laser parameters.

        Args:
            power: Power in mW
            wavelength: Wavelength in nm
        """
        self.green_laser_power = power
        self.green_laser_wavelength = wavelength
        self.logger.info(f"Laser: {power} mW, {wavelength} nm")

    def set_delay_parameters(self, mw_delay: float, green_laser_delay: float, counter_delay: float) -> None:
        """
        this is just for testing: we need to only use json or gui for this
        Set delay parameters.

        Args:
            mw_delay: Microwave delay in ns
            green_laser_delay: green_laser_delay delay in ns
            counter_delay: Counter delay in ns
        """
        self.mw_delay = mw_delay
        self.green_laser_delay = green_laser_delay
        self.counter_delay = counter_delay
        self.logger.info(f"Delays: MW={mw_delay}ns, AOM={green_laser_delay}ns, Counter={counter_delay}ns")

    def configure_gates_v5(self):
        """
        Compute the gate tick targets HERE and push them to the state machine.

        Requires on self (all in ns, all REAL values):
            self.readout_start_ns   trigger -> readout-pulse onset (SAME for every tau;
                                    only init/MW start may move within the segment)
            self.count_time         Y, the count window width (e.g. 300)
            self.reset_time         REAL re-polarization time between the two windows
                                    (keep 2*Y + reset <= readout length!)
            self.trigger_duration   trigger pulse width (e.g. 100-200)
            self.repeat_count
            self.number_of_iterations
        Optional:
            self.gate_align_offset_ns  one-time scope calibration for the fixed
                                       trigger->play latency (default 0)
            self.pd_ticks              Processdelay in CYCLES (default: ~3x the shot)

        Returns the readout pulse length X (ns).
        """
        tpns = float(ADWIN_TICKS_PER_NS)
        Y = int(round(self.count_time))
        rst = int(round(self.reset_time))
        align = int(round(getattr(self, "gate_align_offset_ns", 0)))
        trig_ns = int(round(self.trigger_duration))

        sig_start_ns = int(round(self.readout_start_ns)) + align  # trigger -> signal window
        ref_start_ns = sig_start_ns + Y + rst  # trigger -> reference window (last Y)
        X = readout_length_ns(Y, rst)  # readout pulse length

        # ns -> ticks, done on the PC with the one calibrated constant
        count_ticks = max(1, int(round(Y * tpns)))
        sig_start_ticks = max(1, int(round(sig_start_ns * tpns)))
        ref_start_ticks = int(round(ref_start_ns * tpns))
        trig_ticks = max(1, int(round(trig_ns * tpns)))

        # One Event per shot => Processdelay MUST exceed the shot, or the Event
        # overruns every shot, the shots run back-to-back, and the PC link is
        # starved for the whole sweep (first get_int_var must then survive the
        # entire sweep -> Get_Par timeout once the sweep is long enough). We compute
        # it straight from the shot and DO NOT let anything shrink it. experiment.
        # pd_ticks is intentionally ignored now -- a stray small value was the trap.
        shot_ns = ref_start_ns + Y
        pd_ticks = max(1500, int(round(
            2.0 * shot_ns * tpns)))  # ~2x shot; Event uses half the period, link gets the other half
        pd_us = pd_ticks / tpns / 1000.0
        shot_us = shot_ns / 1000.0
        if getattr(self, "pd_ticks", None) is not None:
            print(f"[v5] IGNORING experiment.pd_ticks={int(self.pd_ticks)} "
                  f"(a small Processdelay starves the link). Using {pd_ticks} cycles.")

        a = self.adwin
        a.set_int_var(1, int(self.repeat_count))  # Par_1
        a.set_int_var(2, int(self.number_of_iterations))  # Par_2
        a.set_int_var(3, count_ticks)  # Par_3  Y (ticks)
        a.set_int_var(4, ref_start_ticks)  # Par_4  trigger->reference start (ticks)
        a.set_int_var(5, sig_start_ticks)  # Par_5  trigger->signal start (ticks)
        a.set_int_var(6, pd_ticks)  # Par_6  Processdelay (cycles, >= shot)
        a.set_int_var(7, trig_ticks)  # Par_7  trigger width (ticks)

        print(f"[v5] ticks/ns={tpns}: sig_start={sig_start_ns}ns->{sig_start_ticks}t, "
              f"ref_start={ref_start_ns}ns->{ref_start_ticks}t, Y={Y}ns->{count_ticks}t, "
              f"readout X={X}ns.")
        print(f"[v5] shot~{shot_us:.1f} us, Processdelay={pd_ticks} cyc (~{pd_us:.1f} us) "
              f"-> Event uses ~{shot_us / pd_us * 100:.0f}% of the period, link gets the rest. "
              f"CHECK Par_71 == {pd_ticks} after this; if it reads small, something else "
              f"is writing Par_6 after configure.")
        return X

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
                print("No sequence loaded")
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

            # Step 6: Run experiment for each frequency
            # 'success': True,
            # 'frequencies': frequency_range,
            results = {
                'signal_counts': [],
                'reference_counts': [],
                'total_counts': []
            }
            # delete later:
            # frequency_range = [2.87e9]
            for freq in frequency_range:
                self.logger.info(f"Running experiment at {freq / 1e9:.3f} GHz")

                # Set SG384 frequency
                # if 'sg384' in self.devices:
                self.sg384.set_frequency(freq)
                self.sg384.set_power(self.microwave_power)
                if self.settings["microwave"]["enable"]:
                    self.sg384.enable_output()
                self.microwave_frequency = freq
                self.logger.info(f"Set SG384 to {freq / 1e9:.3f} GHz, {self.microwave_power} dBm")

                # Run sequence and collect data
                freq_results = self._run_sequence_and_collect_data()
                if not freq_results['success']:
                    return {'success': False, 'error': f'Failed at frequency {freq / 1e9:.3f} GHz'}

                results['signal_counts'].append(freq_results['signal_counts'])
                results['reference_counts'].append(freq_results['reference_counts'])
                results['total_counts'].append(freq_results['total_counts'])
            results['signal_counts'] = np.array(results['signal_counts'], dtype=np.int64)
            results['reference_counts'] = np.array(results['reference_counts'], dtype=np.int64)
            results['total_counts'] = np.array(results['total_counts'], dtype=np.int64)
            results['success'] = True
            results['frequencies'] = frequency_range

            self.logger.info("ODMR Pulsed Experiment completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return {'success': False, 'error': str(e)}

    def run_experiment_averaged(self, frequency_range: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Run run_experiment() 'averages' times and average the count arrays
        term-by-term to beat down noise. Optionally re-optimize the confocal
        position BETWEEN runs (never in the middle of a run).

        Timing matches your description:
            run 1 (repeat_count reps, no optimization)
            -> optimize -> run 2 -> optimize -> run 3 -> ...
        i.e. optimization happens before runs 2..N, not after the last run.

        Returns a dict containing, for each run i (1-based):
            signal_i, reference_i, total_i        (int64,  shape [n_freq, n_iter])
        plus the term-by-term averages:
            signal_avg, reference_avg, total_avg  (float64, same shape)
        The legacy keys signal_counts / reference_counts / total_counts are set
        to the *_avg arrays so the rest of the code (save_hdf5, logging, plots)
        keeps working with no other changes.
        """
        averages = int(self.settings['averaging']['averages'])
        optimize_between = bool(self.settings['averaging']['optimize_between_runs'])
        if averages < 1:
            averages = 1

            # Reset the confocal optimizer's best-known baseline for THIS experiment run.
            # It gets seeded on the first _optimize_position() call (before run 2),
            # i.e. from the good spot run 1 was measured at.
        self._opt_best_counts = None
        self._opt_best_x = None
        self._opt_best_y = None
        self._opt_best_z = None

        results: Dict[str, Any] = {'success': True, 'frequencies': frequency_range}
        signal_runs, ref_runs, total_runs = [], [], []

        for run_idx in range(1, averages + 1):
            msg = f"=== Average run {run_idx}/{averages} ==="
            self.logger.info(msg)
            print(msg)

            # Optimize BEFORE runs 2..N. Never before run 1, never mid-run.
            if optimize_between and run_idx > 1:
                try:
                    self._optimize_position()
                except Exception as e:
                    # Optimization is best-effort: log and keep measuring.
                    self.logger.error(f"Optimization before run {run_idx} failed (continuing): {e}")

            run = self.run_experiment(frequency_range)
            if not run.get('success'):
                self.logger.error(f"Run {run_idx} failed: {run.get('error')}")
                return {'success': False,
                        'error': run.get('error', f'run {run_idx} failed'),
                        'frequencies': frequency_range}

            s = np.asarray(run['signal_counts'], dtype=np.int64)
            r = np.asarray(run['reference_counts'], dtype=np.int64)
            t = np.asarray(run['total_counts'], dtype=np.int64)

            signal_runs.append(s)
            ref_runs.append(r)
            total_runs.append(t)
            results[f'signal_{run_idx}'] = s
            results[f'reference_{run_idx}'] = r
            results[f'total_{run_idx}'] = t

        # Term-by-term average across runs (float; fractional averages are correct).
        signal_avg = np.mean(np.stack(signal_runs, axis=0), axis=0)
        ref_avg = np.mean(np.stack(ref_runs, axis=0), axis=0)
        total_avg = np.mean(np.stack(total_runs, axis=0), axis=0)

        results['signal_avg'] = signal_avg
        results['reference_avg'] = ref_avg
        results['total_avg'] = total_avg

        # Keep the legacy keys pointing at the averages so nothing downstream breaks.
        results['signal_counts'] = signal_avg
        results['reference_counts'] = ref_avg
        results['total_counts'] = total_avg
        results['averages'] = averages

        self.logger.info(f"Averaging complete over {averages} run(s)")
        return results

    def _read_opt_counts(self) -> float:
        """Read the current count rate (cts/s) from the simple counter binary."""
        raw = self.adwin.read_probes('int_var', id=1)
        return raw * 1e3 / self._opt_count_time_ms

    def _optimize_axis(self, axis: str, step: float) -> None:
        """
        3-point quadratic-fit maximization on one nanodrive axis ('x'/'y'/'z').


        Step back 2x first to remove backlash, sample minus/old/plus, parabola-fit.
        The parabola vertex is trusted ONLY when the fit is concave-down (a < 0, a
        real maximum) AND the vertex lands inside the sampled span; otherwise the
        "vertex" is a minimum or a wild extrapolation, so we fall back to the best
        point we actually measured. Out-of-range moves (ARGUMENT_ERROR) are skipped.
        """
        nd = self.nd
        pos_key = f'{axis}_pos'
        opt = self.settings['averaging']
        settle_time = float(opt['opt_settle_time'])
        damping_ratio = float(opt['opt_damping_ratio'])


        def move(target) -> bool:
            try:
                nd.update({pos_key: target})
                time.sleep(settle_time)
                return True
            except Exception as e:
                if "ARGUMENT_ERROR" in str(e):
                    return False
                raise


        def read_pos():
            return nd.read_probes(pos_key)


        # Back off two steps to take out backlash, then sample minus / old / plus.
        move(read_pos() - 2 * step)
        samples = []  # (position, count_rate)
        for _ in range(3):
            if move(read_pos() + step):
                samples.append((read_pos(), self._read_opt_counts()))


        if len(samples) < 3:
            return


        coords = np.array([p for p, _ in samples], dtype=float)
        counts = np.array([c for _, c in samples], dtype=float)
        a, b, _c = np.polyfit(coords, counts, 2)


        best_pos, best_counts = max(samples, key=lambda pc: pc[1])
        lo, hi = float(coords.min()), float(coords.max())


        # Only trust the vertex for a concave-down parabola whose vertex is in range.
        use_vertex = False
        vertex = None
        if a < 0:
            vertex = float(-b / (2 * a))
            if lo <= vertex <= hi:
                use_vertex = True


        if use_vertex:
            if not move(vertex):
                return
            counts_at_vertex = self._read_opt_counts()
            if best_counts > counts_at_vertex:
                move(vertex + damping_ratio * (best_pos - vertex))
        else:
            move(best_pos)

    def _optimize_position(self) -> None:
        """
        One-shot confocal re-optimization between averaging runs, with the
        confocal-point retention net: a best-known (position, counts) baseline is
        remembered across runs; after the z/x/y pass, if counts dropped below
        min_retention_ratio of the best-known counts the stage snaps back toward
        the best position; the baseline is only updated below
        acceptable_counts * acceptable_counts_ratio (rejects noise spikes); and
        optionally the whole pass is skipped when counts are still healthy.


        Reuses the counter binary (Averagable_Trial_Counter.TB1) and does NOT touch
        ADbasic source: the next run_experiment() reloads the pulsed binary via
        _setup_adwin_counting() and reprograms Proteus, so state self-heals.
        """
        nd = getattr(self, 'nd', None)
        if nd is None:
            self.logger.warning("No nanodrive available; skipping optimization.")
            print("No nanodrive available; skipping optimization.")
            return


        opt = self.settings['averaging']
        self._opt_count_time_ms = float(opt['opt_count_time'])
        num_cycles = int(opt['opt_num_cycles'])
        settle_time = float(opt['opt_settle_time'])
        damping_ratio = float(opt['opt_damping_ratio'])
        min_retention_ratio = float(opt['min_retention_ratio'])
        acceptable_counts = float(opt['acceptable_counts'])
        acceptable_counts_ratio = float(opt['acceptable_counts_ratio'])
        min_reoptimize_ratio = float(opt['min_reoptimize_ratio'])
        reoptimize_only_if_needed = bool(opt['reoptimize_only_if_needed'])


        # --- switch ADwin to the simple counter (does NOT touch ADbasic source) ---
        self.adwin.stop_process(1)
        time.sleep(0.1)
        self.adwin.clear_process(1)
        counter_path = get_adwin_binary_path('Averagable_Trial_Counter.TB1')
        self.adwin.update({'process_1': {'load': str(counter_path)}})
        self.adwin.set_int_var(10, num_cycles)
        adwin_delay = round((self._opt_count_time_ms * 1e6) / 3.3)
        self.adwin.update({'process_1': {'delay': adwin_delay, 'running': True}})


        # --- laser on for counting (reprogrammed by the next run) ---
        try:
            self.proteus.set_channel_voltage_high(4, self.settings["Laser Control"])
        except Exception as e:
            self.logger.warning(f"Could not force laser channel high for optimization: {e}")


        time.sleep(0.2)  # let the stage settle and the counter spin up


        # Seed the best-known baseline on the first optimization of this experiment.
        current_counts = self._read_opt_counts()
        if getattr(self, '_opt_best_counts', None) is None:
            self._opt_best_counts = current_counts
            self._opt_best_x = nd.read_probes('x_pos')
            self._opt_best_y = nd.read_probes('y_pos')
            self._opt_best_z = nd.read_probes('z_pos')
            print(f"Seeded best-known baseline: {current_counts:.0f} cts/s.")


        did_optimize = False
        if reoptimize_only_if_needed and current_counts >= (min_reoptimize_ratio * self._opt_best_counts):
            print(f"Counts healthy ({current_counts:.0f} >= "
                  f"{min_reoptimize_ratio:.2f} * {self._opt_best_counts:.0f}); "
                  f"skipping re-optimization -- stage NOT moved.")
        else:
            did_optimize = True
            print("Optimizing confocal position between runs (z -> x -> y)...")
            self._optimize_axis('z', float(opt['opt_zstep']))   # <-- z first
            self._optimize_axis('x', float(opt['opt_xstep']))
            self._optimize_axis('y', float(opt['opt_ystep']))


            # --- retention net (mirrors the confocal-point experiment) ---
            optimized_counts = self._read_opt_counts()
            if optimized_counts > self._opt_best_counts:
                if optimized_counts < (acceptable_counts * acceptable_counts_ratio):
                    self._opt_best_counts = optimized_counts
                    self._opt_best_x = nd.read_probes('x_pos')
                    self._opt_best_y = nd.read_probes('y_pos')
                    self._opt_best_z = nd.read_probes('z_pos')
                    print(f"New best-known position: {optimized_counts:.0f} cts/s.")
                else:
                    print(f"Ignoring implausibly high counts ({optimized_counts:.0f} cts/s) "
                          f"for the baseline (likely a fluctuation).")
            elif optimized_counts < (min_retention_ratio * self._opt_best_counts):
                print(f"Post-opt counts {optimized_counts:.0f} < {min_retention_ratio:.2f} * best "
                      f"{self._opt_best_counts:.0f}; snapping back toward best-known position.")
                cx = nd.read_probes('x_pos')
                cy = nd.read_probes('y_pos')
                cz = nd.read_probes('z_pos')
                try:
                    nd.update({'x_pos': cx + damping_ratio * (self._opt_best_x - cx),
                               'y_pos': cy + damping_ratio * (self._opt_best_y - cy),
                               'z_pos': cz + damping_ratio * (self._opt_best_z - cz)})
                    time.sleep(settle_time)
                except Exception as e:
                    if "ARGUMENT_ERROR" in str(e):
                        print(f"Snap-back skipped (out of range): {e}")
                    else:
                        raise


        if did_optimize:
            try:
                final = self._read_opt_counts()
                print(f"Optimization done: ~{final:.0f} cts/s at "
                      f"x={nd.read_probes('x_pos'):.3f}, "
                      f"y={nd.read_probes('y_pos'):.3f}, "
                      f"z={nd.read_probes('z_pos'):.3f}")
            except Exception:
                pass


        # Stop the counter so the next run starts clean (run_experiment reloads the SM).
        self.adwin.update({'process_1': {'running': False}})
        self.adwin.stop_process(1)
        time.sleep(0.1)
        self.adwin.clear_process(1)

    def cleanup(self):
        """Cleanup experiment resources."""
        # Stop Adwin process
        if self.adwin and self.adwin.is_connected:
            self.adwin.stop_process(1)
            self.adwin.clear_process(1)

        # Disable microwave sweep and output
        if self.sg384 and self.sg384.is_connected:
            self.sg384.disable_modulation()
            self.sg384.disable_output()

        self.log("ODMR Phase Continuous Sweep Experiment cleanup complete")

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
            odmr_pulsed_counter_path = get_adwin_binary_path('test_adwin_digout.TB1')
            # tested proteus delay and digout delay
            odmr_pulsed_counter_path = get_adwin_binary_path('test_adwin_delays.TB1')
            self.adwin_file = 'adwin_triggering_proteus.TB1'
            self.adwin_file = 'T1.TB1'
            self.adwin_file = 'test_adwin_delays_v2.TB1'
            self.adwin_file = 'PULSE_FIXED_WAIT_READ.TB1'
            self.adwin_file = 'TIME_RESOLUTION.TB1'
            self.adwin_file = 'TIME_RESOLUTIONz_T1_transition.TB1'"""
            # ── choose the state-machine binary ──────────────────────────────────────
            # Both binaries pin Processdelay ~1 us. v1 runs signal+gap+ref as ONE Event,
            # which overruns whenever the block is wide, walking the gate ~150 ns/shot --
            # harmless for a wide T1 readout, fatal for a narrow Rabi gate on the onset.
            # v2 splits them into separate Events (gap = bare off-Events) so nothing
            # overruns and the reference can sit ~2 us later in the bright window.
            SHORT_SEQ_US = 50.0
            if (self.sequence_duration / 1000.0) < SHORT_SEQ_US:
                self.adwin_file = 'odmr_pulsed_statemachine_v5.TB1'  # split gates (Rabi)
            else:
                self.adwin_file = 'odmr_pulsed_statemachine_v1.TB1'  # (T1 and similar experiments)

            if self.adwin_file.endswith('v2.TB1') or self.adwin_file.endswith('v4.TB1'):
                expected_signature = 7778
            elif self.adwin_file.endswith('v1.TB1'):
                expected_signature = 7777
            elif self.adwin_file.endswith('v5.TB1'):
                expected_signature = 7783
            elif self.adwin_file.endswith('v3.TB1'):
                expected_signature = 7779
            else:
                expected_signature = 7790
            odmr_pulsed_counter_path = get_adwin_binary_path(self.adwin_file)
            self.adwin.update({'process_1': {'load': str(odmr_pulsed_counter_path)}})
            # Set ADwin parameters for counting
            # OLD SET
            """self.adwin.set_int_var(3, self.count_time )
            self.adwin.set_int_var(4, self.reset_time )
            self.adwin.set_int_var(5, self.repeat_count)
            self.adwin.set_int_var(6, self.number_of_iterations)
            self.adwin.set_int_var(9, self.sequence_duration)
            self.adwin.set_int_var(11, 1)"""
            count_time_10_ns = int(self.count_time / 10)
            ref_delay_10ns = int(self.reset_time / 10) if self.reset_time else 167  # gap between gates
            seq_dur_us = int(self.sequence_duration / 1000)
            TRIG_10NS = 12
            self.adwin.set_int_var(1, int(self.repeat_count))
            self.adwin.set_int_var(2, int(self.number_of_iterations))
            self.adwin.set_int_var(3, count_time_10_ns)  # Par_3 signal gate (10 ns units)
            self.adwin.set_int_var(4, ref_delay_10ns)  # Par_4 inter-gate gap (10 ns units)
            self.adwin.set_int_var(5, count_time_10_ns)  # Par_5 reference gate (10 ns units)
            self.adwin.set_int_var(6, seq_dur_us)  # Par_6 seq_dur_us (informational here)
            self.adwin.set_int_var(7, TRIG_10NS)  # Par_7 trigger width (10 ns units)
            self.adwin.set_int_var(11, 0)  # Par_11 pre_count fine offset (WAS 1 = old start flag)
            self.adwin.set_int_var(12, 0)  # Par_12 fine_delay
            self.adwin.set_int_var(14, 3)  # Par_14 gate countdown -> SET to your test_seq_rabi peak (~readout us)
            # for debug only: self.adwin.set_int_var(15, 200)
            # State machine boots into IDLE; Par_10 starts it (set in _run_sequence_and_collect_data)
            self.adwin.start_process(process_number)
            time.sleep(0.3)
            if self.adwin.get_int_var(80) != expected_signature:
                raise RuntimeError(f"ADwin signature mismatch (loaded {self.adwin_file})")

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

            # Collect data
            # this method has python collecting data and telling aadwin to stop:
            # Collect data
            X = self.configure_gates_v5()
            start_time = datetime.datetime.now()
            self.s_t = start_time.strftime("%m_%d_%Y_%H:%M:%S")

            # ---- Launch this sweep: THIS is what was missing ----
            self.adwin.set_int_var(20, 0)  # clear done flag
            self.adwin.set_int_var(10, 1)  # Par_10 = 1 -> RUN (leaves IDLE, starts firing triggers)

            # ---- Wait on the handshake instead of a fixed sleep ----
            # SM fires repeat_count x n_iterations triggers, counts, then sets Par_20 = 1.
            t_start = time.time();
            t_hb = time.time()
            last_hb = self.adwin.get_int_var(25)
            # ~2 Events per iteration, each = Processdelay. Par_71 echoes the real Processdelay.
            pd_ns = self.adwin.get_int_var(71) / 0.3
            expected_sweep_s = 2 * pd_ns * 1e-9 * self.repeat_count * max(1, self.number_of_iterations)
            timeout_s = max(30.0, expected_sweep_s * 1000.0)
            while self.adwin.get_int_var(20) == 0:
                time.sleep(0.05)
                hb = self.adwin.get_int_var(25)  # heartbeat
                if hb != last_hb:
                    last_hb = hb;
                    t_hb = time.time()
                elif time.time() - t_hb > 15:
                    raise RuntimeError(f"ADwin heartbeat frozen (state Par_26={self.adwin.get_int_var(26)})")
                if time.time() - t_start > timeout_s:
                    raise TimeoutError("State machine never set Par_20 (done)")

            end_time = datetime.datetime.now()
            self.e_t = end_time.strftime("%m_%d_%Y_%H:%M:%S")
            self.number_of_sequences_done_by_adwin = self.adwin.get_int_var(28)  # Par_28 rep counter
            self.adwin_reached_the_end = self.adwin.get_int_var(20)  # Par_20: 1 = done
            signal_counts = np.array(self.adwin.get_int_data(1, self.number_of_iterations), dtype=np.int64)
            """total_counts = np.array(self.adwin.get_int_data(2, self.number_of_iterations), dtype=np.int64)
            ref_counts=total_counts.astype(np.int64)-signal_counts.astype(np.int64)"""
            ref_counts = np.array(self.adwin.get_int_data(2, self.number_of_iterations), dtype=np.int64)
            total_counts = ref_counts.astype(np.int64) + signal_counts.astype(np.int64)
            for signal_counts_val in signal_counts:
                self.logger.info(f"signal_counts_val {signal_counts_val}")
                print(f"signal_counts_val {signal_counts_val}")
            for reference_counts_val in ref_counts:
                self.logger.info(f"reference_counts_val {reference_counts_val}")
                print(f"reference_counts_val {reference_counts_val}")
            for total_counts_val in total_counts:
                self.logger.info(f"total_counts_val {total_counts_val}")
                print(f"total_counts_val {total_counts_val}")
                # ---- ack so the SM idles; the next frequency re-arms via Par_10=1 ----
            a = self.adwin
            sig_start = a.get_int_var(75)  # echoed sig_start_ticks
            ref_start = a.get_int_var(76)  # echoed ref_start_ticks
            capped = a.get_int_var(74)  # spin-loops that hit the cap (want 0)
            pd_ticks = a.get_int_var(71)  # Processdelay actually used
            sig_id = a.get_int_var(80)  # 7783 for direct-tick v5

            print(f"[v5] signature={sig_id} (expect 7783), Processdelay={pd_ticks} cyc")
            print(f"[v5] sig_start={sig_start} ticks, ref_start={ref_start} ticks (echoed back)")
            print(f"[v5] capped spin-loops={capped}  (want 0)")
            if capped:
                print("[v5] WARNING: a spin-loop hit its cap -> a wait never reached target. "
                      "If ADWIN_TICKS_PER_NS is far too large the targets are unreachable; "
                      "recheck the DIO16 high-time calibration.")
            self.adwin.set_int_var(20, 0)
            self.adwin.set_int_var(10, 0)
            # Stop Proteus
            # self.proteus.stop_sequence()
            # self.proteus.driver._close()

            return {
                'success': True,
                'signal_counts': signal_counts,
                'reference_counts': ref_counts,
                'total_counts': total_counts
            }

        except Exception as e:
            self.logger.error(f"Failed to run sequence and collect data: {e}")
            return {'success': False, 'error': str(e)}

    """

sequence: name=odmr_pulsed, type=odmr, duration=1002500ns, sample_rate=1GHz, repeat_count=50000
variable pulse_duration, start=50ns, stop=500ns, steps=20
marker, laser_int_1 on channel 1 at 0ns, 500ns
pi/2 pulse on channel 1 at 500ns, gaussian, pulse_duration, 1.0
pi/2 pulse on channel 2 at 500ns, gaussian, pulse_duration, 1.0
wait pulse on channel 1 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
wait pulse on channel 2 at pulse_duration+0.000000500, square, 2*pulse_duration, 0.0
pi/2 pulse on channel 1 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
pi/2 pulse on channel 2 at 3*pulse_duration+0.000000500, gaussian, pulse_duration, 1.0
marker, laser_readout_1 on channel 1 at 2500ns, 1ms
marker, readout_counts_1 on channel 2 at 2500ns, 300ns
marker, reference_counts_1 on channel 2 at 1002200ns, 300ns
sequence: name=SCC, type=SCC, duration=1400ns, sample_rate=1GHz, repeat=50000
shelving pulse on channel 3 at 0ns, square, 300ns, 0.6
ionization pulse on channel 3 at 300ns, square, 500ns, 1.0
readout pulse on channel 3 at 800ns, square, 600ns, 0.3
    sequence: name=SCC, type=SCC, duration=1400ns, sample_rate=1GHz, repeat=50000
    shelving pulse on channel 1 at 0ns, square, 300ns, 0.6
    ionization pulse on channel 1 at 300ns, square, 500ns, 1.0
    readout pulse on channel 1 at 800ns, square, 600ns, 0.3"""
    """    
    sequence: name=odmr_pulsed, type=odmr, duration=15500ns, sample_rate=1GHz, repeat_count=50000
    variable pulse_duration, start=0ns, stop=1500ns, steps=20
    var_pulse pulse on channel 2 at 0 ns, square, pulse_duration, 0.0
    initialize pulse on channel 4 at 0ns, square, 2000ns, 1.0
    initialize pulse on channel 3 at 0ns, square, 2000ns, 1.0
    signal pulse on channel 4 at 0.000012 + pulse_duration, square, 300ns, 1.0
    signal pulse on channel 3 at 0.000012 + pulse_duration, square, 300ns, 1.0
    ref pulse on channel 4 at 0.0000134 + pulse_duration, square, 300ns, 1.0
    ref pulse on channel 3 at 0.0000134 + pulse_duration, square, 300ns, 1.0

    "sequence: name=T1, type=T1, duration=504000ns, sample_rate=1GHz, repeat_count=50000
variable pulse_duration, start=100ns, stop=500000ns, steps=20
initialize pulse on channel 4 at 0.0005-pulse_duration, square, 2000ns, 1.0
initialize pulse on channel 3 at 0.0005-pulse_duration, square, 2000ns, 1.0
wait pulse on channel 2 at 0.0005-pulse_duration, square, pulse_duration, 0.0
readout pulse on channel 4 at 502000ns, square, 2000ns, 1.0
readout pulse on channel 3 at 502000ns, square, 2000ns, 1.0"
    """

    def create_example_odmr_sequence(self) -> str:
        """
        This is just for testing, please use a file or GUI setting updates
        Create an example ODMR sequence using the sequence language.

        Returns:
            Sequence text in the sequence language format

        PLEASE NOTE: Readout is fixed so please only vary the starting part not the ending part of the sequence.
        #marker, laser_init_1 on channel 4 at 0ns, 2000ns
        #marker, laser_readout_1 on channel 4 at 2900ns, 2000ns
        for chirps:
        Two-pump sequences (e.g. the standing+moving sech/tanh scheme) want the two pulses swept in opposite directions
        to cancel excitation-time dispersion — do that by setting one pulse's bandwidth negative (it just flips the sweep sign).
        For HS: HS{n_left, n_right} works via offset-independent adiabaticity, which reduces exactly to a symmetric HS at n_left=n_right=1
        HS{1,6} matches what a ref paper found best for short traces; flip to n_left=6, n_right=1 to reverse which side is steep.
        """
        sequence_text = """
sequence: name=rabi, type=rabi, duration=12900ns, sample_rate=1GHz, repeat_count=100000
variable pulse_duration, start=4ns, stop=500ns, steps=64
init pulse on channel 4 at 0ns, square, 2000ns, 1.0
MW pulse on channel 1 at 2900ns, square, pulse_duration, 1.0
marker, reference_MW_1 on channel 1 at 2900ns, pulse_duration
readout pulse on channel 4 at 5900ns, square, 7000ns, 1.0
marker, reference_readout_1 on channel 4 at 5900ns, 7000ns
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

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary of experiment configuration.

        Returns:
            Dictionary with experiment summary
        """
        return {
            'name': 'ODMR Pulsed Experiment',
            'sequence_name': self.sequence_description.name if self.sequence_description else 'None',
            'scan_points': len(self.scan_sequences),
            'microwave_frequency_ghz': self.microwave_frequency / 1e9,
            'microwave_power_dbm': self.microwave_power,
            'green_laser_power_mw': self.green_laser_power,
            'green_laser_wavelength_nm': self.green_laser_wavelength,
            'delays_ns': {
                'mw': self.mw_delay,
                'green laser': self.green_laser_delay,
                'counter': self.counter_delay
            },
            'output_directory': str(self.output_dir)
        }

    def _update(self):
        pass


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
        self.window.title("ODMR Pulsed Sequence Preview")
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


# Example usage and testing
if __name__ == "__main__":
    # seq preview part
    """experiment = ODMRPulsedExperiment(name="test_odmr")

    # Set parameters
    experiment.set_microwave_parameters(2.87e9, -10.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)
    # Create and load example sequence
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    # Build sequences
    if experiment.build_scan_sequences():
        print("Scan sequences built successfully")
    else:
        print("Failed to build scan sequences")
    experiment.show_sequence_preview(10)"""

    # independent testing part
    """experiment = ODMRPulsedExperiment(name="test_odmr")

    # Set parameters
    experiment.set_microwave_parameters(2.87e9, -10.0, 25.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)
    # Create and load example sequence
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "="*50 + "\n")

    if experiment.load_sequence_from_text():
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    # Build sequences
    if experiment.build_scan_sequences():
       print("Scan sequences built successfully")
    else:
        print("Failed to build scan sequences")
    # Generate AWG sequences
    if experiment.generate_awg_sequences_awg_triggering_adwin_case():
    #if experiment.generate_awg_task_sequences_adwin_triggering_awg_case():
       print("AWG sequences generated successfully")
    else:
        print("Failed to generate AWG sequences")"""

    # full experiment part
    experiment = ODMRPulsedExperiment(name="test_odmr")

    # Set parameters
    experiment.set_microwave_parameters(2.73632e9, 12.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)
    # Create and load example sequence
    # Run experiment
    print("Running experiment...")


    def build_calibration_overrides():
        return {
            "green_laser_delay": experiment.settings['delays']['green_laser_delay'],
            "mw_delay": experiment.settings['delays']['mw_delay'],
            "iq_delay": experiment.settings['delays']['iq_delay'],
            "counter_delay": experiment.settings['delays']['counter_delay'],
            "trigger_delay": experiment.settings['delays']['trigger_delay'],
            "units": "ns"
        }


    # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
    overrides = build_calibration_overrides()
    experiment.hardware_calibrator.update_calibration_delays(overrides)

    experiment.microwave_power = experiment.settings['microwave']['power']
    #experiment.settings['microwave']['frequency range'] = [2.84e9, 2.8425e9, 2.845e9, 2.8475e9, 2.85e9, 2.8525e9, 2.855e9, 2.8575e9, 2.86e9, 2.8625e9, 2.865e9, 2.8675e9, 2.87e9, 2.8725e9, 2.875e9, 2.8775e9, 2.88e9, 2.8825e9, 2.885e9, 2.8875e9, 2.89e9, 2.8925e9, 2.895e9, 2.8975e9, 2.9e9]
    experiment.mw_delay = experiment.settings['delays']['mw_delay']
    experiment.green_laser_delay = experiment.settings['delays']['green_laser_delay']
    experiment.counter_delay = experiment.settings['delays']['counter_delay']
    experiment.iq_delay = experiment.settings['delays']['iq_delay']
    experiment.trigger_delay = experiment.settings['delays']['trigger_delay']

    experiment.green_laser_power = experiment.settings['green_laser']['power']
    experiment.green_laser_wavelength = experiment.settings['green_laser']['wavelength']

    # Output paths
    experiment.output_dir = experiment.get_output_dir("odmr_pulsed_output")
    experiment.logger.info("ODMR Pulsed Experiment initialized")

    # Create and load sequence
    source = None
    sequence_loaded = False
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        source = "text"
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    if sequence_loaded:
        experiment.logger.info(f"Sequence loaded successfully from {source}")
    else:
        experiment.logger.info(f"Failed to load sequence from {source}")

    experiment.count_time = 340  # then 390 then 415
    experiment.reset_time = 6320
    experiment.readout_start_ns = 6000
    experiment.trigger_duration = 150
    experiment.gate_align_offset_ns = 1800 #1650
    # Run experiment
    print("running exp")
    experiment.logger.info("Running experiment...")
    results = experiment.run_experiment_averaged(experiment.settings['microwave']['frequency range'])
    experiment.data = results
    experiment.save_hdf5()
    experiment.cleanup()
    if results['success']:
        experiment.logger.info("Experiment completed successfully!")
        experiment.logger.info(f"Frequencies scanned: {[f / 1e9 for f in results['frequencies']]} GHz")
        experiment.logger.info(
            f"Signal counts shape: {len(results['signal_counts'])} frequencies x {len(results['signal_counts'][0])} points")
    else:
        experiment.logger.info(f"Experiment failed: {results['error']}")
    #experiment.show_sequence_preview(10)
    experiment.logger.info("\nODMR Pulsed Experiment ready!")
    """# Multiharp tester:
    experiment = ODMRPulsedExperiment(name="test_odmr", mode="testing")  # drop mode= for a real run
    # Set parameters
    experiment.set_microwave_parameters(2.86041e9, 12.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)


    def build_calibration_overrides():
        return {
            "green_laser_delay": experiment.settings['delays']['green_laser_delay'],
            "mw_delay": experiment.settings['delays']['mw_delay'],
            "iq_delay": experiment.settings['delays']['iq_delay'],
            "counter_delay": experiment.settings['delays']['counter_delay'],
            "trigger_delay": experiment.settings['delays']['trigger_delay'],
            "units": "ns"
        }


    # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
    overrides = build_calibration_overrides()
    experiment.hardware_calibrator.update_calibration_delays(overrides)

    experiment.microwave_power = experiment.settings['microwave']['power']
    experiment.mw_delay = experiment.settings['delays']['mw_delay']
    experiment.green_laser_delay = experiment.settings['delays']['green_laser_delay']
    experiment.counter_delay = experiment.settings['delays']['counter_delay']
    experiment.iq_delay = experiment.settings['delays']['iq_delay']
    experiment.trigger_delay = experiment.settings['delays']['trigger_delay']
    experiment.green_laser_power = experiment.settings['green_laser']['power']
    experiment.green_laser_wavelength = experiment.settings['green_laser']['wavelength']

    # Output paths
    experiment.output_dir = experiment.get_output_dir("odmr_pulsed_output")
    experiment.logger.info("ODMR Pulsed Experiment initialized")
    freqs = experiment.settings['microwave']['frequency range']
    # Create and load sequence
    source = None
    sequence_loaded = False
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        source = "text"
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    if sequence_loaded:
        experiment.logger.info(f"Sequence loaded successfully from {source}")
    else:
        experiment.logger.info(f"Failed to load sequence from {source}")

    experiment.count_time = 340  # then 390 then 415
    experiment.reset_time = 6320
    experiment.readout_start_ns = 6000
    experiment.trigger_duration = 150
    experiment.gate_align_offset_ns = 1800  # 1650
    # Run experiment
    print("running exp")
    experiment.logger.info("Running experiment...")
    from multiharp_timing_tester import MultiHarpTimingTester, ChannelConfig, EDGE_RISING

    tester = MultiHarpTimingTester(
        case="init_vs_trigger",
        start_input=2,  # MKR
        stop_input=0,  # DIO26
        start_cfg=ChannelConfig(level_mV=320, edge=EDGE_RISING),
        stop_cfg=ChannelConfig(level_mV=305, edge=EDGE_RISING),
        window_ns=10000.0,  # look 2 µs after each readout edge (preset default)
        output_mode="both",  # "histogram" is fine too
    )
    out = tester.run_with_experiment(experiment, freqs, run_method="run_experiment")
    experiment.timing_result = out["timing"]
    print(out["timing"].summary())  # prints onset + 10–90% rise time
    results = out["odmr"]
    experiment.data = results
    experiment.save_hdf5()
    experiment.cleanup()"""


    """#Multiharp Rabi:
    experiment = ODMRPulsedExperiment(name="test_odmr", mode="testing")  # drop mode= for a real run
    # Set parameters
    experiment.set_microwave_parameters(2.86041e9, 12.0, 0.0)
    experiment.set_green_laser_parameters(1.0, 532)
    experiment.set_delay_parameters(0.0, 0.0, 0.0)


    def build_calibration_overrides():
        return {
            "green_laser_delay": experiment.settings['delays']['green_laser_delay'],
            "mw_delay": experiment.settings['delays']['mw_delay'],
            "iq_delay": experiment.settings['delays']['iq_delay'],
            "counter_delay": experiment.settings['delays']['counter_delay'],
            "trigger_delay": experiment.settings['delays']['trigger_delay'],
            "units": "ns"
        }


    # this part is for when we adjust the settings in the gui, the calibration changes accordingly: so no need to change the json
    overrides = build_calibration_overrides()
    experiment.hardware_calibrator.update_calibration_delays(overrides)

    experiment.microwave_power = experiment.settings['microwave']['power']
    experiment.mw_delay = experiment.settings['delays']['mw_delay']
    experiment.green_laser_delay = experiment.settings['delays']['green_laser_delay']
    experiment.counter_delay = experiment.settings['delays']['counter_delay']
    experiment.iq_delay = experiment.settings['delays']['iq_delay']
    experiment.trigger_delay = experiment.settings['delays']['trigger_delay']
    experiment.green_laser_power = experiment.settings['green_laser']['power']
    experiment.green_laser_wavelength = experiment.settings['green_laser']['wavelength']

    # Output paths
    experiment.output_dir = experiment.get_output_dir("odmr_pulsed_output")
    experiment.logger.info("ODMR Pulsed Experiment initialized")
    freqs = experiment.settings['microwave']['frequency range']
    # Create and load sequence
    source = None
    sequence_loaded = False
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    print("Example ODMR Sequence:")
    print(experiment.sequence_text)
    print("\n" + "=" * 50 + "\n")

    if experiment.load_sequence_from_text():
        source = "text"
        print("Sequence loaded successfully")
    else:
        print("Failed to load sequence")

    if sequence_loaded:
        experiment.logger.info(f"Sequence loaded successfully from {source}")
    else:
        experiment.logger.info(f"Failed to load sequence from {source}")

    experiment.count_time = 340  # then 390 then 415
    experiment.reset_time = 6320
    experiment.readout_start_ns = 6000
    experiment.trigger_duration = 150
    experiment.gate_align_offset_ns = 1800  # 1650
    # Run experiment
    print("running exp")
    experiment.logger.info("Running experiment...")
    from multiharp_timing_tester import MultiHarpTimingTester, ChannelConfig, EDGE_RISING, EDGE_FALLING
    import numpy as np

    tester = MultiHarpTimingTester(
        case="rabi_mw_tagged", start_input=1, stop_input=4,
        mw_input=2, mw_fall_input=3,
        rabi_sig_offset_ns=950, rabi_sig_width_ns=300,  # onset, past the ~900 ns rise
        rabi_ref_offset_ns=5000, rabi_ref_width_ns=300,
        rabi_tau_values_ns=np.linspace(4, 3000, 64),
        decode_chunk_records=50_000_000,  # lower to ~20M if RAM is tight
    )
    out = tester.run_rabi_with_experiment(experiment, raw_records_path = r"D:\Data\rabi_raw.npy")

    results = out["odmr"]
    experiment.data = results
    experiment.save_hdf5()
    experiment.cleanup()"""

    #1. Run IQCalibrationExperiment  →  writes iq_calibration_results.json
    #2. Run ODMRPulsedExperiment     →  reads iq_calibration_results.json at start
    #                                   applies b≈0.89 to channel 2 automatically
    #3. Re-run calibration any time  →  overwrites the JSON
    #                                   next ODMR run picks up the new values

