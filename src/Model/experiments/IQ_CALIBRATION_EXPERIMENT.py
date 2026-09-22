#!/usr/bin/env python3
# Written by <Jannet Trabelsi>
"""
IQ Calibration Experiment for Pulsed ODMR

This experiment calibrates IQ modulation by testing different baseband waveforms
(I only, Q only, I&Q) and recording spectrum analyzer measurements.
Supports DC, Square, and Gaussian modulation modes.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import os
import logging
import time
import numpy as np
import datetime
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from src.core.experiment import Experiment
from src.core import Parameter
from src.Model.sequence_parser import SequenceTextParser
from src.Model.sequence_builder import SequenceBuilder
from src.Model.proteus_hardware_calibrator import ProteusHardwareCalibrator
from src.Model.sequence import Sequence
from src.Controller.Proteus_device import ProteusDevice
from src.Controller.adwin_gold import AdwinGoldDevice
from src.Controller.mux_control import MUXControlDevice
from src.Controller import SG384Generator
from src.Controller.Agilent8596E import Agilent8596E
from src.core.struct_hdf5 import MyStruct, save_data, StructArray
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # or 'TkAgg' if you want interactive windows


class IQCalibrationExperiment(Experiment):
    """
    IQ Calibration Experiment for testing IQ modulator performance.
    
    Tests different baseband waveforms (I only, Q only, I&Q) with different
    modulation modes (DC, Square, Gaussian) and records spectrum analyzer data.
    """
    
    _DEFAULT_SETTINGS = [
        Parameter('iq_calibration', [
            Parameter('mode', 'DC', str, 'Modulation mode: DC, SQUARE, GAUSSIAN'),
            Parameter('amplitude', 1.0, float, 'Baseband amplitude (0-1.0)'),
            Parameter('pulse_duration', 500, float, 'Pulse duration in ns', units='ns'),
            Parameter('amplitude_sweep', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], list,
                      'List of amplitudes to sweep'),
            Parameter('pulse_on_time', 64, float, 'Square pulse ON time in ns', units='ns'),
            Parameter('pulse_off_time', 64, float, 'Square pulse OFF time in ns', units='ns'),
            Parameter('sideband_offset', 10e6, float, 'Sideband offset from LO in Hz', units='Hz'),
            Parameter('sideband_span', 4e6, float, 'SA span for sideband meas (must not show carrier)', units='Hz'),
            Parameter('sideband_rbw', 50e3, float, 'SA RBW for sideband measurement', units='Hz'),
            Parameter('iq_ratio_sweep', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], list,
                      'Q/I amplitude ratios to test (I fixed at 0.98)'),
            Parameter('phase_angles', [0.0, 90.0, 180.0, 270.0], list, 'IQ phase angles to test in degrees'),
            Parameter('dc_offset_sweep', [-0.10, -0.07, -0.05, -0.02, 0.0, 0.02, 0.05, 0.07, 0.10], list,
                      'DC offset deltas ofset to sweep for LO null'),
            Parameter('fg_amplitude', 0.98, float, 'Function generator amplitude'),
            Parameter('fg_frequency', 10e6, float, 'Function generator baseband frequency in Hz', units='Hz'),
            Parameter('q_amplitude_correction', {
                2.7e9: 0.90,  # from ampl_imbalance: best at ratio≈0.90
                2.87e9: 0.90,  # best at ratio≈0.90
                3.0e9: 0.80,  # best at ratio≈0.80
            }, dict, 'Q correction per frequency'),
        ]),
        Parameter('microwave', [
            Parameter('frequency_range', [2.7e9, 2.8e9, 2.87e9, 2.9e9, 3.0e9], list, 
                     'Microwave frequencies in Hz', units='Hz'),
            Parameter('power', 10.0, float, 'Microwave power in dBm', units='dBm'),
        ]),
        Parameter('spectrum_analyzer', [
            Parameter('initial_span', 10e6, float, 'Initial span in Hz', units='Hz'),
            Parameter('initial_rbw', 100e3, float, 'Initial resolution BW in Hz', units='Hz'),
            Parameter('min_peak_height', -60, float, 'Minimum peak height in dBm', units='dBm'),
            Parameter('max_iterations', 5, int, 'Maximum iterations for span/RBW adjustment'),
            Parameter('target_peak_points', 20, int, 'Target number of points across peak for FWHM'),
        ]),
        Parameter('sequence', [
            Parameter('text', """sequence: name=iq_test, type=calibration, duration=2000ns, sample_rate=1GHz, repeat_count=1\n
            variable pulse_duration, start=50ns, stop=50ns, steps=1\n
square pulse on channel 1 at 0ns, square, 500ns, 1.0\n
square pulse on channel 2 at 0ns, square, 500ns, 0\n
debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0""", str, 'Sequence text'),
        ]),
        Parameter('delays', [
            Parameter('iq_delay', 30.0, float, 'IQ delay in ns', units='ns'),
            Parameter('mw_delay', 25.0, float, 'Microwave delay in ns', units='ns'),
        ]),
        Parameter('output', [
            Parameter('output_directory', 'iq_calibration_output', str, 'Output directory'),
            Parameter('save_traces', True, bool, 'Save full spectrum traces'),
        ]),
        Parameter('path', "D:\\Data\\iq_calibration"),
        Parameter('filename', "iq_calibration"),
        Parameter('tag', "iq_calibration_experiment"),
        Parameter('save', False)
    ]
    
    _DEVICES = {
        'proteus': 'proteus',
        'sg384': 'sg384',
        'spectrum_analyzer': 'spectrum_analyzer',
        'adwin': 'adwin',
        'mux_control': 'mux_control'
    }
    
    _EXPERIMENTS = {}
    
    def __init__(self, devices=None, experiments=None, name=None, settings=None, 
                 log_function=None, data_path=None, config_path: Optional[Path] = None):
        """Initialize the IQ Calibration experiment."""
        super().__init__(name=name, settings=settings, devices=devices, 
                        sub_experiments=experiments, log_function=log_function, 
                        data_path=data_path)
        
        self.logger = logging.getLogger(__name__)
        self.repeat_count = None
        
        # Initialize devices
        self.proteus = ProteusDevice()
        self.sg384 = SG384Generator()
        self.mux = MUXControlDevice()
        self.spectrum_analyzer = Agilent8596E()
        
        # Sequence components
        self.sequence_parser = SequenceTextParser()
        self.sequence_builder = SequenceBuilder()
        
        # Hardware calibrator
        connection_file = Path(__file__).parent / "odmr_pulsed_connection.json"
        self.hardware_calibrator = ProteusHardwareCalibrator(connection_file=str(connection_file))
        
        # Data storage
        self.calibration_results = []
        
        # Select trigger mode
        self.mux.select_trigger('pulsed')
        
        # For HDF5 saving
        self.s_t = None
        self.e_t = None
        self.data = None

    def _measure_sideband_powers(self, lo_freq_hz: float) -> Dict[str, float]:
        """
        Measure USB (LO + offset) and LSB (LO - offset) peak powers.
        Span is set narrowly so the carrier is NOT in view.
        Returns dict with keys: usb_dbm, lsb_dbm, irr_db.
        """
        offset_hz = self.settings['iq_calibration']['sideband_offset']
        span_mhz = self.settings['iq_calibration']['sideband_span'] / 1e6
        rbw_mhz = self.settings['iq_calibration']['sideband_rbw'] / 1e6

        results = {}
        for label, center_hz in [('usb', lo_freq_hz + offset_hz),
                                 ('lsb', lo_freq_hz - offset_hz)]:
            self.spectrum_analyzer.set_center_frequency(center_hz / 1e6)
            self.spectrum_analyzer.set_span(span_mhz)
            self.spectrum_analyzer.set_resolution_bw(rbw_mhz)
            self.spectrum_analyzer.set_single_sweep()
            self.spectrum_analyzer.take_sweep()
            time.sleep(0.4)
            _, powers = self._get_trace_from_sa()
            results[f'{label}_dbm'] = float(np.max(powers))

        irr = results['usb_dbm'] - results['lsb_dbm']  # positive = USB is stronger (good)
        results['irr_db'] = irr
        return results

    def _setup_sine_iq(self, i_phase_deg: float, q_phase_deg: float,
                       amplitude: float = 0.98, offset: float = 0.019):
        """
        Configure Proteus function generator for continuous IQ sine output.
        Channels 1 (I) and 2 (Q).
        """
        freq = self.settings['iq_calibration']['fg_frequency']
        # Clamp to safe range
        amplitude = min(amplitude, 1.0)
        self.proteus.set_function_generator(1, 'SIN', freq, amplitude, i_phase_deg, offset)
        self.proteus.set_function_generator(2, 'SIN', freq, amplitude, q_phase_deg, offset)
        time.sleep(0.2)

    def measure_sideband_suppression(self, freq_hz: float) -> Dict[str, Any]:
        """
        For each amplitude in amplitude_sweep, upload a 10 MHz square baseband
        and measure USB, LSB, and IRR for I_ONLY, Q_ONLY, I_AND_Q.
        """
        amplitudes = self.settings['iq_calibration']['amplitude_sweep']
        data = {m: {'amplitudes': [], 'usb': [], 'lsb': [], 'irr': []}
                for m in ('I_ONLY', 'Q_ONLY', 'I_AND_Q')}

        self.sg384.set_frequency(freq_hz)
        self.sg384.set_power(10.0)  # always 10 dBm
        self.sg384.enable_output()

        for amp in amplitudes:
            for mode in ('I_ONLY', 'Q_ONLY', 'I_AND_Q'):
                i_amp = amp if mode in ('I_ONLY', 'I_AND_Q') else 0.0
                q_amp = amp if mode in ('Q_ONLY', 'I_AND_Q') else 0.0
                seq = self._get_square_cycle_sequence(mode, i_amp, q_amp)
                ok, seqs = self._load_and_build_sequence(seq)
                if not ok:
                    print(f"  [SIDEBAND] Sequence build FAILED for mode={mode} amp={amp:.2f}")
                    continue
                self.scan_sequences = seqs
                self.generate_awg_sequences_awg_triggering_adwin_case()
                time.sleep(0.3)

                sb = self._measure_sideband_powers(freq_hz)
                data[mode]['amplitudes'].append(amp)
                data[mode]['usb'].append(sb['usb_dbm'])
                data[mode]['lsb'].append(sb['lsb_dbm'])
                data[mode]['irr'].append(sb['irr_db'])
                print(f"  [{mode}] amp={amp:.2f}  USB={sb['usb_dbm']:.1f} dBm  "
                      f"LSB={sb['lsb_dbm']:.1f} dBm  IRR={sb['irr_db']:.1f} dB")

        return data

    def measure_amplitude_imbalance(self, freq_hz: float) -> Dict[str, Any]:
        """
        Fix I amplitude at fg_amplitude, sweep Q by iq_ratio_sweep factors.
        Ideal ratio = 1.0 (maximum IRR).
        """
        ratios = self.settings['iq_calibration']['iq_ratio_sweep']
        i_amp = self.settings['iq_calibration']['fg_amplitude']
        data = {'ratios': [], 'q_amps': [], 'usb': [], 'lsb': [], 'irr': []}

        self.sg384.set_frequency(freq_hz)
        self.sg384.set_power(10.0)
        self.sg384.enable_output()

        for ratio in ratios:
            q_amp = min(i_amp * ratio, 0.98)  # hard cap at 0.98
            # Standard IQ: I is reference (0°), Q leads by 90°
            self._setup_sine_iq(i_phase_deg=0.0, q_phase_deg=90.0,
                                amplitude=i_amp)  # I amp fixed
            # Override Q amplitude separately
            freq = self.settings['iq_calibration']['fg_frequency']
            self.proteus.set_function_generator(2, 'SIN', freq, q_amp, 90.0, 0.019)
            time.sleep(0.2)

            sb = self._measure_sideband_powers(freq_hz)
            data['ratios'].append(ratio)
            data['q_amps'].append(q_amp)
            data['usb'].append(sb['usb_dbm'])
            data['lsb'].append(sb['lsb_dbm'])
            data['irr'].append(sb['irr_db'])
            print(f"  ratio={ratio:.2f}  Q_amp={q_amp:.3f}  IRR={sb['irr_db']:.1f} dB")

        return data

    def measure_phase_accuracy(self, freq_hz: float) -> Dict[str, Any]:
        """
        Generate I=sin(θ), Q=sin(θ+90°) for each θ via function generator.
        Measures USB, LSB, IRR — constant IRR across θ means good phase accuracy.
        """
        angles = self.settings['iq_calibration']['phase_angles']
        amp = self.settings['iq_calibration']['fg_amplitude']
        data = {'theta_deg': [], 'usb': [], 'lsb': [], 'irr': []}

        self.sg384.set_frequency(freq_hz)
        self.sg384.set_power(10.0)
        self.sg384.enable_output()

        for theta in angles:
            i_phase = theta
            q_phase = theta + 90.0  # ideal quadrature
            self._setup_sine_iq(i_phase, q_phase, amp)
            time.sleep(0.2)

            sb = self._measure_sideband_powers(freq_hz)
            data['theta_deg'].append(theta)
            data['usb'].append(sb['usb_dbm'])
            data['lsb'].append(sb['lsb_dbm'])
            data['irr'].append(sb['irr_db'])
            print(f"  θ={theta:5.1f}°  USB={sb['usb_dbm']:.1f} dBm  "
                  f"LSB={sb['lsb_dbm']:.1f} dBm  IRR={sb['irr_db']:.1f} dB")

        return data

    def measure_dc_offset_null(self, freq_hz: float) -> Dict[str, Any]:
        """
        Sweep DC offset on I and Q independently to find the null that
        minimizes LO leakage. Uses function generator with offset parameter.
        """
        offsets = self.settings['iq_calibration']['dc_offset_sweep']
        amp = self.settings['iq_calibration']['fg_amplitude']
        freq = self.settings['iq_calibration']['fg_frequency']
        base_off = 0.019
        span_mhz = 1.0
        rbw_mhz = 0.01

        self.sg384.set_frequency(freq_hz)
        self.sg384.set_power(10.0)
        self.sg384.enable_output()

        def _measure_carrier(i_off, q_off):
            self.proteus.set_function_generator(1, 'SIN', freq, amp, 0.0, i_off)
            self.proteus.set_function_generator(2, 'SIN', freq, amp, 90.0, q_off)
            time.sleep(0.2)
            self.spectrum_analyzer.set_center_frequency(freq_hz / 1e6)
            self.spectrum_analyzer.set_span(span_mhz)
            self.spectrum_analyzer.set_resolution_bw(rbw_mhz)
            self.spectrum_analyzer.set_single_sweep()
            self.spectrum_analyzer.take_sweep()
            time.sleep(0.5)
            _, powers = self._get_trace_from_sa()
            return float(np.max(powers))

        data = {'i_offsets': [], 'carrier_vs_i': [],
                'q_offsets': [], 'carrier_vs_q': []}

        # Sweep I offset, Q fixed at base
        for delta in offsets:
            i_off = base_off + delta
            power = _measure_carrier(i_off, base_off)
            data['i_offsets'].append(i_off)
            data['carrier_vs_i'].append(power)
            print(f"  I_offset={i_off:.3f}  carrier={power:.1f} dBm")

        # Sweep Q offset, I fixed at base
        for delta in offsets:
            q_off = base_off + delta
            power = _measure_carrier(base_off, q_off)
            data['q_offsets'].append(q_off)
            data['carrier_vs_q'].append(power)
            print(f"  Q_offset={q_off:.3f}  carrier={power:.1f} dBm")

        return data

    def compute_correction_factors(self, imbalance_data: Dict, phase_data: Dict,
                                   dc_data: Dict, freq_hz: float) -> Dict[str, Any]:
        """
        Derive amplitude correction (a, b), phase correction phi, and DC offsets.

        Amplitude correction:  I_corrected = I_raw * a,  Q_corrected = Q_raw * b
        Phase correction:      Q_corrected = I_raw*sin(phi) + Q_raw*cos(phi)
        DC offset correction:  add I_dc, Q_dc to null LO leakage
        """
        corrections = {'frequency_hz': freq_hz}

        # --- Amplitude correction ---
        irr_arr = np.array(imbalance_data['irr'])
        ratio_arr = np.array(imbalance_data['ratios'])
        best_idx = int(np.argmin(irr_arr))
        best_ratio = ratio_arr[best_idx]
        # a=1 by convention; b scales Q to match I
        corrections['amplitude'] = {
            'a': 1.0,
            'b': best_ratio,  # Q_corrected = Q_raw * best_ratio
            'best_irr_db': float(irr_arr[best_idx]),
            'best_q_to_i_ratio': float(best_ratio)
        }

        # --- Phase correction ---
        # IRR should be flat vs theta; deviation reveals phase error.
        # Find worst IRR point → its theta offset from the best tells us phi.
        irr_phase = np.array(phase_data['irr'])
        theta_arr = np.array(phase_data['theta_deg'])
        best_phase_idx = int(np.argmax(irr_phase))
        worst_phase_idx = int(np.argmin(irr_phase))
        # Phase error is roughly half the spread between best and worst theta
        phi_deg = float((theta_arr[worst_phase_idx] - theta_arr[best_phase_idx]) / 2.0)
        phi_rad = np.deg2rad(phi_deg)
        corrections['phase'] = {
            'phi_deg': phi_deg,
            'phi_rad': float(phi_rad),
            'irr_spread_db': float(irr_phase.max() - irr_phase.min()),
            # Apply as: Q_corr = I_raw*sin(phi) + Q_raw*cos(phi)
            'sin_phi': float(np.sin(phi_rad)),
            'cos_phi': float(np.cos(phi_rad))
        }

        # --- DC offset ---
        i_offs = np.array(dc_data['i_offsets'])
        c_vs_i = np.array(dc_data['carrier_vs_i'])
        q_offs = np.array(dc_data['q_offsets'])
        c_vs_q = np.array(dc_data['carrier_vs_q'])
        corrections['dc_offset'] = {
            'i_offset': float(i_offs[int(np.argmin(c_vs_i))]),
            'q_offset': float(q_offs[int(np.argmin(c_vs_q))]),
            'min_i_leakage_dbm': float(c_vs_i.min()),
            'min_q_leakage_dbm': float(c_vs_q.min())
        }

        print(f"\n  Amplitude correction: b={corrections['amplitude']['b']:.4f}  "
              f"(best IRR={corrections['amplitude']['best_irr_db']:.1f} dB)")
        print(f"  Phase correction:    φ={corrections['phase']['phi_deg']:.2f}°")
        print(f"  DC offset:           I={corrections['dc_offset']['i_offset']:.4f}  "
              f"Q={corrections['dc_offset']['q_offset']:.4f}")

        return corrections

    def apply_corrections(self, i_raw: np.ndarray, q_raw: np.ndarray,
                          corrections: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply amplitude, phase, and DC corrections to raw IQ arrays."""
        a = corrections['amplitude']['a']
        b = corrections['amplitude']['b']
        sp = corrections['phase']['sin_phi']
        cp = corrections['phase']['cos_phi']
        i_dc = corrections['dc_offset']['i_offset'] - 0.019  # delta from base
        q_dc = corrections['dc_offset']['q_offset'] - 0.019

        i_corr = i_raw * a + i_dc
        q_corr = i_raw * sp + q_raw * cp * b + q_dc
        return i_corr, q_corr

    def _plot_sideband_suppression(self, freq_hz: float, data: Dict, out_dir: str):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = {'I_ONLY': 'blue', 'Q_ONLY': 'red', 'I_AND_Q': 'green'}
        for mode in ('I_ONLY', 'Q_ONLY', 'I_AND_Q'):
            d = data[mode]
            axes[0].plot(d['amplitudes'], d['usb'], 'o', linestyle='none', color=colors[mode], label=f'{mode} USB')
            axes[0].plot(d['amplitudes'], d['lsb'], 's', linestyle='none', color=colors[mode], alpha=0.5,
                         label=f'{mode} LSB')
            axes[1].plot(d['amplitudes'], d['irr'], 'o', linestyle='none', color=colors[mode], label=mode)
        axes[0].set(xlabel='Amplitude', ylabel='Power (dBm)', title='USB & LSB vs Amplitude')
        axes[1].set(xlabel='Amplitude', ylabel='IRR (dB)', title='Image Rejection Ratio')
        for ax in axes:
            ax.legend(fontsize=7);
            ax.grid(True)
        fig.suptitle(f'Sideband Suppression — {freq_hz / 1e9:.4f} GHz')
        fig.savefig(os.path.join(out_dir, f'sideband_{freq_hz / 1e9:.4f}GHz.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_amplitude_imbalance(self, freq_hz: float, data: Dict, out_dir: str):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(data['ratios'], data['irr'], 'o', linestyle='none', color='purple')
        best_idx = int(np.argmin(data['irr']))
        ax.axvline(data['ratios'][best_idx], color='purple', linestyle='none',
                   label=f"Best ratio={data['ratios'][best_idx]:.2f}")
        ax.set(xlabel='Q/I Amplitude Ratio', ylabel='IRR (dB)',
               title=f'Amplitude Imbalance — {freq_hz / 1e9:.4f} GHz')
        ax.legend();
        ax.grid(True)
        fig.savefig(os.path.join(out_dir, f'ampl_imbalance_{freq_hz / 1e9:.4f}GHz.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_phase_accuracy(self, freq_hz: float, data: Dict, out_dir: str):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].plot(data['theta_deg'], data['usb'], 'bo', linestyle='none',label='USB')
        axes[0].plot(data['theta_deg'], data['lsb'], 'rs', linestyle='none',label='LSB')
        axes[0].set(xlabel='θ (°)', ylabel='Power (dBm)', title='Sideband Power vs IQ Phase')
        axes[1].plot(data['theta_deg'], data['irr'], 'go',linestyle='none')
        axes[1].set(xlabel='θ (°)', ylabel='IRR (dB)', title='IRR vs IQ Phase')
        for ax in axes:
            ax.legend();
            ax.grid(True)
            ax.set_xticks(data['theta_deg'])
        fig.suptitle(f'Phase Accuracy — {freq_hz / 1e9:.4f} GHz')
        fig.savefig(os.path.join(out_dir, f'phase_accuracy_{freq_hz / 1e9:.4f}GHz.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_dc_offset(self, freq_hz: float, data: Dict, out_dir: str):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].plot(data['i_offsets'], data['carrier_vs_i'], 'bo', linestyle='none')
        axes[0].set(xlabel='I offset (V)', ylabel='Carrier (dBm)', title='LO Leakage vs I offset')
        axes[1].plot(data['q_offsets'], data['carrier_vs_q'], 'ro', linestyle='none')
        axes[1].set(xlabel='Q offset (V)', ylabel='Carrier (dBm)', title='LO Leakage vs Q offset')
        for ax in axes:
            ax.grid(True)
        fig.suptitle(f'DC Offset Null — {freq_hz / 1e9:.4f} GHz')
        fig.savefig(os.path.join(out_dir, f'dc_offset_{freq_hz / 1e9:.4f}GHz.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _get_square_cycle_sequence(self, mode: str, i_amplitude: float, q_amplitude: float) -> str:
        on_ns = self.settings['iq_calibration']['pulse_on_time']
        off_ns = self.settings['iq_calibration']['pulse_off_time']
        n_cycles = 64  # 64 × 128 ns = 8192 ns, well above Proteus minimum
        total_ns = (on_ns + off_ns) * n_cycles
        i_raw = i_amplitude if mode in ('I_ONLY', 'I_AND_Q') else 0.0
        q_raw = q_amplitude if mode in ('Q_ONLY', 'I_AND_Q') else 0.0
        i_amp, q_amp = self._correct_amplitudes(i_raw, q_raw)
        return (
            f"sequence: name=iq_square_cycle, type=calibration, duration={total_ns}ns, sample_rate=1GHz, repeat_count=20000\n"
            f"variable pulse_duration, start=50ns, stop=50ns, steps=1\n"
            f"debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0\n"
            f"square pulse on channel 1 at 0ns, square, {on_ns}ns, {i_amp}\n"
            f"square pulse on channel 1 at {on_ns}ns, square, {on_ns}ns, {0.0}\n"
            f"square pulse on channel 2 at 0ns, square, {on_ns}ns, {q_amp}\n"
            f"square pulse on channel 2 at {on_ns}ns, square, {on_ns}ns, {0.0}\n"
        )

    def _sweep_amplitude_at_frequency(self, freq_hz: float, modulation_mode: str,
                                      sequence_fn) -> Dict[str, Any]:
        """
        Sweep amplitude for I_ONLY, Q_ONLY, I_AND_Q at a fixed frequency.
        sequence_fn(mode, i_amp, q_amp) -> sequence_text string.
        Returns dict keyed by mode, each value is a list of (amplitude, peak_power_dbm).
        """
        amplitudes = self.settings['iq_calibration']['amplitude_sweep']
        sweep = {'I_ONLY': [], 'Q_ONLY': [], 'I_AND_Q': []}

        for amp in amplitudes:
            for mode in ('I_ONLY', 'Q_ONLY', 'I_AND_Q'):
                i_amp = amp if mode in ('I_ONLY', 'I_AND_Q') else 0.0
                q_amp = amp if mode in ('Q_ONLY', 'I_AND_Q') else 0.0
                seq_text = sequence_fn(mode, i_amp, q_amp)
                result = self._run_single_test(freq_hz, mode, seq_text)
                power = (result['spectrum'].get('peak_amplitude', -100)
                         if result.get('success') and 'spectrum' in result else -100)
                sweep[mode].append((amp, power))
                print(f"    [{mode}] amp={amp:.2f} → {power:.1f} dBm")

        return sweep

    def _plot_amplitude_sweep(self, freq_hz: float, sweep: Dict, modulation_mode: str,
                              extra_axes=None, fig=None):
        """
        Plot peak power vs amplitude for I, Q, I&Q on one axes.
        If extra_axes is given (for carrier leakage), uses a two-subplot figure.
        Returns (fig, ax_main).
        """
        if fig is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            ax = extra_axes

        colors = {'I_ONLY': 'blue', 'Q_ONLY': 'red', 'I_AND_Q': 'green'}
        labels = {'I_ONLY': 'I only', 'Q_ONLY': 'Q only', 'I_AND_Q': 'I & Q'}

        for mode in ('I_ONLY', 'Q_ONLY', 'I_AND_Q'):
            amps, powers = zip(*sweep[mode])
            ax.plot(amps, powers, marker='o', linestyle = 'none', color=colors[mode], label=labels[mode])

        ax.set_xlabel('Proteus Amplitude (0–1)')
        ax.set_ylabel('SA Peak Power (dBm)')
        ax.set_title(f'{modulation_mode} — {freq_hz / 1e9:.4f} GHz — Sideband Power vs Amplitude')
        ax.legend()
        ax.grid(True)
        return fig, ax

    def _sweep_carrier_leakage_at_frequency(self, freq_hz: float,
                                            sequence_fn) -> Dict[str, Any]:
        """
        For each amplitude, measure power AT the LO (carrier) frequency — this is
        the IQ imbalance / DC offset leakage visible when baseband is pulsed.
        """
        amplitudes = self.settings['iq_calibration']['amplitude_sweep']
        leakage = {'I_ONLY': [], 'Q_ONLY': [], 'I_AND_Q': []}

        sa_settings = self.settings['spectrum_analyzer']
        center_mhz = freq_hz / 1e6
        # Narrow span around the carrier, tight RBW
        span_mhz = 1.0
        rbw_mhz = 0.01

        for amp in amplitudes:
            for mode in ('I_ONLY', 'Q_ONLY', 'I_AND_Q'):
                i_amp = amp if mode in ('I_ONLY', 'I_AND_Q') else 0.0
                q_amp = amp if mode in ('Q_ONLY', 'I_AND_Q') else 0.0
                seq_text = sequence_fn(mode, i_amp, q_amp)

                # Upload waveform (reuse _run_single_test machinery but override SA)
                success, scan_seqs = self._load_and_build_sequence(seq_text)
                if not success:
                    leakage[mode].append((amp, -100))
                    continue
                self.scan_sequences = scan_seqs
                self.generate_awg_sequences_awg_triggering_adwin_case()
                self.sg384.set_frequency(freq_hz)
                self.sg384.set_power(self.settings['microwave']['power'])
                self.sg384.enable_output()
                time.sleep(0.3)

                # Measure carrier leakage
                self.spectrum_analyzer.set_center_frequency(center_mhz)
                self.spectrum_analyzer.set_span(span_mhz)
                self.spectrum_analyzer.set_resolution_bw(rbw_mhz)
                self.spectrum_analyzer.set_single_sweep()
                self.spectrum_analyzer.take_sweep()
                time.sleep(0.5)
                freqs, powers = self._get_trace_from_sa()
                carrier_power = float(np.max(powers))  # power at/near carrier
                leakage[mode].append((amp, carrier_power))
                print(f"    [{mode}] amp={amp:.2f} → carrier leakage {carrier_power:.1f} dBm")

        return leakage

    def _plot_carrier_leakage(self, freq_hz: float, leakage: Dict, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))
        colors = {'I_ONLY': 'blue', 'Q_ONLY': 'red', 'I_AND_Q': 'green'}
        labels = {'I_ONLY': 'I only', 'Q_ONLY': 'Q only', 'I_AND_Q': 'I & Q'}
        for mode in ('I_ONLY', 'Q_ONLY', 'I_AND_Q'):
            amps, powers = zip(*leakage[mode])
            ax.plot(amps, powers, marker='s', linestyle='none',
                    color=colors[mode], label=labels[mode])
        ax.set_xlabel('Proteus Amplitude (0–1)')
        ax.set_ylabel('Carrier Leakage (dBm)')
        ax.set_title(f'SQUARE — {freq_hz / 1e9:.4f} GHz — LO Leakage vs Amplitude')
        ax.legend()
        ax.grid(True)
    
    def _get_trace_from_sa(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frequency and amplitude trace from spectrum analyzer.
        
        Returns:
            Tuple of (frequencies_Hz, amplitudes_dBm)
        """
        # Set trace data format to real numbers (P format)
        self.spectrum_analyzer.set_trace_data_transfer("P")
        
        # Get trace data in dBm
        trace_amplitudes = np.array(self.spectrum_analyzer.get_trace_a_data("P"))
        
        # Calculate frequency points
        center_freq_mhz = self.spectrum_analyzer.get_center_frequency()
        span_mhz = self.spectrum_analyzer.get_span()

        start_freq_mhz = center_freq_mhz - (span_mhz / 2)
        stop_freq_mhz = center_freq_mhz + (span_mhz / 2)
        
        num_points = len(trace_amplitudes)
        frequencies_mhz = np.linspace(start_freq_mhz, stop_freq_mhz, num_points)
        
        # Convert to Hz
        frequencies_hz = frequencies_mhz * 1e6
        
        return frequencies_hz, trace_amplitudes
    
    def _get_sequence_for_mode(self, mode: str, i_amplitude: float, q_amplitude: float, 
                                pulse_duration: float) -> str:
        """
        Generate sequence text for a specific modulation mode.
        
        Args:
            mode: 'I_ONLY', 'Q_ONLY', or 'I_AND_Q'
            i_amplitude: I channel amplitude (0-1.0)
            q_amplitude: Q channel amplitude (0-1.0)
            pulse_duration: Pulse duration in ns

        Returns:
            Sequence text in the sequence language format
        """
        if mode == "I_ONLY":
            sequence_text = f"""sequence: name=iq_test_i_only, type=calibration, duration={pulse_duration + 100}ns, sample_rate=1GHz, repeat_count=1\n
variable pulse_duration, start=50ns, stop=50ns, steps=1\n
square pulse on channel 1 at {0}ns, square, {pulse_duration}ns, {i_amplitude}\n
square pulse on channel 2 at {0}ns, square, {pulse_duration}ns, 0.0\n
square pulse on channel 1 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}\n
debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0"""
            
        elif mode == "Q_ONLY":
            sequence_text = f"""sequence: name=iq_test_q_only, type=calibration, duration={pulse_duration + 100}ns, sample_rate=1GHz, repeat_count=1\n
variable pulse_duration, start=50ns, stop=50ns, steps=1\n
square pulse on channel 1 at {0}ns, square, {pulse_duration}ns, 0.0\n
square pulse on channel 2 at {0}ns, square, {pulse_duration}ns, {q_amplitude}\n
square pulse on channel 2 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}\n
debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0"""
            
        elif mode == "I_AND_Q":
            sequence_text = f"""sequence: name=iq_test_both, type=calibration, duration={pulse_duration + 100}ns, sample_rate=1GHz, repeat_count=1\n
variable pulse_duration, start=50ns, stop=50ns, steps=1\n
square pulse on channel 1 at {0}ns, square, {pulse_duration}ns, {i_amplitude}\n
square pulse on channel 2 at {0}ns, square, {pulse_duration}ns, {q_amplitude}\n
square pulse on channel 1 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}\n
square pulse on channel 2 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}\n
debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0"""
        else:
            raise ValueError(f"Unknown mode: {mode}")
        print(sequence_text)
        return sequence_text
    
    def _get_gaussian_sequence(self, mode: str, i_amplitude: float, q_amplitude: float,
                                pulse_duration: float) -> str:
        """
        Generate sequence text with Gaussian pulses for a specific mode.
        
        Args:
            mode: 'I_ONLY', 'Q_ONLY', or 'I_AND_Q'
            i_amplitude: I channel amplitude (0-1.0)
            q_amplitude: Q channel amplitude (0-1.0)
            pulse_duration: Pulse duration in ns

        Returns:
            Sequence text with Gaussian pulses
        """
        if mode == "I_ONLY":
            sequence_text = f"""sequence: name=iq_test_i_only_gaussian, type=calibration, duration={pulse_duration + 100}ns, sample_rate=1GHz, repeat_count=1\n
            variable pulse_duration, start=50ns, stop=50ns, steps=1\n
    debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0\n
gaussian pulse on channel 1 at {0}ns, gaussian, {pulse_duration}ns, {i_amplitude}\n
square pulse on channel 1 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}\n
square pulse on channel 2 at {0}ns, square, {pulse_duration}ns, 0.0"""
            
        elif mode == "Q_ONLY":
            _, q_amplitude = self._correct_amplitudes(0.0, q_amplitude)
            sequence_text = f"""sequence: name=iq_test_q_only_gaussian, type=calibration, duration={pulse_duration + 100}ns, sample_rate=1GHz, repeat_count=1\n
variable pulse_duration, start=50ns, stop=50ns, steps=1\n
debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0\n
square pulse on channel 1 at {0}ns, square, {pulse_duration}ns, 0.0\n
gaussian pulse on channel 2 at {0}ns, gaussian, {pulse_duration}ns, {q_amplitude}\n
square pulse on channel 2 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}"""
            
        elif mode == "I_AND_Q":
            _, q_amplitude = self._correct_amplitudes(0.0, q_amplitude)
            sequence_text = f"""sequence: name=iq_test_both_gaussian, type=calibration, duration={pulse_duration + 100}ns, sample_rate=1GHz, repeat_count=1\n
variable pulse_duration, start=50ns, stop=50ns, steps=1\n
debug pulse on channel 3 at 0ns, square, pulse_duration, 1.0\n
gaussian pulse on channel 1 at {0}ns, gaussian, {pulse_duration}ns, {i_amplitude}\n
gaussian pulse on channel 2 at {0}ns, gaussian, {pulse_duration}ns, {q_amplitude}\n
square pulse on channel 1 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}\n
square pulse on channel 2 at {pulse_duration}ns, square, {pulse_duration}ns, {0.0}\n"""
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return sequence_text
    
    def _get_square_sequence(self, mode: str, i_amplitude: float, q_amplitude: float,
                              pulse_duration: float) -> str:
        """Alias for _get_sequence_for_mode - keeps square wave shape"""
        i_corr, q_corr = self._correct_amplitudes(i_amplitude, q_amplitude)
        return self._get_sequence_for_mode(mode, i_corr, q_corr, pulse_duration)
    
    def _calculate_fwhm(self, frequencies: np.ndarray, powers: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Full Width at Half Maximum (FWHM) from spectrum data.
        
        Args:
            frequencies: Frequency values in Hz
            powers: Power values in dBm
            
        Returns:
            Dictionary containing peak information
        """
        min_height = self.settings['spectrum_analyzer']['min_peak_height']
        peaks, properties = find_peaks(powers, height=min_height, prominence=5)
        
        if len(peaks) == 0:
            return {
                'success': False,
                'error': 'No peaks found',
                'peak_frequency': None,
                'peak_amplitude': None,
                'fwhm': None
            }
        
        # Get the highest peak
        highest_peak_idx = peaks[np.argmax(powers[peaks])]
        peak_freq = frequencies[highest_peak_idx]
        peak_amplitude = powers[highest_peak_idx]
        
        # Calculate FWHM
        half_max = peak_amplitude - 3.0103  # 3dB down from peak (half power)
        
        # Find left and right crossing points
        left_indices = np.where(powers[:highest_peak_idx] <= half_max)[0]
        right_indices = np.where(powers[highest_peak_idx:] <= half_max)[0]
        
        if len(left_indices) > 0 and len(right_indices) > 0:
            left_freq = frequencies[left_indices[-1]]
            right_freq = frequencies[highest_peak_idx + right_indices[0]]
            fwhm = right_freq - left_freq
        else:
            # Fit Gaussian to estimate FWHM if direct crossing fails
            try:
                start_idx = max(0, highest_peak_idx - 20)
                end_idx = min(len(frequencies), highest_peak_idx + 20)
                freq_region = frequencies[start_idx:end_idx]
                power_region = powers[start_idx:end_idx]
                
                def gaussian(x, amp, mu, sigma, offset):
                    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset
                
                p0 = [peak_amplitude - (-80), peak_freq, 1e6, -80]
                popt, _ = curve_fit(gaussian, freq_region, power_region, p0=p0, maxfev=1000)
                fwhm = 2.35482 * popt[2]
            except Exception:
                fwhm = None
        
        return {
            'success': True,
            'peak_frequency': peak_freq,
            'peak_amplitude': peak_amplitude,
            'fwhm': fwhm,
            'peak_index': highest_peak_idx
        }
    
    def _iterative_spectrum_measurement(self, center_frequency_hz: float) -> Dict[str, Any]:
        """
        Iteratively adjust span and RBW to get good peak resolution.
        
        Args:
            center_frequency_hz: Center frequency in Hz
            
        Returns:
            Dictionary with spectrum measurement results
        """
        sa_settings = self.settings['spectrum_analyzer']
        
        # Convert to MHz for the Agilent
        center_frequency_mhz = center_frequency_hz / 1e6
        span_mhz = sa_settings['initial_span'] / 1e6
        rbw_mhz = sa_settings['initial_rbw'] / 1e6
        
        max_iterations = sa_settings['max_iterations']
        target_points = sa_settings['target_peak_points']
        
        best_result = None
        
        for iteration in range(max_iterations):
            print(f"  Iteration {iteration + 1}: span={span_mhz:.1f}MHz, RBW={rbw_mhz*1e3:.1f}kHz")
            
            # Configure spectrum analyzer
            self.spectrum_analyzer.set_center_frequency(center_frequency_mhz)
            self.spectrum_analyzer.set_span(span_mhz)
            self.spectrum_analyzer.set_resolution_bw(rbw_mhz)
            self.spectrum_analyzer.set_single_sweep()
            self.spectrum_analyzer.take_sweep()
            time.sleep(0.5)  # Wait for sweep to complete
            
            # Get trace data
            frequencies_hz, powers_dbm = self._get_trace_from_sa()
            
            # Calculate FWHM
            result = self._calculate_fwhm(frequencies_hz, powers_dbm)
            result['span_hz'] = span_mhz * 1e6
            result['rbw_hz'] = rbw_mhz * 1e6
            result['center_frequency_hz'] = center_frequency_hz
            result['iteration'] = iteration
            result['trace_frequencies_hz'] = frequencies_hz
            result['trace_powers_dbm'] = powers_dbm
            
            if result['success']:
                if result['fwhm'] is not None:
                    points_across_peak = (span_mhz * 1e6) / result['fwhm'] * 10
                    print(f"    Peak found: {result['peak_frequency']/1e9:.6f} GHz, "
                          f"Amp={result['peak_amplitude']:.1f} dBm, "
                          f"FWHM={result['fwhm']/1e6:.2f} MHz")
                    
                    if points_across_peak >= target_points or iteration == max_iterations - 1:
                        best_result = result
                        break
            
            # Adjust span and RBW for next iteration
            if result['fwhm'] is not None:
                target_span_mhz = (result['fwhm'] / 1e6) * 5
                span_mhz = max(span_mhz * 0.5, target_span_mhz)
                rbw_mhz = max(rbw_mhz * 0.5, (result['fwhm'] / 1e6) / 20)
            else:
                span_mhz = span_mhz * 2
                rbw_mhz = rbw_mhz * 2
            
            # Clamp values (in MHz for Agilent)
            span_mhz = np.clip(span_mhz, 0.1, 3000)  # 100kHz to 3GHz
            rbw_mhz = np.clip(rbw_mhz, 0.00003, 3)  # 30Hz to 3MHz
        
        if best_result is None and result['success']:
            best_result = result
        elif best_result is None:
            best_result = {'success': False, 'error': 'Could not find peak'}
        
        return best_result
    
    def _load_and_build_sequence(self, sequence_text: str) -> Tuple[bool, Optional[List[Sequence]]]:
        """
        Load sequence from text and build scan sequences.
        
        Args:
            sequence_text: Sequence in text format
            
        Returns:
            Tuple of (success, scan_sequences)
        """
        try:
            self.sequence_description = self.sequence_parser.parse_text(sequence_text)
            self.repeat_count = self.sequence_description.repeat_count
            if not self.sequence_description:
                print("Failed to parse sequence text")
                return False, None
            
            scan_sequences = self.sequence_builder.build_scan_sequences(self.sequence_description)
            self.sampling_rate = self.sequence_builder.sample_rate
            
            # Apply hardware calibration
            for i, sequence in enumerate(scan_sequences):
                calibrated_sequence = self.hardware_calibrator.calibrate_sequence(
                    sequence, self.sequence_description.sample_rate
                )
                scan_sequences[i] = calibrated_sequence
            
            return True, scan_sequences
            
        except Exception as e:
            print(f"Error building sequence: {e}")
            return False, None
    
    def _run_single_test(self, frequency_hz: float, mode: str, sequence_text: str) -> Dict[str, Any]:
        """
        Run a single IQ test for a given frequency and sequence.
        
        Args:
            frequency_hz: Microwave frequency in Hz
            mode: Test mode (I_ONLY, Q_ONLY, I_AND_Q)
            sequence_text: Sequence text to run
            
        Returns:
            Dictionary with test results
        """
        result = {
            'frequency_hz': frequency_hz,
            'mode': mode,
            'timestamp': datetime.datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Load and build sequence
            success, scan_sequences = self._load_and_build_sequence(sequence_text)
            if not success:
                result['error'] = "Failed to build sequence"
                return result
            
            # Generate AWG waveforms
            self.scan_sequences = scan_sequences
            self.sequence_duration = scan_sequences[0].length if scan_sequences else 0
            
            if not self.generate_awg_sequences_awg_triggering_adwin_case():
                result['error'] = "Failed to generate AWG sequences"
                return result
            
            # Configure microwave source
            self.sg384.set_frequency(frequency_hz)
            self.sg384.set_power(self.settings['microwave']['power'])
            self.sg384.enable_output()
            time.sleep(0.1)
            
            # Wait for Proteus to output
            time.sleep(0.5)
            
            # Measure spectrum
            spectrum_result = self._iterative_spectrum_measurement(frequency_hz)
            result['spectrum'] = spectrum_result
            
            # Parse amplitude from sequence text
            import re
            i_amp = 0
            q_amp = 0
            # Look for channel 1 amplitude
            ch1_match = re.search(r'channel 1.*?,\s*([0-9.]+)\s*$', sequence_text, re.MULTILINE)
            if ch1_match:
                i_amp = float(ch1_match.group(1))
            # Look for channel 2 amplitude  
            ch2_match = re.search(r'channel 2.*?,\s*([0-9.]+)\s*$', sequence_text, re.MULTILINE)
            if ch2_match:
                q_amp = float(ch2_match.group(1))
            
            result['proteus_settings'] = {
                'i_amplitude': i_amp,
                'q_amplitude': q_amp,
                'pulse_duration_ns': self.settings['iq_calibration']['pulse_duration']
            }
            
            result['success'] = spectrum_result.get('success', False)
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Test failed: {e}")
        
        return result

    def calibrate_dc_mode(self) -> List[Dict[str, Any]]:
        results = []
        frequencies = self.settings['microwave']['frequency_range']
        pulse_duration = self.settings['iq_calibration']['pulse_duration']
        out_dir = self.settings['output']['output_directory']
        os.makedirs(out_dir, exist_ok=True)

        for freq in frequencies:
            print(f"\n{'=' * 60}\nDC mode @ {freq / 1e9:.4f} GHz\n{'=' * 60}")

            def seq_fn(mode, i_amp, q_amp):
                i_corr, q_corr = self._correct_amplitudes(i_amp, q_amp)
                return self._get_sequence_for_mode(mode, i_corr, q_corr, pulse_duration)

            sweep = self._sweep_amplitude_at_frequency(freq, 'DC', seq_fn)
            fig, _ = self._plot_amplitude_sweep(freq, sweep, 'DC')
            fname = os.path.join(out_dir, f"dc_{freq / 1e9:.4f}GHz_sweep.png")
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")

            # Flatten into results list for HDF5
            for mode in ('I_ONLY', 'Q_ONLY', 'I_AND_Q'):
                for amp, power in sweep[mode]:
                    results.append({'frequency_hz': freq, 'mode': mode,
                                    'modulation_mode': 'DC', 'amplitude': amp,
                                    'success': True,
                                    'spectrum': {'peak_amplitude': power}})
        return results

    def calibrate_square_mode(self) -> List[Dict[str, Any]]:
        results = []
        frequencies = self.settings['microwave']['frequency_range']
        out_dir = self.settings['output']['output_directory']
        os.makedirs(out_dir, exist_ok=True)

        for freq in frequencies:
            print(f"\n{'=' * 60}\nSQUARE full calibration @ {freq / 1e9:.4f} GHz\n{'=' * 60}")

            # 1. Sideband suppression vs amplitude (AWG square pulses)
            print("\n-- Sideband Suppression --")
            sb_data = self.measure_sideband_suppression(freq)
            self._plot_sideband_suppression(freq, sb_data, out_dir)

            # 2. Amplitude imbalance (function generator)
            print("\n-- Amplitude Imbalance --")
            imb_data = self.measure_amplitude_imbalance(freq)
            self._plot_amplitude_imbalance(freq, imb_data, out_dir)

            # 3. Phase accuracy (function generator)
            print("\n-- Phase Accuracy --")
            ph_data = self.measure_phase_accuracy(freq)
            self._plot_phase_accuracy(freq, ph_data, out_dir)

            # 4. DC offset null (function generator)
            print("\n-- DC Offset Null --")
            dc_data = self.measure_dc_offset_null(freq)
            self._plot_dc_offset(freq, dc_data, out_dir)

            # 5. Compute corrections
            print("\n-- Computing Correction Factors --")
            corrections = self.compute_correction_factors(imb_data, ph_data, dc_data, freq)

            results.append({
                'frequency_hz': freq,
                'modulation_mode': 'SQUARE',
                'sideband_data': sb_data,
                'imbalance_data': imb_data,
                'phase_data': ph_data,
                'dc_data': dc_data,
                'corrections': corrections,
                'success': True
            })
            time.sleep(0.5)

        return results
    
    def calibrate_gaussian_mode(self) -> List[Dict[str, Any]]:
        """Run Gaussian mode calibration tests."""
        results = []
        frequencies = self.settings['microwave']['frequency_range']
        amplitude = self.settings['iq_calibration']['amplitude']
        pulse_duration = self.settings['iq_calibration']['pulse_duration']
        
        test_modes = ['I_ONLY', 'Q_ONLY', 'I_AND_Q']
        
        for freq in frequencies:
            print(f"\n{'='*60}")
            print(f"Testing GAUSSIAN mode at {freq/1e9:.3f} GHz")
            print(f"{'='*60}")
            
            for mode in test_modes:
                print(f"\n  Mode: {mode}")
                
                if mode == 'I_ONLY':
                    sequence_text = self._get_gaussian_sequence('I_ONLY', amplitude, 0,
                                                                 pulse_duration)
                elif mode == 'Q_ONLY':
                    _, q_corr = self._correct_amplitudes(0, amplitude)  # i_amp irrelevant for correction
                    sequence_text = self._get_gaussian_sequence('Q_ONLY', 0, q_corr,
                                                                 pulse_duration)
                else:
                    _, q_corr = self._correct_amplitudes(0, amplitude)  # i_amp irrelevant for correction
                    sequence_text = self._get_gaussian_sequence('I_AND_Q', amplitude, q_corr,
                                                                 pulse_duration)
                
                result = self._run_single_test(freq, mode, sequence_text)
                result['modulation_mode'] = 'GAUSSIAN'
                result['amplitude'] = amplitude
                results.append(result)
                
                if result['success'] and 'spectrum' in result:
                    spec = result['spectrum']
                    print(f"    Peak freq: {spec.get('peak_frequency', 0)/1e9:.6f} GHz")
                    print(f"    Peak amp: {spec.get('peak_amplitude', -100):.1f} dBm")
                    print(f"    FWHM: {spec.get('fwhm', 0)/1e6:.2f} MHz")
                else:
                    print(f"    Failed: {result.get('error', 'Unknown error')}")
            
            time.sleep(0.5)
        
        return results

    def save_hdf5(self):
        """Save calibration data to HDF5 using MyStruct like pulsed_ODMR_code.py"""
        structure_to_save = MyStruct()

        freq_list = []
        mode_list = []
        modulation_list = []
        peak_freqs = []
        peak_amps = []
        fwhms = []
        i_amps = []
        q_amps = []
        success_list = []
        traces = []

        for result in self.calibration_results:
            freq_list.append(result.get('frequency_hz', 0))
            mode_list.append(result.get('mode', 'N/A'))
            modulation_list.append(result.get('modulation_mode', 'Unknown'))
            i_amps.append(result.get('proteus_settings', {}).get('i_amplitude', 0))
            q_amps.append(result.get('proteus_settings', {}).get('q_amplitude', 0))
            success_list.append(result.get('success', False))

            spec = result.get('spectrum', {})
            peak_freqs.append(spec.get('peak_frequency', 0))
            peak_amps.append(spec.get('peak_amplitude', -100))
            fwhms.append(spec.get('fwhm', 0))

            if self.settings['output']['save_traces']:
                if 'trace_frequencies_hz' in spec and 'trace_powers_dbm' in spec:
                    traces.append({
                        'frequency_hz': spec['trace_frequencies_hz'],
                        'power_dbm': spec['trace_powers_dbm'],
                        'mode': result.get('mode', 'N/A'),
                        'modulation': result.get('modulation_mode', 'Unknown'),
                        'frequency_set_hz': result.get('frequency_hz', 0)
                    })

        # ── Main data struct (must be created before adding attributes to it) ──
        structure_to_save.data = MyStruct(
            frequencies_hz=np.array(freq_list),
            modes=np.array(mode_list, dtype='S'),
            modulation_modes=np.array(modulation_list, dtype='S'),
            i_amplitudes=np.array(i_amps),
            q_amplitudes=np.array(q_amps),
            peak_frequencies_hz=np.array(peak_freqs),
            peak_amplitudes_dbm=np.array(peak_amps),
            fwhm_hz=np.array(fwhms),
            success=np.array(success_list, dtype=bool)
        )

        # ── Traces (only if any were collected) ──
        if self.settings['output']['save_traces'] and traces:
            structure_to_save.data.trace_frequencies_hz = [t['frequency_hz'] for t in traces]
            structure_to_save.data.trace_powers_dbm = [t['power_dbm'] for t in traces]

        # ── Square-mode correction factors (only present after Run 2) ──
        sq_results = [r for r in self.calibration_results if 'corrections' in r]
        if sq_results:
            structure_to_save.corrections = MyStruct(
                frequencies_hz=np.array([r['frequency_hz'] for r in sq_results]),
                amplitude_b=np.array([r['corrections']['amplitude']['b'] for r in sq_results]),
                best_irr_db=np.array([r['corrections']['amplitude']['best_irr_db'] for r in sq_results]),
                phase_phi_deg=np.array([r['corrections']['phase']['phi_deg'] for r in sq_results]),
                irr_spread_db=np.array([r['corrections']['phase']['irr_spread_db'] for r in sq_results]),
                i_dc_offset=np.array([r['corrections']['dc_offset']['i_offset'] for r in sq_results]),
                q_dc_offset=np.array([r['corrections']['dc_offset']['q_offset'] for r in sq_results]),
            )

        # ── Metadata ──
        structure_to_save.meta = MyStruct(
            calibration_mode=self.settings['iq_calibration']['mode'],
            amplitude=self.settings['iq_calibration']['amplitude'],
            pulse_duration_ns=self.settings['iq_calibration']['pulse_duration'],
            microwave_power_dbm=self.settings['microwave']['power'],
            microwave_frequencies_hz=self.settings['microwave']['frequency_range'],
            start_time=self.s_t,
            end_time=self.e_t,
            sequence_text=self.settings['sequence']['text'],
            sampling_rate=getattr(self, 'sampling_rate', 1e9)
        )

        structure_to_save.devices = self.devices
        self.save_hdf_data(structure_to_save)
        print(f"\nData saved to HDF5")
    
    def _function(self):
        """Main experiment function called when experiment is run."""
        print("\n" + "="*80)
        print("IQ CALIBRATION EXPERIMENT")
        print("="*80)
        
        start_time = datetime.datetime.now()
        self.s_t = start_time.strftime("%m_%d_%Y_%H:%M:%S")
        
        mode = self.settings['iq_calibration']['mode'].upper()
        
        print(f"\nMode: {mode}")
        print(f"Amplitude: {self.settings['iq_calibration']['amplitude']}")
        print(f"Pulse duration: {self.settings['iq_calibration']['pulse_duration']} ns")
        print(f"Frequencies: {[f/1e9 for f in self.settings['microwave']['frequency_range']]} GHz")
        print(f"Microwave power: {self.settings['microwave']['power']} dBm")
        
        # Connect devices
        print("\nConnecting devices...")
        if not self.sg384.is_connected:
            self.sg384.connect()
        
        # Initialize spectrum analyzer
        self.spectrum_analyzer.initialize_sa()
        
        # Select trigger mode
        self.mux.select_trigger('pulsed')
        
        # Run calibration based on mode
        if mode == "DC":
            self.calibration_results = self.calibrate_dc_mode()
        elif mode == "SQUARE":
            self.calibration_results = self.calibrate_square_mode()
        elif mode == "GAUSSIAN":
            self.calibration_results = self.calibrate_gaussian_mode()
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented. Use DC, SQUARE, or GAUSSIAN")
        
        end_time = datetime.datetime.now()
        self.e_t = end_time.strftime("%m_%d_%Y_%H:%M:%S")
        
        # Save results to HDF5
        self.data = {'success': True, 'results': self.calibration_results}
        self.save_hdf5()
        self.export_calibration_json()
        
        # Print summary
        self._print_summary()
        
        # Cleanup
        self.sg384.disable_output()
        
        print("\n" + "="*80)
        print("IQ CALIBRATION COMPLETE")
        print("="*80)
    
    def _print_summary(self):
        """Print a summary of calibration results."""
        print("\n" + "="*80)
        print("CALIBRATION SUMMARY")
        print("="*80)
        
        summary_data = {}
        for result in self.calibration_results:
            freq_key = f"{result['frequency_hz']/1e9:.3f} GHz"
            mode_key = result.get('mode', 'N/A')
            mod_key = result.get('modulation_mode', 'Unknown')
            
            if freq_key not in summary_data:
                summary_data[freq_key] = {}
            if mode_key not in summary_data[freq_key]:
                summary_data[freq_key][mode_key] = {}
            
            if result['success'] and 'spectrum' in result:
                spec = result['spectrum']
                summary_data[freq_key][mode_key][mod_key] = {
                    'peak_freq': spec.get('peak_frequency', 0) / 1e9,
                    'peak_amp': spec.get('peak_amplitude', -100),
                    'fwhm': spec.get('fwhm', 0) / 1e6 if spec.get('fwhm') else None
                }
            else:
                summary_data[freq_key][mode_key][mod_key] = {'error': result.get('error', 'Unknown')}
        
        print("\nFrequency | Mode    | I_Q_Type   | Peak Freq (GHz) | Peak Amp (dBm) | FWHM (MHz)")
        print("-" * 80)
        
        for freq in sorted(summary_data.keys()):
            for mode in ['I_ONLY', 'Q_ONLY', 'I_AND_Q']:
                if mode in summary_data[freq]:
                    for mod_type in ['DC', 'SQUARE', 'GAUSSIAN']:
                        if mod_type in summary_data[freq][mode]:
                            data = summary_data[freq][mode][mod_type]
                            if 'error' not in data:
                                fwhm_str = f"{data['fwhm']:.2f}" if data['fwhm'] else "N/A"
                                print(f"{freq:8} | {mode:6} | {mod_type:9} | "
                                      f"{data['peak_freq']:14.6f} | "
                                      f"{data['peak_amp']:14.1f} | {fwhm_str:>8}")

        failures = [r for r in self.calibration_results if not r.get('success', False)]
        if failures:
            for f in failures:
                print(f"  {f.get('frequency_hz', 0) / 1e9:.3f} GHz - {f.get('mode', 'N/A')} - {f.get('error', 'Unknown')}")
    
    def _update(self):
        """Update method required by Experiment base class."""
        pass

    def export_calibration_json(self, output_path: str = None) -> str:
        """Export calibration for use by ODMRPulsedExperiment and other experiments."""
        cal = {
            'timestamp': datetime.datetime.now().isoformat(),
            'frequencies': [],
            'q_amplitude_correction_default': 0.89,  # fallback
        }

        sq_results = [r for r in self.calibration_results if 'corrections' in r]
        for r in sq_results:
            freq_hz = r['frequency_hz']
            c = r['corrections']
            cal['frequencies'].append({
                'frequency_hz': freq_hz,
                'q_amplitude_b': c['amplitude']['b'],
                'best_irr_db': c['amplitude']['best_irr_db'],
                'phase_phi_deg': c['phase']['phi_deg'],
                'i_dc_offset_set_V': c['dc_offset']['i_offset'],
                'q_dc_offset_set_V': c['dc_offset']['q_offset'],
            })

        # Compute a single default b (average across frequencies)
        if sq_results:
            cal['q_amplitude_correction_default'] = float(np.mean(
                [r['corrections']['amplitude']['b'] for r in sq_results]
            ))

        path = output_path or str(Path(__file__).parent / 'iq_calibration_results.json')
        with open(path, 'w') as f:
            json.dump(cal, f, indent=2)
        print(f"\nIQ calibration exported to: {path}")
        return path

    def _correct_amplitudes(self, i_amp: float, q_amp: float,
                            freq_hz: float = 2.87e9) -> tuple:
        corrections = self.settings['iq_calibration']['q_amplitude_correction']
        if isinstance(corrections, dict):
            closest = min(corrections.keys(), key=lambda f: abs(f - freq_hz))
            b = corrections[closest]
        else:
            b = corrections  # backward compatible with single float
        return i_amp, q_amp * b

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
                self.proteus.driver.set_voltage_offset(0)

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
                        self.proteus.driver.set_marker_voltage_offset(0.5)
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
            # self.proteus.driver._close()
            return True
        except Exception as e:
            self.logger.exception("Error generating AWG files")
            return False
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Return the calibration results."""
        return self.calibration_results

    def close(self):
        self.proteus.driver._close()
        self.sg384.close()

if __name__ == "__main__":
    import numpy as np

    FREQUENCIES = [2.7e9, 2.87e9, 3.0e9]  # pick your NV-relevant range
    BASEBAND_FREQ = 1 / (128e-9)  # = 7,812,500 Hz (64on + 64off)

    # ─────────────────────────────────────────────────────────────
    # RUN 1 — DC mode: LO leakage and raw amplitude response
    # ─────────────────────────────────────────────────────────────
    """exp_dc = IQCalibrationExperiment(name="iq_cal_dc")
    exp_dc.settings['iq_calibration']['mode'] = 'DC'
    exp_dc.settings['iq_calibration']['amplitude_sweep'] = list(np.linspace(0.0, 1.0, 11))
    exp_dc.settings['iq_calibration']['pulse_duration'] = 512  # ns, DC-ish long pulse
    exp_dc.settings['microwave']['frequency_range'] = FREQUENCIES
    exp_dc.settings['microwave']['power'] = 10.0
    exp_dc.settings['output']['output_directory'] = 'output_dc'
    exp_dc._function()
    exp_dc.close()"""

    # ─────────────────────────────────────────────────────────────
    # RUN 2 — Square mode: full IQ calibration suite
    # ─────────────────────────────────────────────────────────────
    exp_sq = IQCalibrationExperiment(name="iq_cal_square")
    exp_sq.settings['iq_calibration']['mode'] = 'SQUARE'
    exp_sq.settings['iq_calibration']['pulse_on_time'] = 64  # ns
    exp_sq.settings['iq_calibration']['pulse_off_time'] = 64  # ns  → 7.8125 MHz
    exp_sq.settings['iq_calibration']['sideband_offset'] = BASEBAND_FREQ  # SA looks here
    exp_sq.settings['iq_calibration']['sideband_span'] = 4e6  # Hz, carrier stays invisible
    exp_sq.settings['iq_calibration']['sideband_rbw'] = 30e3  # Hz
    exp_sq.settings['iq_calibration']['amplitude_sweep'] = list(np.linspace(0.1, 1.0, 10))
    exp_sq.settings['iq_calibration']['fg_amplitude'] = 0.98
    exp_sq.settings['iq_calibration']['fg_frequency'] = BASEBAND_FREQ
    exp_sq.settings['iq_calibration']['iq_ratio_sweep'] = [0.5, 0.6, 0.7, 0.8, 0.9,
                                                           1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    exp_sq.settings['iq_calibration']['phase_angles'] = [0.0, 90.0, 180.0, 270.0]
    exp_sq.settings['iq_calibration']['dc_offset_sweep'] = [-0.25, -0.20, -0.17, -0.14, -0.10, -0.07, -0.05, -0.02, 0.0, 0.02, 0.05, 0.07, 0.10]
    exp_sq.settings['microwave']['frequency_range'] = FREQUENCIES
    exp_sq.settings['microwave']['power'] = 10.0
    exp_sq.settings['output']['output_directory'] = 'output_square'

    exp_sq._function()

    # Print correction table across all frequencies
    print("\n\n{'='*60}")
    print("FINAL CORRECTION FACTORS")
    print(f"{'=' * 60}")
    for r in exp_sq.get_results():
        c = r['corrections']
        print(f"\n  {r['frequency_hz'] / 1e9:.4f} GHz")
        print(f"    Amplitude b  = {c['amplitude']['b']:.4f}   "
              f"(best IRR = {c['amplitude']['best_irr_db']:.1f} dB)")
        print(f"    Phase φ      = {c['phase']['phi_deg']:.2f}°  "
              f"(IRR spread = {c['phase']['irr_spread_db']:.1f} dB)")
        print(f"    I DC offset  = {c['dc_offset']['i_offset']:.4f} V  "
              f"(min leakage = {c['dc_offset']['min_i_leakage_dbm']:.1f} dBm)")
        print(f"    Q DC offset  = {c['dc_offset']['q_offset']:.4f} V  "
              f"(min leakage = {c['dc_offset']['min_q_leakage_dbm']:.1f} dBm)")


"""Key flow summary:
_function()
└─ calibrate_dc_mode()  OR  calibrate_square_mode()
   └─ for each frequency:
        _sweep_amplitude_at_frequency()   ← loops amps × 3 modes
            └─ _run_single_test()         ← existing method, unchanged
        [square only] _sweep_carrier_leakage_at_frequency()
        _plot_amplitude_sweep()  +  _plot_carrier_leakage()
        fig.savefig(...)
        
calibrate_square_mode()
└─ per frequency:
   measure_sideband_suppression()   ← AWG square pulses, _measure_sideband_powers ×2
   measure_amplitude_imbalance()    ← _setup_sine_iq, sweep Q ratio
   measure_phase_accuracy()         ← _setup_sine_iq, θ ∈ {0,90,180,270}
   measure_dc_offset_null()         ← set_function_generator offset sweep
   compute_correction_factors()     ← numpy argmax/argmin, returns a/b/φ/dc
   4 × plot + save"""