#!/usr/bin/env python3
# Written by <Jannet Trabelsi>
"""
Standalone ODMR sequence preview.

Build and preview a pulsed-ODMR sequence WITHOUT creating an
ODMRPulsedExperiment. Constructing that experiment instantiates the Proteus,
ADwin, MUX and SG384 drivers (and connects to them); none of that is needed
just to look at the sequence.

This module reuses the real sequence engine -- SequenceTextParser,
SequenceBuilder and ProteusHardwareCalibrator -- so the preview matches what
the experiment would build. It does NOT import the experiment module, so no
device drivers are loaded and no hardware is touched.

Placement: put this file next to odmr_pulsed_experiment.py (i.e. in src/Model/)
so the default connection-file path resolves. Run from your project root.

Adiabaticity check
------------------
For frequency-swept pulses (chirp / hs / asymm_hs) the preview also estimates
the adiabaticity factor Q(delta) = Omega1(t)^2 / |d(delta)/dt| from the pulse's
own I/Q output (Appendix A of the adiabatic-DEER note). Adiabatic inversion
needs Q >> 1 on resonance. You must supply the PEAK Rabi frequency on the pump
channel (Omega1/2pi, in Hz) via rabi_hz -- the waveform alone doesn't know your
drive strength. The report prints to the terminal and appears in the
"Adiabaticity" tab of the preview window.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

# Sequence engine -- pure software, no device connections.
from src.Model.sequence_parser import SequenceTextParser
from src.Model.sequence_builder import SequenceBuilder
from src.Model.proteus_hardware_calibrator import ProteusHardwareCalibrator


# The example sequence from ODMRPulsedExperiment.create_example_odmr_sequence().
EXAMPLE_ODMR_SEQUENCE = """sequence: name=hahn_echo, type=hahn_echo, duration=10700ns, sample_rate=1GHz, repeat_count=250000
variable pulse_duration, start=0ns, stop=797ns, steps=128
init pulse on channel 4 at 0.000001789-2*pulse_duration, square, 2000ns, 1.0
pi_2_1 pulse on channel 1 at 0.000003789-2*pulse_duration, gaussian, 28ns, 1.0
pi pulse on channel 1 at 0.000003817-pulse_duration, gaussian, 55ns, 1.0
pi_2_2 pulse on channel 1 at 3872ns, gaussian, 28ns, 1.0
readout pulse on channel 4 at 3900ns, square, 6800ns, 1.0"""

# A ready-made adiabatic-pump DEER-style sequence you can preview to see the
# Adiabaticity tab populate (chirp pump placed as an I/Q pair on ch 1 / ch 2):
#     preview = ODMRSequencePreview(sequence_text=EXAMPLE_CHIRP_DEER_SEQUENCE, rabi_hz=12e6)
EXAMPLE_CHIRP_DEER_SEQUENCE = """sequence: name=deer_chirp, type=deer, duration=8000ns, sample_rate=1GHz, repeat_count=100000
variable t_pos, start=1500ns, stop=5500ns, steps=64
init pulse on channel 4 at 0ns, square, 2000ns, 1.0
pi_2_1 pulse on channel 1 at 2500ns, square, 28ns, 1.0
pi pulse on channel 1 at 4500ns, square, 55ns, 1.0
chirp_I pulse on channel 1 at t_pos, chirp, 2000ns, 1.0, bandwidth=300MHz, quadrature=I, edge_fraction=0.1
chirp_Q pulse on channel 2 at t_pos, chirp, 2000ns, 1.0, bandwidth=300MHz, quadrature=Q, edge_fraction=0.1
pi_2_2 pulse on channel 1 at 6500ns, square, 28ns, 1.0
readout pulse on channel 4 at 6600ns, square, 6800ns, 1.0"""


# ----------------------------------------------------------------------
# Adiabaticity estimation (pure numpy; works for any frequency-swept pulse)
# ----------------------------------------------------------------------
def _swept_iq(pulse):
    """Return (I, Q, sample_rate) for a frequency-swept pulse, else None.

    Asks the pulse for BOTH IQ quadratures and hands them back so A(t) and f(t)
    can be reconstructed without needing per-shape formulas -- so this works for
    ChirpPulse, HyperbolicSecantPulse, AsymmetricHyperbolicSecantPulse and any
    future swept pulse that exposes `bandwidth`, `quadrature` and `sample_rate`.
    """
    if not (hasattr(pulse, "bandwidth") and hasattr(pulse, "quadrature")):
        return None
    fs = getattr(pulse, "sample_rate", None)
    if not fs:
        return None
    saved = pulse.quadrature
    try:
        pulse.quadrature = "I"
        I = np.asarray(pulse.generate_samples(), dtype=float)
        pulse.quadrature = "Q"
        Q = np.asarray(pulse.generate_samples(), dtype=float)
    finally:
        pulse.quadrature = saved
    if I.size < 8 or I.size != Q.size:
        return None
    return I, Q, float(fs)


def estimate_adiabaticity(sequence, rabi_hz: float, q_threshold: float = 5.0):
    """Per-pulse adiabaticity estimate for the frequency-swept pulses in `sequence`.

    Q(delta) = Omega1(t)^2 / |d(delta)/dt|, with Omega1(t) = 2*pi*rabi_hz*(A(t)/A_peak)
    and delta swept by the pulse. Returns a list of dicts with, per swept pulse:
    name, type, sweep_MHz, Tp_us, Q_center (on resonance), adiab_bw_MHz (detuning span
    where Q >= q_threshold), verdict, and the arrays (detuning_MHz, Q_t, t_us) for plotting.
    """
    results = []
    for _start, pulse in sequence.pulses:
        got = _swept_iq(pulse)
        if got is None:
            continue
        I, Q, fs = got
        amp = np.sqrt(I ** 2 + Q ** 2)
        peak = float(amp.max())
        if peak <= 0:
            continue
        # Restrict to where the envelope is meaningful; below this the phase
        # (atan2) is ill-defined and would inject spurious frequency spikes.
        idx = np.where(amp >= 0.1 * peak)[0]
        if idx.size < 4:
            continue
        sl = slice(idx[0], idx[-1] + 1)
        amp_c = amp[sl]
        phase = np.unwrap(np.arctan2(Q[sl], I[sl]))
        f = np.gradient(phase, 1.0 / fs) / (2.0 * np.pi)     # instantaneous freq (Hz)
        dfdt = np.gradient(f, 1.0 / fs)                       # sweep rate (Hz/s)
        omega1 = 2.0 * np.pi * rabi_hz * (amp_c / peak)       # rad/s, scaled by envelope
        with np.errstate(divide="ignore", invalid="ignore"):
            Q_t = omega1 ** 2 / (2.0 * np.pi * np.abs(dfdt))
        # Drop the two boundary samples (one-sided gradient) to avoid edge artifacts.
        Q_t, f_in, amp_in = Q_t[1:-1], f[1:-1], amp_c[1:-1]
        if f_in.size < 2:
            continue
        f_mid = 0.5 * (f_in.max() + f_in.min())
        Q_center = float(Q_t[int(np.argmin(np.abs(f_in - f_mid)))])   # on-resonance
        adiab = Q_t >= q_threshold
        adiab_bw = float(f_in[adiab].max() - f_in[adiab].min()) / 1e6 if np.any(adiab) else 0.0
        verdict = ("ADIABATIC" if Q_center >= q_threshold else
                   "marginal" if Q_center >= 1.0 else "NOT adiabatic")
        results.append(dict(
            name=pulse.name, type=type(pulse).__name__,
            sweep_MHz=(f_in.max() - f_in.min()) / 1e6,
            Tp_us=pulse.length / fs * 1e6,
            Q_center=Q_center, adiab_bw_MHz=adiab_bw, verdict=verdict,
            detuning_MHz=(f_in - f_mid) / 1e6,
            Q_t=Q_t,
            t_us=np.arange(f_in.size) / fs * 1e6,
        ))
    return results


class ODMRSequencePreview:
    """Parse, build and preview an ODMR sequence with no hardware involved."""

    def __init__(self, sequence_text: Optional[str] = None,
                 connection_file: Optional[str] = None,
                 rabi_hz: float = 12e6):
        self.sequence_text = (
            sequence_text if sequence_text is not None else EXAMPLE_ODMR_SEQUENCE
        )

        # Peak Rabi frequency on the pump channel (Omega1/2pi, Hz), used only for
        # the adiabaticity estimate. Set this to your measured pump-channel Rabi.
        self.rabi_hz = rabi_hz

        # Same connection file the experiment uses for timing calibration.
        # This is only read from disk; it does not open a link to the Proteus.
        if connection_file is None:
            connection_file = str(Path(__file__).parent / "odmr_pulsed_connection.json")

        self.sequence_parser = SequenceTextParser()
        self.sequence_builder = SequenceBuilder()
        self.hardware_calibrator = ProteusHardwareCalibrator(connection_file=connection_file)

        self.sequence_description = None
        self.scan_sequences: List = []
        self.repeat_count = None
        self.number_of_iterations = 0
        self.sampling_rate = None
        self.sequence_duration = None

    def load(self) -> bool:
        """Parse self.sequence_text into a sequence description."""
        try:
            self.sequence_description = self.sequence_parser.parse_text(self.sequence_text)
            self.repeat_count = self.sequence_description.repeat_count
            if self.sequence_description:
                print(f"Sequence loaded: {self.sequence_description.name}")
                print(f"  Variables: {len(self.sequence_description.variables)}")
                print(f"  Pulses: {len(self.sequence_description.pulses)}")
                print(f"  Markers: {len(self.sequence_description.markers)}")
                return True
            print("Failed to parse sequence text")
            return False
        except Exception as e:
            print(f"Error loading sequence: {e}")
            return False

    def build(self) -> bool:
        """Build scan sequences and apply hardware timing calibration."""
        try:
            if not self.sequence_description:
                print("No sequence description loaded")
                return False

            self.scan_sequences = self.sequence_builder.build_scan_sequences(
                self.sequence_description
            )
            self.sampling_rate = self.sequence_builder.sample_rate

            if self.sequence_description.variables:
                self.number_of_iterations = len(self.scan_sequences)
            else:
                self.number_of_iterations = 1

            calibrated_sequence = None
            for i, sequence in enumerate(self.scan_sequences):
                calibrated_sequence = self.hardware_calibrator.calibrate_sequence(
                    sequence,
                    self.sequence_description.sample_rate
                )
                self.scan_sequences[i] = calibrated_sequence

            if calibrated_sequence is not None:
                self.sequence_duration = calibrated_sequence.length

            print(f"Built {len(self.scan_sequences)} scan sequences")

            # Report adiabaticity for any frequency-swept (chirp/HS) pulses.
            self.adiabaticity_report()
            return True
        except Exception as e:
            print(f"Error building scan sequences: {e}")
            return False

    def adiabaticity_report(self, rabi_hz: Optional[float] = None, verbose: bool = True):
        """Estimate & print the adiabaticity of any frequency-swept pulses.

        Uses the first scan point. Returns the list of per-pulse result dicts
        (see estimate_adiabaticity). Call directly from a terminal to re-check
        with a different peak Rabi, e.g. preview.adiabaticity_report(25e6).
        """
        if rabi_hz is None:
            rabi_hz = self.rabi_hz
        if not self.scan_sequences:
            if verbose:
                print("No scan sequences available; call build() first.")
            return []

        results = estimate_adiabaticity(self.scan_sequences[0], rabi_hz)
        if verbose:
            if not results:
                print("Adiabaticity: no frequency-swept (chirp/hs/asymm_hs) pulses in this sequence.")
            else:
                print(f"\nAdiabaticity check  (peak Rabi Omega1/2pi = {rabi_hz / 1e6:.1f} MHz; "
                      f"need Q_center >> 1):")
                print(f"  {'pulse':<16}{'type':<32}{'sweep':>9}{'Tp':>8}"
                      f"{'Q_center':>10}{'adiab.BW(Q>=5)':>16}{'verdict':>15}")
                for r in results:
                    print(f"  {r['name']:<16}{r['type']:<32}{r['sweep_MHz']:>7.0f}MHz"
                          f"{r['Tp_us']:>6.2f}us{r['Q_center']:>10.2f}"
                          f"{r['adiab_bw_MHz']:>13.0f}MHz{r['verdict']:>15}")
                print(f"  (Tp >= 2*pi*sweep/Omega1^2 for a linear chirp; HS needs ~beta x more "
                      f"drive but inverts uniformly across the band.)")
        return results

    def preview(self, num_points: int = 10) -> None:
        """Open the preview window for the first num_points scan points."""
        if not self.scan_sequences:
            messagebox.showerror("Error", "No scan sequences available. Build first.")
            return
        preview_sequences = self.scan_sequences[:min(num_points, len(self.scan_sequences))]
        window = SequencePreviewWindow(preview_sequences, self.sequence_description,
                                       rabi_hz=self.rabi_hz)
        window.show()


class SequencePreviewWindow:
    """Window for previewing sequence scan points.

    Standalone copy of the window defined in odmr_pulsed_experiment.py. It is
    copied here (rather than imported) so this module does not pull in the
    experiment file and its device-driver imports.
    """

    def __init__(self, sequences: List, description, rabi_hz: float = 12e6):
        """Initialize preview window."""
        self.sequences = sequences
        self.description = description
        self.rabi_hz = rabi_hz
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

        # Tab 3: Adiabaticity (only meaningful for chirp / HS pulses)
        adiab_frame = ttk.Frame(notebook)
        notebook.add(adiab_frame, text="Adiabaticity")
        self._create_adiabaticity_tab(adiab_frame)

        # Tab 4: Parameters
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

    def _create_adiabaticity_tab(self, parent):
        """Create the adiabaticity tab: per-pulse verdict + Q(detuning) plot."""
        results = []
        if self.sequences:
            results = estimate_adiabaticity(self.sequences[0], self.rabi_hz)

        info_frame = ttk.LabelFrame(parent, text="Adiabaticity", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(
            info_frame,
            text=f"Peak Rabi Omega1/2pi = {self.rabi_hz / 1e6:.1f} MHz "
                 f"(set rabi_hz= to your measured pump-channel Rabi)"
        ).pack(anchor='w')

        if not results:
            ttk.Label(
                info_frame,
                text="No frequency-swept (chirp / hs / asymm_hs) pulses in this sequence."
            ).pack(anchor='w', pady=(6, 0))
            return

        for r in results:
            ttk.Label(
                info_frame,
                text=(f"{r['name']} [{r['type']}]:  sweep {r['sweep_MHz']:.0f} MHz,  "
                      f"Tp {r['Tp_us']:.2f} us,  Q_center {r['Q_center']:.2f}  ->  "
                      f"{r['verdict']}  (adiabatic over {r['adiab_bw_MHz']:.0f} MHz, Q>=5)")
            ).pack(anchor='w')

        # Plot: instantaneous frequency sweep + Q vs detuning
        fig, (ax_f, ax_q) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill='both', expand=True)

        for r in results:
            ax_f.plot(r['t_us'], r['detuning_MHz'], label=r['name'])
            ax_q.plot(r['detuning_MHz'], r['Q_t'], label=r['name'])

        ax_f.set_xlabel("time (us)")
        ax_f.set_ylabel("detuning (MHz)")
        ax_f.set_title("Instantaneous frequency sweep")
        ax_f.legend(loc='best', fontsize=8)
        ax_f.grid(True, alpha=0.3)

        ax_q.axhline(5.0, color='green', ls='--', lw=1, label='Q=5 (adiabatic)')
        ax_q.axhline(1.0, color='red', ls=':', lw=1, label='Q=1')
        ax_q.set_yscale('log')
        ax_q.set_xlabel("detuning (MHz)")
        ax_q.set_ylabel("adiabaticity Q")
        ax_q.set_title("Adiabaticity across the swept band")
        ax_q.legend(loc='best', fontsize=8)
        ax_q.grid(True, which='both', alpha=0.3)

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
