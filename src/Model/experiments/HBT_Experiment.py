#!/usr/bin/env python3
# Written for the Dutt lab NV setup
"""
Hanbury Brown-Twiss (HBT) g2(tau) Experiment
============================================

Measures the second-order intensity correlation g2(tau) of the light from a
single NV under CW green excitation, using the PicoQuant MultiHarp as a two-
channel T2 time tagger (one input per SPCM of the HBT). A dip g2(0) < 0.5 is the
signature of single-photon emission (a single NV).

How it fits the framework
--------------------------
Subclasses src.core.experiment.Experiment exactly like ODMRPulsedExperiment /
NanodriveAdwinConfocalPoint: defines _DEFAULT_SETTINGS / _DEVICES / _EXPERIMENTS,
implements _function(), save_hdf5(), and the pyqtgraph _plot()/get_axes_layout().
CW illumination is switched the same way the confocal point experiment does it,
by driving Proteus channel 4 to the "laser_control" DC level.

The scientific core (correlation, normalization, fit, simulation) lives in the
hardware-free module hbt_analysis.py, so it can be unit-tested without a rig
(see hbt_selftest.py).

Two run modes
-------------
  mode=None   real measurement: opens the MultiHarp, sets up illumination, tags
              both detectors in blocks, accumulates g2, fits, saves.
  mode="sim"  hardware-free: replaces the MultiHarp acquisition with synthetic
              photon streams (hbt_analysis.simulate_hbt_streams) that carry a
              known g2(0). Runs the *identical* correlation/fit/plot/save path,
              so you can validate the whole pipeline before going to the lab.

Quick hardware-free run:
    exp = HBTExperiment(name="hbt_sim", mode="sim")
    exp.run()                      # or exp._function()
    print(exp.data['g2_zero'])     # should be near settings['simulation']['sim_g2_zero']
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.core.experiment import Experiment
from src.core import Parameter
from src.core.struct_hdf5 import MyStruct

# Hardware-free analysis core (keep hbt_analysis.py alongside this file, or adjust
# this import to your package layout).
import hbt_analysis as hbt

# The MultiHarp device this experiment is built around. Imported defensively so
# the module still loads (for sim mode / GUI registration) if the path differs;
# a real run will raise a clear error from setup() if it is missing.
try:
    from PicoQuant_MultiHarp import PicoQuant_MultiHarp
except Exception:
    try:
        from src.Controller.PicoQuant_MultiHarp import PicoQuant_MultiHarp  # type: ignore
    except Exception:
        PicoQuant_MultiHarp = None

# Optional auxiliary hardware for CW illumination / attenuation. All optional:
# the core measurement never depends on them.
try:
    from src.Controller.Proteus_device import ProteusDevice
except Exception:
    ProteusDevice = None
try:
    from Thorlabs_FW102C_Filter_Wheel import Thorlabs_FW102C
except Exception:
    try:
        from src.Controller.Thorlabs_FW102C_Filter_Wheel import Thorlabs_FW102C  # type: ignore
    except Exception:
        Thorlabs_FW102C = None


class HBTExperiment(Experiment):
    """HBT g2(tau) measurement with a PicoQuant MultiHarp time tagger."""

    _DEFAULT_SETTINGS = [
        Parameter('multiharp', [
            Parameter('device_index', -1, int,
                      'MultiHarp device index; -1 auto-opens the first one'),
            Parameter('dll_path', '', str,
                      'explicit path to MHLib (mhlib64.dll / libmhlib.so); empty = search'),
            Parameter('detector_A_channel', 0, [0, 1, 2, 3, 4, 5, 6, 7],
                      'MultiHarp input wired to SPCM A'),
            Parameter('detector_B_channel', 1, [0, 1, 2, 3, 4, 5, 6, 7],
                      'MultiHarp input wired to SPCM B'),
            Parameter('level_A_mV', 320, int,
                      'SPCM A trigger level in mV (~half the ATTENUATED pulse amplitude)'),
            Parameter('level_B_mV', 320, int,
                      'SPCM B trigger level in mV (~half the ATTENUATED pulse amplitude)'),
            Parameter('edge', 'rising', ['rising', 'falling'],
                      'trigger edge for both detectors'),
        ]),
        Parameter('acquisition', [
            Parameter('integration_time', 60.0, float,
                      'total integration time', units='s'),
            Parameter('block_time', 5.0, float,
                      'per-block acquisition time; g2 accumulates across blocks so '
                      'you see it build up and can abort early', units='s'),
        ]),
        Parameter('correlation', [
            Parameter('tau_window', 200.0, float,
                      'half-width of the g2 window: histogram covers -tau_window..+tau_window',
                      units='ns'),
            Parameter('bin_width', 1.0, float, 'g2 histogram bin width', units='ns'),
        ]),
        Parameter('laser', [
            Parameter('control_laser', True, bool,
                      'let this experiment switch the CW green on/off via Proteus ch4 '
                      '(set False if you keep the laser on from confocal alignment)'),
            Parameter('laser_control', 0.8,
                      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      'Proteus ch4 DC level for the green (AOM), same knob as confocal'),
            Parameter('laser_off_after', True, bool,
                      'turn the green off when the experiment finishes'),
            Parameter('settle_time', 1.0, float,
                      'wait after switching the laser on before tagging', units='s'),
        ]),
        Parameter('Filter Wheel OD', 0, [0, 0.5, 1, 2, 3, 4],
                  'filter-wheel OD (attenuation into the detectors; keep SPCM rates sane)'),
        Parameter('fit', [
            Parameter('do_fit', True, bool,
                      'fit g2(tau) to a two-level antibunching model to extract g2(0)'),
            Parameter('tau0', 0.0, float,
                      'expected zero-delay position (A/B electronic+cable skew)', units='ns'),
            Parameter('tau_bunch', 12.0, float,
                      'initial guess for the antibunching recovery time', units='ns'),
        ]),
        Parameter('simulation', [
            Parameter('sim_g2_zero', 0.15, float,
                      'sim mode only: injected g2(0) (0 = perfect single emitter)'),
            Parameter('sim_count_rate', 200000.0, float,
                      'sim mode only: detected singles rate per detector', units='cps'),
            Parameter('sim_tau_c', 12.0, float,
                      'sim mode only: antibunching recovery time', units='ns'),
            Parameter('sim_jitter_ps', 250.0, float,
                      'sim mode only: detector timing jitter (Gaussian)', units='ps'),
        ]),
        Parameter('path', "D:\\Data"),
        Parameter('filename', "hbt_output"),
        Parameter('tag', "hbtexperiment"),
        Parameter('save', False),
        Parameter('sample', ""),
    ]

    # HBT constructs the hardware it needs directly (like ODMRPulsedExperiment),
    # so it does not require devices to be injected. Empty _DEVICES keeps GUI
    # loading and the base devices/experiments asserts happy for script runs.
    _DEVICES = {}
    _EXPERIMENTS = {}

    def __init__(self, devices=None, experiments=None, name=None, settings=None,
                 log_function=None, data_path=None, mode=None):
        super().__init__(name=name, settings=settings, devices=devices,
                         sub_experiments=experiments, log_function=log_function,
                         data_path=data_path)
        self.logger = logging.getLogger(__name__)
        self.sim_mode = (mode == "sim")
        self.multiharp = None
        self.proteus = None
        self.filter_wheel = None
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # hardware bring-up / teardown (real mode only)
    # ------------------------------------------------------------------ #
    def _get_or_make(self, name, constructor):
        """Prefer an injected device instance; otherwise construct one; otherwise
        return None. Never raises -- auxiliary hardware is optional."""
        inst = None
        if isinstance(self.devices, dict):
            inst = self.devices.get(name, {}).get('instance', None)
        if inst is not None:
            return inst
        if constructor is None:
            return None
        try:
            return constructor()
        except Exception as e:
            self.log(f"[HBT] could not initialize optional device '{name}': {e}")
            return None

    def setup(self):
        """Open the MultiHarp, configure the two detector channels, and switch on
        CW illumination. Called from _function() in real mode only."""
        if PicoQuant_MultiHarp is None:
            raise RuntimeError(
                "PicoQuant_MultiHarp could not be imported. Put PicoQuant_MultiHarp.py "
                "on the path (or fix the import at the top of HBT_Experiment.py).")
        mh = self.settings['multiharp']
        ch_a, ch_b = int(mh['detector_A_channel']), int(mh['detector_B_channel'])
        if ch_a == ch_b:
            raise ValueError("detector_A_channel and detector_B_channel must differ")

        self.multiharp = PicoQuant_MultiHarp(name=f"{self.name}_mh", settings={
            'device_index': int(mh['device_index']),
            'dll_path': mh['dll_path'],
        })
        edge = str(mh['edge'])
        # Enable exactly the two detector inputs (disables all others).
        self.multiharp.configure_channels({
            ch_a: (int(mh['level_A_mV']), edge),
            ch_b: (int(mh['level_B_mV']), edge),
        })
        self.log(f"[HBT] MultiHarp ready: A=ch{ch_a}@{mh['level_A_mV']}mV, "
                 f"B=ch{ch_b}@{mh['level_B_mV']}mV, edge={edge}")

        # optional attenuation
        self.filter_wheel = self._get_or_make('filter_wheel', Thorlabs_FW102C)
        if self.filter_wheel is not None:
            try:
                self.filter_wheel.update({'OD': self.settings['Filter Wheel OD']})
            except Exception as e:
                self.log(f"[HBT] filter-wheel OD set failed: {e}")

        # optional CW illumination via Proteus ch4 (same as confocal point)
        laser = self.settings['laser']
        if laser['control_laser']:
            self.proteus = self._get_or_make('proteus', ProteusDevice)
            if self.proteus is not None:
                try:
                    self.proteus.set_channel_voltage_high(4, laser['laser_control'])
                    self.log(f"[HBT] green ON (Proteus ch4 = {laser['laser_control']})")
                except Exception as e:
                    self.log(f"[HBT] laser ON failed: {e}")
            else:
                self.log("[HBT] control_laser=True but no Proteus available; "
                         "assuming the laser is already on.")
        time.sleep(float(laser['settle_time']))

    def cleanup(self):
        """Switch the green off (if we turned it on) and release the MultiHarp."""
        laser = self.settings['laser']
        if (laser['control_laser'] and laser['laser_off_after']
                and getattr(self, 'proteus', None) is not None):
            try:
                self.proteus.set_channel_voltage_high(4, 0.0)
                self.log("[HBT] green OFF")
            except Exception as e:
                self.log(f"[HBT] laser OFF failed: {e}")
        mh = getattr(self, 'multiharp', None)
        if mh is not None:
            try:
                mh.close()
            finally:
                self.multiharp = None

    # ------------------------------------------------------------------ #
    # acquisition of one block -> two photon-arrival streams (ps)
    # ------------------------------------------------------------------ #
    def _acquire_block(self, block_s: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.sim_mode:
            s = self.settings['simulation']
            return hbt.simulate_hbt_streams(
                duration_s=block_s,
                rate_cps=float(s['sim_count_rate']),
                g2_zero=float(s['sim_g2_zero']),
                tau_c_ns=float(s['sim_tau_c']),
                tau0_ns=float(self.settings['fit']['tau0']),
                jitter_ps=float(s['sim_jitter_ps']),
                rng=self._rng,
            )
        mh = self.settings['multiharp']
        ch_a, ch_b = int(mh['detector_A_channel']), int(mh['detector_B_channel'])
        records = self.multiharp.acquire(block_s)             # passive T2 capture
        decoded = self.multiharp.decode_channels(records, [ch_a, ch_b])
        return decoded[ch_a], decoded[ch_b]

    # ------------------------------------------------------------------ #
    # main measurement: accumulate g2 across blocks, then fit
    # ------------------------------------------------------------------ #
    def run_hbt(self) -> Dict:
        corr = self.settings['correlation']
        acq = self.settings['acquisition']
        bin_ns = float(corr['bin_width'])
        window_ns = float(corr['tau_window'])
        bin_ps = bin_ns * hbt.PS_PER_NS
        window_ps = window_ns * hbt.PS_PER_NS

        integration = float(acq['integration_time'])
        block = max(1e-3, float(acq['block_time']))
        n_blocks = max(1, int(math.ceil(integration / block)))

        hist = None
        denom = 0.0
        counts_a = counts_b = 0
        elapsed = 0.0
        tau_ns = None

        self.log(f"[HBT] starting: {n_blocks} block(s) x {block:g}s "
                 f"(window +/-{window_ns:g} ns, bin {bin_ns:g} ns)"
                 + ("  [SIM]" if self.sim_mode else ""))

        for k in range(n_blocks):
            if self._abort:
                self.log("[HBT] aborted by user")
                break
            t_a, t_b = self._acquire_block(block)
            centers_ps, h = hbt.cross_correlate(t_a, t_b, window_ps, bin_ps)
            hist = h.copy() if hist is None else hist + h
            counts_a += int(t_a.size)
            counts_b += int(t_b.size)
            denom += hbt.coincidence_denominator(t_a.size, t_b.size, bin_ps, block * 1e12)
            elapsed += block
            tau_ns = centers_ps / hbt.PS_PER_NS
            g2 = hbt.normalize_g2(hist, denom)

            # live snapshot so the GUI can plot progressively
            self.data = {
                'tau_ns': tau_ns, 'g2': g2, 'histogram': hist,
                'counts_A': counts_a, 'counts_B': counts_b,
                'elapsed_s': elapsed, 'success': True, 'aborted': self._abort,
            }
            self.progress = int(100 * (k + 1) / n_blocks)
            self.updateProgress.emit(self.progress)
            self.log(f"[HBT] block {k + 1}/{n_blocks}: "
                     f"A={counts_a:,} ({counts_a / max(elapsed, 1e-9):,.0f} cps), "
                     f"B={counts_b:,} ({counts_b / max(elapsed, 1e-9):,.0f} cps)")

        # fit the accumulated g2
        fit: Dict = {}
        if tau_ns is not None and self.settings['fit']['do_fit']:
            fit = hbt.fit_g2(
                tau_ns, self.data['g2'],
                g2_zero_guess=0.3,
                tau_c_guess_ns=float(self.settings['fit']['tau_bunch']),
                tau0_guess_ns=float(self.settings['fit']['tau0']),
            )
            if fit.get('success'):
                self.log(f"[HBT] fit: g2(0)={fit['g2_zero']:.3f} +/- "
                         f"{fit['g2_zero_err']:.3f}, tau_c={fit['tau_c_ns']:.1f} ns")
            else:
                self.log(f"[HBT] fit failed: {fit.get('reason')}")

        results = dict(self.data) if self.data else {'success': False}
        results['fit'] = fit
        got_fit = bool(fit.get('success'))
        results['g2_zero'] = fit['g2_zero'] if got_fit else float('nan')
        results['tau_c_ns'] = fit['tau_c_ns'] if got_fit else float('nan')
        results['tau0_ns_fit'] = fit['tau0_ns'] if got_fit else float('nan')
        results['integration_time_s'] = elapsed
        results['bin_width_ns'] = bin_ns
        results['tau_window_ns'] = window_ns
        results['channel_A'] = int(self.settings['multiharp']['detector_A_channel'])
        results['channel_B'] = int(self.settings['multiharp']['detector_B_channel'])
        return results

    def _function(self):
        """Framework entry point (called by Experiment.run())."""
        if not self.sim_mode:
            self.setup()
        try:
            results = self.run_hbt()
        finally:
            if not self.sim_mode:
                self.cleanup()

        self.data = results
        if self.settings['save']:
            self._save_plot_png()
            self.save_hdf5()
        self.log("[HBT] done")

    # ------------------------------------------------------------------ #
    # saving
    # ------------------------------------------------------------------ #
    def save_hdf5(self):
        """Custom data + metadata, then defer to the base save_hdf_data (which
        also grabs any Get-Basic-Data devices selected in the GUI)."""
        d = self.data or {}
        structure_to_save = MyStruct()
        structure_to_save.data = MyStruct(
            tau_ns=np.asarray(d.get('tau_ns', [])),
            g2=np.asarray(d.get('g2', [])),
            histogram=np.asarray(d.get('histogram', [])),
        )
        fit = d.get('fit', {}) or {}
        structure_to_save.meta = MyStruct(
            g2_zero=d.get('g2_zero', float('nan')),
            tau_c_ns=d.get('tau_c_ns', float('nan')),
            tau0_ns_fit=d.get('tau0_ns_fit', float('nan')),
            fit_success=bool(fit.get('success', False)),
            counts_A=d.get('counts_A', 0),
            counts_B=d.get('counts_B', 0),
            integration_time_s=d.get('integration_time_s', 0.0),
            bin_width_ns=d.get('bin_width_ns', 0.0),
            tau_window_ns=d.get('tau_window_ns', 0.0),
            channel_A=d.get('channel_A', -1),
            channel_B=d.get('channel_B', -1),
            level_A_mV=self.settings['multiharp']['level_A_mV'],
            level_B_mV=self.settings['multiharp']['level_B_mV'],
            edge=self.settings['multiharp']['edge'],
            laser_control=self.settings['laser']['laser_control'],
            filter_wheel_OD=self.settings['Filter Wheel OD'],
            sim_mode=self.sim_mode,
            start_time=str(self.start_time),
            end_time=str(self.end_time),
            success=bool(d.get('success', False)),
            aborted=bool(d.get('aborted', False)),
            sample=self.settings['sample'],
        )
        structure_to_save.devices = self.devices
        self.save_hdf_data(structure_to_save)

    def _save_plot_png(self):
        """Save a matplotlib PNG of g2(tau) (+fit) next to the data. Uses the Agg
        backend so it is safe off the GUI main thread."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            self.log(f"[HBT] matplotlib unavailable, skipping PNG: {e}")
            return
        d = self.data or {}
        if 'g2' not in d:
            return
        tau = np.asarray(d['tau_ns'])
        g2 = np.asarray(d['g2'])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(tau, g2, lw=1.1, color='C0', label='data')
        ax.axhline(1.0, color='gray', ls='--', lw=0.8)
        ax.axhline(0.5, color='red', ls=':', lw=0.9, label='g2 = 0.5')
        fit = d.get('fit', {}) or {}
        if fit.get('success'):
            model = hbt.two_level_g2(tau, fit['g2_zero'], fit['tau_c_ns'],
                                     fit['tau0_ns'], fit.get('offset', 1.0))
            ax.plot(tau, model, color='C1', lw=2,
                    label=f"fit: g2(0)={fit['g2_zero']:.3f}, tau_c={fit['tau_c_ns']:.1f} ns")
        ax.set_xlabel('tau (ns)')
        ax.set_ylabel('g2(tau)')
        ax.set_title("HBT g2  (A={:,}, B={:,}, T={:.0f} s{})".format(
            d.get('counts_A', 0), d.get('counts_B', 0),
            d.get('integration_time_s', 0.0), ", SIM" if self.sim_mode else ""))
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right')
        try:
            base = Path(self.settings['path'])
            base.mkdir(parents=True, exist_ok=True)
            ts = self.start_time.strftime('%y%m%d-%H_%M_%S')
            out = base / f"{ts}_{self.settings['tag']}_g2.png"
            fig.savefig(out, dpi=150, bbox_inches='tight')
            self.log(f"[HBT] saved plot: {out}")
        except Exception as e:
            self.log(f"[HBT] could not save PNG: {e}")
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------ #
    # pyqtgraph plotting (GUI)
    # ------------------------------------------------------------------ #
    def _plot(self, axes_list, data=None):
        if data is None:
            data = self.data
        if not data or 'g2' not in data:
            return
        import pyqtgraph as pg
        tau = np.asarray(data['tau_ns'])
        g2 = np.asarray(data['g2'])
        g2_plot = np.where(np.isfinite(g2), g2, 0.0)

        ax = axes_list[0]
        ax.clear()
        ax.plot(tau, g2_plot, pen=pg.mkPen('c', width=1))
        ax.addLine(y=1.0, pen=pg.mkPen('g', width=1))
        ax.addLine(y=0.5, pen=pg.mkPen('r', width=1))
        fit = data.get('fit', {}) or {}
        if fit.get('success'):
            model = hbt.two_level_g2(tau, fit['g2_zero'], fit['tau_c_ns'],
                                     fit['tau0_ns'], fit.get('offset', 1.0))
            ax.plot(tau, model, pen=pg.mkPen('y', width=2))
        ax.showGrid(x=True, y=True)
        ax.setLabel('bottom', 'tau', units='ns')
        ax.setLabel('left', 'g2(tau)')

        if len(axes_list) > 1 and axes_list[1] is not None:
            g0 = data.get('g2_zero', float('nan'))
            axes_list[1].setText(f"g2(0) = {g0:.3f}"
                                 f"   ({data.get('counts_A', 0):,} / "
                                 f"{data.get('counts_B', 0):,} cts)")

    def get_axes_layout(self, figure_list):
        """One g2 plot, plus a text label showing g2(0) when a second figure is
        provided (mirrors the confocal point layout)."""
        axes_list = []
        if self._plot_refresh is True:
            for graph in figure_list:
                graph.clear()
            axes_list.append(figure_list[0].addPlot(row=0, col=0))
            if len(figure_list) > 1:
                import pyqtgraph as pg
                label = pg.LabelItem(text='', size='12pt', bold=True)
                figure_list[1].addItem(label, row=0, col=0)
                axes_list.append(label)
        else:
            for graph in figure_list:
                axes_list.append(graph.getItem(row=0, col=0))
        return axes_list


if __name__ == '__main__':
    # Hardware-free smoke test: run the full pipeline in simulation and report g2(0).
    logging.basicConfig(level=logging.INFO)
    exp = HBTExperiment(name="hbt_sim", mode="sim")
    exp.settings['acquisition']['integration_time'] = 20.0
    exp.settings['acquisition']['block_time'] = 4.0
    exp.settings['simulation']['sim_g2_zero'] = 0.12
    exp.run()
    d = exp.data
    print(f"\nSIM result: fitted g2(0) = {d.get('g2_zero'):.3f} "
          f"(injected {exp.settings['simulation']['sim_g2_zero']}), "
          f"tau_c = {d.get('tau_c_ns'):.1f} ns, "
          f"A={d.get('counts_A'):,} B={d.get('counts_B'):,}")
    print("For rigorous checks (no framework needed) run:  python hbt_selftest.py")
