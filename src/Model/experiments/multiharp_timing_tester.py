# Written by <Jannet Trabelsi>
#!/usr/bin/env python3
"""
multiharp_timing_tester.py
==========================

Standalone timing-diagnostic harness for the ODMR pulsed experiment, built
around a PicoQuant MultiHarp 150/160 used as a two-channel time tagger.

Purpose
-------
Instead of hand-calibrating delays (running the experiment, eyeballing the
signal/reference with MW off, and nudging the calibration constants), this
class lets the MultiHarp *measure* the timing directly while the ODMR sequence
runs completely unchanged. It listens passively; it does not drive the Proteus,
the ADwin, or the SG384.

It supports two comparisons:

  case="readout_vs_spcm"   (default)
      start = Proteus channel 4 MKR, wired to mimic ONLY the readout pulse
      stop  = SPCM output
      -> builds the photon-arrival histogram relative to the readout edge and
         estimates the rise time (10%->90% of the leading edge). This is the
         "why is the rise time so long" measurement.

  case="init_vs_trigger"
      start = ADwin DIGOUT 28  (a copy of DIGOUT 21, the line that triggers
              the Proteus)
      stop  = Proteus channel 4 MKR, wired to mimic the init pulse
      -> measures trigger -> Proteus-output latency for every shot and reports
         how stable it is across the run (i.e. across tau steps and across
         repetitions).

  case="rabi_vs_spcm"
      start = a once-per-shot reference edge -- the readout laser MARKER is best
              (add `marker, laser_readout_1 on channel 4 at <readout>ns,
              <len>ns` to the sequence so the Proteus emits one clean edge at the
              readout onset); the trigger works too.
      stop  = SPCM output
      -> reconstructs the Rabi curve FROM THE MULTIHARP, with no ADwin gating:
         per shot it counts SPCM photons in a signal window on the bright readout
         onset (and, optionally, a reference window later inside the same readout
         pulse), groups shots by tau, and averages. This is the "is it the ADwin
         or a real drift?" test: a clean oscillation here while the ADwin curve is
         flat pins the problem on the ADwin counting/gate; a flat curve here means
         the problem is upstream (polarization / MW / optics / a genuine drift).
         See run_rabi_with_experiment and analyze_rabi (and the one desync caveat).

  case="rabi_mw_tagged"   (the desync-proof Rabi)
      start = the readout MARKER (per-shot reference + SPCM signal window), as above
      stop  = SPCM output
      mw    = a THIRD input wired to an attenuated copy of the MW line (the pulse
              whose WIDTH is tau). Set mw_input=<ch> (+ optional mw_fall_input=<ch>).
      -> identical to rabi_vs_spcm, except tau for each shot is read from the MW
         pulse itself instead of from shot order (shot i mod n). WIDTH mode (both
         mw_input and mw_fall_input) measures tau = MW-fall - MW-rise directly and
         needs no geometry; GEOMETRY mode (mw_input only, its FALLING edge) derives
         tau from the MW end vs the marker using the fixed MW->readout gap
         (mw_gap_ns). Each measured tau is snapped to the nearest sweep step, so a
         dropped edge drops only that one shot instead of shifting every later one.
         This is the bullet-proof way to run the "ADwin vs real drift" check.

Which physical MultiHarp BNC is "start" and which is "stop" is up to us; set
start_input / stop_input to the 0-based MultiHarp input-channel indices we
plugged into (default 0 and 1). Channel numbering here is 0-based to match
MHLib and the T2 record format; MHLib input channel 0 is the connector labelled
"1" on the front panel.

The MultiHarp inputs are level-trigger inputs with an operating range of only
-1200 mV to +1200 mV (pulse peak into 50 ohm) and a damage level of +/-2500 mV.
That means:
  * A raw SPCM TTL pulse (~2.5 V into 50 ohm for SPCM-AQRH) is at/over the
    damage level. ATTENUATE IT (e.g. a 6-10 dB inline attenuator) before the
    MultiHarp, and set the trigger level to about half the attenuated amplitude.
  * A raw ADwin DIGOUT (3.3 V / 5 V TTL) is over the damage level. ATTENUATE IT
    too.

Set the per-channel trigger levels/edges in the ChannelConfig objects below to
match whatever amplitudes are present.

Usage (from odmr_pulsed.py __main__ or a small test script)
-----------------------------------------------------------
    from multiharp_timing_tester import MultiHarpTimingTester

    experiment = ODMRPulsedExperiment(name="test_odmr", mode="testing")
    # ... set parameters and LOAD THE SEQUENCE exactly as we do for a normal run ...
    experiment.sequence_text = experiment.create_example_odmr_sequence()
    experiment.load_sequence_from_text()

    tester = MultiHarpTimingTester(case="readout_vs_spcm")   # or "init_vs_trigger"
    out = tester.run_with_experiment(
        experiment,
        experiment.settings['microwave']['frequency range'],
    )
    # out["odmr"]     -> the usual dict returned by experiment.run_experiment(...)
    # out["timing"]   -> TimingResult (histogram, stats, rise time / drift, files)

For these timing tests, run a SINGLE frequency (MW on or off doesn't matter for
readout/trigger timing) so the run is short and the drift-vs-time axis maps
cleanly onto the tau sweep.

----------------------------------------------------------------------
measured values (07/20/26)
Proteus Channel 4 MKR: 720 mV
Adwin DO28 (with 10 + 6 dB attenuators): 610 mV
Proteus Channel 3: 700 mV
Adwin DO16 (with 10 + 6 dB attenuators): 620 mV
SPCM (with 6 + 3 dB attenuators): 640 mV
>> all safe for Multiharp
"""

from __future__ import annotations

import ctypes as ct
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Plotting is optional; the harness still returns all numbers if matplotlib
# isn't available or a display isn't present.
try:
    import matplotlib
    matplotlib.use("Agg")  # file output only; safe on headless lab PCs
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


# --------------------------------------------------------------------------- #
# MHLib constants
# --------------------------------------------------------------------------- #
MODE_T2 = 2                     # MH_Initialize measurement mode: T2 time tagging
REFSRC_INTERNAL = 0             # internal clock
TTREADMAX = 1048576             # fixed FIFO read block size (event records)
T2WRAP = 33554432               # 2**25, T2 overflow period in base-res units
EDGE_FALLING = 0
EDGE_RISING = 1
FLAG_FIFOFULL = 0x0002          # MH_GetFlags bit: FIFO overrun (data lost)


@dataclass
class ChannelConfig:
    """Trigger settings for one MultiHarp input (all levels in mV into 50 ohm)."""
    level_mV: int = 200
    edge: int = EDGE_RISING          # EDGE_RISING or EDGE_FALLING
    offset_ps: int = 0               # per-channel cable-delay compensation, +/-100 ns


@dataclass
class TimingResult:
    """Everything the analysis produces for one test run."""
    case: str
    start_input: int
    stop_input: int
    n_start_events: int
    n_stop_events: int
    n_pairs: int
    base_resolution_ps: float

    # Per-pair data (absolute start time and the start->stop delay), so we can
    # slice by tau iteration
    start_times_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    delays_ns: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Histogram of the delay.
    hist_edges_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    hist_counts: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Summary statistics of the delay distribution (ns).
    mean_ns: float = float("nan")
    median_ns: float = float("nan")
    std_ns: float = float("nan")
    fwhm_ns: float = float("nan")

    # Rise-time metrics (case="readout_vs_spcm"): the leading edge of the
    # arrival profile. All in ns; NaN when not applicable.
    onset_ns: float = float("nan")        # t at 10% of plateau
    rise_time_ns: float = float("nan")    # t(90%) - t(10%)
    plateau_counts: float = float("nan")

    # Drift across the run (case="init_vs_trigger" mainly): the run is split
    # into segments in time order; per-segment mean delay lets us confirm the
    # latency is constant over tau / repetitions.
    segment_mid_time_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    segment_mean_delay_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    segment_count: np.ndarray = field(default_factory=lambda: np.empty(0))
    drift_pp_ns: float = float("nan")     # peak-to-peak of segment means

    fifo_overrun: bool = False
    saved_files: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"[MultiHarp timing] case={self.case}  "
            f"start=ch{self.start_input}  stop=ch{self.stop_input}",
            f"  events: start={self.n_start_events}  stop={self.n_stop_events}  "
            f"paired={self.n_pairs}",
            f"  delay: mean={self.mean_ns:.3f} ns  median={self.median_ns:.3f} ns  "
            f"std={self.std_ns:.3f} ns  FWHM={self.fwhm_ns:.3f} ns",
        ]
        if np.isfinite(self.rise_time_ns):
            lines.append(
                f"  rise: onset(10%)={self.onset_ns:.2f} ns  "
                f"rise_time(10-90%)={self.rise_time_ns:.2f} ns  "
                f"plateau={self.plateau_counts:.0f} cts/bin"
            )
        if np.isfinite(self.drift_pp_ns):
            lines.append(f"  drift across run (peak-to-peak of segment means): "
                         f"{self.drift_pp_ns:.3f} ns")
        if self.fifo_overrun:
            lines.append("  WARNING: FIFO overrun occurred -- some events were lost.")
        if self.saved_files:
            lines.append("  saved: " + ", ".join(self.saved_files))
        return "\n".join(lines)


@dataclass
class RabiResult:
    """Everything the Rabi analysis produces for one MultiHarp-only Rabi run.

    The MultiHarp reconstructs the Rabi curve itself: per shot it counts SPCM
    photons in a signal window (and optionally a reference window) placed
    relative to a once-per-shot reference edge (the readout laser marker or the
    trigger), then groups shots by tau step and averages. This is deliberately
    INDEPENDENT of the ADwin gate, so a clean oscillation here with a bad one
    from the ADwin points squarely at the ADwin counting/gating -- and a bad one
    here points upstream (spin / optics / MW / a real setup drift)."""
    start_input: int
    stop_input: int
    n_ref_events: int          # once-per-shot reference edges seen
    n_spcm_events: int         # SPCM photons seen
    n_shots_used: int          # reference edges actually assigned to a tau bin
    n_iterations: int          # number of tau steps in the sweep
    repeat_count: Optional[int]
    base_resolution_ps: float

    # Window definitions (ns, relative to the reference edge), echoed for the record.
    sig_offset_ns: float = float("nan")
    sig_width_ns: float = float("nan")
    ref_offset_ns: float = float("nan")
    ref_width_ns: float = float("nan")
    reference_label: str = ""

    # The Rabi curve, one entry per tau step.
    tau_ns: np.ndarray = field(default_factory=lambda: np.empty(0))
    tau_is_index: bool = False                     # True if tau axis is just 0..n-1
    signal_mean: np.ndarray = field(default_factory=lambda: np.empty(0))     # <cts>/shot in sig window
    signal_sem: np.ndarray = field(default_factory=lambda: np.empty(0))      # standard error of the mean
    reference_mean: np.ndarray = field(default_factory=lambda: np.empty(0))  # <cts>/shot in ref window (NaN if unused)
    contrast: np.ndarray = field(default_factory=lambda: np.empty(0))        # signal/reference (NaN if no ref window)
    shots_per_tau: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Time-resolved photon arrivals (the "keep the whole 7 us" data). For each
    # tau step, the full SPCM arrival-time distribution RELATIVE TO THE MARKER,
    # finely binned over the readout window. This is what lets you re-pick the
    # signal/reference windows offline (e.g. to skip the laser rise) WITHOUT
    # re-running: sum the bins over any range and divide by shots_per_tau. See
    # rabi_curve_from_histogram() / MultiHarpTimingTester.recompute_rabi().
    time_resolved: bool = False
    arrival_hist: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # [n_tau, n_bins] counts
    arrival_edges_ns: np.ndarray = field(default_factory=lambda: np.empty(0))   # n_bins+1 bin edges (ns after marker)
    arrival_profile_ns: np.ndarray = field(default_factory=lambda: np.empty(0)) # tau-independent readout, ALL shots

    # Simple oscillation descriptors (from the signal, or contrast if available).
    modulation_depth: float = float("nan")   # (max-min)/(max+min) of the curve
    best_metric: str = ""                     # "contrast" or "signal_mean"

    phase_ok: bool = True                     # False if shot count desynced from the sweep
    fifo_overrun: bool = False
    saved_files: List[str] = field(default_factory=list)

    # ---- MW-tagged tau assignment (the bullet-proof mode) ----------------- #
    # When an MW copy is wired to a third input, each shot's tau is read from the
    # MW pulse itself instead of from shot order, so a dropped edge can't smear
    # the curve -- every well-formed shot self-identifies its tau.
    tau_assign_source: str = "shot_order"     # "shot_order" | "mw_width" | "mw_geometry"
    mw_inputs: Tuple[int, ...] = ()           # MultiHarp input(s) carrying the MW copy
    n_matched: int = 0                        # shots whose measured tau snapped to a step
    n_unmatched: int = 0                      # shots the MW tag couldn't place (dropped)
    mw_snap_tol_ns: float = float("nan")      # snap tolerance actually used
    order_agreement: float = float("nan")     # fraction where MW tag == shot-order (cross-check)

    def summary(self) -> str:
        lines = [
            f"[MultiHarp Rabi] start=ch{self.start_input} ({self.reference_label})  "
            f"stop=ch{self.stop_input} (SPCM)",
            f"  events: ref-edges={self.n_ref_events}  SPCM={self.n_spcm_events}  "
            f"shots used={self.n_shots_used}  tau steps={self.n_iterations}",
            f"  signal window: [{self.sig_offset_ns:.0f}, "
            f"{self.sig_offset_ns + self.sig_width_ns:.0f}] ns after the reference edge",
        ]
        if np.isfinite(self.ref_width_ns) and self.ref_width_ns > 0:
            lines.append(f"  reference window: [{self.ref_offset_ns:.0f}, "
                         f"{self.ref_offset_ns + self.ref_width_ns:.0f}] ns after the reference edge")
        if self.tau_assign_source == "shot_order":
            lines.append("  tau assignment: shot ORDER (shot i -> tau i mod n_steps)")
        else:
            how = ("MW pulse WIDTH (two-edge, geometry-free)"
                   if self.tau_assign_source == "mw_width"
                   else "MW falling edge vs the readout marker (fixed-gap geometry)")
            mw = ", ".join(f"ch{c}" for c in self.mw_inputs) or "?"
            lines.append(f"  tau assignment: {how}, MW on {mw}")
            lines.append(f"    matched {self.n_matched:,} shots to a tau step, "
                         f"{self.n_unmatched:,} unmatched (dropped, snap tol "
                         f"+/-{self.mw_snap_tol_ns:.0f} ns)")
            if np.isfinite(self.order_agreement):
                lines.append(f"    cross-check: MW tag agrees with shot order for "
                             f"{100.0 * self.order_agreement:.1f}% of matched shots")
        if np.isfinite(self.modulation_depth):
            lines.append(f"  modulation depth of {self.best_metric} "
                         f"(max-min)/(max+min): {self.modulation_depth:.3f}")
        if self.tau_assign_source == "shot_order" and not self.phase_ok:
            lines.append("  WARNING: shots seen != repeat_count x tau_steps -- the "
                         "shot->tau assignment may be desynced (a dropped reference "
                         "edge shifts every later shot). Treat the curve with care; "
                         "wire the MW line to a third input (mw_input=...) to remove "
                         "the ordering assumption entirely.")
        if self.fifo_overrun:
            lines.append("  WARNING: FIFO overrun occurred -- some events were lost.")
        if self.time_resolved and self.arrival_hist.size:
            nb = self.arrival_hist.shape[1]
            span = self.arrival_edges_ns[-1] if self.arrival_edges_ns.size else 0.0
            lines.append(f"  time-resolved: kept the full arrival profile "
                         f"({nb} bins over 0-{span:.0f} ns after the marker). Re-pick "
                         f"windows offline with recompute_rabi() -- no re-run needed.")
        if self.saved_files:
            lines.append("  saved: " + ", ".join(self.saved_files))
        return "\n".join(lines)


def rabi_curve_from_histogram(arrival_hist, arrival_edges_ns, shots_per_tau,
                              sig_window_ns, ref_window_ns=None):
    """Re-derive a Rabi curve from a stored time-resolved arrival histogram, for
    ANY signal/reference window -- the whole point of keeping the 7 us data.

    Nothing has to be re-run: pick the windows you want (e.g. start the signal
    window past the laser rise) and this sums the histogram bins over them.

    Parameters
    ----------
    arrival_hist     : [n_tau, n_bins] int array of SPCM counts vs (tau, arrival
                       time relative to the marker). RabiResult.arrival_hist, or
                       the "arrival_hist" entry of the saved .npz.
    arrival_edges_ns : n_bins+1 bin edges in ns after the marker (arrival_edges_ns).
    shots_per_tau    : shots that went into each tau row (for the per-shot mean).
    sig_window_ns    : (lo, hi) ns after the marker for the SIGNAL window.
    ref_window_ns    : optional (lo, hi) for a REFERENCE window -> contrast.

    Returns a dict: tau_index, signal_sum, signal_mean (per shot), and, if a
    reference window is given, reference_sum, reference_mean, contrast. Windows
    snap to the nearest bin edge; the effective windows actually used are echoed
    back as sig_window_used_ns / ref_window_used_ns.
    """
    H = np.asarray(arrival_hist)
    edges = np.asarray(arrival_edges_ns, dtype=float)
    spt = np.asarray(shots_per_tau, dtype=float)
    if H.ndim != 2 or edges.size != H.shape[1] + 1:
        raise ValueError("arrival_hist must be [n_tau, n_bins] with "
                         "len(arrival_edges_ns) == n_bins + 1")

    def _sum(win):
        lo, hi = float(win[0]), float(win[1])
        i0 = int(np.searchsorted(edges, lo, side="left"))
        i1 = int(np.searchsorted(edges, hi, side="left"))
        i0 = max(0, min(i0, H.shape[1]))
        i1 = max(i0, min(i1, H.shape[1]))
        return H[:, i0:i1].sum(axis=1).astype(float), (edges[i0], edges[min(i1, edges.size - 1)])

    sig_sum, sig_used = _sum(sig_window_ns)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_mean = np.where(spt > 0, sig_sum / spt, np.nan)
    out = {"tau_index": np.arange(H.shape[0]),
           "signal_sum": sig_sum, "signal_mean": sig_mean,
           "sig_window_used_ns": sig_used}
    if ref_window_ns is not None:
        ref_sum, ref_used = _sum(ref_window_ns)
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_mean = np.where(spt > 0, ref_sum / spt, np.nan)
            contrast = np.where(ref_mean > 0, sig_mean / ref_mean, np.nan)
        out.update(reference_sum=ref_sum, reference_mean=ref_mean,
                   contrast=contrast, ref_window_used_ns=ref_used)
    return out


# --------------------------------------------------------------------------- #
# Thin ctypes wrapper around MHLib. All the version-/platform-specific
# assumptions live here so there is exactly one place to adjust.
# --------------------------------------------------------------------------- #
class MHLibWrapper:
    def __init__(self, dll_path: Optional[str] = None):
        self._lib = self._load_library(dll_path)
        self._declare_prototypes()
        self.devidx: Optional[int] = None

    @staticmethod
    def _load_library(dll_path: Optional[str]):
        is_windows = platform.system() == "Windows"
        candidates = []
        if dll_path:
            candidates.append(dll_path)
        if is_windows:
            candidates += ["mhlib64.dll", "mhlib.dll"]
        else:
            candidates += ["libmhlib.so", "libmhlib.so.3"]
        last_err = None
        for name in candidates:
            try:
                # MHLib is cdecl on both platforms -> CDLL.
                return ct.CDLL(name)
            except OSError as e:
                last_err = e
        raise OSError(
            "Could not load MHLib. Tried: %s. Pass dll_path=... or make sure the "
            "MultiHarp software / MHLib is installed. Underlying error: %s"
            % (candidates, last_err)
        )

    def _declare_prototypes(self):
        L = self._lib
        c_int, c_uint, c_double, c_char_p = ct.c_int, ct.c_uint, ct.c_double, ct.c_char_p
        P_int, P_uint, P_double = ct.POINTER(c_int), ct.POINTER(c_uint), ct.POINTER(c_double)

        def sig(fn, argtypes):
            f = getattr(L, fn)
            f.argtypes = argtypes
            f.restype = c_int
            return f

        self.MH_GetLibraryVersion = sig("MH_GetLibraryVersion", [c_char_p])
        self.MH_GetErrorString = sig("MH_GetErrorString", [c_char_p, c_int])
        self.MH_OpenDevice = sig("MH_OpenDevice", [c_int, c_char_p])
        self.MH_CloseDevice = sig("MH_CloseDevice", [c_int])
        self.MH_Initialize = sig("MH_Initialize", [c_int, c_int, c_int])
        self.MH_GetHardwareInfo = sig(
            "MH_GetHardwareInfo", [c_int, c_char_p, c_char_p, c_char_p])
        self.MH_GetNumOfInputChannels = sig(
            "MH_GetNumOfInputChannels", [c_int, P_int])
        self.MH_GetBaseResolution = sig(
            "MH_GetBaseResolution", [c_int, P_double, P_int])
        self.MH_SetSyncDiv = sig("MH_SetSyncDiv", [c_int, c_int])
        self.MH_SetSyncEdgeTrg = sig("MH_SetSyncEdgeTrg", [c_int, c_int, c_int])
        self.MH_SetSyncChannelOffset = sig("MH_SetSyncChannelOffset", [c_int, c_int])
        self.MH_SetInputEdgeTrg = sig(
            "MH_SetInputEdgeTrg", [c_int, c_int, c_int, c_int])
        self.MH_SetInputChannelOffset = sig(
            "MH_SetInputChannelOffset", [c_int, c_int, c_int])
        self.MH_SetInputChannelEnable = sig(
            "MH_SetInputChannelEnable", [c_int, c_int, c_int])
        self.MH_StartMeas = sig("MH_StartMeas", [c_int, c_int])
        self.MH_StopMeas = sig("MH_StopMeas", [c_int])
        self.MH_CTCStatus = sig("MH_CTCStatus", [c_int, P_int])
        self.MH_GetFlags = sig("MH_GetFlags", [c_int, P_int])
        # MH_ReadFiFo (MHLib v3.x): reads up to TTREADMAX records into buffer and
        # returns the count in *nactual. If we are on an OLD MHLib (v1/v2) whose
        # signature is (devidx, buffer, count, *nactual), add a c_int before the
        # pointer here and pass TTREADMAX in read_fifo() below.
        self.MH_ReadFiFo = sig("MH_ReadFiFo", [c_int, P_uint, P_int])

        # Optional: present in v3.1, absent on very old libs -> loaded lazily.
        try:
            self.MH_SetSyncChannelEnable = sig(
                "MH_SetSyncChannelEnable", [c_int, c_int])
        except AttributeError:
            self.MH_SetSyncChannelEnable = None

    # -- error handling ----------------------------------------------------- #
    def _check(self, code: int, where: str):
        if code < 0:
            buf = ct.create_string_buffer(64)
            try:
                self.MH_GetErrorString(buf, code)
                msg = buf.value.decode(errors="replace")
            except Exception:
                msg = "unknown"
            raise RuntimeError(f"MHLib error in {where}: {code} ({msg})")

    # -- lifecycle ---------------------------------------------------------- #
    def library_version(self) -> str:
        buf = ct.create_string_buffer(8)
        self._check(self.MH_GetLibraryVersion(buf), "GetLibraryVersion")
        return buf.value.decode(errors="replace")

    def open_first(self) -> Tuple[int, str]:
        """Open the first responding MultiHarp; return (devidx, serial)."""
        serial = ct.create_string_buffer(8)
        for idx in range(8):  # MAXDEVNUM
            code = self.MH_OpenDevice(idx, serial)
            if code == 0:
                self.devidx = idx
                return idx, serial.value.decode(errors="replace")
        raise RuntimeError("No MultiHarp device could be opened (none found / all busy).")

    def hardware_info(self) -> Tuple[str, str, str]:
        model = ct.create_string_buffer(24)
        partno = ct.create_string_buffer(8)
        version = ct.create_string_buffer(8)
        self._check(self.MH_GetHardwareInfo(self.devidx, model, partno, version),
                    "GetHardwareInfo")
        return (model.value.decode(errors="replace"),
                partno.value.decode(errors="replace"),
                version.value.decode(errors="replace"))

    def initialize_t2(self):
        self._check(self.MH_Initialize(self.devidx, MODE_T2, REFSRC_INTERNAL),
                    "Initialize(T2)")

    def num_input_channels(self) -> int:
        n = ct.c_int()
        self._check(self.MH_GetNumOfInputChannels(self.devidx, ct.byref(n)),
                    "GetNumOfInputChannels")
        return n.value

    def base_resolution_ps(self) -> float:
        res = ct.c_double()
        binsteps = ct.c_int()
        self._check(self.MH_GetBaseResolution(self.devidx, ct.byref(res),
                                              ct.byref(binsteps)),
                    "GetBaseResolution")
        return float(res.value)

    def set_sync_div(self, div: int):
        self._check(self.MH_SetSyncDiv(self.devidx, div), "SetSyncDiv")

    def set_sync_trigger(self, level_mV: int, edge: int):
        self._check(self.MH_SetSyncEdgeTrg(self.devidx, level_mV, edge),
                    "SetSyncEdgeTrg")

    def set_sync_offset(self, offset_ps: int):
        self._check(self.MH_SetSyncChannelOffset(self.devidx, offset_ps),
                    "SetSyncChannelOffset")

    def set_sync_enable(self, enable: bool):
        if self.MH_SetSyncChannelEnable is not None:
            self._check(self.MH_SetSyncChannelEnable(self.devidx, 1 if enable else 0),
                        "SetSyncChannelEnable")

    def set_input_trigger(self, channel: int, level_mV: int, edge: int):
        self._check(self.MH_SetInputEdgeTrg(self.devidx, channel, level_mV, edge),
                    f"SetInputEdgeTrg(ch{channel})")

    def set_input_offset(self, channel: int, offset_ps: int):
        self._check(self.MH_SetInputChannelOffset(self.devidx, channel, offset_ps),
                    f"SetInputChannelOffset(ch{channel})")

    def set_input_enable(self, channel: int, enable: bool):
        self._check(self.MH_SetInputChannelEnable(self.devidx, channel,
                                                  1 if enable else 0),
                    f"SetInputChannelEnable(ch{channel})")

    def start(self, tacq_ms: int):
        self._check(self.MH_StartMeas(self.devidx, tacq_ms), "StartMeas")

    def stop(self):
        self._check(self.MH_StopMeas(self.devidx), "StopMeas")

    def ctc_done(self) -> bool:
        status = ct.c_int()
        self._check(self.MH_CTCStatus(self.devidx, ct.byref(status)), "CTCStatus")
        return status.value != 0

    def flags(self) -> int:
        f = ct.c_int()
        self._check(self.MH_GetFlags(self.devidx, ct.byref(f)), "GetFlags")
        return f.value

    def read_fifo(self, buffer) -> int:
        """Read one FIFO block into `buffer` (a c_uint * TTREADMAX). Returns count."""
        nactual = ct.c_int()
        self._check(self.MH_ReadFiFo(self.devidx, buffer, ct.byref(nactual)),
                    "ReadFiFo")
        # Old-MHLib variant, if we changed the prototype above:
        #   self.MH_ReadFiFo(self.devidx, buffer, TTREADMAX, ct.byref(nactual))
        return nactual.value

    def close(self):
        if self.devidx is not None:
            try:
                self.MH_CloseDevice(self.devidx)
            finally:
                self.devidx = None


# --------------------------------------------------------------------------- #
# The test harness
# --------------------------------------------------------------------------- #
class MultiHarpTimingTester:
    """
    Runs the ODMR sequence unchanged while the MultiHarp time-tags two inputs,
    then computes the delay distribution / rise time / drift.
    """

    # Presets pick the correlation strategy, labels and sensible defaults.
    _PRESETS = {
        "readout_vs_spcm": dict(
            corr_mode="all_in_window",   # collect ALL photons after each readout edge
            window_ns=2000.0,            # look 2 us past the readout edge
            start_label="Proteus ch4 MKR (readout copy)",
            stop_label="SPCM",
        ),
        "init_vs_trigger": dict(
            corr_mode="nearest",         # one init pulse per trigger
            window_ns=4000.0,            # trigger->output should be well under 4 us
            start_label="ADwin DIGOUT 28 (trigger copy of DIGOUT 21)",
            stop_label="Proteus ch4 MKR (init copy)",
        ),
        "adwin_readout_vs_spcm": dict(
            corr_mode="around",
            window_ns=4000.0,            # look this far AFTER the gate-open edge
            pre_window_ns=1000.0,        # ...and this far BEFORE it (negative dt)
            start_label="ADwin DIGOUT 16 gate-open (readout)",   # start = reference
            stop_label="SPCM",
        ),
        # Rabi reconstructed BY THE MULTIHARP (no ADwin gating at all). start is a
        # once-per-shot reference edge -- the readout laser marker is best because
        # it marks the readout directly and (per the init_vs_trigger run) is rock
        # solid; the trigger works too. stop is the SPCM. See analyze_rabi / the
        # RabiResult docstring for how the tau curve is built and its one caveat.
        "rabi_vs_spcm": dict(
            corr_mode="around",          # not used for the curve, but keeps ranges sane
            window_ns=4000.0,
            pre_window_ns=500.0,
            start_label="Proteus ch4 MKR readout (per-shot reference)",
            stop_label="SPCM",
        ),
        # Same as rabi_vs_spcm, but tau is read PER SHOT from a third input wired to
        # the MW line (its pulse width == tau) instead of from shot order. This is
        # the desync-proof version: a dropped edge can no longer smear the curve,
        # because every well-formed shot self-identifies its tau. start is still the
        # readout MARKER (per-shot reference + SPCM signal window), stop is the SPCM,
        # and mw_input (+ optional mw_fall_input) carry the MW copy. See analyze_rabi.
        "rabi_mw_tagged": dict(
            corr_mode="around",
            window_ns=4000.0,
            pre_window_ns=500.0,
            start_label="Proteus ch4 MKR readout (per-shot reference)",
            stop_label="SPCM",
        ),
    }
    # Cases that reconstruct a Rabi curve from MultiHarp tags (no ADwin gating).
    _RABI_CASES = ("rabi_vs_spcm", "rabi_mw_tagged")

    def __init__(
        self,
        case: str = "readout_vs_spcm",
        start_input: int = 0,
        stop_input: int = 1,
        start_cfg: Optional[ChannelConfig] = None,
        stop_cfg: Optional[ChannelConfig] = None,
        window_ns: Optional[float] = None,
        hist_bin_ns: float = 1.0,
        n_drift_segments: int = 20,
        sync_div: int = 1,
        dll_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        make_plots: bool = True,
        output_mode: str = "both",
        save_hdf5: bool = True,
        save_npz: bool = True,
        pre_window_ns: Optional[float] = None,
        gate_ns: Optional[float] = None,
        scan_all_channels: bool = False,
        # ---- Rabi-case parameters (case="rabi_vs_spcm") --------------------- #
        # All in ns, relative to the once-per-shot reference edge on start_input.
        # The signal window should sit on the BRIGHT onset of the readout (first
        # few hundred ns, where NV spin contrast is largest). The reference window
        # (optional) should sit LATER but still INSIDE the readout laser pulse, so
        # dividing signal/reference cancels laser-power and photon-collection drift.
        rabi_sig_offset_ns: float = 0.0,
        rabi_sig_width_ns: float = 300.0,
        rabi_ref_offset_ns: float = 5000.0,
        rabi_ref_width_ns: float = 300.0,
        rabi_n_iterations: Optional[int] = None,     # tau steps; auto-detected if None
        rabi_tau_values_ns: Optional[np.ndarray] = None,  # x-axis; auto-detected if None
        # ---- MW-tagged tau assignment (case="rabi_mw_tagged") -------------- #
        # Wire an attenuated copy of the MW line (the IQ/Proteus channel that
        # carries the variable-width MW pulse) to a THIRD MultiHarp input. Its
        # pulse WIDTH is tau, so each shot's tau is read from the data instead of
        # from shot order -- a dropped edge then drops only that one shot rather
        # than shifting every later one. Two ways to read it:
        #   * WIDTH mode (recommended, geometry-free): give BOTH mw_input and
        #     mw_fall_input. mw_input is forced to the RISING edge (MW start),
        #     mw_fall_input to the FALLING edge (MW end); tau = fall - rise.
        #   * GEOMETRY mode (one BNC): give only mw_input (forced FALLING = MW
        #     end). tau = mw_gap_ns - (marker_time - mw_falling_time), where
        #     mw_gap_ns is the FIXED waveform separation between the MW pulse
        #     START and the readout-marker START (e.g. readout@5900 - MW@2900 =
        #     3000). Exact for square MW pulses (the usual Rabi drive).
        mw_input: Optional[int] = None,              # 3rd input: MW copy (None = tag off)
        mw_cfg: Optional[ChannelConfig] = None,      # its trigger level/edge
        mw_fall_input: Optional[int] = None,         # optional 4th input -> WIDTH mode
        mw_fall_cfg: Optional[ChannelConfig] = None,
        mw_gap_ns: Optional[float] = None,           # readout-start - MW-start (GEOMETRY mode)
        mw_snap_tol_ns: Optional[float] = None,      # max |measured tau - step| to accept
        mw_lookback_ns: Optional[float] = None,      # how far before the marker to hunt MW edges
        # ---- time-resolved arrivals (keep the whole readout, re-window later) #
        # When on, analyze_rabi also builds a per-tau histogram of SPCM arrival
        # times relative to the marker over [0, rabi_tr_window_ns] in
        # rabi_tr_bin_ns bins. Cheap (one extra pass) and it lets you move the
        # signal/reference windows offline -- e.g. to skip the laser rise -- with
        # recompute_rabi(), no re-run needed. Set rabi_time_resolved=False to skip.
        rabi_time_resolved: bool = True,
        rabi_tr_window_ns: float = 7500.0,           # span after the marker to keep (~readout length)
        rabi_tr_bin_ns: float = 10.0,                # arrival-time bin width
        # ---- memory bounds for very large runs ---------------------------- #
        # Long acquisitions can produce billions of T2 records; decoding them all
        # at once would need tens of GB. The Rabi path instead decodes in chunks
        # of this many records and STREAMS the SPCM channel, so peak memory stays
        # small. max_marker_edges guards the one array that must be held whole
        # (the reference/marker edges): if the marker multi-triggers into more
        # than this, analysis stops with a clear message instead of thrashing.
        decode_chunk_records: int = 50_000_000,
        max_marker_edges: int = 200_000_000,
    ):
        if case not in self._PRESETS:
            raise ValueError(f"case must be one of {list(self._PRESETS)}, got {case!r}")
        self.case = case
        preset = self._PRESETS[case]
        self.corr_mode = preset["corr_mode"]
        self.start_label = preset["start_label"]
        self.stop_label = preset["stop_label"]

        # What to plot / save as the primary representation:
        #   "histogram"   -> the start->stop delay histogram (arrival profile / latency)
        #   "time_series" -> the per-shot delay vs time through the run ("data over time")
        #   "both"        -> save everything, plot both panels (default)
        # Aliases accepted for convenience.
        aliases = {
            "hist": "histogram", "histogram": "histogram",
            "time": "time_series", "timeseries": "time_series",
            "time_series": "time_series", "data_over_time": "time_series",
            "both": "both", "all": "both",
        }
        key = str(output_mode).lower().replace(" ", "_")
        if key not in aliases:
            raise ValueError(
                f"output_mode must be 'histogram', 'time_series', or 'both' "
                f"(got {output_mode!r})")
        self.output_mode = aliases[key]
        self.save_hdf5 = bool(save_hdf5)
        self.save_npz = bool(save_npz)

        self.start_input = int(start_input)
        self.stop_input = int(stop_input)
        # Default trigger levels are conservative; OVERRIDE to match our (attenuated!)
        # amplitudes. See the hardware note in the module docstring.
        self.start_cfg = start_cfg or ChannelConfig(level_mV=200, edge=EDGE_RISING)
        self.stop_cfg = stop_cfg or ChannelConfig(level_mV=200, edge=EDGE_RISING)

        self.window_ns = float(window_ns) if window_ns is not None else preset["window_ns"]
        self.pre_window_ns = (float(pre_window_ns) if pre_window_ns is not None
                              else float(preset.get("pre_window_ns", 0.0)))
        self.gate_ns = gate_ns
        self.scan_all_channels = bool(scan_all_channels)

        # Rabi-case configuration (only consulted when case == "rabi_vs_spcm").
        self.rabi_sig_offset_ns = float(rabi_sig_offset_ns)
        self.rabi_sig_width_ns = float(rabi_sig_width_ns)
        self.rabi_ref_offset_ns = float(rabi_ref_offset_ns)
        self.rabi_ref_width_ns = float(rabi_ref_width_ns)
        self.rabi_n_iterations = (int(rabi_n_iterations)
                                  if rabi_n_iterations is not None else None)
        self.rabi_tau_values_ns = (np.asarray(rabi_tau_values_ns, dtype=float)
                                   if rabi_tau_values_ns is not None else None)

        # ---- MW-tagging configuration ------------------------------------- #
        self.mw_input = int(mw_input) if mw_input is not None else None
        self.mw_fall_input = int(mw_fall_input) if mw_fall_input is not None else None
        self.mw_gap_ns = float(mw_gap_ns) if mw_gap_ns is not None else None
        self.mw_snap_tol_ns = (float(mw_snap_tol_ns)
                               if mw_snap_tol_ns is not None else None)
        self.mw_lookback_ns = (float(mw_lookback_ns)
                               if mw_lookback_ns is not None else None)
        # Decide the tagging mode and pin the correct edges (foolproof):
        #   WIDTH mode needs both edges -> mw_input RISING, mw_fall_input FALLING.
        #   GEOMETRY mode uses the single MW END -> mw_input FALLING.
        self.mw_cfg = mw_cfg or ChannelConfig(level_mV=200, edge=EDGE_FALLING)
        self.mw_fall_cfg = mw_fall_cfg or ChannelConfig(level_mV=200, edge=EDGE_FALLING)
        if self.mw_input is not None and self.mw_fall_input is not None:
            self.mw_mode = "width"
            if self.mw_cfg.edge != EDGE_RISING:
                self.mw_cfg = ChannelConfig(level_mV=self.mw_cfg.level_mV,
                                            edge=EDGE_RISING,
                                            offset_ps=self.mw_cfg.offset_ps)
            if self.mw_fall_cfg.edge != EDGE_FALLING:
                self.mw_fall_cfg = ChannelConfig(level_mV=self.mw_fall_cfg.level_mV,
                                                 edge=EDGE_FALLING,
                                                 offset_ps=self.mw_fall_cfg.offset_ps)
        elif self.mw_input is not None:
            self.mw_mode = "geometry"
            if self.mw_cfg.edge != EDGE_FALLING:
                print("[MultiHarp] Rabi MW-tag GEOMETRY mode needs the MW FALLING "
                      "edge (the MW end); overriding mw_cfg edge to falling.")
                self.mw_cfg = ChannelConfig(level_mV=self.mw_cfg.level_mV,
                                            edge=EDGE_FALLING,
                                            offset_ps=self.mw_cfg.offset_ps)
        else:
            self.mw_mode = "off"

        # ---- time-resolved arrival settings ------------------------------- #
        self.rabi_time_resolved = bool(rabi_time_resolved)
        self.rabi_tr_window_ns = float(rabi_tr_window_ns)
        self.rabi_tr_bin_ns = float(rabi_tr_bin_ns)
        self.decode_chunk_records = int(decode_chunk_records)
        self.max_marker_edges = int(max_marker_edges)

        self.hist_bin_ns = float(hist_bin_ns)
        self.n_drift_segments = int(n_drift_segments)
        self.sync_div = int(sync_div)
        self.dll_path = dll_path
        self.output_dir = output_dir
        self.make_plots = make_plots and _HAVE_MPL

        self._mh: Optional[MHLibWrapper] = None
        self._nchan: Optional[int] = None
        self._base_res_ps: float = 5.0
        self._reader: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._chunks: List[np.ndarray] = []
        self._fifo_overrun = False

    # ---- device bring-up -------------------------------------------------- #
    def open_and_configure(self):
        mh = MHLibWrapper(self.dll_path)
        print(f"[MultiHarp] MHLib version {mh.library_version()}")
        idx, serial = mh.open_first()
        print(f"[MultiHarp] opened dev {idx} (S/N {serial})")
        # MH_Initialize MUST come before GetHardwareInfo / GetBaseResolution /
        # GetNumOfInputChannels -- those return MH_ERROR_NOT_INITIALIZED (-22)
        # if the device has not been put into a measurement mode yet.
        mh.initialize_t2()
        model, partno, version = mh.hardware_info()
        self._base_res_ps = mh.base_resolution_ps()
        nchan = mh.num_input_channels()
        print(f"[MultiHarp] {model} (part {partno}, fw {version}) in T2 mode, "
              f"base resolution {self._base_res_ps:.3f} ps, {nchan} input channels")

        # MW-tagging inputs (0, 1 or 2 extra channels) join the active set.
        mw_chans: List[int] = []
        if getattr(self, "mw_mode", "off") != "off":
            mw_chans.append(self.mw_input)
            if self.mw_fall_input is not None:
                mw_chans.append(self.mw_fall_input)
        active = [self.start_input, self.stop_input, *mw_chans]

        for ch in active:
            if not (0 <= ch < nchan):
                mh.close()
                raise ValueError(f"input channel {ch} out of range (device has {nchan})")
        if len(set(active)) != len(active):
            mh.close()
            raise ValueError(f"start/stop/MW inputs must all be different channels "
                             f"(got start={self.start_input}, stop={self.stop_input}, "
                             f"mw={mw_chans})")

        # T2 mode: sync divider must be 1. We don't use the sync channel here.
        mh.set_sync_div(self.sync_div)
        mh.set_sync_trigger(-100, EDGE_FALLING)   # harmless placeholder
        mh.set_sync_offset(0)
        mh.set_sync_enable(False)                 # ignore sync (nothing plugged in)

        self._nchan = nchan
        if self.scan_all_channels:
            edge = 'rising' if self.start_cfg.edge == EDGE_RISING else 'falling'
            print(f"[MultiHarp] SCAN MODE: all {nchan} inputs @ "
                  f"{self.start_cfg.level_mV} mV, {edge} edge (levels equal on all).")
            for ch in range(nchan):
                mh.set_input_enable(ch, True)
                mh.set_input_trigger(ch, self.start_cfg.level_mV, self.start_cfg.edge)
                mh.set_input_offset(ch, 0)
        else:
            for ch in range(nchan):
                mh.set_input_enable(ch, ch in active)
            mh.set_input_trigger(self.start_input, self.start_cfg.level_mV, self.start_cfg.edge)
            mh.set_input_offset(self.start_input, self.start_cfg.offset_ps)
            mh.set_input_trigger(self.stop_input, self.stop_cfg.level_mV, self.stop_cfg.edge)
            mh.set_input_offset(self.stop_input, self.stop_cfg.offset_ps)

        # MW-tag inputs always get their OWN level/edge (even under scan mode), so
        # the width/geometry read is never corrupted by a shared scan level.
        if mw_chans:
            mh.set_input_enable(self.mw_input, True)
            mh.set_input_trigger(self.mw_input, self.mw_cfg.level_mV, self.mw_cfg.edge)
            mh.set_input_offset(self.mw_input, self.mw_cfg.offset_ps)
            if self.mw_fall_input is not None:
                mh.set_input_enable(self.mw_fall_input, True)
                mh.set_input_trigger(self.mw_fall_input, self.mw_fall_cfg.level_mV,
                                     self.mw_fall_cfg.edge)
                mh.set_input_offset(self.mw_fall_input, self.mw_fall_cfg.offset_ps)
            ed = lambda e: "rising" if e == EDGE_RISING else "falling"
            if self.mw_mode == "width":
                print(f"[MultiHarp] Rabi MW-tag WIDTH mode: rise=ch{self.mw_input} "
                      f"({ed(self.mw_cfg.edge)}, {self.mw_cfg.level_mV} mV), "
                      f"fall=ch{self.mw_fall_input} ({ed(self.mw_fall_cfg.edge)}, "
                      f"{self.mw_fall_cfg.level_mV} mV) -> tau = fall - rise.")
            else:
                print(f"[MultiHarp] Rabi MW-tag GEOMETRY mode: MW end on ch{self.mw_input} "
                      f"({ed(self.mw_cfg.edge)}, {self.mw_cfg.level_mV} mV); tau derived "
                      f"vs the readout marker with mw_gap_ns.")

        time.sleep(0.2)  # let the inputs settle after retriggering
        self._mh = mh

    # ---- acquisition (background FIFO reader) ----------------------------- #
    def _read_loop(self):
        buffer = (ct.c_uint * TTREADMAX)()
        while not self._stop_flag.is_set():
            n = self._mh.read_fifo(buffer)
            if n > 0:
                self._chunks.append(np.frombuffer(buffer, dtype=np.uint32,
                                                  count=n).copy())
            else:
                if self._mh.flags() & FLAG_FIFOFULL:
                    self._fifo_overrun = True
                time.sleep(0.001)

    def start_acquisition(self, tacq_ms: int = 360_000_00):
        """Start a long T2 acquisition and spin up the reader thread.

        tacq_ms just needs to outlast the ODMR run; we stop manually when the
        run returns. Default is 100 h (the MHLib maximum in v3.1)."""
        if self._mh is None:
            self.open_and_configure()
        self._chunks = []
        self._fifo_overrun = False
        self._stop_flag.clear()
        self._mh.start(tacq_ms)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        print("[MultiHarp] acquisition started (listening while ODMR runs)")

    def stop_acquisition(self) -> np.ndarray:
        """Stop the reader, stop the measurement, drain the FIFO, return records."""
        self._stop_flag.set()
        if self._reader is not None:
            self._reader.join(timeout=10.0)
        self._mh.stop()
        # Drain whatever is still in the FIFO after StopMeas.
        buffer = (ct.c_uint * TTREADMAX)()
        while True:
            n = self._mh.read_fifo(buffer)
            if n <= 0:
                break
            self._chunks.append(np.frombuffer(buffer, dtype=np.uint32, count=n).copy())
        if self._mh.flags() & FLAG_FIFOFULL:
            self._fifo_overrun = True
        records = (np.concatenate(self._chunks) if self._chunks
                   else np.empty(0, dtype=np.uint32))
        print(f"[MultiHarp] acquisition stopped, {records.size} raw records captured")
        return records

    def close(self):
        if self._mh is not None:
            self._mh.close()
            self._mh = None

    # ---- T2 decoding ------------------------------------------------------ #
    def decode_t2(self, records: np.ndarray) -> Dict[int, np.ndarray]:
        """Decode T2 records into per-input-channel arrival times in ps.

        Returns {channel_index: times_ps}. Only regular photon/event records are
        returned (overflows are folded in, sync and markers are dropped)."""
        if records.size == 0:
            return {self.start_input: np.empty(0), self.stop_input: np.empty(0)}

        r = records.astype(np.uint32)
        special = (r >> 31) & 0x1
        channel = (r >> 25) & 0x3F
        timetag = (r & 0x01FFFFFF).astype(np.uint64)

        # Overflow records: special==1 and channel==0x3F. In the V2/MultiHarp
        # format the timetag field carries the number of overflows (>=1).
        is_ovf = (special == 1) & (channel == 0x3F)
        ovf_step = np.where(is_ovf,
                            np.where(timetag == 0, np.uint64(1), timetag),
                            np.uint64(0))
        ofl = np.cumsum(ovf_step) * np.uint64(T2WRAP)  # running offset per record
        true_units = ofl + timetag                     # base-resolution units
        times_ps = true_units.astype(np.float64) * self._base_res_ps

        is_event = special == 0                         # real input-channel event
        out = {}
        for ch in (self.start_input, self.stop_input):
            mask = is_event & (channel == ch)
            out[ch] = times_ps[mask]                    # already time-ordered
        return out

    def decode_all_channels(self, records: np.ndarray) -> Dict[int, np.ndarray]:
        """Like decode_t2 but returns {channel: times_ps} for EVERY channel."""
        if records.size == 0:
            return {}
        r = records.astype(np.uint32)
        special = (r >> 31) & 0x1
        channel = (r >> 25) & 0x3F
        timetag = (r & 0x01FFFFFF).astype(np.uint64)
        is_ovf = (special == 1) & (channel == 0x3F)  # <- match decode_t2
        ovf_step = np.where(is_ovf, np.where(timetag == 0, np.uint64(1), timetag),
                            np.uint64(0))
        ofl = np.cumsum(ovf_step) * np.uint64(T2WRAP)  # <- match decode_t2
        times_ps = (ofl + timetag).astype(np.float64) * self._base_res_ps
        is_event = special == 0
        out = {}
        for ch in np.unique(channel[is_event]):
            out[int(ch)] = times_ps[is_event & (channel == ch)]
        return out

    # ---- memory-bounded T2 decoding for very large runs ------------------- #
    # decode_t2 / decode_all_channels build several full-length arrays at once
    # (a uint64 cumsum + a float64 time array), which is ~16 bytes/record -- fine
    # for short timing runs but fatal for a 10 h Rabi (billions of records -> tens
    # of GB, MemoryError). The three helpers below decode in fixed-size CHUNKS,
    # carrying the overflow offset across chunk boundaries so the absolute times
    # are identical to the one-shot decode, while peak memory stays ~chunk-sized.
    def _channel_counts(self, records: np.ndarray) -> Dict[int, int]:
        """Count events per channel by streaming (no time arrays). Cheap; used to
        report the channel composition and catch a multi-triggering line early."""
        counts = np.zeros(64, dtype=np.int64)
        chunk = max(1, self.decode_chunk_records)
        for s in range(0, records.size, chunk):
            r = records[s:s + chunk]
            special = (r >> 31) & 0x1
            chan = (r >> 25) & 0x3F
            counts += np.bincount(chan[special == 0], minlength=64).astype(np.int64)
        return {int(ch): int(counts[ch]) for ch in range(64) if counts[ch]}

    def _iter_channel_times(self, records: np.ndarray, channels):
        """Yield {channel: times_ps} for the requested channels, one CHUNK at a
        time, carrying the T2 overflow offset across chunks."""
        channels = [int(c) for c in channels if c is not None]
        base = np.uint64(0)                       # overflow offset so far, base-res units
        chunk = max(1, self.decode_chunk_records)
        for s in range(0, records.size, chunk):
            r = records[s:s + chunk]
            special = (r >> 31) & 0x1
            chan = (r >> 25) & 0x3F
            timetag = (r & 0x01FFFFFF).astype(np.uint64)
            is_ovf = (special == 1) & (chan == 0x3F)
            ovf = np.where(is_ovf, np.where(timetag == 0, np.uint64(1), timetag),
                           np.uint64(0))
            within = np.cumsum(ovf) * np.uint64(T2WRAP)     # cumulative overflow, this chunk
            units = base + within + timetag                 # absolute base-res units
            is_event = special == 0
            out = {}
            for c in channels:
                m = is_event & (chan == c)
                out[c] = units[m].astype(np.float64) * self._base_res_ps
            if within.size:
                base = base + within[-1]           # carry this chunk's total overflow
            yield out

    def _collect_channel_times(self, records: np.ndarray, channels) -> Dict[int, np.ndarray]:
        """Fully materialise times for a few SPARSE channels (markers, MW edges),
        chunk by chunk. Safe only for channels with a bounded number of events."""
        channels = [int(c) for c in channels if c is not None]
        acc = {c: [] for c in channels}
        for chunk_out in self._iter_channel_times(records, channels):
            for c in channels:
                if chunk_out[c].size:
                    acc[c].append(chunk_out[c])
        return {c: (np.concatenate(acc[c]) if acc[c] else np.empty(0)) for c in channels}

    def _report_channel_counts(self, counts: Dict[int, int],
                               expected_markers: Optional[int] = None) -> None:
        if not counts:
            print("[MultiHarp] DIAGNOSTIC: 0 events on ALL channels. Nothing is "
                  "triggering. Check the signals are present, the level sits ~half "
                  "the ATTENUATED amplitude, and the edge matches.")
            return
        print("[MultiHarp] per-channel event counts (0-based input; connector = index+1):")
        for ch in sorted(counts):
            tags = []
            if ch == self.start_input: tags.append("<-START/reference")
            if ch == self.stop_input: tags.append("<-STOP/SPCM")
            if ch == self.mw_input: tags.append("<-MW")
            if ch == self.mw_fall_input: tags.append("<-MW fall")
            print(f"    input {ch}: {counts[ch]:>14,} events  {' '.join(tags)}")
        if expected_markers:
            n_start = counts.get(self.start_input, 0)
            if n_start > 1.5 * expected_markers:
                print(f"[MultiHarp] WARNING: the reference/START channel "
                      f"(input {self.start_input}) has {n_start:,} edges but the sweep "
                      f"only has ~{expected_markers:,} shots (repeat_count x tau steps). "
                      f"That is ~{n_start / max(1, expected_markers):.1f}x too many -- the "
                      f"marker is MULTI-TRIGGERING (ringing/reflection or the level set "
                      f"too low). Raise its trigger level / fix termination so it fires "
                      f"ONCE per shot. The MW tag will reject the spurious edges, but "
                      f"you are wasting acquisition and memory.")

    def _stream_spcm_windows(self, records: np.ndarray, ref_ps: np.ndarray,
                             tau_index: np.ndarray, matched: np.ndarray,
                             n_iterations: int, sig_lo_ps: float, sig_hi_ps: float,
                             use_ref: bool, ref_lo_ps: float, ref_hi_ps: float):
        """Stream the SPCM channel in chunks and accumulate, per tau step:
        signal-window counts, reference-window counts, and (if enabled) the full
        arrival-time histogram -- all WITHOUT ever holding every SPCM time. Each
        photon is attached to the marker at/just-before it (markers are complete
        and sorted), and only photons from matched shots are used."""
        sig_total = np.zeros(n_iterations, dtype=np.int64)
        ref_total = np.zeros(n_iterations, dtype=np.int64)
        do_hist = self.rabi_time_resolved and self.rabi_tr_window_ns > 0
        nbins = max(1, int(round(self.rabi_tr_window_ns / self.rabi_tr_bin_ns))) if do_hist else 0
        edges_ns = np.linspace(0.0, self.rabi_tr_window_ns, nbins + 1) if do_hist else np.empty(0)
        H_flat = np.zeros(n_iterations * nbins, dtype=np.int64) if do_hist else None
        # tau-INDEPENDENT readout profile over ALL shots (matched or not), so the
        # readout is visible for diagnosis even when tau assignment fails.
        prof = np.zeros(nbins, dtype=np.int64) if do_hist else None
        win_ps = self.rabi_tr_window_ns * 1000.0
        bin_ps = self.rabi_tr_bin_ns * 1000.0
        n_spcm = 0
        if ref_ps.size:
            for chunk_out in self._iter_channel_times(records, [self.stop_input]):
                t = chunk_out[self.stop_input]
                n_spcm += t.size
                if t.size == 0:
                    continue
                idx = np.searchsorted(ref_ps, t, side="right") - 1
                keep = idx >= 0
                if not np.any(keep):
                    continue
                idx = idx[keep]; off_all = t[keep] - ref_ps[idx]
                if do_hist:                              # all-shots profile (pre-match)
                    inw = (off_all >= 0.0) & (off_all < win_ps)
                    if np.any(inw):
                        prof += np.bincount(
                            np.minimum((off_all[inw] / bin_ps).astype(np.int64), nbins - 1),
                            minlength=nbins)
                mk = matched[idx]
                if not np.any(mk):
                    continue
                idx = idx[mk]; off = off_all[mk]
                tau_ph = tau_index[idx]
                in_sig = (off >= sig_lo_ps) & (off < sig_hi_ps)
                if np.any(in_sig):
                    sig_total += np.bincount(tau_ph[in_sig], minlength=n_iterations)
                if use_ref:
                    in_ref = (off >= ref_lo_ps) & (off < ref_hi_ps)
                    if np.any(in_ref):
                        ref_total += np.bincount(tau_ph[in_ref], minlength=n_iterations)
                if do_hist:
                    in_win = (off >= 0.0) & (off < win_ps)
                    if np.any(in_win):
                        b = np.minimum((off[in_win] / bin_ps).astype(np.int64), nbins - 1)
                        H_flat += np.bincount(tau_ph[in_win] * nbins + b,
                                              minlength=n_iterations * nbins)
        H = H_flat.reshape(n_iterations, nbins) if do_hist else np.empty((0, 0), dtype=np.int64)
        profile = prof if do_hist else np.empty(0, dtype=np.int64)
        return sig_total, ref_total, H, edges_ns, n_spcm, profile

    def _report_channels(self, records: np.ndarray) -> Dict[int, int]:
        allch = self.decode_all_channels(records)
        if not allch:
            print("[MultiHarp] DIAGNOSTIC: 0 events on ALL channels. Nothing is "
                  "triggering. Check the signals are present during the run, the "
                  "level sits ~half the ATTENUATED amplitude, and the edge matches.")
            return {}
        allt = np.concatenate(list(allch.values()))
        span_s = (allt.max() - allt.min()) / 1e12 if allt.size > 1 else 0.0
        print("[MultiHarp] per-channel event counts (0-based input; connector = index+1):")
        for ch in sorted(allch):
            n = allch[ch].size
            rate = n / span_s if span_s > 0 else 0.0
            tags = []
            if ch == self.start_input: tags.append("<-START")
            if ch == self.stop_input: tags.append("<-STOP")
            print(f"    input {ch}: {n:>10,} events  (~{rate:,.0f}/s)  {' '.join(tags)}")
        return {ch: allch[ch].size for ch in allch}

    # ---- correlation ------------------------------------------------------ #
    def _correlate(self, start_ps: np.ndarray, stop_ps: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (start_times_ps, delays_ps) according to self.corr_mode."""
        window_ps = self.window_ns * 1000.0
        if start_ps.size == 0 or stop_ps.size == 0:
            return np.empty(0), np.empty(0)

        if self.corr_mode == "nearest":
            idx = np.searchsorted(stop_ps, start_ps, side="right")
            valid = idx < stop_ps.size
            s = start_ps[valid]
            d = stop_ps[np.clip(idx[valid], 0, stop_ps.size - 1)] - s
            ok = (d >= 0) & (d <= window_ps)
            return s[ok], d[ok]

        if self.corr_mode == "around":
            pre_ps = self.pre_window_ns * 1000.0
            lo = np.searchsorted(stop_ps, start_ps - pre_ps, side="left")
            hi = np.searchsorted(stop_ps, start_ps + window_ps, side="right")
            starts_out, delays_out = [], []
            for i in range(start_ps.size):
                if hi[i] > lo[i]:
                    seg = stop_ps[lo[i]:hi[i]]
                    delays_out.append(seg - start_ps[i])
                    starts_out.append(np.full(seg.size, start_ps[i]))
            if not delays_out:
                return np.empty(0), np.empty(0)
            return np.concatenate(starts_out), np.concatenate(delays_out)

        # "all_in_window": every stop within (start, start+window] of each start
        lo = np.searchsorted(stop_ps, start_ps, side="right")
        hi = np.searchsorted(stop_ps, start_ps + window_ps, side="right")
        starts_out, delays_out = [], []
        for i in range(start_ps.size):
            if hi[i] > lo[i]:
                seg = stop_ps[lo[i]:hi[i]]
                delays_out.append(seg - start_ps[i])
                starts_out.append(np.full(seg.size, start_ps[i]))
        if not delays_out:
            return np.empty(0), np.empty(0)
        return np.concatenate(starts_out), np.concatenate(delays_out)

    # ---- statistics ------------------------------------------------------- #
    @staticmethod
    def _fwhm_from_hist(centers_ns: np.ndarray, counts: np.ndarray) -> float:
        if counts.size == 0 or counts.max() <= 0:
            return float("nan")
        half = counts.max() / 2.0
        above = np.where(counts >= half)[0]
        if above.size < 2:
            return float("nan")
        return float(centers_ns[above[-1]] - centers_ns[above[0]])

    @staticmethod
    def _rise_time(centers_ns: np.ndarray, counts: np.ndarray
                   ) -> Tuple[float, float, float]:
        """Estimate onset (10%), 10-90 rise, and plateau from an arrival profile.

        Plateau = median of the top 30% of the window (assumes the profile rises
        then flattens/decays slowly). Crossings are linearly interpolated on the
        first rising edge. Returns (onset_ns, rise_ns, plateau)."""
        if counts.size < 3 or counts.max() <= 0:
            return float("nan"), float("nan"), float("nan")
        n = counts.size
        plateau = float(np.median(counts[int(0.7 * n):])) if n >= 4 else float(counts.max())
        if plateau <= 0:
            plateau = float(counts.max())

        def cross(frac):
            thr = frac * plateau
            for i in range(1, n):
                if counts[i - 1] < thr <= counts[i]:
                    c0, c1 = counts[i - 1], counts[i]
                    t0, t1 = centers_ns[i - 1], centers_ns[i]
                    if c1 == c0:
                        return t1
                    return t0 + (thr - c0) * (t1 - t0) / (c1 - c0)
            return float("nan")

        t10, t90 = cross(0.10), cross(0.90)
        rise = (t90 - t10) if (np.isfinite(t10) and np.isfinite(t90)) else float("nan")
        return t10, rise, plateau

    def analyze(self, records: np.ndarray) -> TimingResult:
        chans = self.decode_t2(records)
        self._report_channels(records)
        start_ps = chans[self.start_input]
        stop_ps = chans[self.stop_input]
        s_ps, d_ps = self._correlate(start_ps, stop_ps)

        res = TimingResult(
            case=self.case,
            start_input=self.start_input,
            stop_input=self.stop_input,
            n_start_events=int(start_ps.size),
            n_stop_events=int(stop_ps.size),
            n_pairs=int(d_ps.size),
            base_resolution_ps=self._base_res_ps,
            start_times_ns=s_ps / 1000.0,
            delays_ns=d_ps / 1000.0,
            fifo_overrun=self._fifo_overrun,
        )
        if d_ps.size == 0:
            if start_ps.size == 0 or stop_ps.size == 0:
                dead = ("START (input %d)" % self.start_input if start_ps.size == 0
                        else "STOP (input %d)" % self.stop_input)
                print(f"[MultiHarp] NO PAIRS: the {dead} channel saw 0 events -- a "
                      f"detection problem, not correlation. Fix that channel's level/"
                      f"edge/wiring (attenuate TTL to <=1.2 V, level ~half amplitude). "
                      f"Flipping start/stop won't help while a channel reads zero. "
                      f"Try scan_all_channels=True.")
            else:
                print(f"[MultiHarp] NO PAIRS but both channels have events "
                      f"({start_ps.size}/{stop_ps.size}) -- a timing issue: with "
                      f"corr_mode={self.corr_mode!r} the stop never lands in the "
                      f"window. Use case='adwin_readout_vs_spcm' (corr_mode='around').")
            return res

        d_ns = d_ps / 1000.0
        res.mean_ns = float(np.mean(d_ns))
        res.median_ns = float(np.median(d_ns))
        res.std_ns = float(np.std(d_ns))

        if self.corr_mode == "around":
            lo_ns, hi_ns = -self.pre_window_ns, self.window_ns
        else:
            lo_ns, hi_ns = 0.0, self.window_ns
        nbins = max(10, int(round((hi_ns - lo_ns) / self.hist_bin_ns)))
        counts, edges = np.histogram(d_ns, bins=nbins, range=(lo_ns, hi_ns))
        centers = 0.5 * (edges[:-1] + edges[1:])
        res.hist_edges_ns = edges
        res.hist_counts = counts
        res.fwhm_ns = self._fwhm_from_hist(centers, counts)

        if self.case == "readout_vs_spcm":
            onset, rise, plateau = self._rise_time(centers, counts.astype(float))
            res.onset_ns, res.rise_time_ns, res.plateau_counts = onset, rise, plateau

        # Drift across the run: bin pairs by absolute start time.
        if s_ps.size >= self.n_drift_segments:
            t_s = (s_ps - s_ps.min()) / 1e12  # ps -> s
            seg_edges = np.linspace(t_s.min(), t_s.max(), self.n_drift_segments + 1)
            which = np.clip(np.digitize(t_s, seg_edges) - 1, 0, self.n_drift_segments - 1)
            mids, means, cnts = [], [], []
            for k in range(self.n_drift_segments):
                sel = which == k
                if np.any(sel):
                    mids.append(0.5 * (seg_edges[k] + seg_edges[k + 1]))
                    means.append(float(np.mean(d_ns[sel])))
                    cnts.append(int(np.sum(sel)))
            res.segment_mid_time_s = np.asarray(mids)
            res.segment_mean_delay_ns = np.asarray(means)
            res.segment_count = np.asarray(cnts)
            if means:
                res.drift_pp_ns = float(np.max(means) - np.min(means))

        return res

    # ---- Rabi reconstruction (MultiHarp only) ----------------------------- #
    @staticmethod
    def _counts_in_window(ref_ps: np.ndarray, other_ps: np.ndarray,
                          lo_ps: float, hi_ps: float) -> np.ndarray:
        """For each reference edge, COUNT `other` timestamps in
        [ref+lo_ps, ref+hi_ps). Vectorised; fine for millions of edges.

        Returns an int64 array, one count per reference edge (same order)."""
        n = ref_ps.size
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if other_ps.size == 0:
            return np.zeros(n, dtype=np.int64)
        lo = np.searchsorted(other_ps, ref_ps + lo_ps, side="left")
        hi = np.searchsorted(other_ps, ref_ps + hi_ps, side="left")
        return (hi - lo).astype(np.int64)

    @staticmethod
    def _pick_edge(ref_ps: np.ndarray, edge_ps: np.ndarray,
                   lo_ps: float, hi_ps: float, pick: str = "last") -> np.ndarray:
        """For each reference edge, return ONE `edge` timestamp (ps) inside
        [ref+lo_ps, ref+hi_ps], or NaN if none. pick='first' takes the earliest
        in the window, 'last' the latest. Vectorised over millions of edges.

        Used to grab the MW pulse edges belonging to each shot: with one MW pulse
        per shot there is exactly one edge of each polarity in the window, so
        'first'/'last' just disambiguate rise-vs-fall robustly."""
        n = ref_ps.size
        out = np.full(n, np.nan)
        if n == 0 or edge_ps.size == 0:
            return out
        lo = np.searchsorted(edge_ps, ref_ps + lo_ps, side="left")
        hi = np.searchsorted(edge_ps, ref_ps + hi_ps, side="right")
        has = hi > lo
        idx = (lo[has] if pick == "first" else hi[has] - 1)
        out[has] = edge_ps[idx]
        return out

    def _infer_rabi_axis(self, experiment: Any) -> Tuple[int, np.ndarray, bool]:
        """Work out (n_iterations, tau_values_ns, tau_is_index) for the sweep.

        Explicit values passed to the constructor win. Otherwise we read
        experiment.number_of_iterations and try to rebuild the tau axis from the
        first scan variable. If the variable's units/fields can't be read we fall
        back to a plain 0..n-1 index axis and flag tau_is_index=True (the shape of
        the oscillation is still correct; only the x-axis labels are indices)."""
        # n_iterations
        n = self.rabi_n_iterations
        if n is None:
            n = getattr(experiment, "number_of_iterations", 0) or 0
            if not n:
                n = len(getattr(experiment, "scan_sequences", []) or [])
        n = int(n) if n else 1

        # tau values
        if self.rabi_tau_values_ns is not None:
            tau = np.asarray(self.rabi_tau_values_ns, dtype=float)
            return n, tau, False

        tau = None
        try:
            desc = getattr(experiment, "sequence_description", None)
            variables = getattr(desc, "variables", None)
            if variables:
                v = variables[0] if not isinstance(variables, dict) \
                    else list(variables.values())[0]

                def get(names):
                    for nm in names:
                        if isinstance(v, dict) and nm in v:
                            return v[nm]
                        if hasattr(v, nm):
                            return getattr(v, nm)
                    return None

                start = get(["start", "start_ns", "begin", "min"])
                stop = get(["stop", "stop_ns", "end", "max"])
                steps = get(["steps", "num_steps", "n_steps", "count", "n"])
                if start is not None and stop is not None and steps:
                    tau = np.linspace(float(start), float(stop), int(steps))
        except Exception:
            tau = None

        if tau is None or (n and tau.size != n):
            tau = np.arange(n, dtype=float)
            return n, tau, True
        return n, tau, False

    def _mw_measure_tau(self, allch: Dict[int, np.ndarray], ref_ps: np.ndarray
                        ) -> Tuple[np.ndarray, str, Tuple[int, ...], float]:
        """Measure tau PER SHOT from the MW copy on the third input.

        Returns (tau_meas_ns, source, mw_inputs, lookback_ns). tau_meas_ns has one
        entry per reference (marker) edge; NaN where the MW pulse for that shot was
        not found. `source` is 'mw_width' or 'mw_geometry'."""
        gap = self.mw_gap_ns if self.mw_gap_ns is not None else 3000.0
        lookback = self.mw_lookback_ns if self.mw_lookback_ns is not None else (gap + 500.0)
        lo_ps = -lookback * 1000.0
        mw_ps = allch.get(self.mw_input, np.empty(0))
        if self.mw_mode == "width":
            fall_ps = allch.get(self.mw_fall_input, np.empty(0))
            # one MW pulse/shot -> one rising (MW start) and one falling (MW end)
            rise_t = self._pick_edge(ref_ps, mw_ps, lo_ps, 0.0, pick="first")
            fall_t = self._pick_edge(ref_ps, fall_ps, lo_ps, 0.0, pick="last")
            tau_meas = (fall_t - rise_t) / 1000.0     # ps -> ns; == pulse width == tau
            return tau_meas, "mw_width", (self.mw_input, self.mw_fall_input), lookback
        # geometry mode: single MW END edge, tau from the fixed MW->readout gap
        fall_t = self._pick_edge(ref_ps, mw_ps, lo_ps, 0.0, pick="last")
        tau_meas = gap - (ref_ps - fall_t) / 1000.0
        return tau_meas, "mw_geometry", (self.mw_input,), lookback

    @staticmethod
    def _snap_to_steps(tau_meas_ns: np.ndarray, tau_steps_ns: np.ndarray,
                       tol_ns: float) -> Tuple[np.ndarray, np.ndarray]:
        """Snap each measured tau to the nearest sweep step. Returns
        (step_index, matched) where step_index[i] is the index into tau_steps_ns
        of the nearest step and matched[i] is True iff within tol_ns (and finite).
        tau_steps_ns[k] is the tau of step k, so step_index IS the tau-step index."""
        n = tau_meas_ns.size
        step_index = np.zeros(n, dtype=np.int64)
        matched = np.zeros(n, dtype=bool)
        if n == 0 or tau_steps_ns.size == 0:
            return step_index, matched
        order = np.argsort(tau_steps_ns)
        ts = np.asarray(tau_steps_ns, dtype=float)[order]
        finite = np.isfinite(tau_meas_ns)
        tm = np.where(finite, tau_meas_ns, ts[0])          # placeholder for NaNs
        pos = np.clip(np.searchsorted(ts, tm), 1, ts.size - 1)
        left, right = ts[pos - 1], ts[pos]
        take_right = (tm - left) > (right - tm)
        nearest_sorted = np.where(take_right, pos, pos - 1)
        dist = np.abs(tm - ts[nearest_sorted])
        step_index = order[nearest_sorted]
        matched = finite & (dist <= tol_ns)
        return step_index, matched

    def _build_arrival_hist(self, ref_ps: np.ndarray, spcm_ps: np.ndarray,
                            tau_index: np.ndarray, matched: np.ndarray,
                            n_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram SPCM arrivals relative to the marker, per tau step.

        Vectorised: each photon is attached to the marker at/just-before it, its
        offset is binned, and only photons from matched shots inside
        [0, tr_window] are kept. Returns (H[n_tau, n_bins], edges_ns)."""
        nbins = max(1, int(round(self.rabi_tr_window_ns / self.rabi_tr_bin_ns)))
        edges_ns = np.linspace(0.0, self.rabi_tr_window_ns, nbins + 1)
        H = np.zeros((n_iterations, nbins), dtype=np.int64)
        if ref_ps.size == 0 or spcm_ps.size == 0:
            return H, edges_ns
        win_ps = self.rabi_tr_window_ns * 1000.0
        # marker at/just-before each photon (markers are time-ordered)
        idx = np.searchsorted(ref_ps, spcm_ps, side="right") - 1
        ok = idx >= 0
        off = np.full(spcm_ps.shape, -1.0)
        off[ok] = spcm_ps[ok] - ref_ps[idx[ok]]
        good = ok & (off >= 0.0) & (off < win_ps)
        good[good] = matched[idx[good]]              # drop photons from unmatched shots
        if not np.any(good):
            return H, edges_ns
        ph_tau = tau_index[idx[good]].astype(np.int64)
        ph_bin = np.minimum((off[good] / (self.rabi_tr_bin_ns * 1000.0)).astype(np.int64),
                            nbins - 1)
        # accumulate into the flat [n_tau*nbins] grid
        flat = ph_tau * nbins + ph_bin
        counts = np.bincount(flat, minlength=n_iterations * nbins)
        H = counts.reshape(n_iterations, nbins).astype(np.int64)
        return H, edges_ns

    @staticmethod
    def recompute_rabi(result_or_npz, sig_window_ns, ref_window_ns=None):
        """Re-derive the Rabi curve for new windows from a RabiResult or a saved
        .npz path (or an npz handle / dict). Thin wrapper over
        rabi_curve_from_histogram so you can slide windows offline, e.g.:

            r = tester.run_with_experiment(exp)["rabi"]
            new = MultiHarpTimingTester.recompute_rabi(
                      r, sig_window_ns=(950, 1250), ref_window_ns=(5000, 5300))
            # ... or from disk, after the fact:
            new = MultiHarpTimingTester.recompute_rabi(
                      "multiharp_rabi_mw_tagged_20260819_120000.npz",
                      sig_window_ns=(950, 1250))
        """
        if isinstance(result_or_npz, RabiResult):
            H = result_or_npz.arrival_hist
            edges = result_or_npz.arrival_edges_ns
            spt = result_or_npz.shots_per_tau
        else:
            d = (np.load(result_or_npz, allow_pickle=True)
                 if isinstance(result_or_npz, str) else result_or_npz)
            H, edges, spt = d["arrival_hist"], d["arrival_edges_ns"], d["shots_per_tau"]
        if np.asarray(H).size == 0:
            raise ValueError("no time-resolved data present (run with "
                             "rabi_time_resolved=True to enable re-windowing)")
        return rabi_curve_from_histogram(H, edges, spt, sig_window_ns, ref_window_ns)

    def analyze_rabi(self, records: np.ndarray, n_iterations: int,
                     tau_values_ns: np.ndarray, tau_is_index: bool = False
                     ) -> RabiResult:
        """Build a Rabi curve from raw T2 records, using ONLY the MultiHarp.

        start_input = a once-per-shot reference edge (readout laser marker best,
        or the trigger). stop_input = the SPCM. Per shot we count SPCM photons in
        the signal window (bright readout onset) and, if configured, a reference
        window later inside the same readout pulse. Shots are assigned to tau
        steps by their order (interleaved task table -> shot i has
        tau_index = i mod n_iterations), then averaged per step.

        TWO WAYS TO ASSIGN tau:
          * shot ORDER (default, case="rabi_vs_spcm"): the interleaved task table
            means shot i has tau_index = i mod n_iterations. Simple, but a single
            MISSED reference edge shifts every later shot by one tau and smears the
            curve, so the reference edge must be a clean once-per-shot TTL
            (attenuated to <=1.2 V, level ~half amplitude, correct edge). We
            sanity-check the shot count against repeat_count x n_iterations.
          * MW TAG (case="rabi_mw_tagged", or any time mw_input is set): tau is read
            PER SHOT from the MW pulse on a third input -- its WIDTH is tau (two-edge
            WIDTH mode) or it is derived from the MW end vs the readout marker
            (one-BNC GEOMETRY mode). Each measured tau is snapped to the nearest
            sweep step; a dropped edge now drops only that shot instead of shifting
            all the later ones. This removes the ordering assumption entirely and is
            the recommended way to settle "is it the ADwin or a real drift?"."""
        n_iterations = max(1, int(n_iterations))
        rc = getattr(self, "_rabi_repeat_count", None)
        expected_markers = int(rc) * n_iterations if rc else None

        # ---- 1) cheap streaming count pass: composition + multi-trigger check
        counts = self._channel_counts(records)
        self._report_channel_counts(counts, expected_markers)
        n_start = counts.get(self.start_input, 0)
        n_spcm_ct = counts.get(self.stop_input, 0)

        res = RabiResult(
            start_input=self.start_input,
            stop_input=self.stop_input,
            n_ref_events=int(n_start),
            n_spcm_events=int(n_spcm_ct),
            n_shots_used=0,
            n_iterations=n_iterations,
            repeat_count=(int(rc) if rc else None),
            base_resolution_ps=self._base_res_ps,
            sig_offset_ns=self.rabi_sig_offset_ns,
            sig_width_ns=self.rabi_sig_width_ns,
            reference_label=self.start_label,
            tau_ns=np.asarray(tau_values_ns, dtype=float),
            tau_is_index=bool(tau_is_index),
            fifo_overrun=self._fifo_overrun,
        )

        if n_start == 0 or n_spcm_ct == 0:
            dead = ("reference/start (input %d)" % self.start_input
                    if n_start == 0 else "SPCM/stop (input %d)" % self.stop_input)
            print(f"[MultiHarp] Rabi: the {dead} channel saw 0 events -- no curve. "
                  f"Fix that channel's level/edge/wiring (attenuate TTL to <=1.2 V, "
                  f"level ~half amplitude). Try scan_all_channels=True to see who fires.")
            return res

        # Guard the one array we must hold whole (the marker edges). If the marker
        # multi-triggers into billions of edges, holding them (and the MW arrays)
        # would thrash; stop with a clear message rather than crash.
        if n_start > self.max_marker_edges:
            raise RuntimeError(
                f"[MultiHarp] Rabi: the reference/START channel (input "
                f"{self.start_input}) has {n_start:,} edges, above the "
                f"max_marker_edges={self.max_marker_edges:,} safety limit. This almost "
                f"always means the marker is MULTI-TRIGGERING (ringing/reflection, or "
                f"the trigger level set too low) -- fix that first so it fires once per "
                f"shot. If you are sure the count is legitimate and you have the RAM, "
                f"re-run with a larger max_marker_edges.")

        # ---- 2) sparse pass: materialise ONLY marker + MW edge times -------
        want = [self.start_input]
        if self.mw_mode != "off":
            want += [self.mw_input, self.mw_fall_input]
        sparse = self._collect_channel_times(records, want)
        ref_ps = sparse.get(self.start_input, np.empty(0))   # once-per-shot reference edges
        n_shots = ref_ps.size
        order_index = np.arange(n_shots) % n_iterations      # shot-order fallback / cross-check

        # ---- 3) assign shots -> tau steps (MW tag if available, else order)
        want_mw = self.mw_mode != "off"
        n_mw = counts.get(self.mw_input, 0) if self.mw_input is not None else 0
        n_mwf = counts.get(self.mw_fall_input, 0) if self.mw_fall_input is not None else 0
        mw_ready = want_mw and not tau_is_index and n_mw > 0
        # WIDTH mode needs BOTH edges; a dead falling-edge input makes every width
        # undefined (this is the usual cause of "0 matched").
        if want_mw and self.mw_mode == "width" and n_mwf == 0:
            print(f"[MultiHarp] Rabi MW-tag WIDTH mode needs both edges, but the "
                  f"FALLING-edge input {self.mw_fall_input} saw 0 events -- tau = "
                  f"fall - rise cannot be formed. Check that BNC's cable/level/edge "
                  f"(or use GEOMETRY mode: drop mw_fall_input and set mw_gap_ns).")
            mw_ready = False
        if want_mw and not mw_ready and not (self.mw_mode == "width" and n_mwf == 0):
            why = ("the tau axis is only step indices (pass rabi_tau_values_ns in ns "
                   "so measured widths can be matched)" if tau_is_index
                   else f"input {self.mw_input} (MW copy) saw no events -- check its "
                        f"level/edge/wiring")
            print(f"[MultiHarp] Rabi MW-tag requested but unavailable: {why}.")

        use_mw_result = False
        if mw_ready:
            tau_meas, source, mw_chans, lookback = self._mw_measure_tau(sparse, ref_ps)
            steps = np.asarray(tau_values_ns, dtype=float)
            if self.mw_snap_tol_ns is not None:
                tol = float(self.mw_snap_tol_ns)
            else:
                ds = np.diff(np.sort(steps[np.isfinite(steps)]))
                spacing = float(np.median(ds)) if ds.size else 1.0
                tol = max(1.0, 0.5 * spacing)             # half a step by default
            tau_index_mw, matched_mw = self._snap_to_steps(tau_meas, steps, tol)
            n_match = int(np.count_nonzero(matched_mw))
            if n_match >= max(1, int(0.02 * n_shots)):     # MW tagging is working
                use_mw_result = True
                tau_index, matched = tau_index_mw, matched_mw
                res.tau_assign_source = source
                res.mw_inputs = tuple(int(c) for c in mw_chans)
                res.mw_snap_tol_ns = tol
                res.n_matched = n_match
                res.n_unmatched = n_shots - n_match
                res.order_agreement = float(
                    np.mean(order_index[matched] == tau_index[matched]))
                print(f"[MultiHarp] Rabi MW-tag ({source}, tol +/-{tol:.0f} ns): matched "
                      f"{res.n_matched:,}/{n_shots:,} shots; {res.n_unmatched:,} unmatched. "
                      f"MW-tag vs shot-order agree on "
                      f"{100.0 * res.order_agreement:.1f}% of matched shots.")
            else:
                # MW tagging effectively failed -- explain WHY before falling back.
                finite = np.isfinite(tau_meas)
                nfin = int(np.count_nonzero(finite))
                if nfin:
                    fw = tau_meas[finite]
                    print(f"[MultiHarp] Rabi MW-tag matched only {n_match:,}/{n_shots:,} "
                          f"shots. Measured widths: {nfin:,} finite, range "
                          f"{np.min(fw):.0f}..{np.max(fw):.0f} ns (median {np.median(fw):.0f}); "
                          f"the sweep spans {np.nanmin(steps):.0f}..{np.nanmax(steps):.0f} ns, "
                          f"tol +/-{tol:.0f} ns. The widths do not line up with the sweep -- "
                          f"check the MW edges/level, or that mw_input/mw_fall_input aren't "
                          f"swapped, or widen mw_snap_tol_ns.")
                else:
                    print(f"[MultiHarp] Rabi MW-tag matched 0/{n_shots:,}: every measured "
                          f"width is undefined (a MW edge is missing on essentially every "
                          f"shot). Verify BOTH MW inputs fire exactly once per shot.")

        if not use_mw_result:
            tau_index = order_index
            matched = np.ones(n_shots, dtype=bool)
            res.tau_assign_source = "shot_order"
            res.mw_inputs = ()
            res.n_matched = int(n_shots)
            res.n_unmatched = 0
            if want_mw:
                clean = (expected_markers and abs(n_shots - expected_markers) <= n_iterations)
                print(f"[MultiHarp] Falling back to SHOT-ORDER (shot i -> tau i mod "
                      f"{n_iterations}). " + ("The marker looks clean (~1 edge/shot), so "
                      "this should give a valid curve." if clean else "NOTE: the marker "
                      "count does not match repeat_count x tau steps, so shot-order may be "
                      "desynced -- fix the MW inputs and re-run for a trustworthy curve."))

        res.n_shots_used = int(np.count_nonzero(matched))

        # ---- 4) stream the SPCM channel: window counts + arrival histogram --
        sig_lo = self.rabi_sig_offset_ns * 1000.0
        sig_hi = (self.rabi_sig_offset_ns + self.rabi_sig_width_ns) * 1000.0
        use_ref = bool(self.rabi_ref_width_ns and self.rabi_ref_width_ns > 0)
        ref_lo = self.rabi_ref_offset_ns * 1000.0
        ref_hi = (self.rabi_ref_offset_ns + self.rabi_ref_width_ns) * 1000.0
        if use_ref:
            res.ref_offset_ns = self.rabi_ref_offset_ns
            res.ref_width_ns = self.rabi_ref_width_ns
        sig_total, ref_total, H, edges_ns, n_spcm_used, profile = self._stream_spcm_windows(
            records, ref_ps, tau_index, matched, n_iterations,
            sig_lo, sig_hi, use_ref, ref_lo, ref_hi)

        shots_per = np.bincount(tau_index[matched], minlength=n_iterations).astype(np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig_mean = np.where(shots_per > 0, sig_total / shots_per, np.nan)
            # Poisson standard error on the mean (exact for these low per-shot counts).
            sig_sem = np.where(shots_per > 0, np.sqrt(sig_total) / shots_per, np.nan)
            ref_mean = (np.where(shots_per > 0, ref_total / shots_per, np.nan)
                        if use_ref else np.full(n_iterations, np.nan))
        res.signal_mean = sig_mean
        res.signal_sem = sig_sem
        res.shots_per_tau = shots_per
        if use_ref:
            res.reference_mean = ref_mean
            with np.errstate(divide="ignore", invalid="ignore"):
                res.contrast = np.where(ref_mean > 0, sig_mean / ref_mean, np.nan)
        if self.rabi_time_resolved and H.size:
            res.arrival_hist = H
            res.arrival_edges_ns = edges_ns
            res.time_resolved = True
        if profile.size:
            res.arrival_profile_ns = profile         # tau-independent readout (all shots)

        # If no shot contributed, still surface what the readout looked like so the
        # run is diagnosable rather than a silent empty file.
        if res.n_shots_used == 0:
            tot = int(profile.sum()) if profile.size else 0
            if tot > 0:
                ctr = 0.5 * (edges_ns[:-1] + edges_ns[1:])
                on = ctr[np.argmax(profile > 0.25 * np.max(profile))] if np.max(profile) else 0.0
                print(f"[MultiHarp] Rabi: 0 shots were assigned a tau, so there is no "
                      f"curve -- but the SPCM DID see {tot:,} photons in the readout "
                      f"(light turns on ~{on:.0f} ns after the marker). The problem is the "
                      f"tau ASSIGNMENT (MW inputs), not the photon path. Fix the MW inputs "
                      f"(or re-run case='rabi_vs_spcm' now that the marker is clean).")
            else:
                print(f"[MultiHarp] Rabi: 0 shots assigned AND 0 SPCM photons in the "
                      f"readout window -- check the SPCM (input {self.stop_input}) and the "
                      f"signal-window placement.")

        # ---- oscillation descriptor -------------------------------------- #
        curve = res.contrast if (use_ref and np.isfinite(res.contrast).any()) else sig_mean
        res.best_metric = "contrast" if (use_ref and np.isfinite(res.contrast).any()) else "signal_mean"
        good = curve[np.isfinite(curve)]
        if good.size >= 2:
            hi, lo = float(np.max(good)), float(np.min(good))
            res.modulation_depth = (hi - lo) / (hi + lo) if (hi + lo) > 0 else float("nan")

        # ---- phase / desync sanity check --------------------------------- #
        # Only meaningful for shot-ORDER assignment; MW-tagged runs tolerate
        # dropped shots by construction (they just go unmatched), so we leave
        # phase_ok True there.
        rc = getattr(self, "_rabi_repeat_count", None)
        res.repeat_count = int(rc) if rc else None
        if rc and res.tau_assign_source == "shot_order":
            expected = int(rc) * n_iterations
            # allow the last (partial) cycle: within one full cycle is fine.
            res.phase_ok = abs(n_shots - expected) <= n_iterations
            if not res.phase_ok:
                print(f"[MultiHarp] Rabi WARNING: saw {n_shots} reference edges but "
                      f"expected ~{expected} (= repeat_count {rc} x {n_iterations} taus). "
                      f"Shot->tau assignment may be desynced by dropped/extra edges. "
                      f"Consider case='rabi_mw_tagged' (wire the MW line to a 3rd input).")
        return res

    # ---- HDF5 saving (for the data-analyzer GUI) -------------------------- #
    #
    # ============================ WHERE TO SAVE ============================= #
    # This is the method that writes the MultiHarp timing data to disk in the
    # same /data + /meta group layout ODMRPulsedExperiment.save_hdf5 uses,
    # so data-analyzer GUI can open it the same way. It is called from
    # _save_outputs(), which run_with_experiment() calls automatically after the
    # run.
    # ======================================================================= #
    def _save_hdf5(self, res: TimingResult, base: str) -> Optional[str]:
        try:
            import h5py
        except Exception as e:
            print(f"[MultiHarp] h5py not available, skipping HDF5 save ({e}). "
                  f"The .npz still holds all arrays.")
            return None
        from src.core.struct_hdf5 import MyStruct, save_data
        s = MyStruct()
        s.data = MyStruct(start_times_ns=res.start_times_ns, delays_ns=res.delays_ns, segment_mid_time_s = res.segment_mid_time_s, segment_mean_delay_ns = res.segment_mean_delay_ns, segment_count = res.segment_count,
                              hist_counts=res.hist_counts, hist_edges_ns=res.hist_edges_ns)
        s.meta = MyStruct(case=self.case,
                    output_mode=self.output_mode,
                    correlation_mode=self.corr_mode,
                    start_input=self.start_input,
                    stop_input=self.stop_input,
                    start_label=self.start_label,
                    stop_label=self.stop_label,
                    base_resolution_ps=res.base_resolution_ps,
                    window_ns=self.window_ns,
                    hist_bin_ns=self.hist_bin_ns,
                    n_start_events=res.n_start_events,
                    n_stop_events=res.n_stop_events,
                    n_pairs=res.n_pairs,
                    mean_ns=res.mean_ns,
                    median_ns=res.median_ns,
                    std_ns=res.std_ns,
                    fwhm_ns=res.fwhm_ns,
                    onset_ns=res.onset_ns,
                    rise_time_ns=res.rise_time_ns,
                    plateau_counts=res.plateau_counts,
                    drift_pp_ns=res.drift_pp_ns,
                    fifo_overrun=int(res.fifo_overrun),
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_data(base + ".h5", s)

    # ---- plotting / saving ------------------------------------------------ #
    def _plot_histogram(self, ax, res: TimingResult) -> None:
        centers = 0.5 * (res.hist_edges_ns[:-1] + res.hist_edges_ns[1:])
        ax.bar(centers, res.hist_counts,
               width=(centers[1] - centers[0]) if centers.size > 1 else 1.0,
               align="center")
        ax.set_xlabel(f"delay: {self.stop_label} - {self.start_label} (ns)")
        ax.set_ylabel("counts")
        title = f"{self.case}: median={res.median_ns:.2f} ns, FWHM={res.fwhm_ns:.2f} ns"
        if self.corr_mode == "around":
            ax.set_xlabel(f"{self.stop_label} arrival relative to {self.start_label} (ns)")
            ax.axvline(0.0, color="k", lw=1.2)  # gate opens
            if self.gate_ns:
                ax.axvspan(0.0, self.gate_ns, alpha=0.15, color="green")
                ax.axvline(self.gate_ns, color="k", ls=":", lw=1)  # gate closes
                frac_in = np.mean((res.delays_ns >= 0) & (res.delays_ns <= self.gate_ns)) * 100
                title += f"\nphotons inside gate [0,{self.gate_ns:.0f}] ns: {frac_in:.0f}%"
        else:
            ax.set_xlabel(f"delay: {self.stop_label} - {self.start_label} (ns)")
            if np.isfinite(res.rise_time_ns):
                title += f"\nrise(10-90%)={res.rise_time_ns:.1f} ns"
                for t in (res.onset_ns, res.onset_ns + res.rise_time_ns):
                    if np.isfinite(t):
                        ax.axvline(t, ls="--", lw=1)
        ax.set_title(title)

    def _plot_time_series(self, ax, res: TimingResult) -> None:
        # Per-shot delay vs time through the run ("data over time"), with the
        # binned mean overlaid so drift is easy to see.
        if res.delays_ns.size:
            t_s = (res.start_times_ns - res.start_times_ns.min()) / 1e9
            n = t_s.size
            step = max(1, n // 200_000)          # keep the scatter light
            ax.plot(t_s[::step], res.delays_ns[::step], ".", ms=2, alpha=0.25,
                    label="per shot")
        if res.segment_mean_delay_ns.size:
            ax.plot(res.segment_mid_time_s, res.segment_mean_delay_ns,
                    "-o", lw=1.5, label="binned mean")
        ax.set_xlabel("time through run (s)  ~ tau-sweep progression")
        ax.set_ylabel(f"delay: {self.stop_label} - {self.start_label} (ns)")
        pp = res.drift_pp_ns if np.isfinite(res.drift_pp_ns) else float("nan")
        ax.set_title(f"data over time (drift peak-to-peak {pp:.2f} ns)")
        ax.legend(loc="best", fontsize=8)

    def _save_outputs(self, res: TimingResult, tag: str) -> None:
        outdir = self.output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"multiharp_{self.case}_{self.output_mode}_{stamp}")

        if self.save_npz:
            npz = base + ".npz"
            np.savez_compressed(
                npz,
                case=self.case, output_mode=self.output_mode,
                start_input=self.start_input, stop_input=self.stop_input,
                base_resolution_ps=self._base_res_ps,
                start_times_ns=res.start_times_ns, delays_ns=res.delays_ns,
                hist_edges_ns=res.hist_edges_ns, hist_counts=res.hist_counts,
                segment_mid_time_s=res.segment_mid_time_s,
                segment_mean_delay_ns=res.segment_mean_delay_ns,
                segment_count=res.segment_count,
            )
            res.saved_files.append(npz)

        if self.save_hdf5:
            self._save_hdf5(res, base)

        if not self.make_plots or res.n_pairs == 0:
            return
        try:
            if self.output_mode == "both":
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                self._plot_histogram(axes[0], res)
                self._plot_time_series(axes[1], res)
            elif self.output_mode == "histogram":
                fig, ax = plt.subplots(figsize=(7, 4.5))
                self._plot_histogram(ax, res)
            else:  # time_series
                fig, ax = plt.subplots(figsize=(7, 4.5))
                self._plot_time_series(ax, res)
            fig.tight_layout()
            png = base + ".png"
            fig.savefig(png, dpi=130)
            plt.close(fig)
            res.saved_files.append(png)
        except Exception as e:  # never let plotting kill a good measurement
            print(f"[MultiHarp] plot warning: {e}")

    # ---- Rabi plotting / saving ------------------------------------------ #
    def _plot_rabi(self, res: RabiResult) -> "Optional[Any]":
        if not self.make_plots:
            return None
        two = np.isfinite(res.contrast).any() if res.contrast.size else False
        fig, axes = plt.subplots(2 if two else 1, 1, sharex=True,
                                 figsize=(7.5, 6.0 if two else 3.8))
        if not two:
            axes = [axes]
        xlabel = ("tau step index" if res.tau_is_index
                  else "MW pulse duration tau (ns)")

        ax0 = axes[0]
        m = np.isfinite(res.signal_mean)
        ax0.errorbar(res.tau_ns[m], res.signal_mean[m],
                     yerr=(res.signal_sem[m] if res.signal_sem.size else None),
                     fmt="-o", ms=4, lw=1.3, capsize=2, label="signal window")
        ax0.set_ylabel("SPCM counts / shot\n(signal window)")
        if res.tau_assign_source == "mw_width":
            tau_src = "  |  tau from MW pulse width (3rd input)"
        elif res.tau_assign_source == "mw_geometry":
            tau_src = "  |  tau from MW end vs marker (3rd input)"
        else:
            tau_src = ""
        ax0.set_title(
            f"MultiHarp Rabi (no ADwin gating)  |  {res.n_shots_used:,} shots, "
            f"{res.n_iterations} taus{tau_src}\nsignal [{res.sig_offset_ns:.0f},"
            f"{res.sig_offset_ns + res.sig_width_ns:.0f}] ns after "
            f"{res.reference_label}")
        ax0.legend(loc="best", fontsize=8)
        if not res.phase_ok:
            ax0.text(0.5, 0.02, "shot->tau desync suspected (see log)", color="crimson",
                     ha="center", va="bottom", transform=ax0.transAxes, fontsize=9)

        if two:
            ax1 = axes[1]
            mc = np.isfinite(res.contrast)
            ax1.plot(res.tau_ns[mc], res.contrast[mc], "-o", ms=4, lw=1.3, color="C1",
                     label="signal / reference")
            ax1.set_ylabel("contrast\n(signal / reference)")
            ax1.legend(loc="best", fontsize=8)
            ax1.set_xlabel(xlabel)
            if np.isfinite(res.modulation_depth):
                ax1.set_title(f"modulation depth (max-min)/(max+min) = "
                              f"{res.modulation_depth:.3f}", fontsize=9)
        else:
            ax0.set_xlabel(xlabel)
            if np.isfinite(res.modulation_depth):
                ax0.text(0.98, 0.02,
                         f"mod. depth = {res.modulation_depth:.3f}",
                         ha="right", va="bottom", transform=ax0.transAxes, fontsize=9)
        fig.tight_layout()
        return fig

    def _plot_rabi_map(self, res: RabiResult) -> "Optional[Any]":
        """Heatmap of SPCM counts/shot vs (arrival time after marker, tau), with
        the current signal/reference windows drawn on top. Shows exactly where
        the laser rise sits so you can re-pick windows with recompute_rabi()."""
        if not self.make_plots or res.arrival_hist.size == 0:
            return None
        H = res.arrival_hist.astype(float)
        spt = res.shots_per_tau.astype(float).copy()
        spt[spt == 0] = np.nan
        per_shot = H / spt[:, None]                    # counts/shot per (tau, time-bin)
        edges = res.arrival_edges_ns
        ycancel = res.tau_is_index
        y = res.tau_ns if res.tau_ns.size == H.shape[0] else np.arange(H.shape[0])
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        extent = [edges[0], edges[-1], y[-1], y[0]]    # origin upper; tau on y
        im = ax.imshow(per_shot, aspect="auto", extent=extent,
                       interpolation="nearest", cmap="viridis")
        cb = fig.colorbar(im, ax=ax); cb.set_label("SPCM counts / shot")
        # overlay the signal (and reference) windows currently in use
        s0 = res.sig_offset_ns; s1 = res.sig_offset_ns + res.sig_width_ns
        ax.axvline(s0, color="w", lw=1.4); ax.axvline(s1, color="w", lw=1.4, ls="--")
        ax.text(0.5 * (s0 + s1), y[0], " signal", color="w", va="top", ha="center",
                fontsize=8, rotation=90)
        if np.isfinite(res.ref_width_ns) and res.ref_width_ns > 0:
            r0 = res.ref_offset_ns; r1 = res.ref_offset_ns + res.ref_width_ns
            ax.axvline(r0, color="orange", lw=1.4); ax.axvline(r1, color="orange", lw=1.4, ls="--")
            ax.text(0.5 * (r0 + r1), y[0], " reference", color="orange", va="top",
                    ha="center", fontsize=8, rotation=90)
        ax.set_xlabel(f"SPCM arrival time after {res.reference_label}  [ns]")
        ax.set_ylabel("tau step index" if ycancel else "MW pulse duration tau (ns)")
        ax.set_title("Time-resolved Rabi map (full readout kept)\n"
                     "white = signal window, orange = reference; "
                     "re-pick offline with recompute_rabi()")
        fig.tight_layout()
        return fig

    def _save_rabi_outputs(self, res: RabiResult) -> None:
        outdir = self.output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"multiharp_{self.case}_{stamp}")

        if self.save_npz:
            npz = base + ".npz"
            np.savez_compressed(
                npz,
                case=self.case,
                start_input=self.start_input, stop_input=self.stop_input,
                base_resolution_ps=self._base_res_ps,
                n_iterations=res.n_iterations, repeat_count=(res.repeat_count or 0),
                sig_offset_ns=res.sig_offset_ns, sig_width_ns=res.sig_width_ns,
                ref_offset_ns=res.ref_offset_ns, ref_width_ns=res.ref_width_ns,
                tau_ns=res.tau_ns, tau_is_index=int(res.tau_is_index),
                signal_mean=res.signal_mean, signal_sem=res.signal_sem,
                reference_mean=res.reference_mean, contrast=res.contrast,
                shots_per_tau=res.shots_per_tau,
                modulation_depth=res.modulation_depth,
                phase_ok=int(res.phase_ok),
                # MW-tagging provenance
                tau_assign_source=res.tau_assign_source,
                mw_inputs=np.asarray(res.mw_inputs, dtype=np.int64),
                n_matched=res.n_matched, n_unmatched=res.n_unmatched,
                mw_snap_tol_ns=res.mw_snap_tol_ns, order_agreement=res.order_agreement,
                # time-resolved arrivals (re-window offline with recompute_rabi)
                time_resolved=int(res.time_resolved),
                arrival_hist=res.arrival_hist, arrival_edges_ns=res.arrival_edges_ns,
                arrival_profile_ns=res.arrival_profile_ns,
            )
            res.saved_files.append(npz)

        if self.save_hdf5:
            try:
                import h5py  # noqa: F401
                from src.core.struct_hdf5 import MyStruct, save_data
                s = MyStruct()
                s.data = MyStruct(tau_ns=res.tau_ns, signal_mean=res.signal_mean,
                                  signal_sem=res.signal_sem,
                                  reference_mean=res.reference_mean,
                                  contrast=res.contrast,
                                  shots_per_tau=res.shots_per_tau,
                                  arrival_hist=res.arrival_hist,
                                  arrival_edges_ns=res.arrival_edges_ns)
                s.meta = MyStruct(case=self.case,
                                  start_input=self.start_input,
                                  stop_input=self.stop_input,
                                  start_label=res.reference_label,
                                  stop_label=self.stop_label,
                                  base_resolution_ps=res.base_resolution_ps,
                                  n_iterations=res.n_iterations,
                                  repeat_count=(res.repeat_count or 0),
                                  sig_offset_ns=res.sig_offset_ns,
                                  sig_width_ns=res.sig_width_ns,
                                  ref_offset_ns=res.ref_offset_ns,
                                  ref_width_ns=res.ref_width_ns,
                                  tau_is_index=int(res.tau_is_index),
                                  modulation_depth=res.modulation_depth,
                                  best_metric=res.best_metric,
                                  phase_ok=int(res.phase_ok),
                                  fifo_overrun=int(res.fifo_overrun),
                                  tau_assign_source=res.tau_assign_source,
                                  mw_inputs=",".join(str(c) for c in res.mw_inputs),
                                  n_matched=res.n_matched,
                                  n_unmatched=res.n_unmatched,
                                  mw_snap_tol_ns=res.mw_snap_tol_ns,
                                  order_agreement=res.order_agreement,
                                  timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
                save_data(base + ".h5", s)
            except Exception as e:
                print(f"[MultiHarp] Rabi HDF5 save skipped ({e}); the .npz has everything.")

        if self.make_plots and res.n_shots_used:
            try:
                fig = self._plot_rabi(res)
                if fig is not None:
                    png = base + ".png"
                    fig.savefig(png, dpi=130)
                    plt.close(fig)
                    res.saved_files.append(png)
            except Exception as e:
                print(f"[MultiHarp] Rabi plot warning: {e}")
            if res.time_resolved and res.arrival_hist.size:
                try:
                    fig = self._plot_rabi_map(res)
                    if fig is not None:
                        pmap = base + "_map.png"
                        fig.savefig(pmap, dpi=130)
                        plt.close(fig)
                        res.saved_files.append(pmap)
                except Exception as e:
                    print(f"[MultiHarp] Rabi map plot warning: {e}")

    def run_rabi_with_experiment(
        self,
        experiment: Any,
        frequency_range: Optional[List[float]] = None,
        save: bool = True,
        run_method: str = "run_experiment",
        raw_records_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a Rabi sweep and reconstruct the Rabi curve FROM THE MULTIHARP.

        This is the answer to "is it the ADwin, or a real drift on my setup?":
        the curve here never touches the ADwin gate. Wire start_input to the
        readout laser marker (recommended -- add a `marker, laser_readout_1 on
        channel 4 at <readout_start>ns, <readout_len>ns` line to the sequence so
        the Proteus emits one clean edge at the readout onset) and stop_input to
        the (attenuated) SPCM. Then:

          * clean oscillation here + bad ADwin result  -> the ADwin counting/gate
            is the problem (misaligned or jittering gate; see the diagnosis).
          * flat/noisy here too                        -> the problem is upstream
            (polarization, MW, optics, or a genuine slow drift).

        The signal window must sit on the bright readout onset; use a reference
        window later but INSIDE the same readout pulse for a drift-robust ratio.

        raw_records_path: for long runs, dump the raw T2 records to this .npy path
        BEFORE analysis. Cheap insurance -- if anything downstream fails you keep
        the data and can re-analyze offline with analyze_records_file() (which
        memory-maps the file and decodes in chunks, so it never loads it whole)."""
        if self.case not in self._RABI_CASES:
            print(f"[MultiHarp] note: case is {self.case!r}; running the Rabi "
                  f"reconstruction anyway. Pass case='rabi_vs_spcm' (shot order) or "
                  f"case='rabi_mw_tagged' (MW third input) to silence this.")
        if frequency_range is None:
            frequency_range = experiment.settings['microwave']['frequency range']
        if not hasattr(experiment, run_method):
            raise AttributeError(f"experiment has no method {run_method!r}")
        if self.output_dir is None:
            self.output_dir = getattr(experiment, "output_dir", None) or os.getcwd()

        # Stash repeat_count for the desync sanity check (best-effort).
        self._rabi_repeat_count = getattr(experiment, "repeat_count", None)

        self.open_and_configure()
        odmr_result: Dict[str, Any] = {}
        try:
            self.start_acquisition()
            try:
                odmr_result = getattr(experiment, run_method)(frequency_range)
            finally:
                records = self.stop_acquisition()
            # Insurance: persist the raw stream before touching it, so a long run
            # is never lost to an analysis error.
            if raw_records_path:
                try:
                    np.save(raw_records_path, records)
                    print(f"[MultiHarp] raw records saved to {raw_records_path} "
                          f"({records.size:,} records). Re-analyze later with "
                          f"analyze_records_file().")
                except Exception as e:
                    print(f"[MultiHarp] WARNING: could not save raw records "
                          f"({e}); continuing to analysis.")
            n_iter, tau_ns, tau_is_index = self._infer_rabi_axis(experiment)
            self._rabi_axis_cache = (n_iter, tau_ns, tau_is_index)   # for offline re-analysis
            rabi = self.analyze_rabi(records, n_iter, tau_ns, tau_is_index)
            if save:
                self._save_rabi_outputs(rabi)
        finally:
            self.close()

        print(rabi.summary())
        return {"odmr": odmr_result, "rabi": rabi}

    def analyze_records_file(self, path: str, n_iterations: int,
                             tau_values_ns, tau_is_index: bool = False,
                             repeat_count: Optional[int] = None,
                             save: bool = True) -> "RabiResult":
        """Re-run the Rabi analysis on a raw-records .npy saved by
        run_rabi_with_experiment(raw_records_path=...). The file is MEMORY-MAPPED
        and decoded in chunks, so even a 10-hour, tens-of-GB capture is analyzed
        without loading it whole. Pass the same n_iterations/tau_values_ns you
        used for the run (and repeat_count for the desync check)."""
        records = np.load(path, mmap_mode="r")
        if repeat_count is not None:
            self._rabi_repeat_count = repeat_count
        if self.output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(path)) or os.getcwd()
        rabi = self.analyze_rabi(records, n_iterations, tau_values_ns, tau_is_index)
        if save:
            self._save_rabi_outputs(rabi)
        print(rabi.summary())
        return rabi

    # ---- top-level entry point ------------------------------------------- #
    def run_with_experiment(
        self,
        experiment: Any,
        frequency_range: Optional[List[float]] = None,
        save: bool = True,
        run_method: str = "run_experiment",
    ) -> Dict[str, Any]:
        """Open the MultiHarp, start tagging, run the ODMR experiment unchanged,
        stop, analyze, and return {'odmr': <run result>, 'timing': TimingResult}.

        run_method selects which experiment method to run while tagging:
          "run_experiment"          -> a single pass (default)
          "run_experiment_averaged" -> averaged run; the MultiHarp tags
                                       across all averages, so the drift/
                                       time-series view then spans every average.

        For the Rabi cases ("rabi_vs_spcm", "rabi_mw_tagged") this transparently
        delegates to run_rabi_with_experiment and returns {'odmr':..., 'rabi': ...}.
        """
        if self.case in self._RABI_CASES:
            return self.run_rabi_with_experiment(
                experiment, frequency_range=frequency_range,
                save=save, run_method=run_method)

        if frequency_range is None:
            frequency_range = experiment.settings['microwave']['frequency range']
        if not hasattr(experiment, run_method):
            raise AttributeError(f"experiment has no method {run_method!r}")

        # Prefer the experiment's own output directory if it exposes one.
        if self.output_dir is None:
            self.output_dir = (getattr(experiment, "output_dir", None)
                               or os.getcwd())

        self.open_and_configure()
        odmr_result: Dict[str, Any] = {}
        try:
            self.start_acquisition()
            try:
                # ODMR runs exactly as in a normal run -- we don't touch it.
                odmr_result = getattr(experiment, run_method)(frequency_range)
            finally:
                records = self.stop_acquisition()
            timing = self.analyze(records)
            if save:
                self._save_outputs(timing, tag=getattr(experiment, "tag", "odmr"))
        finally:
            self.close()

        print(timing.summary())
        return {"odmr": odmr_result, "timing": timing}

    # ---- multi-channel timing map ---------------------------------------- #
    # Enable several inputs at once, run the sequence, and histogram EVERY
    # channel relative to ONE chosen reference channel. In a single run this
    # draws the full timing diagram of a pulse sequence -- trigger, laser
    # marker, counter-open/close proxies, SPCM, ... -- so alignment generalizes
    # to any sequence. For the counter-alignment run, set reference_input to the
    # counter-OPEN proxy (the digout toggled right after Cnt_Enable): the SPCM
    # histogram then shows the photon burst relative to the ACTUAL counter
    # opening, and the counter-CLOSE proxy shows the true integration window
    # (Cnt_Enable -> Cnt_Latch), independent of whatever count_time you told
    # ADwin. `channels` maps input index -> (label, level_mV, edge); remember to
    # attenuate TTLs to <=1.2 V and set each level to ~half its amplitude.
    def _open_and_configure_map(self, channels: Dict[int, Tuple[str, int, int]]):
        """Open the device and enable exactly the inputs listed in `channels`."""
        mh = MHLibWrapper(self.dll_path)
        print(f"[MultiHarp] MHLib version {mh.library_version()}")
        idx, serial = mh.open_first()
        print(f"[MultiHarp] opened dev {idx} (S/N {serial})")
        mh.initialize_t2()                        # must precede the queries below
        model, partno, version = mh.hardware_info()
        self._base_res_ps = mh.base_resolution_ps()
        nchan = mh.num_input_channels()
        self._nchan = nchan
        print(f"[MultiHarp] {model} (part {partno}, fw {version}) in T2 mode, "
              f"base resolution {self._base_res_ps:.3f} ps, {nchan} input channels")
        for ch in channels:
            if not (0 <= ch < nchan):
                mh.close()
                raise ValueError(f"input channel {ch} out of range (device has {nchan})")
        mh.set_sync_div(self.sync_div)
        mh.set_sync_trigger(-100, EDGE_FALLING)
        mh.set_sync_offset(0)
        mh.set_sync_enable(False)
        for ch in range(nchan):
            on = ch in channels
            mh.set_input_enable(ch, on)
            if on:
                _label, level_mV, edge = channels[ch]
                mh.set_input_trigger(ch, level_mV, edge)
                mh.set_input_offset(ch, 0)
        time.sleep(0.2)
        self._mh = mh

    @staticmethod
    def _events_around(ref_ps: np.ndarray, other_ps: np.ndarray,
                       pre_ps: float, post_ps: float) -> np.ndarray:
        """All `other` timestamps within [ref-pre, ref+post] of each ref edge,
        returned as signed dt (other - ref) in ps."""
        if ref_ps.size == 0 or other_ps.size == 0:
            return np.empty(0)
        lo = np.searchsorted(other_ps, ref_ps - pre_ps, side="left")
        hi = np.searchsorted(other_ps, ref_ps + post_ps, side="right")
        out = []
        for i in range(ref_ps.size):
            if hi[i] > lo[i]:
                out.append(other_ps[lo[i]:hi[i]] - ref_ps[i])
        return np.concatenate(out) if out else np.empty(0)

    def _report_channels_map(self, allch, channels, reference_input):
        if not allch:
            print("[MultiHarp] DIAGNOSTIC: 0 events on ALL channels -- nothing "
                  "triggered. Check trigger levels/edges/wiring.")
            return
        allt = np.concatenate(list(allch.values()))
        span_s = (allt.max() - allt.min()) / 1e12 if allt.size > 1 else 0.0
        print("[MultiHarp] per-channel event counts:")
        for ch in sorted(allch):
            n = allch[ch].size
            rate = n / span_s if span_s > 0 else 0.0
            lbl = channels.get(ch, ("(not configured)", 0, 0))[0]
            tag = "  <-REFERENCE" if ch == reference_input else ""
            print(f"    input {ch}: {n:>10,} events (~{rate:,.0f}/s)  {lbl}{tag}")

    def analyze_timing_map(self, records, channels, reference_input,
                           pre_ns=1000.0, post_ns=4000.0, hist_bin_ns=5.0):
        """Histogram every non-reference channel relative to reference_input.
        Returns {input_index: {...}} and prints a summary table."""
        allch = self.decode_all_channels(records)
        self._report_channels_map(allch, channels, reference_input)
        if reference_input not in allch or allch[reference_input].size == 0:
            print(f"[MultiHarp] reference channel {reference_input} "
                  f"({channels[reference_input][0]}) saw NO events -- cannot build a "
                  f"map. Fix its trigger level/edge/wiring.")
            return {}
        ref_ps = allch[reference_input]
        pre_ps, post_ps = pre_ns * 1000.0, post_ns * 1000.0
        lo_ns, hi_ns = -pre_ns, post_ns
        nbins = max(20, int(round((hi_ns - lo_ns) / hist_bin_ns)))
        results = {}
        print(f"\n[MultiHarp] timing map -- all channels relative to "
              f"'{channels[reference_input][0]}' (input {reference_input}) at t=0:")
        print(f"    {'channel':<34} {'events':>9} {'center_ns':>10} "
              f"{'peak_ns':>9} {'FWHM_ns':>8}")
        for ch in sorted(channels):
            if ch == reference_input:
                continue
            dt_ns = self._events_around(ref_ps, allch.get(ch, np.empty(0)),
                                        pre_ps, post_ps) / 1000.0
            entry = dict(label=channels[ch][0], input=ch, n=int(dt_ns.size),
                         delays_ns=dt_ns, hist_counts=np.empty(0),
                         hist_edges_ns=np.empty(0), center_ns=float("nan"),
                         peak_ns=float("nan"), fwhm_ns=float("nan"))
            if dt_ns.size:
                counts, edges = np.histogram(dt_ns, bins=nbins, range=(lo_ns, hi_ns))
                ctr = 0.5 * (edges[:-1] + edges[1:])
                entry.update(hist_counts=counts, hist_edges_ns=edges,
                             center_ns=float(np.median(dt_ns)),
                             peak_ns=float(ctr[np.argmax(counts)]),
                             fwhm_ns=self._fwhm_from_hist(ctr, counts))
                print(f"    {channels[ch][0]:<34} {dt_ns.size:>9,} "
                      f"{entry['center_ns']:>10.1f} {entry['peak_ns']:>9.1f} "
                      f"{entry['fwhm_ns']:>8.1f}")
            else:
                print(f"    {channels[ch][0]:<34} {'0':>9}   (no events in window)")
            results[ch] = entry
        return results

    def _save_timing_map(self, results, channels, reference_input, tag="map"):
        outdir = self.output_dir or os.getcwd()
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, f"multiharp_timingmap_{stamp}")
        save = {"reference_input": reference_input,
                "reference_label": channels[reference_input][0]}
        for ch, e in results.items():
            save[f"ch{ch}_label"] = e["label"]
            save[f"ch{ch}_delays_ns"] = e["delays_ns"]
            save[f"ch{ch}_hist_counts"] = e["hist_counts"]
            save[f"ch{ch}_hist_edges_ns"] = e["hist_edges_ns"]
        npz = base + ".npz"
        np.savez_compressed(npz, **save)
        files = [npz]
        if self.make_plots and results:
            try:
                chs = sorted(results)
                fig, axes = plt.subplots(len(chs), 1, sharex=True,
                                         figsize=(10, 1.9 * len(chs) + 0.6))
                if len(chs) == 1:
                    axes = [axes]
                for ax, ch in zip(axes, chs):
                    e = results[ch]
                    if e["hist_counts"].size:
                        ctr = 0.5 * (e["hist_edges_ns"][:-1] + e["hist_edges_ns"][1:])
                        ax.fill_between(ctr, e["hist_counts"], step="mid", alpha=.6)
                        ax.axvline(e["peak_ns"], ls="--", lw=1, color="k")
                    ax.axvline(0, color="red", lw=1.2)     # reference at t=0
                    ax.set_ylabel(e["label"], fontsize=8, rotation=0,
                                  ha="right", va="center")
                    ax.set_yticks([])
                axes[-1].set_xlabel(f"time relative to "
                                    f"'{channels[reference_input][0]}' "
                                    f"(input {reference_input})  [ns]")
                axes[0].set_title("MultiHarp timing map  (red = reference at t=0)")
                fig.tight_layout()
                png = base + ".png"
                fig.savefig(png, dpi=130)
                plt.close(fig)
                files.append(png)
            except Exception as ex:
                print(f"[MultiHarp] timing-map plot warning: {ex}")
        print("[MultiHarp] timing map saved: " + ", ".join(files))
        return files

    def run_timing_map(self, experiment, channels: Dict[int, Tuple[str, int, int]],
                       reference_input: int, frequency_range=None,
                       pre_ns: float = 1000.0, post_ns: float = 4000.0,
                       hist_bin_ns: float = 5.0, run_method: str = "run_experiment",
                       save: bool = True) -> Dict[str, Any]:
        """One run, many channels -> the full timing diagram of the sequence.

        channels        : {input_index: (label, level_mV, edge)} -- what each BNC
                          carries and its (attenuated!) trigger level and edge.
        reference_input : the input used as t=0 for every histogram. Set it to the
                          counter-OPEN proxy for the counter-alignment run, or to
                          the trigger to see the whole sequence laid out forward.
        pre_ns/post_ns  : how far before/after each reference edge to collect.
        """
        if reference_input not in channels:
            raise ValueError("reference_input must be one of the channels")
        if frequency_range is None:
            frequency_range = experiment.settings['microwave']['frequency range']
        if not hasattr(experiment, run_method):
            raise AttributeError(f"experiment has no method {run_method!r}")
        if self.output_dir is None:
            self.output_dir = getattr(experiment, "output_dir", None) or os.getcwd()

        self._open_and_configure_map(channels)     # sets self._mh
        odmr_result: Dict[str, Any] = {}
        try:
            self.start_acquisition()                # reuses the already-open device
            try:
                odmr_result = getattr(experiment, run_method)(frequency_range)
            finally:
                records = self.stop_acquisition()
            results = self.analyze_timing_map(records, channels, reference_input,
                                              pre_ns=pre_ns, post_ns=post_ns,
                                              hist_bin_ns=hist_bin_ns)
            if save:
                self._save_timing_map(results, channels, reference_input,
                                      tag=getattr(experiment, "tag", "map"))
        finally:
            self.close()
        return {"odmr": odmr_result, "map": results}
