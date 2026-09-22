#!/usr/bin/env python3
"""
hbt_analysis.py
===============

Scientific core for a Hanbury Brown-Twiss (HBT) g2(tau) measurement, kept as
pure numpy/scipy functions with NO dependency on the lab framework (src.*) or on
any hardware. That separation is deliberate:

  * HBT_Experiment.py imports these functions to do the real measurement, and
  * hbt_selftest.py exercises them with known-answer and statistical tests that
    run anywhere (no MultiHarp, no src.core).

Everything here works on photon arrival-time arrays in PICOSECONDS (which is what
PicoQuant_MultiHarp.decode_* returns), sorted ascending.

Conventions
-----------
g2(tau) is built from the cross-correlation histogram of (t_B - t_A) for the two
detectors of the HBT, normalized so that uncorrelated light gives g2 -> 1 and a
single emitter shows an antibunching dip g2(0) < 0.5 at tau = 0.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

PS_PER_NS = 1000.0


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #
def cross_correlate(t_a: np.ndarray,
                    t_b: np.ndarray,
                    tau_window_ps: float,
                    bin_ps: float,
                    chunk: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram delays (t_b - t_a) for all pairs with |t_b - t_a| <= tau_window_ps.

    This is the start-stop HBT correlation. Both inputs must be sorted ascending
    and in ps. Returns (bin_centers_ps, hist_counts). Memory is bounded by
    processing the start photons in chunks; within a chunk it is fully vectorized.
    """
    t_a = np.asarray(t_a, dtype=np.float64)
    t_b = np.asarray(t_b, dtype=np.float64)
    nbins = int(round(2.0 * tau_window_ps / bin_ps))
    if nbins < 1:
        raise ValueError("tau_window_ps / bin_ps too small: need at least 1 bin")
    edges = -tau_window_ps + bin_ps * np.arange(nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(nbins, dtype=np.int64)
    if t_a.size == 0 or t_b.size == 0:
        return centers, hist

    for s in range(0, t_a.size, chunk):
        ta = t_a[s:s + chunk]
        lo = np.searchsorted(t_b, ta - tau_window_ps, side="left")
        hi = np.searchsorted(t_b, ta + tau_window_ps, side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            continue
        # Build the concatenated list of stop indices for every (start, stop) pair
        # without a Python loop: base offset per start + within-group index.
        base = np.repeat(np.cumsum(counts) - counts, counts)
        within = np.arange(total) - base
        stop_idx = np.repeat(lo, counts) + within
        dt = t_b[stop_idx] - np.repeat(ta, counts)     # ps, in [-window, +window]
        bidx = np.floor((dt + tau_window_ps) / bin_ps).astype(np.int64)
        np.clip(bidx, 0, nbins - 1, out=bidx)
        hist += np.bincount(bidx, minlength=nbins)
    return centers, hist


def coincidence_denominator(counts_a: int, counts_b: int,
                            bin_ps: float, total_time_ps: float) -> float:
    """Expected accidental coincidences PER BIN for uncorrelated light:
    N_A * N_B * bin_width / T. Dividing the raw histogram by this normalizes
    g2 -> 1 for uncorrelated sources. For block accumulation, sum this per block."""
    if total_time_ps <= 0:
        return 0.0
    return counts_a * counts_b * bin_ps / total_time_ps


def normalize_g2(hist: np.ndarray, denominator: float) -> np.ndarray:
    """g2 = hist / (summed per-bin accidental expectation). NaN if denom <= 0."""
    if denominator <= 0:
        return np.full(np.shape(hist), np.nan)
    return hist.astype(np.float64) / denominator


# --------------------------------------------------------------------------- #
# Antibunching model + fit
# --------------------------------------------------------------------------- #
def two_level_g2(tau_ns: np.ndarray, g2_zero: float, tau_c_ns: float,
                 tau0_ns: float = 0.0, offset: float = 1.0) -> np.ndarray:
    """CW two-level antibunching: offset - (offset - g2_zero) * exp(-|tau - tau0| / tau_c).

    g2(tau0) = g2_zero (the dip), g2 -> offset (=1 for an ideal normalization) at
    large |tau|. tau_c is the antibunching recovery time (~ 1 / (pump + decay))."""
    tau_ns = np.asarray(tau_ns, dtype=np.float64)
    return offset - (offset - g2_zero) * np.exp(-np.abs(tau_ns - tau0_ns) / tau_c_ns)


def fit_g2(tau_ns: np.ndarray, g2: np.ndarray,
           g2_zero_guess: float = 0.3, tau_c_guess_ns: float = 12.0,
           tau0_guess_ns: float = 0.0, fit_offset: bool = False) -> Dict:
    """Least-squares fit of g2(tau) to two_level_g2. Returns a dict with g2_zero,
    tau_c_ns, tau0_ns (and offset if fit_offset), 1-sigma errors, and success flag.
    Degrades gracefully (success=False) if scipy is missing or the fit fails."""
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return {"success": False, "reason": "scipy not available"}

    tau_ns = np.asarray(tau_ns, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)
    m = np.isfinite(g2) & np.isfinite(tau_ns)
    if int(m.sum()) < 5:
        return {"success": False, "reason": "not enough finite points to fit"}
    x, y = tau_ns[m], g2[m]
    span = float(np.max(x) - np.min(x)) or 1.0

    if fit_offset:
        def model(t, g0, tc, t0, off):
            return two_level_g2(t, g0, tc, t0, off)
        p0 = [g2_zero_guess, tau_c_guess_ns, tau0_guess_ns, 1.0]
        bounds = ([0.0, 1e-3, x.min(), 0.5], [2.0, span, x.max(), 2.0])
    else:
        def model(t, g0, tc, t0):
            return two_level_g2(t, g0, tc, t0, 1.0)
        p0 = [g2_zero_guess, tau_c_guess_ns, tau0_guess_ns]
        bounds = ([0.0, 1e-3, x.min()], [2.0, span, x.max()])

    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as e:
        return {"success": False, "reason": f"curve_fit failed: {e}"}

    perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) \
        else np.full(len(popt), np.nan)
    out = {
        "success": True,
        "g2_zero": float(popt[0]),
        "g2_zero_err": float(perr[0]),
        "tau_c_ns": float(popt[1]),
        "tau0_ns": float(popt[2]),
        "offset": float(popt[3]) if fit_offset else 1.0,
    }
    return out


# --------------------------------------------------------------------------- #
# Simulation (used by sim-mode of the experiment AND by the self-test)
# --------------------------------------------------------------------------- #
def simulate_hbt_streams(duration_s: float,
                         rate_cps: float,
                         g2_zero: float,
                         tau_c_ns: float = 12.0,
                         tau0_ns: float = 0.0,
                         jitter_ps: float = 300.0,
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two detector photon streams (times in ps, sorted) whose A-B
    cross-correlation follows two_level_g2(tau; g2_zero, tau_c_ns, tau0_ns).

    Method: start from two independent Poisson singles streams at `rate_cps`
    (so both detectors see the requested singles rate, as real ones do), then
    thin A-B coincidences: each B photon is dropped with probability
    (1 - g2_zero) * exp(-|dt - tau0| / tau_c) using its nearest A partner. Because
    real detected rates are far below the emitter rate, coincidences are rare and
    this reproduces the physical g2 shape while remaining O(N) and memory-light.
    Detector timing jitter is added as Gaussian noise. Returns (t_A_ps, t_B_ps)."""
    if rng is None:
        rng = np.random.default_rng()
    t_ps = duration_s * 1e12
    n_a = int(rng.poisson(rate_cps * duration_s))
    n_b = int(rng.poisson(rate_cps * duration_s))
    t_a = np.sort(rng.uniform(0.0, t_ps, n_a))
    t_b = np.sort(rng.uniform(0.0, t_ps, n_b))

    tau0_ps = tau0_ns * PS_PER_NS
    tau_c_ps = tau_c_ns * PS_PER_NS
    if t_a.size and t_b.size and g2_zero < 1.0:
        shifted = t_b - tau0_ps                    # compare A against (B - tau0)
        idx = np.searchsorted(t_a, shifted)
        left = np.clip(idx - 1, 0, t_a.size - 1)
        right = np.clip(idx, 0, t_a.size - 1)
        d_min = np.minimum(np.abs(shifted - t_a[left]), np.abs(shifted - t_a[right]))
        p_remove = (1.0 - g2_zero) * np.exp(-d_min / tau_c_ps)
        keep = rng.random(t_b.size) >= p_remove
        t_b = t_b[keep]

    if jitter_ps > 0:
        t_a = np.sort(t_a + rng.normal(0.0, jitter_ps, t_a.size))
        t_b = np.sort(t_b + rng.normal(0.0, jitter_ps, t_b.size))
    return t_a, t_b


# --------------------------------------------------------------------------- #
# Convenience: full g2 from two streams (single block)
# --------------------------------------------------------------------------- #
def g2_single_block(t_a: np.ndarray, t_b: np.ndarray, duration_s: float,
                    tau_window_ns: float, bin_ns: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Correlate one block and normalize. Returns
    (tau_ns, g2, hist_counts, denominator). Useful for quick checks / the
    experiment accumulates hist and denominator across blocks instead."""
    bin_ps = bin_ns * PS_PER_NS
    window_ps = tau_window_ns * PS_PER_NS
    centers_ps, hist = cross_correlate(t_a, t_b, window_ps, bin_ps)
    denom = coincidence_denominator(t_a.size, t_b.size, bin_ps, duration_s * 1e12)
    g2 = normalize_g2(hist, denom)
    return centers_ps / PS_PER_NS, g2, hist, denom
