#!/usr/bin/env python3
"""
hbt_selftest.py
===============

Rigorous, hardware-free tests of the HBT scientific core (hbt_analysis.py). No
MultiHarp, no src.core -- runs anywhere numpy/scipy are installed:

    python hbt_selftest.py

Two flavors, mirroring the multiharp_device_check offline test:
  * Deterministic known-answer tests of the correlator (exact integer checks).
  * Statistical tests of the simulator + normalization + fit: an antibunched
    source must produce a dip and the fit must recover the injected g2(0).

Exits non-zero if any test fails, so it doubles as a regression check.
"""

from __future__ import annotations

import sys

import numpy as np

import hbt_analysis as H

NS = H.PS_PER_NS  # 1000 ps per ns


def _report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
# Deterministic correlator tests
# --------------------------------------------------------------------------- #
def test_correlator_known_answer():
    t_a = np.array([0, 100]) * NS
    t_b = np.array([10, 20, 105]) * NS
    centers_ps, hist = H.cross_correlate(t_a, t_b, tau_window_ps=50 * NS, bin_ps=5 * NS)
    # A=0 -> B={10,20} (Δ=+10,+20); A=100 -> B={105} (Δ=+5); B=105 vs A=0 is +105 (out)
    got = {round(centers_ps[i] / NS, 1): int(hist[i]) for i in np.nonzero(hist)[0]}
    ok = (got == {7.5: 1, 12.5: 1, 22.5: 1}) and int(hist.sum()) == 3
    return _report("correlator known answer", ok, f"nonzero bins={got}")


def test_correlator_empty():
    c, h = H.cross_correlate(np.array([]), np.array([1.0, 2.0]) * NS, 50 * NS, 5 * NS)
    ok1 = (h.sum() == 0 and h.size == 20)
    c, h = H.cross_correlate(np.array([1.0]) * NS, np.array([]), 50 * NS, 5 * NS)
    ok2 = (h.sum() == 0 and h.size == 20)
    return _report("correlator empty inputs", ok1 and ok2)


def test_correlator_chunking_invariance():
    rng = np.random.default_rng(1)
    t_a = np.sort(rng.uniform(0, 1e9, 5000))
    t_b = np.sort(rng.uniform(0, 1e9, 5000))
    _, h_full = H.cross_correlate(t_a, t_b, 5000.0, 100.0, chunk=10 ** 9)
    _, h_chunked = H.cross_correlate(t_a, t_b, 5000.0, 100.0, chunk=137)
    ok = np.array_equal(h_full, h_chunked)
    return _report("correlator chunking invariance", ok,
                   f"identical over {h_full.sum():,} pairs")


def test_correlator_swap_symmetry():
    # hist(B, A) must equal the mirror image of hist(A, B): swapping detectors
    # flips the sign of every delay.
    rng = np.random.default_rng(2)
    t_a = np.sort(rng.uniform(0, 1e9, 4000))
    t_b = np.sort(rng.uniform(0, 1e9, 4000))
    _, h_ab = H.cross_correlate(t_a, t_b, 3000.0, 60.0)
    _, h_ba = H.cross_correlate(t_b, t_a, 3000.0, 60.0)
    ok = np.array_equal(h_ab, h_ba[::-1])
    return _report("correlator swap symmetry", ok)


# --------------------------------------------------------------------------- #
# Statistical tests: simulator + normalization + fit
# --------------------------------------------------------------------------- #
def _accumulate_g2(g2_zero, tau_c_ns=12.0, tau0_ns=0.0, jitter_ps=250.0,
                   rate=4e5, block_s=4.0, n_blocks=6, window_ns=100.0,
                   bin_ns=1.0, seed=42):
    rng = np.random.default_rng(seed)
    bin_ps = bin_ns * NS
    window_ps = window_ns * NS
    hist = None
    denom = 0.0
    centers_ps = None
    for _ in range(n_blocks):
        t_a, t_b = H.simulate_hbt_streams(block_s, rate, g2_zero,
                                          tau_c_ns=tau_c_ns, tau0_ns=tau0_ns,
                                          jitter_ps=jitter_ps, rng=rng)
        centers_ps, h = H.cross_correlate(t_a, t_b, window_ps, bin_ps)
        hist = h.copy() if hist is None else hist + h
        denom += H.coincidence_denominator(t_a.size, t_b.size, bin_ps, block_s * 1e12)
    tau_ns = centers_ps / NS
    g2 = H.normalize_g2(hist, denom)
    return tau_ns, g2


def test_normalization_uncorrelated():
    # g2_zero=1 -> no dip -> flat g2 ~ 1 everywhere (tests the normalization).
    tau_ns, g2 = _accumulate_g2(g2_zero=1.0, jitter_ps=0.0, n_blocks=4, seed=7)
    mean = float(np.nanmean(g2))
    ok = abs(mean - 1.0) < 0.03
    return _report("normalization: uncorrelated -> g2~1", ok,
                   f"mean g2 = {mean:.3f}")


def test_antibunching_recovery():
    all_ok = True
    for target in (0.05, 0.2, 0.4):
        tau_ns, g2 = _accumulate_g2(g2_zero=target, tau_c_ns=12.0, seed=42)
        fit = H.fit_g2(tau_ns, g2, g2_zero_guess=0.3, tau_c_guess_ns=12.0)
        ok = (fit.get('success')
              and abs(fit['g2_zero'] - target) < 0.05
              and 6.0 < fit['tau_c_ns'] < 24.0)
        all_ok &= _report(
            f"antibunching recovery (target g2(0)={target})", ok,
            "fit g2(0)={:.3f}±{:.3f}, tau_c={:.1f} ns".format(
                fit.get('g2_zero', float('nan')),
                fit.get('g2_zero_err', float('nan')),
                fit.get('tau_c_ns', float('nan'))))
    return all_ok


def test_tau0_offset_recovery():
    # Put the dip at tau0 = +20 ns and confirm the fit localizes it there.
    tau_ns, g2 = _accumulate_g2(g2_zero=0.2, tau_c_ns=10.0, tau0_ns=20.0, seed=11)
    fit = H.fit_g2(tau_ns, g2, g2_zero_guess=0.3, tau_c_guess_ns=10.0, tau0_guess_ns=0.0)
    ok = fit.get('success') and abs(fit['tau0_ns'] - 20.0) < 4.0 and abs(fit['g2_zero'] - 0.2) < 0.06
    return _report("nonzero tau0 recovery (injected +20 ns)", ok,
                   "fit tau0={:.1f} ns, g2(0)={:.3f}".format(
                       fit.get('tau0_ns', float('nan')), fit.get('g2_zero', float('nan'))))


def test_model_roundtrip():
    tau = np.linspace(-100, 100, 401)
    y = H.two_level_g2(tau, g2_zero=0.15, tau_c_ns=12.0, tau0_ns=0.0, offset=1.0)
    ok = (abs(y[np.argmin(np.abs(tau))] - 0.15) < 1e-9
          and abs(y[0] - 1.0) < 1e-3 and abs(y[-1] - 1.0) < 1e-3)
    return _report("two_level_g2 model roundtrip", ok)


def main():
    print("HBT analysis self-test\n" + "-" * 40)
    print("Deterministic correlator:")
    results = [
        test_correlator_known_answer(),
        test_correlator_empty(),
        test_correlator_chunking_invariance(),
        test_correlator_swap_symmetry(),
    ]
    print("\nModel + simulator + fit:")
    results += [
        test_model_roundtrip(),
        test_normalization_uncorrelated(),
        test_antibunching_recovery(),
        test_tau0_offset_recovery(),
    ]
    n_pass = sum(bool(r) for r in results)
    n_tot = len(results)
    print("-" * 40)
    print(f"{n_pass}/{n_tot} test groups passed")
    if n_pass != n_tot:
        sys.exit(1)


if __name__ == '__main__':
    main()
