"""
significance_test.py — is the feature real, or is the fitter chasing noise?
===========================================================================

Companion to plot_pulsed_hdf5.py (keep both in the same folder). Pops up the
same file picker + mode chooser, then runs the appropriate "is it real" test
and prints a false-alarm probability (FAP) with a plain-language verdict.

Which test per mode:
  * Rabi (time / amplitude), Ramsey  -> Lomb-Scargle periodogram + a
        permutation test on the peak. FAP = fraction of random shuffles of the
        same data whose strongest periodogram peak is at least as tall as the
        one in our data. Small FAP => a real oscillation at that frequency.
  * Pulsed ODMR (a dip), Hahn echo (a decay) -> a periodogram is the wrong
        tool (no oscillation frequency). Instead: fit the expected shape
        (Lorentzian / stretched exponential), then permute the data and refit
        many times. FAP = fraction of shuffles whose fit explains the data
        (R^2) at least as well as the fit to our real data.

Verdict thresholds (same bar I used to separate our real Rabi/Ramsey from the
noise fits):  FAP < 1%  -> real;  1-5% -> marginal (average more);  >5% -> noise.

Run:  python significance_test.py                       (picker + popup)
  or  python significance_test.py file.h5 [mode] [n_perm]
      mode in: rabi_time rabi_amp odmr ramsey echo
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.signal import lombscargle
from scipy.optimize import curve_fit

OSC_MODES = {"rabi_time", "rabi_amp", "ramsey"}     # periodogram applies
FIT_MODES = {"odmr", "echo"}                        # shape-fit permutation


# ── models (same forms as the plotter) ────────────────────────────────
def lorentz1(f, C, d, f0, hw):
    return C - d * (hw ** 2) / ((f - f0) ** 2 + hw ** 2)

def stretched_decay(x, A, C, T, p):
    return C + A * np.exp(-(x / T) ** p)


# ── periodogram + permutation FAP (oscillation modes) ─────────────────
def _freq_grid(mode, x, npts=1400):
    """Physical frequency array + the angular-frequency array scipy wants,
    given the sweep x in its native unit (ns for time, V for amplitude)."""
    dx = np.median(np.diff(np.sort(x)))
    if mode == "rabi_amp":                     # x in V -> cycles/V
        f_nyq = 0.5 / dx
        f = np.linspace(0.1, min(10.0, 0.95 * f_nyq), npts)
        ang = 2 * np.pi * f                    # cycles/V -> rad per V
        unit = "cyc/V"
    else:                                      # x in ns -> MHz
        f_nyq = 1000.0 / (2 * dx)
        f = np.linspace(0.5, min(50.0, 0.95 * f_nyq), npts)
        ang = 2 * np.pi * f / 1000.0           # MHz -> rad per ns
        unit = "MHz"
    return f, ang, unit, f_nyq


def periodogram_fap(x, y, ang, n_perm=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    y = y - np.mean(y)
    pg = lombscargle(x, y, ang, normalize=True)
    obs = pg.max()
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = lombscargle(x, rng.permutation(y), ang, normalize=True).max()
    fap = (np.sum(null >= obs) + 1) / (n_perm + 1)     # +1: never report exactly 0
    return pg, obs, fap, np.percentile(null, 95)


# ── shape-fit + permutation FAP (dip / decay modes) ───────────────────
def _fit_lorentz(f_hz, y):
    fM = f_hz / 1e6
    C0 = np.median(y); d0 = C0 - np.min(y); f0 = fM[int(np.argmin(y))]
    span = fM.max() - fM.min()
    p, _ = curve_fit(lorentz1, fM, y, p0=[C0, max(d0, 1e-6), f0, max(span / 20, 1.0)],
                     bounds=([-2, 0, fM.min(), 0.1], [2, 2, fM.max(), span]), maxfev=40000)
    return lorentz1(fM, *p), p

def _fit_stretched(x_ns, y):
    xu = x_ns / 1e3
    rising = np.mean(y[:3]) < np.mean(y[-3:])
    A0 = (np.min(y) - np.max(y)) if rising else (np.max(y) - np.min(y))
    p, _ = curve_fit(stretched_decay, xu, y,
                     p0=[A0, float(np.median(y)), 0.4 * (xu.max() - xu.min()) + 1e-3, 1.5],
                     bounds=([-2, -2, 1e-4, 0.5], [2, 2, 1e6, 4.0]), maxfev=40000)
    return stretched_decay(xu, *p), p

def fit_r2_fap(x, y, fit_fn, n_perm=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    def r2_of(xx, yy):
        yh, p = fit_fn(xx, yy)
        return 1 - np.sum((yy - yh) ** 2) / np.sum((yy - np.mean(yy)) ** 2), p
    r2, p = r2_of(x, y)
    null = np.empty(n_perm)
    for i in range(n_perm):
        try:
            null[i] = r2_of(x, rng.permutation(y))[0]
        except Exception:
            null[i] = 0.0
    fap = (np.sum(null >= r2) + 1) / (n_perm + 1)
    return r2, fap, np.percentile(null, 95), p


def verdict(fap):
    if fap < 0.01: return "REAL  (p < 1%)"
    if fap < 0.05: return f"MARGINAL  (p = {fap*100:.1f}%) — average more before trusting it"
    return f"NOT significant  (p = {fap*100:.0f}%) — consistent with noise"


def _drop_dead(x, sig, ref):
    """Drop startup points where signal ~ 0 (S/R garbage), like tau=4/6.5 ns."""
    med = np.median(sig[sig > 0]) if np.any(sig > 0) else 0
    live = sig > 0.2 * med
    n_dead = int((~live).sum())
    return x[live], sig[live], ref[live], n_dead


# ── main ──────────────────────────────────────────────────────────────
def main():
    try:
        from plot_pulsed_hdf5 import (pick_file, pick_mode, load_hdf5, detect_mode,
                                       build_time_axis, build_amp_axis)
    except Exception as e:
        print("Could not import plot_pulsed_hdf5.py — keep it in the same folder.\n", e)
        return

    path = sys.argv[1] if len(sys.argv) > 1 else pick_file()
    if not path:
        print("No file selected."); return
    n_perm = int(sys.argv[3]) if len(sys.argv) > 3 else None

    d = load_hdf5(path)
    sig, ref, n = d["sig"], d["ref"], d["n"]
    auto = detect_mode(d["seq_text"], d["has_freq_array"])
    mode = sys.argv[2] if len(sys.argv) > 2 else pick_mode(auto)

    # x-axis for the chosen mode
    if mode == "odmr":
        x = d["freq_axis"]; xlabel = d["freq_label"]
    elif mode == "rabi_amp":
        x, xlabel = build_amp_axis(d["seq_text"], n)
    else:
        x, xlabel = build_time_axis(d["seq_text"], n)

    x, sig, ref, n_dead = _drop_dead(x, sig, ref)
    if n_dead:
        print(f"  dropped {n_dead} dead startup point(s) (signal ~ 0)")
    y = sig / ref
    rng = np.random.default_rng(0)
    print(f"  mode = {mode}   {len(y)} live points   ref CV = "
          f"{100*np.std(ref)/np.mean(ref):.2f}%")

    import matplotlib; matplotlib.use("TkAgg" if sys.stdout.isatty() else "Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))

    if mode in OSC_MODES:
        nperm = n_perm or 2000
        f, ang, unit, fnyq = _freq_grid(mode, x)
        pg, obs, fap, thr = periodogram_fap(x, y, ang, n_perm=nperm, rng=rng)
        fpk = f[int(np.argmax(pg))]
        print(f"\n  Lomb-Scargle peak = {fpk:.2f} {unit}   power = {obs:.3f}"
              f"   (Nyquist {fnyq:.1f} {unit}, {nperm} shuffles)")
        print(f"  95% noise power   = {thr:.3f}")
        print(f"  FALSE-ALARM PROB  = {fap*100:.2f}%   ->  {verdict(fap)}")
        if mode == "rabi_time":
            print(f"  => f_Rabi = {fpk:.2f} MHz,  pi = {500/fpk:.0f} ns,  pi/2 = {250/fpk:.0f} ns")
        elif mode == "ramsey":
            print(f"  => detuning = {fpk:.2f} MHz  (fringe frequency; T2* from the envelope fit)")
        elif mode == "rabi_amp":
            print(f"  => A_pi = {0.5/fpk:.3f} V,  A_pi/2 = {0.25/fpk:.3f} V")
        ax.plot(f, pg, color="navy", lw=1.2, label="periodogram")
        ax.axhline(thr, color="crimson", ls="--", lw=1, label="95% noise floor")
        ax.plot(fpk, obs, "o", color="orange", ms=8, label=f"peak {fpk:.2f} {unit}")
        ax.set_xlabel(f"frequency ({unit})"); ax.set_ylabel("Lomb-Scargle power")
        ax.set_title(f"{mode}: peak {fpk:.2f} {unit},  FAP {fap*100:.2f}%")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    elif mode in FIT_MODES:
        nperm = n_perm or 1000
        fit_fn = _fit_lorentz if mode == "odmr" else _fit_stretched
        r2, fap, thr, p = fit_r2_fap(x, y, fit_fn, n_perm=nperm, rng=rng)
        print(f"\n  shape-fit R^2 = {r2:.3f}   (95% of noise-shuffles reach {thr:.3f}, "
              f"{nperm} shuffles)")
        print(f"  FALSE-ALARM PROB = {fap*100:.2f}%   ->  {verdict(fap)}")
        if mode == "odmr":
            print(f"  => resonance = {p[2]/1e3:.5f} GHz,  FWHM = {2*p[3]:.2f} MHz")
            xp = x / 1e9 if x.max() > 1e6 else x; xlab = "frequency (GHz)"
        else:
            print(f"  => T2 = {p[2]*1e3:.0f} ns  (stretch n = {p[3]:.2f})")
            xp = x; xlab = xlabel
        yh, _ = fit_fn(x, y)
        order = np.argsort(x)
        ax.plot(xp[order], y[order], "o", ms=4, color="darkorange", label="S/R")
        ax.plot(xp[order], yh[order], "-k", lw=1.5, label="shape fit")
        ax.axhline(np.mean(y), color="gray", ls=":", label="flat (null)")
        ax.set_xlabel(xlab); ax.set_ylabel("Signal / Reference")
        ax.set_title(f"{mode}: R^2 {r2:.3f},  FAP {fap*100:.2f}%")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    else:
        print("Unknown mode:", mode); return

    import os
    out = os.path.splitext(path)[0] + f"_{mode}_significance.png"
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"\n  saved -> {out}")
    except Exception as e:
        print(f"  (could not save: {e})")
    plt.show()


if __name__ == "__main__":
    main()
