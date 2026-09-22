"""
#Written by <Jannet Trabelsi>
plot_pulsed_hdf5.py — pick a saved pulsed-experiment HDF5, choose the
experiment type in a popup, and fit + plot it.
========================================================================

Handles five experiment types with the right x-axis, fit, and readout:

  * Rabi (time)      : sweep pulse_duration  -> damped cosine
                       reports f_Rabi (MHz), pi (ns), pi/2 (ns)
  * Rabi (amplitude) : sweep pulse_amplitude -> damped cosine in amplitude
                       reports A_pi (V), A_pi/2 (V)
  * Pulsed ODMR      : sweep frequency       -> Lorentzian dip(s)
                       reports resonance (GHz) and FWHM (MHz)
  * Ramsey           : sweep free-evolution  -> decaying cosine
                       reports detuning (MHz) and T2* (us)
  * Hahn echo        : sweep free-evolution  -> stretched-exp decay
                       reports T2 (us)

It stays layout-agnostic: it walks the whole HDF5, finds signal / reference
by name, and finds sequence_text / repeat_count / a frequencies array
wherever they sit (dataset or attribute). The experiment type is
auto-detected from the stored sequence text ("type=..." and the "variable"
line); a popup lets the user confirm or override the guess.

Needs: h5py, numpy, scipy, matplotlib, tkinter
Run it:  python plot_pulsed_hdf5.py        (file picker + mode popup)
     or  python plot_pulsed_hdf5.py file.h5 [rabi_time|rabi_amp|odmr|ramsey|echo]
"""
from __future__ import annotations
import os, re, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import tkinter as tk
from tkinter import filedialog
import h5py

MODES = ["rabi_time", "rabi_amp", "odmr", "ramsey", "echo"]
MODE_LABEL = {
    "rabi_time": "Rabi  (sweep pulse duration)",
    "rabi_amp":  "Rabi  (sweep pulse amplitude)",
    "odmr":      "Pulsed ODMR  (sweep frequency)",
    "ramsey":    "Ramsey  (T2*)",
    "echo":      "Hahn echo  (T2)",
}


# ──────────────────────────────────────────────────────────────────────
# File picker + mode popup
# ──────────────────────────────────────────────────────────────────────
def pick_file() -> str:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a pulsed-experiment HDF5 file",
        filetypes=[("HDF5 files", "*.h5 *.hdf5 *.he5 *.hf5"), ("All files", "*.*")])
    root.destroy()
    return path


def pick_mode(default: str) -> str:
    """Small radio-button popup. Returns the chosen mode string."""
    root = tk.Tk()
    root.title("Which experiment?")
    root.attributes("-topmost", True)
    choice = tk.StringVar(value=default if default in MODES else "rabi_time")
    tk.Label(root, text="Detected: %s\nConfirm or change:" % MODE_LABEL.get(default, "?"),
             justify="left", padx=12, pady=8).pack(anchor="w")
    for m in MODES:
        tk.Radiobutton(root, text=MODE_LABEL[m], variable=choice, value=m,
                       anchor="w", padx=12).pack(fill="x")
    done = {"go": False}
    def ok(): done["go"] = True; root.quit()
    tk.Button(root, text="Plot", command=ok, padx=20, pady=4).pack(pady=8)
    root.protocol("WM_DELETE_WINDOW", ok)
    root.mainloop()
    val = choice.get(); root.destroy()
    return val


# ──────────────────────────────────────────────────────────────────────
# HDF5 walking / lookup (layout-agnostic)  — from plot_rabi_hdf5.py
# ──────────────────────────────────────────────────────────────────────
def _all_datasets(f):
    out = {}
    f.visititems(lambda name, obj: out.__setitem__(name, obj)
                 if isinstance(obj, h5py.Dataset) else None)
    return out


def _print_tree(f):
    print("\n  HDF5 contents:")
    def show(name, obj):
        kind = "group" if isinstance(obj, h5py.Group) else \
               f"dataset shape={obj.shape} dtype={obj.dtype}"
        print(f"    /{name}   [{kind}]")
        for k in obj.attrs: print(f"        .{k} (attr)")
    for k in f.attrs: print(f"    /.{k} (attr)")
    f.visititems(show)


def _get_dataset(datasets, *name_options):
    for opt in name_options:
        for path, ds in datasets.items():
            if opt in path.lower():
                return ds
    return None


def _get_value(f, datasets, *name_options):
    ds = _get_dataset(datasets, *name_options)
    if ds is not None:
        return ds[()]
    hits = []
    def check(obj):
        for k, v in obj.attrs.items():
            if any(opt in k.lower() for opt in name_options):
                hits.append(v)
    check(f); f.visititems(lambda n, o: check(o))
    return hits[0] if hits else None


def _to_text(v):
    if v is None: return None
    if isinstance(v, (bytes, np.bytes_)): return v.decode("utf-8", "ignore")
    if isinstance(v, np.ndarray):
        v = v.ravel()
        if v.size == 0: return None
        parts = [x.decode("utf-8", "ignore") if isinstance(x, (bytes, np.bytes_)) else str(x)
                 for x in v]
        return "\n".join(parts)          # join lines; don't keep only the first
    return str(v)


def _to_scalar(v):
    if v is None: return None
    try: return float(np.asarray(v).ravel()[0])
    except Exception: return None


_SEQ_MARK = re.compile(r"variable\s+\w+\s*,\s*start|sequence:\s*name|pulse\s+on\s+channel", re.I)

def _find_seq_text(f, datasets):
    """Return the real multi-line sequence text. Prefer an exact 'sequence_text'
    key, but validate it actually looks like a sequence (a bare 'sequence'
    substring can match unrelated fields like number_of_sequence_done_by_adwin).
    Fall back to scanning every string attr/dataset and taking the longest match."""
    t = _to_text(_get_value(f, datasets, "sequence_text"))
    if t and _SEQ_MARK.search(t):
        return t
    best = None
    def consider(v):
        nonlocal best
        s = _to_text(v)
        if s and _SEQ_MARK.search(s) and (best is None or len(s) > len(best)):
            best = s
    for _, v in f.attrs.items():
        consider(v)
    def visit(name, obj):
        for _, v in obj.attrs.items():
            consider(v)
        if isinstance(obj, h5py.Dataset):
            try: consider(obj[()])
            except Exception: pass
    f.visititems(visit)
    return best


def _as_1d(v):
    a = np.asarray(v).squeeze()
    if a.ndim == 2:            # (n_outer, n_inner) -> first row
        a = a[0]
    return a.astype(float).ravel()


# ──────────────────────────────────────────────────────────────────────
# x-axis construction
# ──────────────────────────────────────────────────────────────────────
def _num(token, default_unit="ns"):
    m = re.search(r"([0-9]*\.?[0-9]+)\s*(ns|us|µs|μs|ms|v|mv|ghz|mhz|khz|hz)?",
                  token, re.IGNORECASE)
    if not m: return float("nan"), default_unit
    val = float(m.group(1)); unit = (m.group(2) or default_unit).lower()
    return val, unit


def _variable_line(seq_text):
    """Return (name, start_token, stop_token) from the 'variable ...' line,
    ignoring commented (#) copies."""
    if not seq_text: return None
    m = re.search(r"(?<!#)\bvariable\s+(\w+)\s*,\s*start\s*=\s*([^,]+),\s*stop\s*=\s*([^,]+)",
                  seq_text, re.IGNORECASE)
    if not m: return None
    return m.group(1).lower(), m.group(2), m.group(3)


def build_time_axis(seq_text, n):
    vl = _variable_line(seq_text)
    if vl:
        _, s_tok, e_tok = vl
        s, su = _num(s_tok, "ns"); e, eu = _num(e_tok, "ns")
        sc = {"ns": 1.0, "us": 1e3, "µs": 1e3, "μs": 1e3, "ms": 1e6}
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s * sc.get(su, 1), e * sc.get(eu, 1), n), "tau (ns)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_amp_axis(seq_text, n):
    vl = _variable_line(seq_text)
    if vl:
        _, s_tok, e_tok = vl
        s, _ = _num(s_tok, "v"); e, _ = _num(e_tok, "v")
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s, e, n), "pulse amplitude (V)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_freq_axis(f, datasets, seq_text, n):
    """Prefer a stored frequency array; else reconstruct from start/stop
    (+ step or n). Returns (freqs_Hz, label)."""
    arr = _get_dataset(datasets, "frequencies", "freq_list", "freq_axis")
    if arr is not None:
        fa = _as_1d(arr[()])
        if len(fa) >= n:
            return fa[:n], "frequency (GHz)"
    start = _to_scalar(_get_value(f, datasets, "start_freq", "freq_start")) 
    stop  = _to_scalar(_get_value(f, datasets, "stop_freq",  "freq_stop"))
    if start is None or stop is None:      # nested frequency_range [start, stop]
        fr = _get_value(f, datasets, "frequency_range")
        try:
            fr = np.asarray(fr).ravel().astype(float)
            if fr.size >= 2: start, stop = fr[0], fr[-1]
        except Exception:
            pass
    if start is not None and stop is not None:
        return np.linspace(start, stop, n), "frequency (GHz)"
    return np.arange(n, dtype=float), "scan point (index)"


# ──────────────────────────────────────────────────────────────────────
# mode detection
# ──────────────────────────────────────────────────────────────────────
def detect_mode(seq_text, has_freq_array):
    t = (seq_text or "").lower()
    typ = None
    m = re.search(r"type\s*=\s*(\w+)", t)
    if m: typ = m.group(1)
    vl = _variable_line(seq_text)
    varname = vl[0] if vl else ""

    if typ in ("ramsey",): return "ramsey"
    if typ in ("echo", "hahn", "hahn_echo", "spinecho"): return "echo"
    if typ in ("odmr", "pulsed_odmr", "podmr") or has_freq_array: return "odmr"
    if "amplitude" in varname: return "rabi_amp"
    if "duration" in varname or "width" in varname or typ == "rabi": return "rabi_time"
    return "rabi_time"


# ──────────────────────────────────────────────────────────────────────
# models
# ──────────────────────────────────────────────────────────────────────
def damped_cos(x, f, A, phi, C, decay):
    return C + A * np.cos(2 * np.pi * f * x + phi) * np.exp(-x / decay)

def gauss_cos(x, f, A, phi, C, T):          # Ramsey: gaussian envelope
    return C + A * np.cos(2 * np.pi * f * x + phi) * np.exp(-(x / T) ** 2)

def stretched_decay(x, A, C, T, p):         # Hahn echo
    return C + A * np.exp(-(x / T) ** p)

def lorentz1(f, C, d, f0, hw):
    return C - d * (hw ** 2) / ((f - f0) ** 2 + hw ** 2)

def lorentz2(f, C, d1, f1, w1, d2, f2, w2):
    return (C - d1 * (w1 ** 2) / ((f - f1) ** 2 + w1 ** 2)
              - d2 * (w2 ** 2) / ((f - f2) ** 2 + w2 ** 2))


def _osc_freq_seed(x, y):
    order = np.argsort(x); t = np.asarray(x)[order]; v = np.asarray(y)[order]
    grid = np.linspace(t.min(), t.max(), max(64, 4 * len(t)))
    vi = np.interp(grid, t, v) - np.mean(v)
    spec = np.abs(np.fft.rfft(vi)); spec[0] = 0.0
    fr = np.fft.rfftfreq(len(grid), d=grid[1] - grid[0])
    f = fr[int(np.argmax(spec))]
    return f if f > 0 else 1.0 / (t.max() - t.min() + 1e-9)


# ──────────────────────────────────────────────────────────────────────
# fitters (each returns a dict with a text summary + a dense curve)
# ──────────────────────────────────────────────────────────────────────
def fit_rabi(x, y, amplitude=False):
    """x in native units (ns for time, V for amp)."""
    xu = x / 1e3 if not amplitude else x        # time: ns -> us so f is in MHz
    f0 = _osc_freq_seed(xu, y)
    p0 = [f0, 0.5 * (np.nanmax(y) - np.nanmin(y)), 0.0, float(np.nanmean(y)),
          max(1e-3, 3 * (xu.max() - xu.min()))]
    lo = [1e-4, 0, -np.pi, -2, 1e-4]; hi = [1e4, 2, np.pi, 2, 1e9]
    popt, _ = curve_fit(damped_cos, xu, y, p0=p0, bounds=(lo, hi), maxfev=40000)
    xf = np.linspace(x.min(), x.max(), 600)
    yf = damped_cos(xf / 1e3 if not amplitude else xf, *popt)
    f = popt[0]
    if amplitude:
        half = 0.5 / f                          # amplitude for pi (period/2)
        title = f"amplitude Rabi:  A_pi = {half:.3f} V   A_pi/2 = {half/2:.3f} V"
        summary = (f"  amplitude period 1/f = {1/f:.3f} V\n"
                   f"  A_pi   = {half:.3f} V\n  A_pi/2 = {half/2:.3f} V")
    else:
        pi_ns = 500.0 / f
        title = f"f_Rabi = {f:.2f} MHz   pi = {pi_ns:.0f} ns   pi/2 = {pi_ns/2:.0f} ns"
        summary = (f"  f_Rabi = {f:.3f} MHz\n  pi     = {pi_ns:.1f} ns\n"
                   f"  pi/2   = {pi_ns/2:.1f} ns\n  envelope decay = {popt[4]:.3f} us")
    return dict(xf=xf, yf=yf, title=title, summary=summary)


def fit_ramsey(x, y):
    xu = x / 1e3                                # ns -> us
    f0 = _osc_freq_seed(xu, y)
    p0 = [f0, 0.5 * (np.nanmax(y) - np.nanmin(y)), 0.0, float(np.nanmean(y)),
          max(1e-3, 0.5 * (xu.max() - xu.min()))]
    lo = [0, 0, -np.pi, -2, 1e-4]; hi = [1e4, 2, np.pi, 2, 1e6]
    popt, _ = curve_fit(gauss_cos, xu, y, p0=p0, bounds=(lo, hi), maxfev=40000)
    xf = np.linspace(x.min(), x.max(), 600); yf = gauss_cos(xf / 1e3, *popt)
    title = f"Ramsey:  detuning = {popt[0]:.2f} MHz   T2* = {popt[4]*1e3:.0f} ns"
    summary = (f"  detuning = {popt[0]:.3f} MHz\n  T2*      = {popt[4]*1e3:.1f} ns "
               f"({popt[4]:.3f} us)")
    return dict(xf=xf, yf=yf, title=title, summary=summary)


def fit_echo(x, y):
    xu = x / 1e3                                # ns -> us
    rising = np.nanmean(y[:3]) < np.nanmean(y[-3:])
    A0 = (np.nanmin(y) - np.nanmax(y)) if rising else (np.nanmax(y) - np.nanmin(y))
    p0 = [A0, float(np.nanmedian(y)), max(1e-3, 0.4 * (xu.max() - xu.min())), 1.5]
    lo = [-2, -2, 1e-4, 0.5]; hi = [2, 2, 1e6, 4.0]
    popt, _ = curve_fit(stretched_decay, xu, y, p0=p0, bounds=(lo, hi), maxfev=40000)
    xf = np.linspace(x.min(), x.max(), 600); yf = stretched_decay(xf / 1e3, *popt)
    title = f"Hahn echo:  T2 = {popt[2]*1e3:.0f} ns   (n = {popt[3]:.2f})"
    summary = f"  T2 = {popt[2]*1e3:.1f} ns ({popt[2]:.3f} us)\n  stretch n = {popt[3]:.2f}"
    return dict(xf=xf, yf=yf, title=title, summary=summary)


def fit_odmr(f_hz, y):
    """f_hz in Hz; fit in MHz for stability. Try 2 dips, fall back to 1."""
    fM = f_hz / 1e6
    C0 = np.nanmedian(y); depth0 = C0 - np.nanmin(y)
    fmin = fM[int(np.nanargmin(y))]
    span = fM.max() - fM.min()
    # single
    p1 = [C0, max(depth0, 1e-6), fmin, max(span / 20, 1.0)]
    try:
        p1, _ = curve_fit(lorentz1, fM, y, p0=p1,
                          bounds=([-2, 0, fM.min(), 0.1], [2, 2, fM.max(), span]),
                          maxfev=40000)
        sse1 = np.sum((lorentz1(fM, *p1) - y) ** 2)
    except Exception:
        p1, sse1 = None, np.inf
    # double (seed two dips symmetric about the minimum)
    p2 = [C0, depth0, fmin - span / 8, span / 20, depth0, fmin + span / 8, span / 20]
    try:
        p2, _ = curve_fit(lorentz2, fM, y, p0=p2,
                          bounds=([-2, 0, fM.min(), 0.1, 0, fM.min(), 0.1],
                                  [2, 2, fM.max(), span, 2, fM.max(), span]),
                          maxfev=60000)
        sse2 = np.sum((lorentz2(fM, *p2) - y) ** 2)
    except Exception:
        p2, sse2 = None, np.inf

    xf = np.linspace(fM.min(), fM.max(), 800)
    if p2 is not None and sse2 < 0.7 * sse1:        # prefer double only if clearly better
        yf = lorentz2(xf, *p2)
        title = (f"resonances: {p2[2]/1e3:.4f} & {p2[5]/1e3:.4f} GHz   "
                 f"FWHM {2*p2[3]:.1f}, {2*p2[6]:.1f} MHz")
        summary = (f"  f1 = {p2[2]/1e3:.5f} GHz  FWHM {2*p2[3]:.2f} MHz\n"
                   f"  f2 = {p2[5]/1e3:.5f} GHz  FWHM {2*p2[6]:.2f} MHz\n"
                   f"  splitting = {abs(p2[5]-p2[2]):.2f} MHz")
    elif p1 is not None:
        yf = lorentz1(xf, *p1)
        title = f"resonance = {p1[2]/1e3:.5f} GHz   FWHM = {2*p1[3]:.1f} MHz"
        summary = f"  f0   = {p1[2]/1e3:.5f} GHz\n  FWHM = {2*p1[3]:.2f} MHz"
    else:
        return None
    return dict(xf=xf * 1e6, yf=yf, title=title, summary=summary)   # xf back to Hz


# ──────────────────────────────────────────────────────────────────────
# load
# ──────────────────────────────────────────────────────────────────────
def load_hdf5(path):
    with h5py.File(path, "r") as f:
        datasets = _all_datasets(f)
        sig_raw = _get_dataset(datasets, "signal_counts", "signal")
        ref_raw = _get_dataset(datasets, "reference_counts", "reference")
        if sig_raw is None or ref_raw is None:
            _print_tree(f)
            raise KeyError("Could not find signal/reference arrays; see tree above.")
        sig = _as_1d(sig_raw[()]); ref = _as_1d(ref_raw[()])
        seq_text = _find_seq_text(f, datasets)
        reps = _to_scalar(_get_value(f, datasets, "repeat_count"))
        has_freq_array = _get_dataset(datasets, "frequencies", "freq_list") is not None
        n = min(len(sig), len(ref))
        sig, ref = sig[:n], ref[:n]
        # x-axes are built lazily by the caller once the mode is known,
        # but ODMR needs the file handle, so build the freq axis here too.
        freq_axis, freq_label = build_freq_axis(f, datasets, seq_text, n)
    return dict(sig=sig, ref=ref, seq_text=seq_text, reps=reps, n=n,
                has_freq_array=has_freq_array,
                freq_axis=freq_axis, freq_label=freq_label)


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else pick_file()
    if not path:
        print("No file selected."); return
    print(f"\n  Loading {path}")
    d = load_hdf5(path)
    sig, ref, n = d["sig"], d["ref"], d["n"]

    auto = detect_mode(d["seq_text"], d["has_freq_array"])
    mode = sys.argv[2] if len(sys.argv) > 2 else pick_mode(auto)
    if mode not in MODES: mode = auto
    print(f"  mode = {mode}   (auto-detected {auto})   n = {n} points")

    # x-axis for the chosen mode
    if mode == "odmr":
        x, xlabel = d["freq_axis"], d["freq_label"]
        x_plot = x / 1e9 if x.max() > 1e6 else x        # GHz for display
    elif mode == "rabi_amp":
        x, xlabel = build_amp_axis(d["seq_text"], n); x_plot = x
    else:                                               # rabi_time / ramsey / echo
        x, xlabel = build_time_axis(d["seq_text"], n); x_plot = x

    contrast = np.where(ref > 0, sig / ref, np.nan)
    ref_cv = 100 * np.nanstd(ref) / np.nanmean(ref) if np.nanmean(ref) > 0 else float("nan")
    if d["reps"] and d["reps"] > 0:
        sig_disp, ref_disp, ylab = sig / d["reps"], ref / d["reps"], "counts / rep"
    else:
        sig_disp, ref_disp, ylab = sig, ref, "raw counts"
    print(f"  reference CV = {ref_cv:.1f}%   reps = {int(d['reps']) if d['reps'] else '?'}")

    # fit
    res = None
    v = np.isfinite(contrast)
    try:
        if mode == "rabi_time":  res = fit_rabi(x[v], contrast[v], amplitude=False)
        elif mode == "rabi_amp": res = fit_rabi(x[v], contrast[v], amplitude=True)
        elif mode == "ramsey":   res = fit_ramsey(x[v], contrast[v])
        elif mode == "echo":     res = fit_echo(x[v], contrast[v])
        elif mode == "odmr":     res = fit_odmr(x[v], contrast[v])
    except Exception as e:
        print(f"  fit failed: {e}")
    if res:
        print("\n" + res["summary"])

    # plot
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(os.path.basename(path), fontsize=10)
    ax_r.plot(x_plot, sig_disp, "o", color="steelblue", label="Signal")
    ax_r.plot(x_plot, ref_disp, "s", color="tomato", alpha=0.8, label="Reference")
    ax_r.set_xlabel(xlabel); ax_r.set_ylabel(ylab)
    ax_r.set_title(f"Raw counts  (ref CV = {ref_cv:.1f}%)")
    ax_r.legend(fontsize=9); ax_r.grid(True, alpha=0.3)

    ax_c.plot(x_plot, contrast, "o", color="darkorange", label="Signal / Reference")
    if res:
        xf_plot = res["xf"] / 1e9 if (mode == "odmr" and res["xf"].max() > 1e6) else res["xf"]
        ax_c.plot(xf_plot, res["yf"], "--k", lw=1.5, label="Fit")
        ax_c.set_title(res["title"])
    else:
        ax_c.set_title(MODE_LABEL[mode])
    ax_c.set_xlabel(xlabel); ax_c.set_ylabel("Signal / Reference")
    ax_c.legend(fontsize=9); ax_c.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.splitext(path)[0] + f"_{mode}_plot.png"
    try:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"\n  saved -> {out_png}")
    except Exception as e:
        print(f"  (could not save PNG: {e})")
    plt.show()


if __name__ == "__main__":
    main()
