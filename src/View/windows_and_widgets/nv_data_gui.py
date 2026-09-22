"""
Written by <Jannet Trabelsi>
nv_data_gui.py — a Tkinter + matplotlib GUI for visualising, analysing and
fitting our lab's HDF5 data (pulsed ODMR / Rabi / Ramsey / echo, CW-ODMR
frequency sweeps, confocal raster scans, and camera image stacks).
=============================================================================

What it does
------------
* Open any HDF5 saved by our struct_hdf5.py and browse the whole struct as a
  tree (groups, struct-arrays, datasets, and scalar attributes).
* Auto-detects the four data types we have and picks a sensible first plot:
    - signal_counts / reference_counts   -> pulsed experiment (1D)
    - counts + frequencies               -> CW ODMR sweep (1D)
    - count_img                          -> confocal image (2D)
    - image_1.. / camera_image / 3-D arr -> image stack (movie)
* SELECT PART OF THE DATA and act only on it:
    - 1D: drag a span on the plot -> zoom / fit / stats / export just that range
    - 2D: drag a rectangle ROI    -> crop / row+column profile / stats
* Fit 1D data (full trace OR the selected span) with a library of models
  (Lorentzian & Gaussian peaks/dips, damped cosine/Rabi, Ramsey, stretched-exp
  echo, exponential, polynomials...), with parameter uncertainties and R^2.
* A dedicated "Pulsed" tab that calls our fitters in plot_pulsed_hdf5.py and
  the false-alarm-probability test in significance_test.py when those files are
  importable, and otherwise uses built-in equivalents.
* Image tab: colormaps, percentile contrast clip, ROI crop/profiles, and a movie
  player (frame slider + play/pause + mean/max/sum projections) for stacks.

Needs: numpy, scipy, matplotlib, h5py, tkinter
plot_pulsed_hdf5.py and significance_test.py are optional; keep them next to
this file (or on PYTHONPATH) to enable the specialised pulsed workflow.

Design note: everything numerical lives in plain module-level functions so it
can be imported and unit-tested without a display. All GUI / h5py / pyplot
imports are deferred into the functions that need them.
"""
from __future__ import annotations

import os
import re
import numpy as np
from scipy.optimize import curve_fit

APP_TITLE = "NV / ODMR Data Explorer"


# =============================================================================
# HDF5 -> nested-tree loader (self-contained; mirrors struct_hdf5 conventions)
# =============================================================================
def load_hdf5_tree(path):
    """Read an HDF5 file into a nested python structure.

    Mapping (matches our struct_hdf5.py save format):
      * group attributes            -> scalar/str leaves
      * datasets                    -> numpy arrays (or scalars)
      * subgroup whose keys are all digits -> list (a MATLAB-style struct array)
      * any other subgroup          -> nested dict

    Returns a dict.  h5py is imported lazily so this module stays importable
    without it.
    """
    import h5py

    def read_group(g):
        node = {}
        # scalar attributes first
        for k, v in g.attrs.items():
            node[k] = _clean_scalar(v)
        # then child datasets / groups
        for name, obj in g.items():
            if isinstance(obj, h5py.Dataset):
                node[name] = _clean_dataset(obj[()])
            elif isinstance(obj, h5py.Group):
                keys = list(obj.keys())
                if keys and all(k.isdigit() for k in keys):
                    node[name] = [read_group(obj[k])
                                  for k in sorted(keys, key=int)]
                else:
                    node[name] = read_group(obj)
        return node

    with h5py.File(path, "r", libver="latest", swmr=False) as f:
        try:
            return read_group(f)
        except OSError:
            # SWMR files written with swmr=True occasionally need the flag
            pass
    with h5py.File(path, "r", libver="latest", swmr=True) as f:
        return read_group(f)


def _clean_scalar(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "ignore")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(-1)[0].item()
    return v


def _clean_dataset(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "ignore")
    arr = np.asarray(v)
    if arr.dtype.kind in ("S", "O", "U"):
        # array of bytes/strings -> joined text (sequence_text etc.)
        flat = arr.ravel()
        parts = [x.decode("utf-8", "ignore") if isinstance(x, (bytes, np.bytes_))
                 else str(x) for x in flat]
        return "\n".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
    if arr.ndim == 0:
        return arr.item()
    return arr


# =============================================================================
# Tree helpers: flatten, classify, find text (for auto-detection)
# =============================================================================
def flatten_tree(node, prefix=""):
    """Yield (path, value) for every leaf. Lists index as name[i]."""
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{prefix}/{k}" if prefix else str(k)
            yield from flatten_tree(v, p)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            p = f"{prefix}[{i}]"
            yield from flatten_tree(item, p)
    else:
        yield prefix, node


def classify_value(v):
    """Return one of: 'scalar', 'text', '1d', 'image', 'stack', 'array', 'group'."""
    if isinstance(v, (dict, list)):
        return "group"
    if isinstance(v, str):
        return "text"
    if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
        return "scalar"
    if isinstance(v, np.ndarray):
        a = np.squeeze(v)
        if a.ndim == 0 or a.size == 1:
            return "scalar"
        if not np.issubdtype(a.dtype, np.number):
            return "array"
        if a.ndim == 1:
            return "1d"
        if a.ndim == 2 and a.shape[0] > 1 and a.shape[1] > 1:
            return "image"
        if a.ndim == 3 and a.shape[-1] not in (3, 4) and min(a.shape[1:]) > 1:
            return "stack"
        if a.ndim == 3 and a.shape[-1] in (3, 4):
            return "image"      # RGB(A)
        return "array"
    return "scalar"


def describe_value(v):
    """Short human description for the info box / tree label."""
    kind = classify_value(v)
    if kind == "group":
        n = len(v)
        return f"[{'list' if isinstance(v, list) else 'group'}: {n} item{'s' if n != 1 else ''}]"
    if kind == "text":
        s = v.replace("\n", " ")
        return f'"{s[:40]}{"…" if len(s) > 40 else ""}"'
    if kind == "scalar":
        try:
            return f"{float(v):.6g}"
        except (TypeError, ValueError):
            return str(v)
    if isinstance(v, np.ndarray):
        return f"{kind} {tuple(np.squeeze(v).shape)} {v.dtype}"
    return str(type(v).__name__)


def find_text(tree, *needles):
    """First string leaf whose *value* looks like it contains any needle."""
    for _, v in flatten_tree(tree):
        if isinstance(v, str):
            low = v.lower()
            if any(n in low for n in needles):
                return v
    return None


def get_by_name(tree, *name_options, want=("1d",)):
    """First leaf whose PATH contains any of name_options and whose kind is in
    `want`. Case-insensitive. Returns (path, value) or (None, None)."""
    for path, v in flatten_tree(tree):
        low = path.lower()
        if any(opt in low for opt in name_options) and classify_value(v) in want:
            return path, v
    return None, None


# =============================================================================
# Image-stack detection (handles the several ways stacks can be stored)
# =============================================================================
_NUMBERED = re.compile(r"^(.*?)(\d+)$")


def find_image_stacks(tree):
    """Return a list of dicts {name, frames (N,H,W ndarray), source} describing
    every image stack we can assemble, across the layouts our files use:
      1) a single 3-D array,
      2) >=3 sibling 2-D arrays that share a name stem + number (image_1..N),
      3) a list of dicts that each hold a common 2-D image field.
    """
    stacks = []

    # 1) 3-D arrays anywhere
    for path, v in flatten_tree(tree):
        if isinstance(v, np.ndarray):
            a = np.squeeze(v)
            if a.ndim == 3 and a.shape[-1] not in (3, 4) and min(a.shape[1:]) > 1:
                stacks.append({"name": f"{path} (3-D, {a.shape[0]} frames)",
                               "frames": a, "source": path})

    # 2) sibling numbered 2-D arrays inside each dict level
    def _stack_cohort(items_kv):
        """items_kv: list of (key, 2-D array). Stack the largest set that shares
        one shape (>=3). Returns (frames, n_used, n_total) or None."""
        from collections import Counter
        shape_counts = Counter(im.shape for _, im in items_kv)
        best_shape, cnt = shape_counts.most_common(1)[0]
        if cnt < 3:
            return None
        cohort = [im for _, im in items_kv if im.shape == best_shape]
        return np.stack(cohort, axis=0), len(cohort), len(items_kv)

    def _stack_name(p, n_used, n_total):
        if n_used == n_total:
            return f"{p} ({n_used} frames)"
        return f"{p} ({n_used} frames, {n_total - n_used} odd-sized skipped)"

    def scan_dicts(node, prefix=""):
        if isinstance(node, dict):
            # 2) sibling numbered 2-D arrays directly under this dict
            groups = {}
            for k, v in node.items():
                if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
                    m = _NUMBERED.match(k)
                    stem = m.group(1) if m else k
                    groups.setdefault(stem, []).append((k, np.squeeze(v)))
            for stem, items in groups.items():
                if len(items) >= 3:
                    def keyfn(kv):
                        m = _NUMBERED.match(kv[0])
                        return int(m.group(2)) if m else 0
                    items = sorted(items, key=keyfn)
                    res = _stack_cohort(items)
                    if res is not None:
                        frames, nu, nt = res
                        p = f"{prefix}/{stem}*" if prefix else f"{stem}*"
                        stacks.append({"name": _stack_name(p, nu, nt),
                                       "frames": frames, "source": p})

            # 4) sibling sub-GROUPS that each hold a common 2-D field, e.g.
            #    image_1/camera_image, image_2/camera_image, ...  (each frame is
            #    its own group with metadata alongside the picture).
            subgroups = [(k, v) for k, v in node.items() if isinstance(v, dict)]
            if len(subgroups) >= 3:
                field_frames = {}          # field name -> [(order, key, 2-D array)]
                for k, sub in subgroups:
                    m = _NUMBERED.match(k)
                    order = int(m.group(2)) if m else 0
                    for fk, fv in sub.items():
                        if isinstance(fv, np.ndarray) and np.squeeze(fv).ndim == 2:
                            field_frames.setdefault(fk, []).append(
                                (order, k, np.squeeze(fv)))
                for field, items in field_frames.items():
                    if len(items) >= 3:
                        items = sorted(items, key=lambda t: t[0])
                        res = _stack_cohort([(kk, im) for _, kk, im in items])
                        if res is not None:
                            frames, nu, nt = res
                            stems = {(_NUMBERED.match(kk).group(1)
                                      if _NUMBERED.match(kk) else kk)
                                     for _, kk, _ in items}
                            stem = stems.pop() if len(stems) == 1 else "group_"
                            p = (f"{prefix}/{stem}*/{field}" if prefix
                                 else f"{stem}*/{field}")
                            stacks.append({"name": _stack_name(p, nu, nt),
                                           "frames": frames, "source": p})

            for k, v in node.items():
                scan_dicts(v, f"{prefix}/{k}" if prefix else k)
        elif isinstance(node, list):
            # 3) list of dicts sharing a 2-D field
            fields = {}
            for item in node:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
                            fields.setdefault(k, []).append((k, np.squeeze(v)))
            for field, imgs in fields.items():
                if len(imgs) >= 3:
                    res = _stack_cohort(imgs)
                    if res is not None:
                        frames, nu, nt = res
                        p = f"{prefix}[*].{field}"
                        stacks.append({"name": _stack_name(p, nu, nt),
                                       "frames": frames, "source": p})
            for i, item in enumerate(node):
                scan_dicts(item, f"{prefix}[{i}]")

    scan_dicts(tree)
    # de-duplicate by source
    seen, out = set(), []
    for s in stacks:
        if s["source"] not in seen:
            seen.add(s["source"])
            out.append(s)
    return out


# =============================================================================
# Camera scan with light-source modalities: sibling image_N groups, each holding
# a 'camera_image' (laser) and/or 'white_light_camera_image' (white light).
# =============================================================================
_CAM_MODALITIES = [("laser", "camera_image"),
                   ("white_light", "white_light_camera_image")]
_CAM_LABEL = {"laser": "laser", "white_light": "white light"}


def find_camera_scan(tree):
    """Detect a camera scan stored as sibling image_N groups, each holding a
    'camera_image' (laser) and/or a 'white_light_camera_image' (white light).
    Returns {'modalities': {label: frames (N,H,W)}, 'present': [labels...]} or
    None if the tree does not match this layout."""
    if not isinstance(tree, dict):
        return None
    subgroups = [(k, v) for k, v in tree.items()
                 if isinstance(v, dict) and _NUMBERED.match(k)]
    if not subgroups:
        return None
    subgroups.sort(key=lambda kv: int(_NUMBERED.match(kv[0]).group(2)))

    def _point_meta(sub):
        # every field on the point that isn't a 2-D image (coords, timestamps…)
        return {k: v for k, v in sub.items()
                if not (isinstance(v, np.ndarray) and np.squeeze(v).ndim >= 2)}

    out, meta = {}, {}
    for label, attr in _CAM_MODALITIES:
        pairs = []
        for _, sub in subgroups:
            v = sub.get(attr)
            if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
                pairs.append((np.squeeze(v), _point_meta(sub)))
        if pairs:
            from collections import Counter
            shp = Counter(f.shape for f, _ in pairs).most_common(1)[0][0]
            cohort = [(f, m) for f, m in pairs if f.shape == shp]
            if cohort:
                out[label] = np.stack([f for f, _ in cohort], axis=0)
                meta[label] = [m for _, m in cohort]
    if not out:
        return None
    present = [lab for lab, _ in _CAM_MODALITIES if lab in out]
    return {"modalities": out, "meta": meta, "present": present}


def format_point_meta(d):
    """One-line-ish readable summary of a scan point's coordinates + metadata."""
    if not d:
        return ""
    order = ["micro_x", "micro_y", "micro_z", "nano_z", "timestamp"]
    keys = [k for k in order if k in d] + [k for k in d if k not in order]
    parts = []
    for k in keys:
        v = d[k]
        if isinstance(v, np.ndarray):
            v = np.squeeze(v)
            v = v.item() if v.size == 1 else v
        parts.append(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}")
    return "     ".join(parts)


# =============================================================================
# Averaged vs non-averaged pulsed data + Poisson / empirical error statistics
# =============================================================================
def find_field(tree, name):
    """First leaf whose final path component equals `name` (any type)."""
    for path, v in flatten_tree(tree):
        if path.split("/")[-1] == name:
            return v
    return None


def find_averaging(tree):
    """Detect averaged pulsed data that also stored the individual runs.
    New files save signal_counts/reference_counts/total_counts = the AVERAGE and
    signal_1..signal_N / reference_1.. / total_1.. = the individual acquisitions.
    Returns {'navg', 'runs': {ch: [arrays]}, 'avg': {ch: array}} or None (old
    single-acquisition files, treated as averaging = 1)."""
    flat = {}
    for path, v in flatten_tree(tree):
        if isinstance(v, np.ndarray):
            flat[path.split("/")[-1]] = np.squeeze(v).astype(float)
    channels = ("signal", "reference", "total")
    runs = {}
    for ch in channels:
        idx = [(int(m.group(1)), nm) for nm in flat
               if (m := re.fullmatch(rf"{ch}_(\d+)", nm))]
        idx.sort()
        if idx:
            runs[ch] = [flat[nm] for _, nm in idx]
    if "signal" not in runs:
        return None
    avg = {}
    for ch in channels:
        a = flat.get(f"{ch}_counts", flat.get(f"{ch}_avg"))
        if a is not None:
            avg[ch] = a
    return {"navg": len(runs["signal"]), "runs": runs, "avg": avg}


def poisson_error(y):
    """Shot-noise (Poisson) 1-sigma error for a counts array: sqrt(N)."""
    return np.sqrt(np.abs(np.asarray(y, float)))


def stack_runs(runs):
    """List of 1-D run arrays -> (navg, n) array trimmed to the common length."""
    n = min(len(r) for r in runs)
    return np.array([np.asarray(r, float)[:n] for r in runs])


def error_on_mean(runs):
    """Per-point empirical standard error of the mean across runs."""
    a = stack_runs(runs)
    navg = a.shape[0]
    if navg < 2:
        return poisson_error(a[0])
    return np.std(a, axis=0, ddof=1) / np.sqrt(navg)


def error_stats(y, runs=None):
    """Poisson stats for counts array y; if the individual `runs` are supplied,
    also the empirical error-on-mean and its ratio to the Poisson expectation
    (ratio ~1 => the data scales as Poissonian shot noise)."""
    y = np.asarray(y, float)
    yy = y[np.isfinite(y)]
    m = float(np.mean(yy)) if yy.size else float("nan")
    out = {"mean": m, "poisson_sigma": float(np.sqrt(abs(m))),
           "rel_poisson": float(np.sqrt(abs(m)) / abs(m)) if m else float("nan")}
    if runs and len(runs) >= 2:
        a = stack_runs(runs)
        navg = a.shape[0]
        emp = np.std(a, axis=0, ddof=1) / np.sqrt(navg)      # error on the mean
        mean_curve = np.mean(a, axis=0)
        pois = np.sqrt(np.abs(mean_curve) / navg)            # Poisson expectation
        good = np.isfinite(emp) & np.isfinite(pois) & (pois > 0)
        mm = float(np.median(mean_curve[np.isfinite(mean_curve)])) if mean_curve.size else float("nan")
        out.update(navg=navg,
                   emp_sigma_mean=float(np.median(emp[good])) if good.any() else float("nan"),
                   poisson_sigma_mean=float(np.median(pois[good])) if good.any() else float("nan"),
                   ratio_emp_poisson=float(np.median(emp[good] / pois[good])) if good.any() else float("nan"),
                   rel_emp_mean=float(np.median(emp[good]) / abs(mm)) if (good.any() and mm) else float("nan"))
    return out


def quantity_error(q, sig, ref, sig_err, ref_err):
    """Propagated 1-sigma error for the pulsed quantity `q` given per-point
    signal/reference values and their errors."""
    sig = np.asarray(sig, float); ref = np.asarray(ref, float)
    se = np.asarray(sig_err, float); re_ = np.asarray(ref_err, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if q == "signal":
            return se
        if q == "reference":
            return re_
        if q == "S-R":
            return np.sqrt(se ** 2 + re_ ** 2)
        if q == "(S-R)/R":
            return np.sqrt((se / ref) ** 2 + (sig * re_ / ref ** 2) ** 2)
        r = sig / ref
        return np.abs(r) * np.sqrt((se / sig) ** 2 + (re_ / ref) ** 2)


# =============================================================================
# MultiHarp timing sub-test: photon-arrival histogram + rise-time / FWHM
# =============================================================================
def find_multiharp(tree):
    """Detect a MultiHarp timing file (has hist_counts + hist_edges_ns). Returns
    {'centers','counts','meta'} where meta carries the stored stats, or None."""
    _, hc = get_by_name(tree, "hist_counts", want=("1d",))
    _, he = get_by_name(tree, "hist_edges_ns", want=("1d",))
    if hc is None or he is None:
        return None
    hc = np.squeeze(hc).astype(float)
    he = np.squeeze(he).astype(float)
    if he.size == hc.size + 1:
        centers = 0.5 * (he[:-1] + he[1:])
    elif he.size == hc.size:
        centers = he
    else:
        centers = np.arange(hc.size, dtype=float)
    meta = {}
    for key in ("rise_time_ns", "fwhm_ns", "onset_ns", "plateau_counts",
                "mean_ns", "median_ns", "std_ns", "drift_pp_ns", "window_ns",
                "hist_bin_ns", "base_resolution_ps", "start_label", "stop_label",
                "case", "timestamp"):
        v = find_field(tree, key)
        if v is not None:
            meta[key] = v
    return {"centers": centers, "counts": hc, "meta": meta}


def rise_time(centers, counts, lo_frac=0.10, hi_frac=0.90, low=0.0, high=None):
    """Leading-edge rise time between lo_frac and hi_frac of the level range.
    Auto (low=0, high=None): high = median of the top 30% of the window and the
    crossings are frac*high -- identical to the MultiHarp acquisition code.
    Manual: pass low = avg of a baseline region and high = avg of a plateau
    region to measure the rise between two user-picked levels. Crossings are
    linearly interpolated on the first rising edge."""
    t = np.asarray(centers, float)
    c = np.asarray(counts, float)
    n = c.size
    nan = float("nan")
    if n < 3 or not np.isfinite(c).any() or np.nanmax(c) <= 0:
        return {"t_lo": nan, "t_hi": nan, "rise": nan, "low": low, "high": high,
                "lo_frac": lo_frac, "hi_frac": hi_frac}
    if high is None:
        high = float(np.median(c[int(0.7 * n):])) if n >= 4 else float(np.nanmax(c))
        if high <= 0:
            high = float(np.nanmax(c))

    def cross(frac):
        thr = low + frac * (high - low)
        for i in range(1, n):
            if c[i - 1] < thr <= c[i]:
                c0, c1 = c[i - 1], c[i]
                t0, t1 = t[i - 1], t[i]
                return t1 if c1 == c0 else t0 + (thr - c0) * (t1 - t0) / (c1 - c0)
        return nan

    t_lo, t_hi = cross(lo_frac), cross(hi_frac)
    rise = (t_hi - t_lo) if (np.isfinite(t_lo) and np.isfinite(t_hi)) else nan
    return {"t_lo": t_lo, "t_hi": t_hi, "rise": rise, "low": float(low),
            "high": float(high), "lo_frac": lo_frac, "hi_frac": hi_frac}


def fwhm_from_hist(centers, counts, baseline=None, peak=None):
    """Full width at half maximum. Auto: half = max/2 (matches the acquisition
    code). Manual: pass baseline + peak levels and the half level is taken
    midway between them."""
    t = np.asarray(centers, float)
    c = np.asarray(counts, float)
    nan = float("nan")
    if c.size == 0 or np.nanmax(c) <= 0:
        return {"fwhm": nan, "t0": nan, "t1": nan, "half": nan}
    base = 0.0 if baseline is None else float(baseline)
    pk = float(np.nanmax(c)) if peak is None else float(peak)
    half = base + 0.5 * (pk - base)
    above = np.where(c >= half)[0]
    if above.size < 2:
        return {"fwhm": nan, "t0": nan, "t1": nan, "half": half}
    return {"fwhm": float(t[above[-1]] - t[above[0]]),
            "t0": float(t[above[0]]), "t1": float(t[above[-1]]), "half": half}


# =============================================================================
# Confocal raster-scan geometry: physical x/y axes (microns), z-slice
# coordinates and orientation, from the nanodrive scan metadata.
# =============================================================================
def find_node(tree, *names):
    """Value stored at the first key matching any of `names`, whether it is a
    leaf, a list (numeric-keyed group, e.g. point_a = [x, y]), or a dict.
    Unlike find_field this does NOT require the value to be a leaf, so 2-element
    positions saved as groups are still found."""
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in names:
                    return v
                r = walk(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for it in node:
                r = walk(it)
                if r is not None:
                    return r
        return None
    return walk(tree)


def _as_num_vec(v):
    """Best-effort numeric 1-D vector from a value that may be an array, a list,
    a dict ({'value': ...} or {'x':.., 'y':..}), or a string like '[55, 90]'."""
    if v is None:
        return None
    if isinstance(v, dict):
        for key in ("value", "val", "data"):
            if key in v:
                return _as_num_vec(v[key])
        if "x" in v and "y" in v:
            try:
                return np.array([float(v["x"]), float(v["y"])], float)
            except Exception:
                return None
        return None
    if isinstance(v, (list, tuple)):
        try:
            return np.array([float(z) for z in v], float)
        except Exception:
            return None
    if isinstance(v, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        return np.array([float(x) for x in nums], float) if nums else None
    try:
        a = np.atleast_1d(np.squeeze(np.asarray(v, float)))
        return a if a.size else None
    except Exception:
        return None


def _pair(v, idx):
    """Component idx of a small position array/list; the scalar if size 1."""
    a = _as_num_vec(v)
    if a is None or a.size == 0:
        return None
    return float(a[0]) if a.size == 1 else float(a[min(idx, a.size - 1)])


def find_confocal(tree):
    """Detect a confocal raster scan (has count_img) and collect the scan
    geometry the nanodrive stored: per-axis position vectors (x_pos / y_pos /
    z_pos), scan corners (point_a / point_b and, if a sub-region was picked,
    min/max points), the step (resolution), and z-stack limits (z_min / z_max /
    z_step). Returns a dict of raw pieces or None."""
    _, img = get_by_name(tree, "count_img", "count_image", "raw_img",
                         want=("image", "stack"))
    if img is None:
        return None

    def vec(*names):
        return _as_num_vec(find_node(tree, *names))

    def arr(*names):
        v = vec(*names)
        return v if (v is not None and v.size >= 1) else None

    def sca(*names):
        v = vec(*names)
        return float(v.flat[0]) if (v is not None and v.size) else None

    def boo(*names):
        v = find_node(tree, *names)
        if isinstance(v, dict):
            v = v.get("value", v.get("val"))
        if v is None:
            return None
        try:
            return bool(np.squeeze(np.asarray(v)))
        except Exception:
            return bool(v)

    return {
        "img": np.squeeze(np.asarray(img, float)),
        "x_pos": arr("x_pos", "x_positions", "xpos"),
        "y_pos": arr("y_pos", "y_positions", "ypos"),
        "z_pos": arr("z_pos", "z_positions", "zpos"),
        "point_a": vec("point_a", "point_A", "pointa", "point_1"),
        "point_b": vec("point_b", "point_B", "pointb", "point_2"),
        "min_pt": vec("min_point", "min_pos", "min_pt", "minimum_point"),
        "max_pt": vec("max_point", "max_pos", "max_pt", "maximum_point"),
        "resolution": sca("resolution"), "num_datapoints": sca("num_datapoints"),
        "z_min": sca("z_min"), "z_max": sca("z_max"), "z_step": sca("z_step"),
        "sel_min": boo("Select MIN?", "Select MIN", "select_min"),
        "sel_max": boo("Select MAX?", "Select MAX", "select_max"),
        "unit": "\u00b5m",
    }


def _axis_from_corners(lo, hi, n, step=None):
    """Length-n coordinate vector lo->hi; else walk from lo by step; else pixels."""
    if lo is not None and hi is not None and n > 1:
        return np.linspace(lo, hi, n)
    if lo is not None and step:
        return lo + np.arange(n) * step
    if step:
        return np.arange(n) * step
    return np.arange(n, dtype=float)


def confocal_axes(conf, nx, ny, source="auto"):
    """Return (x_vec[nx], y_vec[ny], xlabel, ylabel, note) for a confocal image.
    The scan corners carry the absolute offset: point_a = (min x, min y),
    point_b = (max x, max y). If 'Select MIN?' / 'Select MAX?' were on, the
    corresponding corner is taken from the selected min/max point instead.
    'auto' uses those corners first (so the axes start at the real position),
    then absolute x_pos/y_pos vectors, then the step size, then pixels."""
    unit = conf.get("unit", "\u00b5m")
    step = conf.get("resolution")
    xp, yp = conf.get("x_pos"), conf.get("y_pos")

    def match(v, n):
        return v is not None and v.size == n

    # effective corners (selected min/max override point_a / point_b) ----------
    a, b = conf.get("point_a"), conf.get("point_b")
    amin = conf.get("min_pt") if conf.get("sel_min") else None
    bmax = conf.get("max_pt") if conf.get("sel_max") else None
    lo_x = _pair(amin if amin is not None else a, 0)
    lo_y = _pair(amin if amin is not None else a, 1)
    hi_x = _pair(bmax if bmax is not None else b, 0)
    hi_y = _pair(bmax if bmax is not None else b, 1)
    have_corners = None not in (lo_x, lo_y, hi_x, hi_y)
    csrc = ("selected min / max points"
            if (amin is not None or bmax is not None) else "point A \u2192 point B")

    def from_corners():
        xv = _axis_from_corners(lo_x, hi_x, nx, step)
        yv = _axis_from_corners(lo_y, hi_y, ny, step)
        note = (f"axes from {csrc}: x {lo_x:g}\u2192{hi_x:g}, "
                f"y {lo_y:g}\u2192{hi_y:g} {unit}")
        return xv, yv, f"x ({unit})", f"y ({unit})", note

    # explicit dropdown choices ------------------------------------------------
    if source == "pixels":
        return (np.arange(nx, dtype=float), np.arange(ny, dtype=float),
                "x (pixels)", "y (pixels)", "pixel index")
    if source == "x_pos / y_pos":
        xv = xp if match(xp, nx) else (yp if match(yp, nx) else np.arange(nx, dtype=float))
        yv = yp if match(yp, ny) else (xp if match(xp, ny) else np.arange(ny, dtype=float))
        return xv, yv, f"x ({unit})", f"y ({unit})", "axes from x_pos / y_pos"
    if source in ("point A \u2192 B", "min / max points") and have_corners:
        return from_corners()

    # auto: corners first (they include the absolute offset) -------------------
    if have_corners:
        return from_corners()
    if match(xp, nx) and match(yp, ny):
        return xp, yp, f"x ({unit})", f"y ({unit})", "axes from x_pos / y_pos"
    if step:
        return (np.arange(nx) * step, np.arange(ny) * step,
                f"x ({unit})", f"y ({unit})", f"axes from step = {step:g} {unit}")
    return (np.arange(nx, dtype=float), np.arange(ny, dtype=float),
            "x (pixels)", "y (pixels)", "pixel index (no scan geometry found)")


def confocal_z(conf, nframes):
    """Length-nframes z-coordinate vector for a z-stack, from z_pos, or
    z_min + k*z_step, or linspace(z_min, z_max). None if unavailable."""
    if conf is None or nframes < 1:
        return None
    zp = conf.get("z_pos")
    if zp is not None and zp.size == nframes:
        return zp
    zmin, zmax, zstep = conf.get("z_min"), conf.get("z_max"), conf.get("z_step")
    if zmin is not None and zstep:
        return zmin + np.arange(nframes) * zstep
    if zmin is not None and zmax is not None and nframes > 1:
        return np.linspace(zmin, zmax, nframes)
    return None


# =============================================================================
# Auto-detection of the default view when a file is opened
# =============================================================================
def detect_default_view(tree):
    """Return dict describing the best first plot:
       {'mode': 'pulsed'|'cw_odmr'|'image'|'stack'|'1d'|'none', ...}."""
    _, sig = get_by_name(tree, "signal_counts", "signal", want=("1d", "image"))
    _, ref = get_by_name(tree, "reference_counts", "reference", want=("1d", "image"))
    if sig is not None and ref is not None:
        return {"mode": "pulsed"}

    fpath, freqs = get_by_name(tree, "frequencies", "freq_list", "freq_axis",
                               want=("1d",))
    cpath, counts = get_by_name(tree, "counts_averaged", "all_counts_forward",
                                "counts_forward", "counts", want=("1d",))
    if freqs is not None and counts is not None:
        return {"mode": "cw_odmr", "freq_path": fpath, "count_path": cpath}

    ipath, img = get_by_name(tree, "count_img", "count_image", "image",
                             want=("image",))
    stacks = find_image_stacks(tree)
    if stacks:
        best = max(stacks, key=lambda s: s["frames"].shape[0])
        return {"mode": "stack", "stack": best, "all_stacks": stacks}
    if img is not None:
        return {"mode": "image", "path": ipath}

    # fall back to the first 1-D array
    p, v = None, None
    for path, val in flatten_tree(tree):
        if classify_value(val) == "1d":
            p, v = path, val
            break
    if p is not None:
        return {"mode": "1d", "path": p}
    return {"mode": "none"}


# =============================================================================
# ParameterSweep coordinate picker support: a hdf5_single file is a list of
# points, each carrying its 2-D image plus the swept-axis values (OD / laser /
# frequency ...) as scalars. The default view already stacks those images (see
# find_image_stacks case 3); these helpers let the GUI offer one dropdown per
# swept axis that jumps the movie player to the matching frame.
# =============================================================================
_SWEEP_IMAGE_KEYS = ("count_img", "count_image", "image", "raw_img")
_SWEEP_BOOKKEEPING = ("index", "status")


def _sweep_point_image(point):
    """The point's 2-D image (squeezed) or None. Prefers the known image keys,
    then any 2-D array on the point, then anything 2-D under aux/."""
    if not isinstance(point, dict):
        return None
    for k in _SWEEP_IMAGE_KEYS:
        v = point.get(k)
        if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
            return np.squeeze(v)
    for v in point.values():
        if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
            return np.squeeze(v)
    aux = point.get("aux")
    if isinstance(aux, dict):
        for v in aux.values():
            if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
                return np.squeeze(v)
    return None


def sweep_frame_info(tree):
    """If `tree` is a ParameterSweep hdf5_single file, return a dict:
        {'axis_paths': [...], 'values': {axis: [sorted unique...]},
         'frame_coords': [{axis: value}, ...], 'lookup': {tuple: frame_index}}
    where frame i lines up with the image stack the movie player shows (points
    that carry a 2-D image, in order, at the modal image shape - matching
    find_image_stacks case 3). Returns None for non-sweep files."""
    if not isinstance(tree, dict):
        return None
    points = tree.get("points")
    meta = tree.get("meta")
    # signature of our writer: a points list plus a meta block
    if not isinstance(points, list) or not isinstance(meta, dict):
        return None
    if not any(k in meta for k in ("axis_paths", "n_points_total", "n_points_done")):
        return None

    cand = []  # (coords, image_shape) for points that have an image, in order
    for p in points:
        img = _sweep_point_image(p)
        if img is None:
            continue
        coords = {k: v for k, v in p.items()
                  if k not in _SWEEP_BOOKKEEPING and k != "aux" and np.isscalar(v)}
        cand.append((coords, np.squeeze(img).shape))
    if not cand:
        return None

    from collections import Counter
    shp = Counter(s for _, s in cand).most_common(1)[0][0]
    frame_coords = [c for c, s in cand if s == shp]
    if not frame_coords:
        return None

    axis_paths = []
    ap = meta.get("axis_paths")
    if isinstance(ap, str) and ap.strip():
        axis_paths = [s.strip() for s in ap.split("|") if s.strip()]
    if not axis_paths:
        common = set(frame_coords[0].keys())
        for c in frame_coords[1:]:
            common &= set(c.keys())
        axis_paths = sorted(common)
    # keep the axes that are present everywhere; drop any that never vary
    axis_paths = [ax for ax in axis_paths
                  if all(ax in c for c in frame_coords)]
    varying = [ax for ax in axis_paths
               if len({c[ax] for c in frame_coords}) > 1]
    if varying:
        axis_paths = varying
    if not axis_paths:
        return None

    values = {ax: sorted({c[ax] for c in frame_coords}) for ax in axis_paths}
    lookup = {}
    for i, c in enumerate(frame_coords):
        lookup.setdefault(tuple(c[ax] for ax in axis_paths), i)
    return {"axis_paths": axis_paths, "values": values,
            "frame_coords": frame_coords, "lookup": lookup}


def sweep_fmt(v):
    """Compact label for a coordinate value in a dropdown."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%g" % float(v)
    return str(v)


def find_odmr_channels(tree):
    """Map a display name -> array search-token for a CW ODMR sweep, for each of
    the forward / reverse / averaged channels that is present. 1-D only, so the
    per-run 2-D arrays (all_counts_forward, ...) are ignored."""
    out = {}
    spec = [("averaged", ("counts_averaged", "counts_avg")),
            ("forward", ("counts_forward", "counts_fwd")),
            ("reverse (backward)", ("counts_reverse", "counts_backward",
                                    "counts_rev"))]
    for disp, toks in spec:
        for tok in toks:
            _, v = get_by_name(tree, tok, want=("1d",))
            if v is not None:
                out[disp] = tok
                break
    return out


def odmr_run_matrix(tree, channel):
    """Per-run 2-D matrix (runs x steps) for a CW ODMR channel, or None.
    'averaged' = mean of the forward and reverse per-run matrices. Rows are runs
    (the smaller axis is assumed to be the run index)."""
    def m2d(*names):
        _, v = get_by_name(tree, *names, want=("image", "stack", "1d"))
        if v is None:
            return None
        a = np.asarray(v, float)
        if a.ndim == 1:
            return a[None, :]
        if a.ndim == 2 and a.shape[0] > a.shape[1]:
            a = a.T                      # make runs the first (smaller) axis
        return a
    fwd = m2d("all_counts_forward", "all_counts_fwd")
    rev = m2d("all_counts_reverse", "all_counts_reversed", "all_counts_rev")
    ch = (channel or "").lower()
    if "forward" in ch:
        return fwd
    if "revers" in ch or "backward" in ch:
        return rev
    if fwd is not None and rev is not None:                    # averaged
        k = min(fwd.shape[0], rev.shape[0]); w = min(fwd.shape[1], rev.shape[1])
        return (fwd[:k, :w] + rev[:k, :w]) / 2.0
    return fwd if fwd is not None else rev


def combine_odmr_runs(specs, channel):
    """specs = [(tree, [run numbers, 1-based]), ...].  Returns
    (frequencies, averaged_trace, n_runs_used, labels), averaging the chosen runs
    (matched by step index) across all files.  Raises ValueError on trouble."""
    traces, labels, nsteps, freqs = [], [], None, None
    for idx, (tree, runs) in enumerate(specs):
        mat = odmr_run_matrix(tree, channel)
        if mat is None:
            raise ValueError(f"file #{idx + 1}: no per-run '{channel}' data "
                             f"(needs all_counts_forward / all_counts_reverse).")
        if freqs is None:
            _, fr = get_by_name(tree, "frequencies", "freq_list", "freq_axis",
                                want=("1d",))
            if fr is not None:
                freqs = np.squeeze(fr).astype(float)
        for r in runs:
            j = int(r) - 1                                     # 1-based -> 0-based
            if not (0 <= j < mat.shape[0]):
                raise ValueError(f"file #{idx + 1}: run {r} is out of range "
                                 f"(that file has {mat.shape[0]} runs).")
            row = mat[j].astype(float)
            nsteps = row.size if nsteps is None else min(nsteps, row.size)
            traces.append(row); labels.append(f"f{idx + 1}:r{r}")
    if not traces:
        raise ValueError("no runs selected.")
    traces = [t[:nsteps] for t in traces]
    avg = np.mean(np.vstack(traces), axis=0)
    freqs = (freqs[:nsteps] if freqs is not None else np.arange(nsteps, dtype=float))
    return freqs, avg, len(traces), labels


# =============================================================================
# x-axis reconstruction for pulsed sequences (adapted from plot_pulsed_hdf5.py)
# =============================================================================
def _num(token, default_unit="ns"):
    m = re.search(r"([0-9]*\.?[0-9]+)\s*(ns|us|µs|μs|ms|v|mv|ghz|mhz|khz|hz)?",
                  str(token), re.IGNORECASE)
    if not m:
        return float("nan"), default_unit
    return float(m.group(1)), (m.group(2) or default_unit).lower()


def _variable_line(seq_text):
    if not seq_text:
        return None
    m = re.search(
        r"(?<!#)\bvariable\s+(\w+)\s*,\s*start\s*=\s*([^,]+),\s*stop\s*=\s*([^,]+)",
        seq_text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower(), m.group(2), m.group(3)


def sweep_kind(seq_text):
    """Classify what the pulsed sequence actually sweeps: 'time' | 'amplitude'
    | 'frequency' | None.  The explicit experiment type wins — a pulsed-ODMR
    sweeps frequency even though its pulse template still declares a (fixed)
    duration variable.  Otherwise fall back to the variable name + its unit."""
    t = (seq_text or "").lower()
    m = re.search(r"type\s*=\s*(\w+)", t)
    typ = m.group(1) if m else ""
    if typ in ("odmr", "podmr", "pulsed_odmr", "cwodmr"):
        return "frequency"
    vl = _variable_line(seq_text)
    if not vl:
        return None
    name = vl[0].lower()
    _, unit = _num(vl[1], "")            # unit on the start token, if any
    # order matters: phase / amplitude / frequency keywords are checked before
    # time, because "pulse_amplitude"/"pulse_duration" both contain "pulse".
    if "phase" in name:
        return "phase"
    if any(k in name for k in ("amp", "power", "volt")) or unit in ("v", "mv"):
        return "amplitude"
    if any(k in name for k in ("freq", "detun")) or unit in ("ghz", "mhz", "khz", "hz"):
        return "frequency"
    if (any(k in name for k in ("dur", "tau", "time", "width", "delay",
                                "wait", "free", "gate"))
            or unit in ("ns", "us", "µs", "μs", "ms")):
        return "time"
    return "time"                        # sensible default for a pulsed scan


def build_phase_axis(seq_text, n):
    """Phase sweep reconstructed from the sequence text (radians)."""
    vl = _variable_line(seq_text)
    if vl and "phase" in vl[0].lower():
        _, s_tok, e_tok = vl
        s, _ = _num(s_tok, "")
        e, _ = _num(e_tok, "")
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s, e, n), "phase (rad)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_time_axis(seq_text, n):
    vl = _variable_line(seq_text)
    if vl and sweep_kind(seq_text) == "time":
        _, s_tok, e_tok = vl
        s, su = _num(s_tok, "ns")
        e, eu = _num(e_tok, "ns")
        sc = {"ns": 1.0, "us": 1e3, "µs": 1e3, "μs": 1e3, "ms": 1e6}
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s * sc.get(su, 1), e * sc.get(eu, 1), n), "tau (ns)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_amp_axis(seq_text, n):
    vl = _variable_line(seq_text)
    if vl and sweep_kind(seq_text) == "amplitude":
        _, s_tok, e_tok = vl
        s, su = _num(s_tok, "v")
        e, eu = _num(e_tok, "v")
        sc = {"v": 1.0, "mv": 1e-3}
        if np.isfinite(s) and np.isfinite(e):
            return np.linspace(s * sc.get(su, 1), e * sc.get(eu, 1), n), "pulse amplitude (V)"
    return np.arange(n, dtype=float), "scan point (index)"


def build_freq_axis(seq_text, n):
    """Frequency sweep reconstructed from the sequence text (returns Hz).  Only
    fires when the variable line is genuinely a frequency (freq/detuning name or
    a Hz/kHz/MHz/GHz unit) — it will NOT misread a duration variable as Hz."""
    vl = _variable_line(seq_text)
    if vl:
        name = vl[0].lower()
        _, u0 = _num(vl[1], "")
        if ("freq" in name or "detun" in name
                or u0 in ("ghz", "mhz", "khz", "hz")):
            s, su = _num(vl[1], "hz")
            e, eu = _num(vl[2], "hz")
            sc = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
            if np.isfinite(s) and np.isfinite(e):
                return np.linspace(s * sc.get(su, 1), e * sc.get(eu, 1), n), "frequency (Hz)"
    return np.arange(n, dtype=float), "scan point (index)"


def detect_pulsed_mode(seq_text, has_freq_array):
    t = (seq_text or "").lower()
    typ = None
    m = re.search(r"type\s*=\s*(\w+)", t)
    if m:
        typ = m.group(1)
    vl = _variable_line(seq_text)
    varname = vl[0] if vl else ""
    if "phase" in varname:               # phase sweep of any experiment type
        return "phase"
    if typ in ("ramsey",):
        return "ramsey"
    if typ in ("echo", "hahn", "hahn_echo", "spinecho"):
        return "echo"
    if typ in ("odmr", "pulsed_odmr", "podmr") or has_freq_array:
        return "odmr"
    if "amplitude" in varname:
        return "rabi_amp"
    if "duration" in varname or "width" in varname or typ == "rabi":
        return "rabi_time"
    return "rabi_time"


# =============================================================================
# Pulse-sequence preview: parse the sequence_text into pulses and evaluate their
# timing at a chosen scan point, so the GUI can draw a channel-vs-time diagram.
# This mirrors the sequence-text format used by our SequenceTextParser, but is
# fully self-contained (it does not import our src/ engine).
# =============================================================================
_SEQ_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
              "sqrt": np.sqrt, "abs": abs, "pi": np.pi, "e": np.e}
_TIME_UNIT_S = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ms": 1e-3, "s": 1.0}


def parse_sequence_text(text):
    """Parse a sequence_text block into {name, type, duration_ns, sample_rate,
    variables:[{name,start,stop,unit,steps}], pulses:[{name,channel,at,shape,
    width,amp}]}.  Time/width/amp are kept as raw expression strings and only
    evaluated later (they may depend on the swept variable)."""
    if not text or not isinstance(text, str):
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    info = {"name": None, "type": None, "duration_ns": None,
            "sample_rate": None, "variables": [], "pulses": []}
    for ln in lines:
        low = ln.lower()
        if low.startswith("sequence:"):
            for key in ("name", "type"):
                m = re.search(key + r"\s*=\s*([^,]+)", ln, re.I)
                if m:
                    info[key] = m.group(1).strip()
            m = re.search(r"duration\s*=\s*([0-9.]+)\s*(ns|us|µs|μs|ms|s)?", ln, re.I)
            if m:
                u = (m.group(2) or "ns").lower()
                info["duration_ns"] = float(m.group(1)) * _TIME_UNIT_S.get(u, 1e-9) * 1e9
            m = re.search(r"sample_rate\s*=\s*([^,]+)", ln, re.I)
            if m:
                info["sample_rate"] = m.group(1).strip()
        elif low.startswith("variable"):
            m = re.search(r"variable\s+(\w+)\s*,\s*start\s*=\s*([^,]+),\s*"
                          r"stop\s*=\s*([^,]+?)(?:,\s*steps\s*=\s*(\d+))?\s*$", ln, re.I)
            if m:
                s, su = _num(m.group(2), "")
                e, eu = _num(m.group(3), "")
                info["variables"].append(
                    {"name": m.group(1).lower(), "start": s, "stop": e,
                     "unit": (su or eu or ""), "steps": int(m.group(4) or 1)})
        elif "pulse" in low and "channel" in low:
            m = re.match(r"(?P<name>\S+)\s+pulse\s+on\s+channel\s+(?P<ch>\d+)"
                         r"\s+at\s+(?P<rest>.+)$", ln, re.I)
            if m:
                parts = [p.strip() for p in m.group("rest").split(",")]
                if len(parts) >= 4:
                    info["pulses"].append(
                        {"name": m.group("name"), "channel": int(m.group("ch")),
                         "at": parts[0], "shape": parts[1],
                         "width": parts[2], "amp": parts[3]})
    return info


def is_sequence_text(val):
    """True if a value looks like a pulse-sequence description we can preview."""
    if not isinstance(val, str):
        return False
    low = val.lower()
    return ("pulse" in low and "channel" in low) or \
           ("sequence:" in low and "variable" in low)


def _seq_units_to_seconds(expr):
    def repl(m):
        return repr(float(m.group(1)) * _TIME_UNIT_S[m.group(2).lower()])
    return re.sub(r"([0-9]*\.?[0-9]+)\s*(ns|us|µs|μs|ms|s)\b", repl, str(expr))


def _eval_seq_expr(expr, subs):
    """Safely evaluate a sequence expression.  Bare numbers are treated as
    seconds; unit-suffixed numbers convert to seconds; names come from subs or
    the small math whitelist.  Returns a float (seconds for time expressions)."""
    e = _seq_units_to_seconds(expr)
    if "__" in e or not re.fullmatch(r"[0-9a-zA-Z_.eE+\-*/() ,]*", e or ""):
        raise ValueError(f"unsafe or unparseable expression: {expr!r}")
    ns = dict(_SEQ_FUNCS)
    ns.update(subs)
    return float(eval(e, {"__builtins__": {}}, ns))  # noqa: S307 (whitelisted)


def sequence_scan_value(parsed, scan_index):
    """Value of the (first) swept variable at a given scan-point index."""
    vs = parsed.get("variables") if parsed else None
    if not vs:
        return None, None, "", 1, 0
    v = vs[0]
    steps = max(1, int(v["steps"]))
    i = int(np.clip(scan_index, 0, steps - 1))
    if steps > 1:
        val = v["start"] + (v["stop"] - v["start"]) * i / (steps - 1)
    else:
        val = v["start"]
    return v["name"], val, v["unit"], steps, i


def sequence_pulse_geometry(parsed, scan_index=0):
    """Evaluate every pulse at the given scan point; return numeric geometry
    (start/width in ns, amplitude) plus the sorted channel list and time span."""
    if not parsed or not parsed.get("pulses"):
        return None
    var_name, var_val, var_unit, steps, idx = sequence_scan_value(parsed, scan_index)
    subs_time, subs_amp = {}, {}
    for k, v in enumerate(parsed.get("variables", [])):
        val = (v["start"] + (v["stop"] - v["start"]) * idx / (steps - 1)
               if (k == 0 and steps > 1) else v["start"])
        subs_time[v["name"]] = val * _TIME_UNIT_S.get(v["unit"], 1.0)
        subs_amp[v["name"]] = val
    pulses = []
    for p in parsed["pulses"]:
        try:
            start_ns = _eval_seq_expr(p["at"], subs_time) * 1e9
            width_ns = _eval_seq_expr(p["width"], subs_time) * 1e9
        except Exception:
            continue
        try:
            amp = _eval_seq_expr(p["amp"], subs_amp)
        except Exception:
            amp = 1.0
        pulses.append({"name": p["name"], "channel": p["channel"],
                       "start_ns": float(start_ns),
                       "width_ns": float(max(width_ns, 0.0)), "amp": float(amp)})
    channels = sorted({p["channel"] for p in pulses})
    t_max = max((p["start_ns"] + p["width_ns"] for p in pulses), default=0.0)
    if parsed.get("duration_ns"):
        t_max = max(t_max, parsed["duration_ns"])
    return {"var_name": var_name, "var_value": var_val, "var_unit": var_unit,
            "scan_index": idx, "steps": steps, "pulses": pulses,
            "channels": channels, "t_max_ns": t_max}


# =============================================================================
# Fit-model library (pure). Each model reports params, 1-sigma errors, R^2.
# =============================================================================
def _as_xy(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _osc_freq_seed(x, y):
    order = np.argsort(x)
    t, v = np.asarray(x)[order], np.asarray(y)[order]
    if len(t) < 4 or (t.max() - t.min()) == 0:
        return 1.0
    grid = np.linspace(t.min(), t.max(), max(64, 4 * len(t)))
    vi = np.interp(grid, t, v) - np.mean(v)
    spec = np.abs(np.fft.rfft(vi))
    spec[0] = 0.0
    fr = np.fft.rfftfreq(len(grid), d=grid[1] - grid[0])
    f = fr[int(np.argmax(spec))]
    return f if f > 0 else 1.0 / (t.max() - t.min() + 1e-9)


# --- model functions -------------------------------------------------------
def m_constant(x, C):
    return np.full_like(np.asarray(x, float), C)


def m_linear(x, a, b):
    return a * x + b


def m_gauss_peak(x, C, A, x0, s):
    return C + A * np.exp(-((x - x0) ** 2) / (2 * s ** 2))


def m_gauss_dip(x, C, A, x0, s):
    return C - A * np.exp(-((x - x0) ** 2) / (2 * s ** 2))


def m_gauss2_peak(x, C, A1, x1, s1, A2, x2, s2):
    return (C + A1 * np.exp(-((x - x1) ** 2) / (2 * s1 ** 2))
              + A2 * np.exp(-((x - x2) ** 2) / (2 * s2 ** 2)))


def m_lorentz_peak(x, C, A, x0, hw):
    return C + A * (hw ** 2) / ((x - x0) ** 2 + hw ** 2)


def m_lorentz_dip(x, C, A, x0, hw):
    return C - A * (hw ** 2) / ((x - x0) ** 2 + hw ** 2)


def m_lorentz2_dip(x, C, A1, x1, w1, A2, x2, w2):
    return (C - A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
              - A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2))


def m_exp_decay(x, C, A, tau):
    return C + A * np.exp(-x / tau)


def m_exp_rise(x, C, A, tau):
    return C + A * (1.0 - np.exp(-x / tau))


def m_damped_cos(x, C, A, f, phi, tau):
    return C + A * np.cos(2 * np.pi * f * x + phi) * np.exp(-x / tau)


def m_gauss_cos(x, C, A, f, phi, T):
    return C + A * np.cos(2 * np.pi * f * x + phi) * np.exp(-((x / T) ** 2))


def m_stretched(x, C, A, T, p):
    return C + A * np.exp(-((x / T) ** p))


def m_sine(x, C, A, f, phi):
    return C + A * np.sin(2 * np.pi * f * x + phi)


def m_phase_cos(x, C, A, phi0):
    # phase sweep: one cosine fringe over the swept phase (x in radians)
    return C + A * np.cos(x - phi0)


def _span(x):
    return (x.max() - x.min()) or 1.0


def _seed_constant(x, y):
    return [np.mean(y)], (None, None)


def _seed_linear(x, y):
    if len(x) > 1 and _span(x) > 0:
        a, b = np.polyfit(x, y, 1)
        return [float(a), float(b)], (None, None)
    return [0.0, float(np.mean(y))], (None, None)


def _seed_gauss(x, y, dip):
    C = float(np.median(y))
    amp = float((C - y.min()) if dip else (y.max() - C))
    x0 = float(x[np.argmin(y)] if dip else x[np.argmax(y)])
    s = _span(x) / 10
    lo = [-np.inf, 0, x.min(), 1e-9 * _span(x) + 1e-12]
    hi = [np.inf, np.inf, x.max(), _span(x)]
    return [C, max(amp, 1e-9), x0, s], (lo, hi)


def _seed_gauss2(x, y):
    C = float(np.median(y)); amp = float(y.max() - C)
    x1 = x.min() + _span(x) / 3; x2 = x.min() + 2 * _span(x) / 3
    s = _span(x) / 12
    return [C, max(amp, 1e-9), x1, s, max(amp, 1e-9), x2, s], (None, None)


def _seed_lorentz(x, y, dip):
    C = float(np.median(y))
    amp = float((C - y.min()) if dip else (y.max() - C))
    x0 = float(x[np.argmin(y)] if dip else x[np.argmax(y)])
    hw = _span(x) / 20
    lo = [-np.inf, 0, x.min(), 1e-6 * _span(x) + 1e-12]
    hi = [np.inf, np.inf, x.max(), _span(x)]
    return [C, max(amp, 1e-9), x0, hw], (lo, hi)


def _seed_lorentz2(x, y):
    C = float(np.median(y)); d = float(max(C - y.min(), 1e-9))
    x1 = float(x.min() + _span(x) / 3)          # generic two-dip guess
    x2 = float(x.min() + 2 * _span(x) / 3)       # (e.g. Zeeman-split ODMR)
    w = _span(x) / 20
    wmin = 1e-6 * _span(x) + 1e-12
    lo = [-np.inf, 0, x.min(), wmin, 0, x.min(), wmin]
    hi = [np.inf, np.inf, x.max(), _span(x), np.inf, x.max(), _span(x)]
    return [C, d, x1, w, d, x2, w], (lo, hi)


def _seed_exp(x, y, rise):
    C = float(y[-1] if not rise else y[0])
    A = float((y[0] - y[-1]) if not rise else (y[-1] - y[0]))
    tau = max(_span(x) / 3, 1e-9)
    return [C, A, tau], ([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf])


def _seed_damped_cos(x, y):
    f = _osc_freq_seed(x, y)
    A = 0.5 * (np.max(y) - np.min(y))
    return ([float(np.mean(y)), float(A), float(f), 0.0, max(_span(x), 1e-9)],
            ([-np.inf, 0, 0, -np.pi, 1e-9],
             [np.inf, np.inf, np.inf, np.pi, np.inf]))


def _seed_gauss_cos(x, y):
    f = _osc_freq_seed(x, y)
    A = 0.5 * (np.max(y) - np.min(y))
    return ([float(np.mean(y)), float(A), float(f), 0.0, max(_span(x) / 2, 1e-9)],
            ([-np.inf, 0, 0, -np.pi, 1e-9],
             [np.inf, np.inf, np.inf, np.pi, np.inf]))


def _seed_stretched(x, y):
    rising = np.mean(y[:3]) < np.mean(y[-3:])
    A = (y.min() - y.max()) if rising else (y.max() - y.min())
    return ([float(np.median(y)), float(A), max(_span(x) / 2, 1e-9), 1.5],
            ([-np.inf, -np.inf, 1e-9, 0.3], [np.inf, np.inf, np.inf, 5.0]))


def _seed_sine(x, y):
    f = _osc_freq_seed(x, y)
    A = 0.5 * (np.max(y) - np.min(y))
    return ([float(np.mean(y)), float(A), float(f), 0.0],
            ([-np.inf, 0, 0, -np.pi], [np.inf, np.inf, np.inf, np.pi]))


def _seed_phase_cos(x, y):
    C = float(np.mean(y))
    A = 0.5 * (np.max(y) - np.min(y))
    phi0 = float(np.asarray(x)[int(np.argmax(y))])   # phase where signal peaks
    return ([C, A, phi0], ([-np.inf, 0, -2 * np.pi], [np.inf, np.inf, 2 * np.pi]))


def _rep_lorentz(p, dip):
    C, A, x0, hw = p
    return [f"center x0 = {x0:.6g}", f"FWHM = {2*abs(hw):.6g} (x-units)",
            f"amplitude = {A:.4g}", f"offset C = {C:.4g}"]


def _rep_lorentz2(p):
    C, A1, x1, w1, A2, x2, w2 = p
    return [f"peak 1 center = {x1:.6g},  FWHM = {2*abs(w1):.6g}",
            f"peak 2 center = {x2:.6g},  FWHM = {2*abs(w2):.6g}",
            f"splitting = {abs(x2-x1):.6g} (x-units)"]


def _rep_gauss(p, dip):
    C, A, x0, s = p
    return [f"center x0 = {x0:.6g}", f"sigma = {abs(s):.6g}",
            f"FWHM = {2.3548*abs(s):.6g} (x-units)", f"amplitude = {A:.4g}"]


def _rep_osc(p):
    C, A, f, phi, tau = p
    out = [f"frequency f = {f:.6g} (cycles per x-unit)",
           f"period 1/f = {1/f:.6g} (x-units)" if f else "period = inf",
           f"amplitude = {A:.4g}", f"decay = {tau:.6g} (x-units)"]
    return out


def _rep_stretched(p):
    C, A, T, n = p
    return [f"decay constant T = {T:.6g} (x-units)", f"stretch n = {n:.3f}",
            f"amplitude = {A:.4g}"]


def _rep_exp(p):
    C, A, tau = p
    return [f"tau = {tau:.6g} (x-units)", f"amplitude = {A:.4g}",
            f"offset C = {C:.4g}"]


def _rep_phase_cos(p):
    C, A, phi0 = p
    phi0w = (phi0 + np.pi) % (2 * np.pi) - np.pi          # wrap to [-pi, pi]
    vis = abs(A) / abs(C) if C else float("nan")
    return [f"fringe amplitude A = {abs(A):.4g}",
            f"phase offset phi0 = {phi0w:.4g} rad  ({np.degrees(phi0w):.1f} deg)",
            f"offset C = {C:.4g}",
            f"visibility A/C = {vis:.3g}"]


MODELS = {
    # name: (func, pnames, seed_fn, report_fn)
    "Constant":           (m_constant, ["C"], _seed_constant, None),
    "Linear":             (m_linear, ["a", "b"], _seed_linear, None),
    "Gaussian peak":      (m_gauss_peak, ["C", "A", "x0", "sigma"],
                           lambda x, y: _seed_gauss(x, y, False),
                           lambda p: _rep_gauss(p, False)),
    "Gaussian dip":       (m_gauss_dip, ["C", "A", "x0", "sigma"],
                           lambda x, y: _seed_gauss(x, y, True),
                           lambda p: _rep_gauss(p, True)),
    "Double Gaussian peak": (m_gauss2_peak,
                           ["C", "A1", "x1", "s1", "A2", "x2", "s2"],
                           _seed_gauss2, None),
    "Lorentzian peak":    (m_lorentz_peak, ["C", "A", "x0", "hw"],
                           lambda x, y: _seed_lorentz(x, y, False),
                           lambda p: _rep_lorentz(p, False)),
    "Lorentzian dip (ODMR)": (m_lorentz_dip, ["C", "A", "x0", "hw"],
                           lambda x, y: _seed_lorentz(x, y, True),
                           lambda p: _rep_lorentz(p, True)),
    "Double Lorentzian dip": (m_lorentz2_dip,
                           ["C", "A1", "x1", "w1", "A2", "x2", "w2"],
                           _seed_lorentz2, _rep_lorentz2),
    "Exponential decay":  (m_exp_decay, ["C", "A", "tau"],
                           lambda x, y: _seed_exp(x, y, False), _rep_exp),
    "Exponential rise":   (m_exp_rise, ["C", "A", "tau"],
                           lambda x, y: _seed_exp(x, y, True), _rep_exp),
    "Damped cosine (Rabi)": (m_damped_cos, ["C", "A", "f", "phi", "tau"],
                           _seed_damped_cos, _rep_osc),
    "Ramsey (Gaussian×cos)": (m_gauss_cos, ["C", "A", "f", "phi", "T"],
                           _seed_gauss_cos,
                           lambda p: [f"fringe f = {p[2]:.6g}/x-unit",
                                      f"T2* = {p[4]:.6g} (x-units)"]),
    "Stretched exp (echo)": (m_stretched, ["C", "A", "T", "n"],
                           _seed_stretched, _rep_stretched),
    "Sine":               (m_sine, ["C", "A", "f", "phi"], _seed_sine, None),
    "Cosine (phase sweep)": (m_phase_cos, ["C", "A", "phi0"],
                           _seed_phase_cos, _rep_phase_cos),
    "Polynomial (deg 2)": ("poly2", None, None, None),
    "Polynomial (deg 3)": ("poly3", None, None, None),
    "Polynomial (deg 4)": ("poly4", None, None, None),
}


def r_squared(y, yhat):
    y = np.asarray(y, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_model(name, x, y, n_dense=800):
    """Fit `name` (a key in MODELS) to (x, y). Returns a dict with keys:
       ok, params, perr, pnames, r2, x_dense, y_dense, y_fit, summary, error."""
    x, y = _as_xy(x, y)
    if len(x) < 2:
        return {"ok": False, "error": "not enough finite points to fit"}
    spec = MODELS.get(name)
    if spec is None:
        return {"ok": False, "error": f"unknown model {name!r}"}
    func, pnames, seed_fn, report_fn = spec

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xd = np.linspace(xs.min(), xs.max(), n_dense)

    # polynomial special case
    if isinstance(func, str) and func.startswith("poly"):
        deg = int(func[-1])
        if len(xs) <= deg:
            return {"ok": False, "error": f"need > {deg} points for degree {deg}"}
        coeffs = np.polyfit(xs, ys, deg)
        yhat = np.polyval(coeffs, xs)
        pn = [f"c{deg-i}" for i in range(deg + 1)]
        rep = [f"{pn[i]} = {coeffs[i]:.6g}" for i in range(len(coeffs))]
        return {"ok": True, "params": coeffs, "perr": np.full(len(coeffs), np.nan),
                "pnames": pn, "r2": r_squared(ys, yhat),
                "x_dense": xd, "y_dense": np.polyval(coeffs, xd),
                "y_fit": np.polyval(coeffs, x), "report": rep,
                "summary": _format_summary(name, pn, coeffs,
                                           np.full(len(coeffs), np.nan),
                                           r_squared(ys, yhat), rep)}

    p0, bounds = seed_fn(xs, ys)
    kw = dict(p0=p0, maxfev=60000)
    if bounds and bounds[0] is not None:
        kw["bounds"] = bounds
    try:
        popt, pcov = curve_fit(func, xs, ys, **kw)
    except Exception as e:
        return {"ok": False, "error": f"fit did not converge: {e}"}
    perr = np.sqrt(np.abs(np.diag(pcov))) if pcov is not None else \
        np.full(len(popt), np.nan)
    yhat = func(xs, *popt)
    rep = report_fn(popt) if report_fn else None
    return {"ok": True, "params": popt, "perr": perr, "pnames": pnames,
            "r2": r_squared(ys, yhat), "x_dense": xd, "y_dense": func(xd, *popt),
            "y_fit": func(x, *popt), "report": rep,
            "summary": _format_summary(name, pnames, popt, perr,
                                       r_squared(ys, yhat), rep)}


def _format_summary(name, pnames, popt, perr, r2, report):
    lines = [f"Model: {name}", f"R\u00b2 = {r2:.5f}", "", "parameters:"]
    for nm, val, err in zip(pnames, popt, perr):
        if np.isfinite(err):
            lines.append(f"  {nm:>5} = {val:.6g}  \u00b1 {err:.3g}")
        else:
            lines.append(f"  {nm:>5} = {val:.6g}")
    if report:
        lines += ["", "derived:"] + [f"  {r}" for r in report]
    return "\n".join(lines)


def suggest_models(mode):
    """Order the model list so the most likely one for `mode` is first."""
    order = list(MODELS.keys())
    pref = {
        "cw_odmr": ["Lorentzian dip (ODMR)", "Double Lorentzian dip"],
        "odmr":    ["Lorentzian dip (ODMR)", "Double Lorentzian dip"],
        "rabi_time": ["Damped cosine (Rabi)"],
        "rabi_amp":  ["Damped cosine (Rabi)"],
        "ramsey":  ["Ramsey (Gaussian×cos)"],
        "echo":    ["Stretched exp (echo)"],
    }.get(mode, [])
    for p in reversed(pref):
        if p in order:
            order.remove(p)
            order.insert(0, p)
    return order


# =============================================================================
# Selection / region analysis (pure)
# =============================================================================
def span_mask(x, x0, x1):
    x = np.asarray(x, float)
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    return (x >= lo) & (x <= hi)


def region_stats_1d(x, y, x0=None, x1=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x0 is not None and x1 is not None:
        m = span_mask(x, x0, x1)
        x, y = x[m], y[m]
    good = np.isfinite(y)
    x, y = x[good], y[good]
    if len(y) == 0:
        return {"n": 0}
    order = np.argsort(x)
    area = float(np.trapezoid(y[order], x[order])) if len(y) > 1 else 0.0
    return {"n": int(len(y)), "x_min": float(x.min()), "x_max": float(x.max()),
            "mean": float(np.mean(y)), "std": float(np.std(y)),
            "min": float(np.min(y)), "max": float(np.max(y)),
            "sum": float(np.sum(y)), "area": area,
            "argmax_x": float(x[np.argmax(y)]), "argmin_x": float(x[np.argmin(y)])}


def format_stats_1d(s):
    if s.get("n", 0) == 0:
        return "selection is empty"
    return ("\n".join([
        f"points            {s['n']}",
        f"x range           {s['x_min']:.6g} … {s['x_max']:.6g}",
        f"mean ± std        {s['mean']:.6g} ± {s['std']:.4g}",
        f"min / max         {s['min']:.6g} / {s['max']:.6g}",
        f"x at max / min    {s['argmax_x']:.6g} / {s['argmin_x']:.6g}",
        f"sum               {s['sum']:.6g}",
        f"integral (trapz)  {s['area']:.6g}",
    ]))


def image_crop(img, x0, x1, y0, y1):
    """Crop a 2-D image by pixel-coordinate box (floats accepted). Returns
    (sub, (c0, c1, r0, r1)) with integer bounds actually used."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    c0, c1 = sorted((int(round(x0)), int(round(x1))))
    r0, r1 = sorted((int(round(y0)), int(round(y1))))
    c0 = max(0, min(c0, w - 1)); c1 = max(c0 + 1, min(c1, w))
    r0 = max(0, min(r0, h - 1)); r1 = max(r0 + 1, min(r1, h))
    return img[r0:r1, c0:c1], (c0, c1, r0, r1)


def image_profiles(img, box=None):
    """Return (x_idx, col_profile, y_idx, row_profile) averaged over the box
    (or the whole image). col_profile is mean over rows -> vs x; row_profile is
    mean over cols -> vs y."""
    img = np.asarray(img, float)
    if box is not None:
        sub, (c0, c1, r0, r1) = image_crop(img, *box)
    else:
        sub, (c0, c1, r0, r1) = img, (0, img.shape[1], 0, img.shape[0])
    col = np.nanmean(sub, axis=0)          # vs x (columns)
    row = np.nanmean(sub, axis=1)          # vs y (rows)
    return (np.arange(c0, c1), col, np.arange(r0, r1), row)


def region_stats_2d(img, box=None):
    img = np.asarray(img, float)
    if box is not None:
        sub, bounds = image_crop(img, *box)
    else:
        sub, bounds = img, (0, img.shape[1], 0, img.shape[0])
    c0, c1, r0, r1 = bounds
    v = sub[np.isfinite(sub)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "cols": f"{c0}:{c1}", "rows": f"{r0}:{r1}",
            "mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "sum": float(np.sum(v))}


def format_stats_2d(s):
    if s.get("n", 0) == 0:
        return "ROI is empty"
    return "\n".join([
        f"pixels        {s['n']}   (rows {s['rows']}, cols {s['cols']})",
        f"mean ± std    {s['mean']:.6g} ± {s['std']:.4g}",
        f"min / max     {s['min']:.6g} / {s['max']:.6g}",
        f"sum           {s['sum']:.6g}",
    ])


def stack_projection(frames, how="mean"):
    frames = np.asarray(frames, float)
    if how == "mean":
        return np.nanmean(frames, axis=0)
    if how == "max":
        return np.nanmax(frames, axis=0)
    if how == "sum":
        return np.nansum(frames, axis=0)
    return frames[0]


# =============================================================================
# Significance test (uses our significance_test.py if importable; else built-in)
# =============================================================================
def run_significance(mode, x, y, n_perm=None, seed=0):
    """Return a dict describing an 'is it real?' test for the given mode.
    Oscillation modes (rabi/ramsey) -> Lomb-Scargle permutation FAP.
    Shape modes (odmr dip / echo decay) -> shape-fit R^2 permutation FAP."""
    x, y = _as_xy(x, y)
    if len(x) < 6:
        return {"ok": False, "error": "need at least ~6 points for a permutation test"}
    rng = np.random.default_rng(seed)

    # try the user's module first (identical method, their tuning)
    user = None
    try:
        import significance_test as st  # noqa
        user = st
    except Exception:
        user = None

    osc = mode in ("rabi_time", "rabi_amp", "ramsey", "phase")
    if osc:
        nperm = n_perm or 2000
        use_user = user is not None and mode != "phase"   # no phase grid upstream
        if use_user:
            f, ang, unit, fnyq = user._freq_grid(mode, x)
            pg, obs, fap, thr = user.periodogram_fap(x, y, ang, n_perm=nperm, rng=rng)
            fpk = f[int(np.argmax(pg))]
        else:
            f, ang, unit, fnyq = _freq_grid(mode, x)
            pg, obs, fap, thr = _periodogram_fap(x, y, ang, n_perm=nperm, rng=rng)
            fpk = f[int(np.argmax(pg))]
        return {"ok": True, "kind": "periodogram", "unit": unit, "peak": float(fpk),
                "power": float(obs), "thr": float(thr), "fap": float(fap),
                "f": f, "pg": pg, "verdict": _verdict(fap), "nyquist": float(fnyq),
                "n_perm": nperm}
    else:
        nperm = n_perm or 1000
        if user is not None:
            fit_fn = user._fit_lorentz if mode == "odmr" else user._fit_stretched
            r2, fap, thr, p = user.fit_r2_fap(x, y, fit_fn, n_perm=nperm, rng=rng)
        else:
            fit_fn = _fit_lorentz if mode == "odmr" else _fit_stretched
            r2, fap, thr, p = _fit_r2_fap(x, y, fit_fn, n_perm=nperm, rng=rng)
        return {"ok": True, "kind": "shapefit", "r2": float(r2), "thr": float(thr),
                "fap": float(fap), "verdict": _verdict(fap), "params": p,
                "n_perm": nperm}


def _verdict(fap):
    if fap < 0.01:
        return "REAL  (p < 1%)"
    if fap < 0.05:
        return f"MARGINAL  (p = {fap*100:.1f}%) — average more before trusting it"
    return f"NOT significant  (p = {fap*100:.0f}%) — consistent with noise"


def _freq_grid(mode, x, npts=1400):
    dx = np.median(np.diff(np.sort(x)))
    if mode == "phase":                          # x is phase in radians
        f_nyq = 0.5 / dx
        f = np.linspace(0.02, min(2.0, 0.95 * f_nyq), npts)
        return f, 2 * np.pi * f, "cyc/rad", f_nyq
    if mode == "rabi_amp":
        f_nyq = 0.5 / dx
        f = np.linspace(0.1, min(10.0, 0.95 * f_nyq), npts)
        return f, 2 * np.pi * f, "cyc/V", f_nyq
    f_nyq = 1000.0 / (2 * dx)
    f = np.linspace(0.5, min(50.0, 0.95 * f_nyq), npts)
    return f, 2 * np.pi * f / 1000.0, "MHz", f_nyq


def _periodogram_fap(x, y, ang, n_perm=2000, rng=None):
    from scipy.signal import lombscargle
    rng = rng or np.random.default_rng(0)
    y = y - np.mean(y)
    pg = lombscargle(x, y, ang, normalize=True)
    obs = pg.max()
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = lombscargle(x, rng.permutation(y), ang, normalize=True).max()
    fap = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return pg, obs, fap, np.percentile(null, 95)


def _fit_lorentz(x_hz, y):
    fM = x_hz / 1e6 if np.nanmax(np.abs(x_hz)) > 1e5 else x_hz
    C0 = np.median(y); d0 = C0 - np.min(y); f0 = fM[int(np.argmin(y))]
    span = fM.max() - fM.min()
    p, _ = curve_fit(m_lorentz_dip, fM, y,
                     p0=[C0, max(d0, 1e-6), f0, max(span / 20, 1.0)],
                     bounds=([-2, 0, fM.min(), 0.1], [2, 2, fM.max(), span]),
                     maxfev=40000)
    return m_lorentz_dip(fM, *p), p


def _fit_stretched(x_ns, y):
    xu = x_ns / 1e3 if np.nanmax(np.abs(x_ns)) > 100 else x_ns
    rising = np.mean(y[:3]) < np.mean(y[-3:])
    A0 = (np.min(y) - np.max(y)) if rising else (np.max(y) - np.min(y))
    p, _ = curve_fit(m_stretched, xu, y,
                     p0=[float(np.median(y)), A0,
                         0.4 * (xu.max() - xu.min()) + 1e-3, 1.5],
                     bounds=([-2, -2, 1e-4, 0.5], [2, 2, 1e6, 4.0]), maxfev=40000)
    return m_stretched(xu, *p), p


def _fit_r2_fap(x, y, fit_fn, n_perm=1000, rng=None):
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

def gaussian_2d(coords, bg, amp, x0, y0, sx, sy, theta):
    """Elliptical 2-D Gaussian evaluated on flattened meshgrid (X, Y)."""
    X, Y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr = (X - x0) * ct + (Y - y0) * st
    yr = -(X - x0) * st + (Y - y0) * ct
    return (bg + amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))).ravel()




def fit_gaussian_2d(Z, x=None, y=None, circular=True):
    """Fit a 2-D Gaussian to image Z over physical axes x (columns) / y (rows).
    circular=True -> one sigma (round 'spherical' spot); False -> elliptical.
    Returns center (x0,y0), sigma(s), FWHM(s), amplitude, background, R^2."""
    Z = np.asarray(Z, float)
    ny, nx = Z.shape
    x = np.arange(nx, dtype=float) if x is None else np.asarray(x, float)
    y = np.arange(ny, dtype=float) if y is None else np.asarray(y, float)
    X, Y = np.meshgrid(x, y)
    good = np.isfinite(Z)
    if good.sum() < 6:
        return {"ok": False, "error": "not enough finite pixels"}
    Zf = np.where(good, Z, np.nanmedian(Z[good]))
    bg0 = float(np.nanpercentile(Z[good], 10))
    amp0 = float(np.nanmax(Z[good])) - bg0
    dip = amp0 < 0
    iy, ix = np.unravel_index(np.nanargmax(Zf if not dip else -Zf), Z.shape)
    x0, y0 = float(x[ix]), float(y[iy])
    sx0 = max((x.max() - x.min()) / 8.0, 1e-9)
    sy0 = max((y.max() - y.min()) / 8.0, 1e-9)
    if dip:
        amp0 = float(np.nanmin(Z[good])) - bg0
    Zr = Zf.ravel()
    try:
        if circular:
            def f(c, bg, amp, x0, y0, s):
                return gaussian_2d(c, bg, amp, x0, y0, s, s, 0.0)
            popt, _ = curve_fit(f, (X, Y), Zr,
                                p0=[bg0, amp0, x0, y0, (sx0 + sy0) / 2], maxfev=20000)
            bg, amp, x0, y0, s = popt
            sx = sy = abs(s); theta = 0.0
        else:
            popt, _ = curve_fit(gaussian_2d, (X, Y), Zr,
                                p0=[bg0, amp0, x0, y0, sx0, sy0, 0.0], maxfev=20000)
            bg, amp, x0, y0, sx, sy, theta = popt
            sx, sy = abs(sx), abs(sy)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    Zfit = gaussian_2d((X, Y), bg, amp, x0, y0, sx, sy, theta).reshape(Z.shape)
    ss_res = np.nansum((Z[good] - Zfit[good]) ** 2)
    ss_tot = np.nansum((Z[good] - np.nanmean(Z[good])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    k = 2.3548200450309493  # FWHM = 2*sqrt(2 ln2) * sigma
    return {"ok": True, "bg": bg, "amp": amp, "x0": x0, "y0": y0,
            "sigma_x": sx, "sigma_y": sy, "sigma": (sx + sy) / 2, "theta": theta,
            "fwhm_x": k * sx, "fwhm_y": k * sy, "fwhm": k * (sx + sy) / 2,
            "r2": r2, "circular": circular}


def guess_background_1d(x, y):
    """Constant background = average of the off-feature points (the peak/dip
    minority, far from the robust baseline, is excluded)."""
    y = np.asarray(y, float)
    good = np.isfinite(y)
    if good.sum() < 3:
        return float(np.nanmean(y)) if good.any() else 0.0
    yv = y[good]
    med = np.median(yv)
    mad = np.median(np.abs(yv - med)) + 1e-12
    keep = np.abs(yv - med) <= 3.0 * mad
    return float(np.mean(yv[keep])) if keep.any() else float(med)




def guess_background_2d(Z):
    """Constant background = average of the non-bright pixels (the NV spot's
    bright/dark outliers are excluded)."""
    v = np.asarray(Z, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float(np.mean(v)) if v.size else 0.0
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    keep = np.abs(v - med) <= 3.0 * mad
    return float(np.mean(v[keep])) if keep.any() else float(med)




def eval_monomial(text, x):
    """Evaluate a background string as a function of x (numpy namespace). A bare
    number -> constant. Returns an array/scalar, or None if blank/invalid."""
    text = (text or "").strip()
    if not text:
        return None
    ns = {"x": np.asarray(x, float), "np": np, "pi": np.pi, "sqrt": np.sqrt,
          "exp": np.exp, "log": np.log, "abs": np.abs}
    try:
        return np.asarray(eval(text, {"__builtins__": {}}, ns), float)
    except Exception:
        return None


# =============================================================================
# ============================  GUI LAYER  ====================================
# (All tkinter / matplotlib imports live here so the module stays importable.)
# =============================================================================
def _moving_average(y, w):
    """Moving average over w points; edges handled by 'nearest'."""
    y = np.asarray(y, float)
    if w is None or w < 2 or y.size < 2:
        return y
    try:
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(y, int(w), mode="nearest")
    except Exception:
        k = int(w); pad = k // 2
        yp = np.pad(y, pad, mode="edge")
        return np.convolve(yp, np.ones(k) / k, mode="same")[pad:pad + y.size]


def parse_regions(text):
    """Parse 'a b, c d' (also a-b, a:b, a to b) into [(lo, hi), ...] in x-units."""
    out = []
    for chunk in re.split(r"[,;]", text or ""):
        c = chunk.strip()
        if not c:
            continue
        c = c.replace("to", " ").replace(":", " ").replace("..", " ")
        c = re.sub(r"(?<=[0-9])\s*-\s*(?=[0-9])", " ", c)   # 2.85-2.89 -> 2.85 2.89
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", c)
        if len(nums) >= 2:
            a, b = float(nums[0]), float(nums[1])
            out.append((min(a, b), max(a, b)))
    return out


def background_1d(x, y, regions, mode="outside"):
    """Constant background from selected x-regions. mode 'outside' -> average the
    points OUTSIDE the regions (regions mark the feature/resonance); 'inside' ->
    average the points INSIDE the regions (regions are the baseline). Falls back
    to the auto off-feature guess if nothing usable."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if regions:
        m = np.zeros(x.shape, bool)
        for lo, hi in regions:
            m |= (x >= lo) & (x <= hi)
        sel = (~m) if mode == "outside" else m
        pts = y[sel & np.isfinite(y)]
        if pts.size >= 1:
            return float(np.mean(pts))
    return guess_background_1d(x, y)


_CAM_IMG_RE = re.compile(r"^image_0*\d")


def find_camera_sweep(tree):
    """Detect an ODMR camera sweep: per-frequency image groups image_* each with
    camera_image (2-D) + frequency_ghz/hz. Returns {'freqs' (GHz), 'images'
    (list of 2-D), 'n'} sorted by frequency; falls back to a 3-D image_cube_mean
    + frequencies; else None."""
    imgs = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if (_CAM_IMG_RE.match(str(k)) and isinstance(v, dict)
                        and "camera_image" in v):
                    ci = v.get("camera_image")
                    if isinstance(ci, np.ndarray) and np.squeeze(ci).ndim == 2:
                        fg, fh, ix = (v.get("frequency_ghz"), v.get("frequency_hz"),
                                      v.get("index"))
                        if fg is not None:
                            freq = float(np.squeeze(fg))
                        elif fh is not None:
                            freq = float(np.squeeze(fh)) / 1e9
                        else:
                            freq = None
                        oi = int(np.squeeze(ix)) if ix is not None else None
                        imgs.append((oi, freq, np.squeeze(ci).astype(float)))
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for it in node:
                walk(it)
    walk(tree)
    if imgs:
        imgs.sort(key=lambda t: (t[1] if t[1] is not None
                                 else (t[0] if t[0] is not None else 0)))
        freqs = np.array([(t[1] if t[1] is not None else (t[0] or 0))
                          for t in imgs], float)
        return {"freqs": freqs, "images": [t[2] for t in imgs], "n": len(imgs)}
    _, cube = get_by_name(tree, "image_cube_mean", "image_cube", want=("stack",))
    if cube is not None:
        cube = np.asarray(cube, float)
        _, fr = get_by_name(tree, "frequencies", "freq_list", want=("1d",))
        if fr is not None:
            fr = np.squeeze(fr).astype(float)
            freqs = fr / 1e9 if np.nanmax(fr) > 1e6 else fr
        else:
            freqs = np.arange(cube.shape[0], dtype=float)
        m = cube.shape[0]
        return {"freqs": freqs[:m], "images": [cube[i] for i in range(m)], "n": m}
    return None


def subtract_row_poly(img, degree):
    """Fit a polynomial of `degree` to each row and subtract it -- removes the
    smooth per-row (systematic) background while keeping localized features such
    as the NV spot. degree 0 = constant, 1 = linear, 2 = quadratic, ..."""
    img = np.asarray(img, float)
    if img.ndim != 2:
        return img
    H, W = img.shape
    x = np.arange(W, dtype=float)
    out = np.array(img, float)
    deg = max(0, int(degree))
    for r in range(H):
        y = img[r]
        good = np.isfinite(y)
        if int(good.sum()) > deg + 1:
            try:
                out[r] = y - np.polyval(np.polyfit(x[good], y[good], deg), x)
            except Exception:
                pass
    return out


def fft_lowpass(img, cutoff_frac=0.3):
    """2-D FFT low-pass: keep spatial frequencies within cutoff_frac (0-1) of the
    spectrum radius and zero the rest, to suppress white noise."""
    img = np.asarray(img, float)
    if img.ndim != 2:
        return img
    fill = float(np.nanmedian(img[np.isfinite(img)])) if np.isfinite(img).any() else 0.0
    F = np.fft.fftshift(np.fft.fft2(np.nan_to_num(img, nan=fill)))
    H, W = img.shape
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt(((Y - H / 2.0) / (H / 2.0)) ** 2 + ((X - W / 2.0) / (W / 2.0)) ** 2)
    F[r > float(cutoff_frac)] = 0.0
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def _odmr_model(x, coeffs, dips):
    """Polynomial background (coeffs, highest power first) minus Lorentzian dips.
    dips = flat [A1, x1, w1, A2, x2, w2, ...] (A > 0 = dip depth, w = HWHM)."""
    y = np.polyval(np.asarray(coeffs, float), x)
    d = list(dips)
    for i in range(0, len(d) - 2, 3):
        A, x0, w = d[i], d[i + 1], d[i + 2]
        y = y - A / (1.0 + ((x - x0) / (w if w != 0 else 1e-9)) ** 2)
    return y


def fit_odmr_dips(x, y, n_dips=1, bg_degree=0, centers=None):
    """Fit a polynomial background (degree bg_degree) minus n_dips Lorentzians.
    `centers` are initial dip-position guesses (list, x-units); if omitted they
    are auto-detected as the deepest excursions. Returns x_dense/y_dense/centers/
    widths/depths/r2/summary (keys compatible with the plot overlay)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    if x.size < 4:
        return {"ok": False, "error": "too few points"}
    order = np.argsort(x); x, y = x[order], y[order]
    span = float(x.max() - x.min()) or 1.0
    base0 = np.polyfit(x, y, max(0, int(bg_degree)))
    resid = np.polyval(base0, x) - y                 # positive at dips
    if not centers:
        centers, rr = [], resid.copy()
        for _ in range(n_dips):
            i = int(np.argmax(rr))
            centers.append(float(x[i]))
            rr[np.abs(x - x[i]) < span / 25.0] = -np.inf
    centers = [float(c) for c in centers][:n_dips]
    while len(centers) < n_dips:
        centers.append(float(x.min() + span * (len(centers) + 1) / (n_dips + 1)))
    depth0 = max(float(np.nanmax(resid)), 1e-9)
    w0 = span / 60.0
    nb = max(0, int(bg_degree)) + 1
    p0 = list(base0)
    for c in centers:
        p0 += [depth0, c, w0]

    def model(xx, *p):
        return _odmr_model(xx, p[:nb], list(p[nb:]))

    try:
        popt, _ = curve_fit(model, x, y, p0=p0, maxfev=40000)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    coeffs, dips = popt[:nb], list(popt[nb:])
    xd = np.linspace(x.min(), x.max(), 800)
    yhat = model(x, *popt)
    ss_res = float(np.nansum((y - yhat) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    cent = [float(dips[i + 1]) for i in range(0, len(dips), 3)]
    wid = [abs(float(dips[i + 2])) for i in range(0, len(dips), 3)]
    dep = [float(dips[i]) for i in range(0, len(dips), 3)]
    summary = (f"ODMR fit: {n_dips} Lorentzian dip(s) + poly bg deg {bg_degree}\n"
               + "\n".join(f"  dip {k+1}: center = {cent[k]:.6g}, "
                            f"FWHM = {2*wid[k]:.4g}, depth = {dep[k]:.4g}"
                            for k in range(len(cent)))
               + f"\n  R\u00b2 = {r2:.4f}")
    return {"ok": True, "x_dense": xd, "y_dense": _odmr_model(xd, coeffs, dips),
            "centers": cent, "widths": wid, "depths": dep, "r2": r2,
            "summary": summary}


def main(argv=None):
    import sys
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    from matplotlib.widgets import SpanSelector, RectangleSelector

    COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis",
                 "gray", "hot", "jet", "turbo", "twilight", "coolwarm"]

    def _make_span(ax, cb):
        try:
            return SpanSelector(ax, cb, "horizontal", useblit=True,
                                interactive=True, drag_from_anywhere=True,
                                props=dict(alpha=0.2, facecolor="tab:orange"))
        except TypeError:
            return SpanSelector(ax, cb, "horizontal", useblit=True,
                                rectprops=dict(alpha=0.2, facecolor="tab:orange"))

    def _make_rect(ax, cb):
        try:
            return RectangleSelector(ax, cb, useblit=True, button=[1],
                                     interactive=True, minspanx=2, minspany=2,
                                     spancoords="pixels",
                                     props=dict(edgecolor="white", fill=False, lw=1.3))
        except TypeError:
            return RectangleSelector(ax, cb, useblit=True, button=[1],
                                     interactive=True, minspanx=2, minspany=2,
                                     spancoords="pixels",
                                     rectprops=dict(edgecolor="white", fill=False, lw=1.3))

    # ---------------------------------------------------------------- app ----
    class App:
        def __init__(self, root):
            self.root = root
            root.title(APP_TITLE)
            root.geometry("1320x820")
            root.minsize(1060, 640)

            self.tree_data = None          # nested dict from load_hdf5_tree
            self.filepath = None
            self.item_values = {}          # treeview item id -> value
            self.stacks = []               # detected image stacks
            self.camera_scan = None        # {modalities, present} for laser/white-light scans
            self.averaging = None          # {navg, runs, avg} for averaged pulsed data
            self.multiharp = None          # {centers, counts, meta} for MultiHarp files

            # current plot state
            self.view_kind = None          # '1d' | 'image' | 'stack' | 'sequence' | None
            self.x = None
            self.y = None
            self.curves = None             # list of (x, y, label) for overlays; None = single
            self.xlabel = ""
            self.ylabel = ""
            self.cur_image = None          # 2-D array currently shown
            self.cur_frames = None         # (N,H,W) for stacks
            self.frame_idx = 0
            self._sweep = None             # sweep_frame_info() for the loaded file, or None
            self._sweep_combos = {}        # axis path -> ttk.Combobox
            self._sweep_busy = False       # guard while syncing dropdowns to the frame
            self.playing = False
            self.seq_parsed = None         # parsed sequence_text for the preview
            self.seq_index = 0             # current scan point in the preview
            self.sel_x0 = self.sel_x1 = None       # 1-D span (orange)
            self.sel_box = None                    # 2-D ROI (x0,x1,y0,y1)
            self.cur_yerr = None                   # per-point error for the primary curve
            self.extra_spans = []                  # [(x0,x1,color,alpha)] highlights
            self.extra_vlines = []                 # [(x,color)] marker lines
            self.mh = {}                           # manual multiharp regions/levels
            self.confocal = None                   # find_confocal(): scan geometry
            self._is_confocal = False              # current image is a confocal scan
            self.frame_z = None                    # per-frame z (µm) for z-stacks
            self._disp_data = None                 # oriented 2-D array now shown
            self._img_gauss = None  # last 2-D Gaussian fit on the image
            self.camera_sweep = None               # find_camera_sweep(): per-freq images
            self._sweep_fits = None                # per-frame 2-D Gaussian params
            self._sweep_freqs = None
            self._frame_coord_label = "z"          # movie frame-coordinate label
            self._frame_coord_unit = None
            self.img_x = None                      # physical x per column (shown)
            self.img_y = None                      # physical y per row (shown)
            self.img_xlabel = "x (pixels)"
            self.img_ylabel = "y (pixels)"
            self._cross_xy = None                  # crosshair position (data coords)
            self._cross_cuts = None                # {'x':(x,y),'y':(x,y),...}
            self._cross_win = None                 # (unused) reserved
            self._cross_fit_x = None               # fit result for the x line-out
            self._cross_fit_y = None               # fit result for the y line-out
            self.bg_tree = None                    # loaded background-file tree
            self.bg_on = False                     # background subtraction active
            self.bg_path = None
            self.cur_image_name = ""               # name of the shown image array
            self._rerender = None                  # re-render current view (bg toggle)
            self.odmr_channels = {}                # CW ODMR sweep channels present
            self._bg_status = ""                   # last background-subtraction note
            self.pulsed_mode = "rabi_time"
            self.pulsed_cache = None       # dict(sig, ref, seq_text, has_freq...)

            self._build_layout()
            self._new_axes()

            args = argv if argv is not None else sys.argv[1:]
            if args and os.path.exists(args[0]):
                self.open_path(args[0])

        # ----------------------------------------------------------- layout --
        def _build_layout(self):
            top = ttk.Frame(self.root, padding=(8, 6))
            top.pack(side="top", fill="x")
            ttk.Button(top, text="Open HDF5…", command=self.open_dialog).pack(side="left")
            ttk.Button(top, text="Reload", command=self.reload).pack(side="left", padx=(6, 0))
            self.file_lbl = ttk.Label(top, text="no file loaded", foreground="#555")
            self.file_lbl.pack(side="left", padx=12)
            self.status = ttk.Label(top, text="", foreground="#0a6")
            self.status.pack(side="right")

            paned = ttk.PanedWindow(self.root, orient="horizontal")
            paned.pack(side="top", fill="both", expand=True)

            # -- left: data browser -------------------------------------------
            left = ttk.Frame(paned, padding=(6, 4))
            paned.add(left, weight=0)
            ttk.Label(left, text="Data browser", font=("", 10, "bold")).pack(anchor="w")
            tree_wrap = ttk.Frame(left)
            tree_wrap.pack(fill="both", expand=True)
            self.tree = ttk.Treeview(tree_wrap, show="tree", selectmode="extended")
            ysb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=ysb.set)
            self.tree.pack(side="left", fill="both", expand=True)
            ysb.pack(side="right", fill="y")
            self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
            self.tree.bind("<Double-1>", lambda e: self.plot_selected_node())

            ttk.Label(left, text="x-axis for 1-D:").pack(anchor="w", pady=(8, 0))
            self.xaxis_combo = ttk.Combobox(left, state="readonly", width=32, values=[])
            self.xaxis_combo.pack(fill="x")
            self.xaxis_combo.bind("<<ComboboxSelected>>", lambda e: self.replot_1d())

            ttk.Button(left, text="Plot selected node",
                       command=self.plot_selected_node).pack(fill="x", pady=(6, 2))

            bgf = ttk.LabelFrame(left, text="background subtraction", padding=(6, 4))
            bgf.pack(fill="x", pady=(6, 0))
            self.bg_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(bgf, text="subtract background",
                            variable=self.bg_var, command=self._toggle_bg
                            ).pack(anchor="w")
            ttk.Button(bgf, text="choose background file\u2026",
                       command=self._change_bg).pack(fill="x", pady=(2, 0))
            self.bg_lbl = ttk.Label(bgf, text="(no background loaded)",
                                    foreground="#777", wraplength=210)
            self.bg_lbl.pack(anchor="w", pady=(2, 0))

            ttk.Label(left, text="ODMR sweep channel:").pack(anchor="w", pady=(8, 0))
            self.odmr_combo = ttk.Combobox(left, state="disabled", width=32, values=[])
            self.odmr_combo.pack(fill="x")
            self.odmr_combo.bind("<<ComboboxSelected>>",
                                 lambda e: self.on_odmr_channel())
            ttk.Button(left, text="Combine runs across files\u2026",
                       command=self.combine_odmr_dialog).pack(fill="x", pady=(2, 0))

            ttk.Label(left, text="Info", font=("", 9, "bold")).pack(anchor="w", pady=(6, 0))
            self.info = tk.Text(left, height=8, width=40, wrap="word",
                                font=("TkFixedFont", 9))
            self.info.pack(fill="x")
            self.info.configure(state="disabled")

            # -- right: figure + controls -------------------------------------
            right = ttk.Frame(paned)
            paned.add(right, weight=1)

            self.fig = Figure(figsize=(7.6, 5.4), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
            toolbar = NavigationToolbar2Tk(self.canvas, right)
            toolbar.update()

            nb = ttk.Notebook(right)
            nb.pack(side="bottom", fill="x")
            self.nb = nb
            self._build_tab_plot(nb)
            self._build_tab_fit(nb)
            self._build_tab_pulsed(nb)
            self._build_tab_image(nb)
            self._build_tab_sequence(nb)
            self._seq_tab_index = nb.index("end") - 1
            self._build_tab_multiharp(nb)
            self._mh_tab_index = nb.index("end") - 1

        def _build_tab_plot(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Plot & selection")

            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="style:").pack(side="left")
            self.style_var = tk.StringVar(value="line+markers")
            self.style_combo = ttk.Combobox(row, textvariable=self.style_var, width=13,
                                            state="readonly", values=["line", "markers", "line+markers"])
            self.style_combo.pack(side="left", padx=(2, 12))
            self.style_combo.bind("<<ComboboxSelected>>",
                                  lambda e: self.draw_1d(keep_lims=True))
            self.logy_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="log y", variable=self.logy_var,
                            command=self.replot_1d).pack(side="left")
            ttk.Label(row, text="normalise:").pack(side="left", padx=(12, 2))
            self.norm_var = tk.StringVar(value="none")
            ttk.Combobox(row, textvariable=self.norm_var, width=16, state="readonly",
                         values=["none", "divide by max",
                                 "subtract selection baseline"]).pack(side="left")
            ttk.Button(row, text="Apply", command=self.replot_1d).pack(side="left", padx=6)
            ttk.Button(row, text="Plot selected together",
                       command=self.plot_selected_together).pack(side="left", padx=(16, 0))
            self.errbar_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="error bars", variable=self.errbar_var,
                            command=self.replot_1d).pack(side="left", padx=(16, 0))

            rowe = ttk.Frame(f); rowe.pack(fill="x", pady=(4, 0))
            ttk.Label(rowe, text="error:").pack(side="left")
            self.err_lbl = ttk.Label(rowe, text="(select a counts array to see its "
                                     "shot-noise error)", foreground="#068")
            self.err_lbl.pack(side="left", padx=6)

            row2 = ttk.Frame(f); row2.pack(fill="x", pady=(8, 0))
            ttk.Label(row2, text="Selection:").pack(side="left")
            self.sel_lbl = ttk.Label(row2, text="(drag on the plot to select a range)",
                                     foreground="#a60")
            self.sel_lbl.pack(side="left", padx=6)

            row3 = ttk.Frame(f); row3.pack(fill="x", pady=(6, 0))
            for txt, cmd in [("Zoom to selection", self.zoom_selection),
                             ("Selection stats", self.selection_stats),
                             ("Export selection CSV", self.export_selection),
                             ("Clear selection", self.clear_selection),
                             ("Reset view", self.reset_view)]:
                ttk.Button(row3, text=txt, command=cmd).pack(side="left", padx=(0, 6))

        def _build_tab_fit(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Fit")
            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="model:").pack(side="left")
            self.model_combo = ttk.Combobox(row, state="readonly", width=26,
                                             values=list(MODELS.keys()))
            self.model_combo.current(0)
            self.model_combo.pack(side="left", padx=(2, 12))
            self.fit_sel_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="fit selection only",
                            variable=self.fit_sel_var).pack(side="left")
            ttk.Button(row, text="Fit", command=self.do_fit).pack(side="left", padx=8)
            ttk.Button(row, text="Clear fit", command=self.clear_fit).pack(side="left")
            ttk.Button(row, text="Export fit CSV",
                       command=self.export_fit).pack(side="left", padx=(8, 0))
            srow = ttk.Frame(f); srow.pack(fill="x", pady=(2, 0))
            self.smooth_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(srow, text="smooth (moving average)",
                            variable=self.smooth_var,
                            command=self.replot_1d).pack(side="left")
            ttk.Label(srow, text="window (points):").pack(side="left", padx=(8, 0))
            self.smooth_win = ttk.Entry(srow, width=5)
            self.smooth_win.insert(0, "5")
            self.smooth_win.pack(side="left", padx=(2, 6))
            self.smooth_win.bind("<Return>", lambda e: self.replot_1d())
            ttk.Label(srow, text="(applies to the plot and the fit; ODMR & pulsed)",
                      foreground="#777").pack(side="left", padx=(8, 0))
            orow = ttk.Frame(f); orow.pack(fill="x", pady=(2, 0))
            ttk.Label(orow, text="ODMR dips:").pack(side="left")
            self.odmr_ndip = ttk.Combobox(orow, state="readonly", width=3,
                                          values=["1", "2"])
            self.odmr_ndip.set("1"); self.odmr_ndip.pack(side="left", padx=(2, 8))
            self.odmr_ndip.bind("<<ComboboxSelected>>", lambda e: self.fit_odmr())
            ttk.Label(orow, text="bg poly:").pack(side="left")
            self.odmr_bgdeg = ttk.Combobox(orow, state="readonly", width=3,
                                           values=["0", "1", "2", "3"])
            self.odmr_bgdeg.set("0"); self.odmr_bgdeg.pack(side="left", padx=(2, 8))
            self.odmr_bgdeg.bind("<<ComboboxSelected>>", lambda e: self.fit_odmr())
            ttk.Label(orow, text="center(s) [x-units]:").pack(side="left")
            self.odmr_centers = ttk.Entry(orow, width=16)
            self.odmr_centers.pack(side="left", padx=(2, 6))
            self.odmr_centers.bind("<Return>", lambda e: self.fit_odmr())
            ttk.Button(orow, text="Fit ODMR dips",
                       command=self.fit_odmr).pack(side="left", padx=(4, 0))
            ttk.Label(orow, text="(blank = auto guess; edit + Enter to guide)",
                      foreground="#777").pack(side="left", padx=(6, 0))
            self.fit_text = tk.Text(f, height=8, wrap="none", font=("TkFixedFont", 9))
            self.fit_text.pack(fill="both", expand=True, pady=(8, 0))
            self._last_fit = None

        def _build_tab_pulsed(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Pulsed (Rabi / ODMR / echo)")
            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="experiment:").pack(side="left")
            self.pmode_combo = ttk.Combobox(row, state="readonly", width=12,
                                            values=["rabi_time", "rabi_amp",
                                                    "odmr", "ramsey", "echo",
                                                    "phase"])
            self.pmode_combo.current(0)
            self.pmode_combo.pack(side="left", padx=(2, 10))
            ttk.Label(row, text="dataset:").pack(side="left")
            self.dataset_combo = ttk.Combobox(row, state="disabled", width=11,
                                              values=["averaged"])
            self.dataset_combo.set("averaged")
            self.dataset_combo.pack(side="left", padx=(2, 10))
            self.dataset_combo.bind("<<ComboboxSelected>>",
                                    lambda e: self.on_dataset_change())
            ttk.Label(row, text="quantity:").pack(side="left")
            self.pq_combo = ttk.Combobox(row, state="readonly", width=10,
                                         values=["S/R", "(S-R)/R", "signal",
                                                 "reference", "S-R"])
            self.pq_combo.current(0)
            self.pq_combo.pack(side="left", padx=(2, 10))
            self.drop_dead_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text="drop dead startup pts",
                            variable=self.drop_dead_var).pack(side="left")
            self.pulsed_sel_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="fit selection only",
                            variable=self.pulsed_sel_var).pack(side="left", padx=(10, 0))

            row2 = ttk.Frame(f); row2.pack(fill="x", pady=(6, 0))
            ttk.Button(row2, text="Analyse & fit",
                       command=self.pulsed_analyse).pack(side="left")
            ttk.Button(row2, text="Significance test",
                       command=self.pulsed_significance).pack(side="left", padx=8)
            ttk.Label(row2, text="(quantity + selection apply to both; select a "
                      "span on the Plot tab first)", foreground="#777").pack(side="left")
            self.pulsed_text = tk.Text(f, height=8, wrap="none", font=("TkFixedFont", 9))
            self.pulsed_text.pack(fill="both", expand=True, pady=(8, 0))

        def _build_tab_image(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Image / movie")
            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Label(row, text="colormap:").pack(side="left")
            self.cmap_combo = ttk.Combobox(row, state="readonly", width=10,
                                           values=COLORMAPS)
            self.cmap_combo.current(0)
            self.cmap_combo.pack(side="left", padx=(2, 12))
            self.cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())
            ttk.Label(row, text="contrast clip %:").pack(side="left")
            self.clip_var = tk.DoubleVar(value=0.0)
            self.clip_scale = ttk.Scale(row, from_=0, to=10, variable=self.clip_var,
                                        length=140, command=lambda e: self.redraw_image())
            self.clip_scale.pack(side="left", padx=(2, 12))
            self.equal_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text="equal aspect", variable=self.equal_var,
                            command=self.redraw_image).pack(side="left")
            ttk.Label(row, text="modality:").pack(side="left", padx=(12, 2))
            self.modality_combo = ttk.Combobox(row, state="disabled", width=11, values=[])
            self.modality_combo.pack(side="left")
            self.modality_combo.bind("<<ComboboxSelected>>",
                                     lambda e: self.on_modality_change())

            row_or = ttk.Frame(f); row_or.pack(fill="x", pady=(6, 0))
            ttk.Label(row_or, text="axes:").pack(side="left")
            self.coord_src = ttk.Combobox(row_or, state="readonly", width=15,
                values=["auto", "x_pos / y_pos", "min / max points",
                        "point A → B", "pixels"])
            self.coord_src.current(0)
            self.coord_src.pack(side="left", padx=(2, 10))
            self.coord_src.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())
            self.flipx_var = tk.BooleanVar(value=False)
            self.flipy_var = tk.BooleanVar(value=False)
            self.transpose_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row_or, text="flip X", variable=self.flipx_var,
                            command=self.redraw_image).pack(side="left")
            ttk.Checkbutton(row_or, text="flip Y", variable=self.flipy_var,
                            command=self.redraw_image).pack(side="left", padx=(6, 0))
            ttk.Checkbutton(row_or, text="transpose (swap X/Y)",
                            variable=self.transpose_var,
                            command=self.redraw_image).pack(side="left", padx=(6, 0))
            self.coord_lbl = ttk.Label(row_or, text="", foreground="#036")
            self.coord_lbl.pack(side="left", padx=(10, 0))

            row2 = ttk.Frame(f); row2.pack(fill="x", pady=(8, 0))
            ttk.Label(row2, text="ROI:").pack(side="left")
            self.roi_lbl = ttk.Label(row2, text="(drag a box on the image)",
                                     foreground="#a60")
            self.roi_lbl.pack(side="left", padx=6)
            for txt, cmd in [("Crop to ROI", self.crop_roi),
                             ("X profile", lambda: self.roi_profile("x")),
                             ("Y profile", lambda: self.roi_profile("y")),
                             ("ROI stats", self.roi_stats),
                             ("Fit 2-D Gaussian", self.fit_image_gaussian),
                             ("Clear ROI", self.clear_roi)]:
                ttk.Button(row2, text=txt, command=cmd).pack(side="left", padx=(6, 0))

            xrow = ttk.Frame(f); xrow.pack(fill="x", pady=(6, 0))
            self.crosshair_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(xrow, text="crosshair", variable=self.crosshair_var,
                            command=self._toggle_crosshair).pack(side="left")
            ttk.Label(xrow, text="click the image to drop the crosshair \u2014 the "
                      "counts-vs-x (top) and counts-vs-y (right) line-outs appear; "
                      "fit them below.", foreground="#777"
                      ).pack(side="left", padx=(6, 0))

            xfit = ttk.Frame(f); xfit.pack(fill="x", pady=(2, 0))
            ttk.Label(xfit, text="fit line-outs:").pack(side="left")
            self.cross_model = ttk.Combobox(xfit, state="readonly", width=22,
                                            values=list(MODELS.keys()))
            self.cross_model.set("Gaussian peak")
            self.cross_model.pack(side="left", padx=(2, 8))
            ttk.Button(xfit, text="Fit counts-vs-x",
                       command=lambda: self._fit_cut("x")).pack(side="left")
            ttk.Button(xfit, text="Fit counts-vs-y",
                       command=lambda: self._fit_cut("y")).pack(side="left", padx=6)
            ttk.Button(xfit, text="clear fits",
                       command=self._clear_cut_fits).pack(side="left")
            self.cross_fit_lbl = ttk.Label(xfit, text="", foreground="#036")
            self.cross_fit_lbl.pack(side="left", padx=(8, 0))

            irow = ttk.Frame(f); irow.pack(fill="x", pady=(2, 0))
            ttk.Label(irow, text="line-out width (px):").pack(side="left")
            self.lineout_w = ttk.Entry(irow, width=5)
            self.lineout_w.insert(0, "1")
            self.lineout_w.pack(side="left", padx=(2, 12))
            self.lineout_w.bind("<Return>", lambda e: self.redraw_image())
            self.img_bg_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(irow, text="subtract background (image)",
                            variable=self.img_bg_var,
                            command=self.redraw_image).pack(side="left")
            ttk.Label(irow, text=" from:").pack(side="left")
            self.img_bg_mode = ttk.Combobox(irow, state="readonly", width=20,
                values=["auto (exclude bright)", "all pixels",
                        "outside 2-D Gaussian", "outside ROI"])
            self.img_bg_mode.current(0)
            self.img_bg_mode.pack(side="left", padx=(2, 8))
            self.img_bg_mode.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())
            ttk.Label(irow, text="bg =").pack(side="left")
            self.img_bg_entry = ttk.Entry(irow, width=12)
            self.img_bg_entry.pack(side="left", padx=(2, 6))
            self.img_bg_entry.bind("<Return>", lambda e: self.redraw_image())
            ttk.Button(irow, text="guess", command=self._guess_img_bg).pack(side="left")

            srow2 = ttk.Frame(f); srow2.pack(fill="x", pady=(2, 0))
            self.img_smooth_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(srow2, text="smooth background (keep 2-D Gaussian sharp)",
                            variable=self.img_smooth_var,
                            command=self.redraw_image).pack(side="left")
            ttk.Label(srow2, text="sigma (px):").pack(side="left", padx=(8, 0))
            self.img_smooth_sigma = ttk.Entry(srow2, width=5)
            self.img_smooth_sigma.insert(0, "1.5")
            self.img_smooth_sigma.pack(side="left", padx=(2, 6))
            self.img_smooth_sigma.bind("<Return>", lambda e: self.redraw_image())
            ttk.Label(srow2, text="Gaussian circle radius (\u00d7FWHM):"
                      ).pack(side="left", padx=(12, 0))
            self.gauss_radius = ttk.Entry(srow2, width=5)
            self.gauss_radius.insert(0, "1.5")
            self.gauss_radius.pack(side="left", padx=(2, 6))
            self.gauss_radius.bind("<Return>", lambda e: self.redraw_image())

            prow = ttk.Frame(f); prow.pack(fill="x", pady=(2, 0))
            self.rowbg_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(prow, text="subtract per-row polynomial",
                            variable=self.rowbg_var,
                            command=self.redraw_image).pack(side="left")
            ttk.Label(prow, text="degree:").pack(side="left", padx=(4, 0))
            self.rowbg_deg = ttk.Combobox(prow, state="readonly", width=4,
                                          values=["0", "1", "2", "3", "4"])
            self.rowbg_deg.set("1")
            self.rowbg_deg.pack(side="left", padx=(2, 14))
            self.rowbg_deg.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())
            self.denoise_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(prow, text="denoise", variable=self.denoise_var,
                            command=self.redraw_image).pack(side="left")
            self.denoise_method = ttk.Combobox(prow, state="readonly", width=13,
                                               values=["average", "FFT low-pass"])
            self.denoise_method.set("average")
            self.denoise_method.pack(side="left", padx=(2, 4))
            self.denoise_method.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())
            ttk.Label(prow, text="strength:").pack(side="left")
            self.denoise_param = ttk.Entry(prow, width=5)
            self.denoise_param.insert(0, "1.5")
            self.denoise_param.pack(side="left", padx=(2, 6))
            self.denoise_param.bind("<Return>", lambda e: self.redraw_image())
            ttk.Label(prow, text="(avg: blur \u03c3 px; FFT: keep-fraction 0-1)",
                      foreground="#777").pack(side="left", padx=(4, 0))

            csrow = ttk.Frame(f); csrow.pack(fill="x", pady=(4, 0))
            ttk.Button(csrow, text="Fit 2-D Gaussian: all frames (sweep)",
                       command=self.fit_sweep).pack(side="left")
            ttk.Label(csrow, text="plot:").pack(side="left", padx=(10, 0))
            self.sweep_plot_combo = ttk.Combobox(csrow, state="readonly", width=26,
                values=["images (scroll & verify)",
                        "Gaussian integral (A\u00b72\u03c0\u03c3\u00b2)",
                        "peak pixel (max)", "sigma (px)", "center x (px)",
                        "center y (px)", "amplitude (counts)", "background (counts)",
                        "ROI mean (counts)", "fit R\u00b2"])
            self.sweep_plot_combo.set("images (scroll & verify)")
            self.sweep_plot_combo.pack(side="left", padx=(2, 6))
            self.sweep_plot_combo.bind("<<ComboboxSelected>>",
                                       lambda e: self._plot_sweep_quantity())
            ttk.Label(csrow, text="(ODMR camera sweep: draw an ROI around the NV, "
                      "then Fit all frames)", foreground="#777").pack(side="left", padx=(6, 0))

            self.movie_row = ttk.Frame(f); self.movie_row.pack(fill="x", pady=(8, 0))
            ttk.Label(self.movie_row, text="frame:").pack(side="left")
            self.frame_scale = ttk.Scale(self.movie_row, from_=0, to=0,
                                         command=self.on_frame_scale, length=260)
            self.frame_scale.pack(side="left", padx=6)
            self.frame_lbl = ttk.Label(self.movie_row, text="-")
            self.frame_lbl.pack(side="left")
            ttk.Button(self.movie_row, text="◀", width=3,
                       command=lambda: self.step_frame(-1)).pack(side="left", padx=(10, 0))
            self.play_btn = ttk.Button(self.movie_row, text="▶ Play", command=self.toggle_play)
            self.play_btn.pack(side="left", padx=4)
            ttk.Button(self.movie_row, text="▶|", width=3,
                       command=lambda: self.step_frame(1)).pack(side="left")
            ttk.Label(self.movie_row, text="fps:").pack(side="left", padx=(10, 2))
            self.fps_var = tk.IntVar(value=8)
            ttk.Spinbox(self.movie_row, from_=1, to=60, width=4,
                        textvariable=self.fps_var).pack(side="left")
            ttk.Label(self.movie_row, text="projection:").pack(side="left", padx=(10, 2))
            self.proj_combo = ttk.Combobox(self.movie_row, state="readonly", width=8,
                                           values=["frame", "mean", "max", "sum"])
            self.proj_combo.current(0)
            self.proj_combo.pack(side="left")
            self.proj_combo.bind("<<ComboboxSelected>>", lambda e: self.redraw_image())

            # ParameterSweep coordinate picker: one dropdown per swept axis (OD /
            # laser / frequency ...). Built dynamically when such a file is loaded;
            # hidden otherwise. Selecting values jumps the movie player to that point.
            self.sweep_row = ttk.Frame(f)      # packed/unpacked by _refresh_sweep_picker
            self.meta_lbl = ttk.Label(f, text="", foreground="#036",
                                      font=("TkFixedFont", 9), wraplength=900,
                                      justify="left")
            self.meta_lbl.pack(anchor="w", pady=(8, 0))

        def _build_tab_sequence(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="Sequence")
            ttk.Label(f, text="Pulse-timing preview — select the sequence_text "
                      "node in the browser to draw it here.",
                      foreground="#555").pack(anchor="w")
            self.seq_row = ttk.Frame(f); self.seq_row.pack(fill="x", pady=(8, 0))
            ttk.Label(self.seq_row, text="scan point:").pack(side="left")
            self.seq_scale = ttk.Scale(self.seq_row, from_=0, to=0,
                                       command=self.on_seq_scale, length=280)
            self.seq_scale.pack(side="left", padx=6)
            ttk.Button(self.seq_row, text="◀", width=3,
                       command=lambda: self.step_seq(-1)).pack(side="left", padx=(6, 0))
            ttk.Button(self.seq_row, text="▶|", width=3,
                       command=lambda: self.step_seq(1)).pack(side="left", padx=(2, 8))
            self.seq_overlay_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.seq_row, text="overlay all scan points",
                            variable=self.seq_overlay_var,
                            command=self.draw_sequence).pack(side="left")
            self.seq_lbl = ttk.Label(f, text="(no sequence selected)",
                                     foreground="#a60")
            self.seq_lbl.pack(anchor="w", pady=(8, 0))

        def _build_tab_multiharp(self, nb):
            f = ttk.Frame(nb, padding=8)
            nb.add(f, text="MultiHarp")
            r1 = ttk.Frame(f); r1.pack(fill="x")
            ttk.Label(r1, text="Auto rise / FWHM \u2014 levels:").pack(side="left")
            self.mh_levels = ttk.Combobox(r1, state="readonly", width=8,
                                          values=["10\u201390%", "5\u201395%"])
            self.mh_levels.current(0)
            self.mh_levels.pack(side="left", padx=(2, 10))
            ttk.Button(r1, text="Measure rise time",
                       command=self.mh_auto_rise).pack(side="left")
            ttk.Button(r1, text="Measure FWHM",
                       command=self.mh_auto_fwhm).pack(side="left", padx=6)
            ttk.Button(r1, text="Clear marks", command=self.mh_clear).pack(side="left")

            r2 = ttk.Frame(f); r2.pack(fill="x", pady=(6, 0))
            self.mh_manual_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(r2, text="manual", variable=self.mh_manual_var).pack(side="left")
            ttk.Button(r2, text="Set region 1 (low)",
                       command=lambda: self.mh_set_region(1)).pack(side="left", padx=(8, 0))
            ttk.Button(r2, text="Set region 2 (high)",
                       command=lambda: self.mh_set_region(2)).pack(side="left", padx=6)
            ttk.Button(r2, text="Rise between avgs",
                       command=self.mh_manual_rise).pack(side="left")
            ttk.Button(r2, text="FWHM between avgs",
                       command=self.mh_manual_fwhm).pack(side="left", padx=6)

            ttk.Label(f, text="blue = region 1 (low avg)   \u00b7   red = region 2 "
                      "(high avg)   \u00b7   yellow = measured span   \u00b7   orange = "
                      "the usual drag-select for zoom / fit", foreground="#777"
                      ).pack(anchor="w", pady=(6, 0))
            self.mh_text = tk.Text(f, height=9, wrap="none", font=("TkFixedFont", 9))
            self.mh_text.pack(fill="both", expand=True, pady=(6, 0))

        def show_multiharp(self):
            mh = self.multiharp
            if not mh:
                return
            self.mh = {}
            self.set_1d(mh["centers"], mh["counts"], xlabel="delay (ns)",
                        ylabel="counts", title="MultiHarp arrival-time histogram")
            self._mh_show_stored()
            try:
                self.nb.select(self._mh_tab_index)
            except Exception:
                pass

        def _mh_levels(self):
            return (0.05, 0.95) if self.mh_levels.get().startswith("5") else (0.10, 0.90)

        def _mh_show_stored(self):
            m = self.multiharp["meta"] if self.multiharp else {}

            def g(k, unit="ns", fmt="%.3f"):
                v = m.get(k)
                if v is None:
                    return "\u2014"
                try:
                    return (fmt % float(np.squeeze(v))) + (f" {unit}" if unit else "")
                except Exception:
                    return str(v)

            lbl = f"{m.get('start_label', 'start')} \u2192 {m.get('stop_label', 'stop')}"
            lines = [f"MultiHarp timing   ({lbl})",
                     "values stored in the file at acquisition time:",
                     f"  rise (10\u201390%) = {g('rise_time_ns')}",
                     f"  FWHM          = {g('fwhm_ns')}",
                     f"  onset (10%)   = {g('onset_ns')}",
                     f"  plateau       = {g('plateau_counts', 'cts', '%.0f')}",
                     f"  median = {g('median_ns')}   mean = {g('mean_ns')}   "
                     f"std = {g('std_ns')}",
                     f"  drift (p-p)   = {g('drift_pp_ns')}",
                     "",
                     "re-measure below \u2014 auto (pick levels) or manual (set two "
                     "regions, then measure between their averages)."]
            self.mh_text.delete("1.0", "end")
            self.mh_text.insert("end", "\n".join(lines))

        def _mh_redraw(self, measured=None, vlines=None):
            spans = []
            if self.mh.get("r1"):
                spans.append((self.mh["r1"][0], self.mh["r1"][1], "tab:blue", 0.20))
            if self.mh.get("r2"):
                spans.append((self.mh["r2"][0], self.mh["r2"][1], "tab:red", 0.20))
            if measured is not None:
                spans.append((measured[0], measured[1], "gold", 0.30))
            self.extra_spans = spans
            self.extra_vlines = vlines or []
            self.draw_1d()

        def mh_set_region(self, which):
            if not self.multiharp:
                messagebox.showinfo("MultiHarp", "Open a MultiHarp file first.")
                return
            if self.sel_x0 is None:
                messagebox.showinfo("MultiHarp", "Drag a span on the plot first "
                                    "(the orange selection), then click Set region.")
                return
            c, y = self.multiharp["centers"], self.multiharp["counts"]
            m = span_mask(c, self.sel_x0, self.sel_x1)
            avg = float(np.mean(y[m])) if m.any() else float("nan")
            self.mh[f"r{which}"] = (self.sel_x0, self.sel_x1)
            self.mh[f"avg{which}"] = avg
            self._mh_redraw()
            self.mh_text.insert("end", f"\nregion {which} average = {avg:.5g} cts   "
                                f"(x \u2208 [{self.sel_x0:.4g}, {self.sel_x1:.4g}] ns)")

        def _mh_report_rise(self, r, header):
            lo, hi = int(round(r["lo_frac"] * 100)), int(round(r["hi_frac"] * 100))
            self.mh_text.insert("end",
                f"\n\n{header}\n"
                f"  low level  = {r['low']:.5g} cts\n"
                f"  high level = {r['high']:.5g} cts\n"
                f"  t({lo}%) = {r['t_lo']:.4f} ns\n"
                f"  t({hi}%) = {r['t_hi']:.4f} ns\n"
                f"  RISE TIME = {r['rise']:.4f} ns")

        def _rise_marks(self, r):
            ok = np.isfinite(r["t_lo"]) and np.isfinite(r["t_hi"])
            measured = (r["t_lo"], r["t_hi"]) if ok else None
            vlines = ([(r["t_lo"], "goldenrod"), (r["t_hi"], "goldenrod")] if ok else [])
            return measured, vlines

        def mh_auto_rise(self):
            if not self.multiharp:
                messagebox.showinfo("MultiHarp", "Open a MultiHarp file first.")
                return
            lo, hi = self._mh_levels()
            r = rise_time(self.multiharp["centers"], self.multiharp["counts"], lo, hi)
            measured, vlines = self._rise_marks(r)
            self._mh_redraw(measured=measured, vlines=vlines)
            self._mh_report_rise(r, f"AUTO RISE  ({lo*100:.0f}\u2013{hi*100:.0f}% of plateau)")

        def mh_manual_rise(self):
            if not self.multiharp:
                messagebox.showinfo("MultiHarp", "Open a MultiHarp file first.")
                return
            if "avg1" not in self.mh or "avg2" not in self.mh:
                messagebox.showinfo("MultiHarp", "Set region 1 (low) and region 2 "
                                    "(high) first \u2014 drag a span, click Set region 1, "
                                    "drag again, click Set region 2.")
                return
            lo, hi = self._mh_levels()
            r = rise_time(self.multiharp["centers"], self.multiharp["counts"], lo, hi,
                          low=self.mh["avg1"], high=self.mh["avg2"])
            measured, vlines = self._rise_marks(r)
            self._mh_redraw(measured=measured, vlines=vlines)
            self._mh_report_rise(r, f"MANUAL RISE  ({lo*100:.0f}\u2013{hi*100:.0f}% "
                                    f"between region averages)")

        def mh_auto_fwhm(self):
            if not self.multiharp:
                messagebox.showinfo("MultiHarp", "Open a MultiHarp file first.")
                return
            fw = fwhm_from_hist(self.multiharp["centers"], self.multiharp["counts"])
            self._mh_mark_fwhm(fw, "AUTO FWHM  (half of max)")

        def mh_manual_fwhm(self):
            if not self.multiharp:
                messagebox.showinfo("MultiHarp", "Open a MultiHarp file first.")
                return
            if "avg1" not in self.mh or "avg2" not in self.mh:
                messagebox.showinfo("MultiHarp", "Set region 1 (baseline) and region "
                                    "2 (peak) first.")
                return
            fw = fwhm_from_hist(self.multiharp["centers"], self.multiharp["counts"],
                                baseline=self.mh["avg1"], peak=self.mh["avg2"])
            self._mh_mark_fwhm(fw, "MANUAL FWHM  (half between region averages)")

        def _mh_mark_fwhm(self, fw, header):
            ok = np.isfinite(fw["t0"]) and np.isfinite(fw["t1"])
            measured = (fw["t0"], fw["t1"]) if ok else None
            vlines = ([(fw["t0"], "goldenrod"), (fw["t1"], "goldenrod")] if ok else [])
            self._mh_redraw(measured=measured, vlines=vlines)
            self.mh_text.insert("end",
                f"\n\n{header}\n"
                f"  half level = {fw['half']:.5g} cts\n"
                f"  left  = {fw['t0']:.4f} ns\n"
                f"  right = {fw['t1']:.4f} ns\n"
                f"  FWHM  = {fw['fwhm']:.4f} ns")

        def mh_clear(self):
            self.mh = {}
            self.extra_spans = []
            self.extra_vlines = []
            if self.view_kind == "1d":
                self.draw_1d()
            self._mh_show_stored()

        # ------------------------------------------------------------ files --
        def open_dialog(self):
            path = filedialog.askopenfilename(
                title="Select an HDF5 file",
                filetypes=[("HDF5 files", "*.h5 *.hdf5 *.he5 *.hf5"),
                           ("All files", "*.*")])
            if path:
                self.open_path(path)

        def reload(self):
            if self.filepath:
                self.open_path(self.filepath)

        def open_path(self, path):
            self._set_status(f"loading {os.path.basename(path)} …")
            self.root.update_idletasks()
            try:
                self.tree_data = load_hdf5_tree(path)
            except Exception as e:
                messagebox.showerror("Load failed", f"Could not read file:\n{e}")
                self._set_status("")
                return
            self.filepath = path
            self.file_lbl.configure(text=os.path.basename(path))
            self.stacks = find_image_stacks(self.tree_data)
            self.camera_scan = find_camera_scan(self.tree_data)
            self.averaging = find_averaging(self.tree_data)
            self.multiharp = find_multiharp(self.tree_data)
            self.confocal = find_confocal(self.tree_data)
            self.camera_sweep = find_camera_sweep(self.tree_data)
            self._sweep_fits = None
            self._sweep = sweep_frame_info(self.tree_data)
            self._update_modality_combo()
            self._update_dataset_combo()
            self._update_odmr_combo()
            self._populate_tree()
            self._auto_view()
            self._set_status("loaded")

        # ------------------------------------------------------------- tree --
        def _populate_tree(self):
            self.tree.delete(*self.tree.get_children())
            self.item_values.clear()

            # synthetic node listing detected image stacks (easy movie access)
            if self.stacks:
                sroot = self.tree.insert("", "end",
                                         text="🎞 detected image stacks", open=True)
                for s in self.stacks:
                    iid = self.tree.insert(sroot, "end", text=s["name"])
                    self.item_values[iid] = ("__stack__", s)

            def add(parent, key, val):
                label = f"{key}   {describe_value(val)}"
                iid = self.tree.insert(parent, "end", text=label,
                                       open=(parent == "" ))
                self.item_values[iid] = ("value", val)
                if isinstance(val, dict):
                    for k, v in val.items():
                        add(iid, k, v)
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        add(iid, f"[{i}]", item)

            for k, v in self.tree_data.items():
                add("", k, v)

        def on_tree_select(self, _evt):
            iid = self._sel_iid()
            if iid is None:
                return
            tag, val = self.item_values[iid]
            self._show_info(iid, tag, val)
            self._update_xaxis_choices(val if tag == "value" else None)
            if tag == "value" and is_sequence_text(val):
                self.show_sequence(val)

        def _sel_iid(self):
            sel = self.tree.selection()
            return sel[0] if sel else None

        def _show_info(self, iid, tag, val):
            self.info.configure(state="normal")
            self.info.delete("1.0", "end")
            if tag == "__stack__":
                fr = val["frames"]
                self.info.insert("end", f"image stack\nsource: {val['source']}\n"
                                        f"frames: {fr.shape[0]}\n"
                                        f"frame size: {fr.shape[1]}×{fr.shape[2]}\n"
                                        f"dtype: {fr.dtype}\n\n"
                                        "double-click to open in the movie player.")
            else:
                kind = classify_value(val)
                self.info.insert("end", f"kind: {kind}\n")
                if isinstance(val, np.ndarray):
                    a = np.squeeze(val)
                    self.info.insert("end", f"shape: {tuple(a.shape)}\ndtype: {val.dtype}\n")
                    if a.size:
                        finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.number) else a
                        if getattr(finite, "size", 0):
                            self.info.insert("end",
                                f"min/max: {np.min(finite):.6g} / {np.max(finite):.6g}\n"
                                f"mean: {np.mean(finite):.6g}\n")
                    name = self.tree.item(iid, "text").split()[0] if iid else ""
                    res = (self._error_summary_for(name, a)
                           if (a.ndim == 1 and np.issubdtype(a.dtype, np.number)
                               and self._is_countlike(name)) else None)
                    if res:
                        self.info.insert("end", "\n— shot-noise error —\n" + res[0] + "\n")
                        self.err_lbl.configure(text=res[1])
                    else:
                        self.err_lbl.configure(text="(select a counts array to see "
                                               "its shot-noise error)")
                elif isinstance(val, str):
                    self.info.insert("end", "\n" + val[:1200])
                elif isinstance(val, (dict, list)):
                    self.info.insert("end", describe_value(val))
                else:
                    self.info.insert("end", f"value: {val}")
            self.info.configure(state="disabled")

        @staticmethod
        def _is_countlike(name):
            low = (name or "").lower()
            return any(k in low for k in ("count", "signal", "reference", "total"))

        def _error_summary_for(self, name, arr):
            """(info_text, short_label) describing the shot-noise error of a counts
            array. If it's the averaged channel and the individual runs are on
            disk, also report the empirical error-on-mean vs the Poisson value."""
            a = np.squeeze(np.asarray(arr, float))
            if a.ndim != 1 or not np.isfinite(a).any():
                return None
            runs = None
            av = self.averaging
            if av:
                for ch in ("signal", "reference", "total"):
                    if name in (f"{ch}_counts", f"{ch}_avg"):
                        runs = av["runs"].get(ch)
                        break
            es = error_stats(a, runs=runs)
            lines = [f"⟨N⟩ = {es['mean']:.6g}",
                     f"Poisson σ = √N = {es['poisson_sigma']:.4g}  "
                     f"({100 * es['rel_poisson']:.3g}% relative)"]
            short = (f"⟨N⟩={es['mean']:.4g},  σ=√N={es['poisson_sigma']:.4g}  "
                     f"({100 * es['rel_poisson']:.2g}%)")
            if "navg" in es:
                lines += [f"averaged over {es['navg']} runs:",
                          f"  empirical σ_mean = {es['emp_sigma_mean']:.4g}  "
                          f"({100 * es['rel_emp_mean']:.3g}% relative)",
                          f"  Poisson-expected σ_mean = {es['poisson_sigma_mean']:.4g}",
                          f"  ratio emp/Poisson = {es['ratio_emp_poisson']:.3f}  "
                          f"(≈1 ⇒ shot-noise limited)"]
                short = (f"averaged (n={es['navg']}): σ_mean={es['emp_sigma_mean']:.4g} "
                         f"({100 * es['rel_emp_mean']:.2g}%),  "
                         f"emp/Poisson={es['ratio_emp_poisson']:.2f}")
            return "\n".join(lines), short

        def _update_xaxis_choices(self, val):
            """Populate the x-axis combobox for the selected 1-D array."""
            if not (isinstance(val, np.ndarray) and classify_value(val) == "1d"):
                self.xaxis_combo.configure(values=[])
                self.xaxis_combo.set("")
                return
            n = int(np.squeeze(val).size)
            choices = ["index"]
            self._xaxis_map = {"index": np.arange(n, dtype=float)}
            freq_choices = []
            # sibling 1-D numeric arrays of equal length; a frequency-like one
            # (e.g. frequency_range / frequencies) is presented in GHz.
            for path, v in flatten_tree(self.tree_data):
                if isinstance(v, np.ndarray) and classify_value(v) == "1d":
                    if int(np.squeeze(v).size) == n and v is not val:
                        name = path.split("/")[-1]
                        arr = np.squeeze(v).astype(float)
                        if "freq" in name.lower() and np.nanmax(np.abs(arr)) > 1e6:
                            label = f"{name} (GHz)"
                            arr = arr / 1e9
                        else:
                            label = name
                        if label not in self._xaxis_map:
                            choices.append(label)
                            self._xaxis_map[label] = arr
                            if "freq" in label.lower():
                                freq_choices.append(label)
            # confocal physical position axes (with the point_a offset), offered
            # whenever the selected 1-D array spans the scan width or height.
            conf = getattr(self, "confocal", None)
            if conf is not None and conf.get("img") is not None:
                shp = np.asarray(conf["img"]).shape
                if len(shp) >= 2:
                    try:
                        cxv, cyv, cxl, cyl, _ = confocal_axes(conf, shp[1], shp[0],
                                                              "auto")
                    except Exception:
                        cxv = cyv = None
                    if (cxv is not None and cxv.size == n
                            and cxl not in self._xaxis_map):
                        choices.append(cxl); self._xaxis_map[cxl] = cxv
                    if (cyv is not None and cyv.size == n
                            and cyl not in self._xaxis_map):
                        choices.append(cyl); self._xaxis_map[cyl] = cyv
            # reconstructed pulsed axis from sequence_text — only the ONE that
            # matches what the sequence actually sweeps (tau / amplitude / freq)
            seq = find_text(self.tree_data, "sequence:", "variable ", "type=")
            kind = sweep_kind(seq) if seq else None
            if kind == "time":
                tax, tlab = build_time_axis(seq, n)
                if tlab.startswith("tau"):
                    choices.append("tau (from sequence)")
                    self._xaxis_map["tau (from sequence)"] = tax
            elif kind == "amplitude":
                aax, alab = build_amp_axis(seq, n)
                if alab.startswith("pulse amplitude"):
                    choices.append("amplitude (from sequence)")
                    self._xaxis_map["amplitude (from sequence)"] = aax
            elif kind == "frequency":
                fax, flab = build_freq_axis(seq, n)
                if flab.startswith("frequency"):
                    choices.append("frequency (from sequence, GHz)")
                    self._xaxis_map["frequency (from sequence, GHz)"] = fax / 1e9
                    freq_choices.append("frequency (from sequence, GHz)")
            elif kind == "phase":
                pax, plab = build_phase_axis(seq, n)
                if plab.startswith("phase"):
                    choices.append("phase (from sequence, rad)")
                    self._xaxis_map["phase (from sequence, rad)"] = pax
            self.xaxis_combo.configure(values=choices)
            # default: honour the swept variable. time->tau, amplitude->amp,
            # phase->phase, frequency (or unknown) -> a real frequency array.
            default = "index"
            if kind == "time" and "tau (from sequence)" in choices:
                default = "tau (from sequence)"
            elif kind == "amplitude" and "amplitude (from sequence)" in choices:
                default = "amplitude (from sequence)"
            elif kind == "phase" and "phase (from sequence, rad)" in choices:
                default = "phase (from sequence, rad)"
            elif freq_choices:
                default = freq_choices[0]
            elif "tau (from sequence)" in choices:
                default = "tau (from sequence)"
            self.xaxis_combo.set(default)

        # --------------------------------------------------- plotting nodes --
        def plot_selected_node(self):
            iid = self._sel_iid()
            if iid is None:
                return
            tag, val = self.item_values[iid]
            if tag == "__stack__":
                self.show_stack(val["frames"], title=val["name"])
                return
            kind = classify_value(val)
            if kind == "1d":
                self._plot_from_node(np.squeeze(val).astype(float),
                                     self.tree.item(iid, "text").split("   ")[0])
            elif kind == "image":
                nm = self.tree.item(iid, "text").split("   ")[0]
                self.show_image(np.squeeze(val).astype(float), title=nm, name=nm)
            elif kind == "stack":
                self.show_stack(np.squeeze(val).astype(float),
                                title=self.tree.item(iid, "text").split("   ")[0])
            elif is_sequence_text(val):
                self.show_sequence(val)
            else:
                messagebox.showinfo("Not plottable",
                                    f"This node is '{kind}'. Pick a 1-D array, an "
                                    "image, or a stack (or a scalar to inspect).")

        def _plot_from_node(self, yarr, name):
            # CW ODMR sweep channels -> use the dedicated trimmed / zoom-stable view
            chans = getattr(self, "odmr_channels", None) or {}
            leaf = str(name).split("/")[-1]
            for disp, tok in chans.items():
                if leaf == tok or name == tok:
                    self.show_cw_odmr(disp)
                    return
            self._rerender = lambda y=yarr, nm=name: self._plot_from_node(y, nm)
            xchoice = self.xaxis_combo.get() or "index"
            xmap = getattr(self, "_xaxis_map", {"index": np.arange(len(yarr))})
            x = xmap.get(xchoice, np.arange(len(yarr), dtype=float))
            xl = xchoice if xchoice != "index" else "index"
            y = self._bg_apply(name, yarr)
            self.set_1d(x, y, xlabel=xl, ylabel=name, title=name)

        # ---------------------------------------------------- bg subtraction --
        def _bg_find(self, base):
            """Background array whose leaf name equals `base` (exact preferred,
            else a substring match); None if not found."""
            if self.bg_tree is None:
                return None
            fallback = None
            for path, v in flatten_tree(self.bg_tree):
                if isinstance(v, np.ndarray):
                    leaf = path.split("/")[-1]
                    if leaf == base:
                        return np.squeeze(np.asarray(v, float))
                    if fallback is None and base and base in leaf:
                        fallback = np.squeeze(np.asarray(v, float))
            return fallback

        def _bg_apply(self, name, arr):
            """Subtract the matching background array from `arr`: by leaf name,
            with a unique-same-shape fallback (for images whose name we don't
            track). Returns `arr` unchanged when background is off / not loaded /
            no compatible match, so it is always safe to call."""
            if not (self.bg_on and self.bg_tree is not None) or arr is None:
                return arr
            a = np.asarray(arr, float)
            b = self._bg_find(str(name).split("/")[-1]) if name else None
            if b is None:
                same = [np.squeeze(np.asarray(v, float))
                        for _, v in flatten_tree(self.bg_tree)
                        if isinstance(v, np.ndarray)
                        and np.squeeze(np.asarray(v)).shape == a.shape]
                if len(same) == 1:
                    b = same[0]
            if b is None:
                return arr
            try:
                return a - b       # exact shape, or NumPy broadcast (2-D bg vs 3-D)
            except Exception:
                return arr

        def _pick_bg(self):
            path = filedialog.askopenfilename(
                title="Select background HDF5 file",
                filetypes=[("HDF5 files", "*.h5 *.hdf5 *.he5"), ("All files", "*.*")])
            if not path:
                return False
            try:
                self.bg_tree = load_hdf5_tree(path)
                self.bg_path = path
                self.bg_lbl.configure(text="bg: " + os.path.basename(path))
            except Exception as e:
                messagebox.showerror("Background",
                                     f"Could not load background file:\n{e}")
                return False
            return True

        def _change_bg(self):
            if self._pick_bg():
                self.bg_var.set(True)
                self.bg_on = True
                self._bg_refresh()

        def _toggle_bg(self):
            if self.bg_var.get():
                if self.bg_tree is None and not self._pick_bg():
                    self.bg_var.set(False)
                    return
                self.bg_on = True
            else:
                self.bg_on = False
            self._bg_refresh()

        def _bg_refresh(self):
            """Re-render the current view so an on/off change takes effect."""
            self._cross_fit_x = self._cross_fit_y = None
            self.pulsed_cache = None
            if self.view_kind in ("image", "stack"):
                self.redraw_image()
            elif self._rerender is not None:
                try:
                    self._rerender()
                except Exception:
                    pass

        def plot_selected_together(self):
            """Overlay every selected 1-D array on one plot (Ctrl/Shift-click to
            pick several, e.g. signal_counts + reference_counts)."""
            xchoice = self.xaxis_combo.get() or "index"
            xmap = getattr(self, "_xaxis_map", {})
            curves = []
            for iid in self.tree.selection():
                tag, val = self.item_values.get(iid, (None, None))
                if tag != "value" or classify_value(val) != "1d":
                    continue
                y = np.squeeze(val).astype(float)
                name = self.tree.item(iid, "text").split("   ")[0]
                xc = xmap.get(xchoice)
                x = xc if (xc is not None and len(xc) == len(y)) \
                    else np.arange(len(y), dtype=float)
                curves.append((x, y, name))
            if not curves:
                messagebox.showinfo(
                    "Plot selected together",
                    "Select one or more 1-D arrays in the browser first "
                    "(hold Ctrl or Shift to pick several), then click this button.\n\n"
                    "e.g. select signal_counts and reference_counts to overlay them.")
                return
            xl = xchoice if xchoice != "index" else "index"
            if len(curves) == 1:
                self.set_1d(curves[0][0], curves[0][1], xlabel=xl,
                            ylabel=curves[0][2], title=curves[0][2])
            else:
                self.set_multi(curves, xlabel=xl,
                               title="overlay: " + ", ".join(c[2] for c in curves))
            try:
                self.nb.select(0)
            except Exception:
                pass

        def replot_1d(self):
            if self.view_kind == "1d" and self.x is not None:
                self.draw_1d(keep_lims=getattr(self, "_odmr_view", False))

        # -------------------------------------------------- auto first view --
        def _auto_view(self):
            if self.multiharp:                   # MultiHarp timing histogram
                self.show_multiharp()
                return
            if self.camera_scan:                 # laser / white-light camera scan
                self.show_camera_scan()
                self.nb.select(3)
                return
            if self.camera_sweep:                # ODMR camera sweep (image per freq)
                self.show_camera_sweep()
                return
            info = detect_default_view(self.tree_data)
            mode = info["mode"]
            if mode == "pulsed":
                self._prepare_pulsed()
                self.nb.select(2)
                self.pulsed_plot_raw()
            elif mode == "cw_odmr":
                self._update_odmr_combo()
                self.show_cw_odmr()
            elif mode == "stack":
                frames = info["stack"]["frames"]
                zc = confocal_z(self.confocal, frames.shape[0]) if self.confocal else None
                self.show_stack(frames, title=info["stack"]["name"],
                                confocal=(self.confocal is not None), zcoords=zc,
                                name=info["stack"]["name"])
                self.nb.select(3)
            elif mode == "image":
                _, img = get_by_name(self.tree_data, "count_img", "count_image",
                                     "image", want=("image",))
                self.show_image(np.squeeze(img).astype(float),
                                title="confocal scan", confocal=True,
                                name="count_img")
                self.nb.select(3)
            elif mode == "1d":
                _, v = get_by_name(self.tree_data, info["path"].split("/")[-1],
                                   want=("1d",))
                if v is None:
                    for path, val in flatten_tree(self.tree_data):
                        if classify_value(val) == "1d":
                            v = val; break
                if v is not None:
                    self.set_1d(np.arange(len(np.squeeze(v))),
                                np.squeeze(v).astype(float),
                                xlabel="index", ylabel="value", title=info["path"])
            else:
                self._new_axes()
                self.ax.text(0.5, 0.5, "No obvious dataset to plot.\n"
                             "Pick something from the data browser on the left.",
                             ha="center", va="center", transform=self.ax.transAxes,
                             color="#888")
                self.canvas.draw_idle()

        # -------------------------------------------------------- 1-D engine --
        def _kill_selectors(self):
            """Deactivate + disconnect any live Span/Rectangle selectors so their
            drawn spans/boxes and event handlers don't leak across redraws."""
            for attr in ("span", "rect"):
                sel = getattr(self, attr, None)
                if sel is not None:
                    for meth in ("set_active", "set_visible", "disconnect_events"):
                        try:
                            fn = getattr(sel, meth, None)
                            if fn:
                                fn(False) if meth != "disconnect_events" else fn()
                        except Exception:
                            pass
                setattr(self, attr, None)

        def _new_axes(self):
            self._kill_selectors()
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.span = None
            self.rect = None
            self.imobj = None
            self.cbar = None
            self.fig.tight_layout()
            self.canvas.draw_idle()

        def set_1d(self, x, y, xlabel="", ylabel="", title="", yerr=None):
            self._stop_play()
            self.view_kind = "1d"
            self._odmr_view = False
            self.curves = None                    # single-curve mode
            self.x = np.asarray(x, float).ravel()
            self.y = np.asarray(y, float).ravel()
            n = min(len(self.x), len(self.y))
            self.x, self.y = self.x[:n], self.y[:n]
            self.cur_yerr = np.asarray(yerr, float).ravel()[:n] if yerr is not None else None
            self.xlabel, self.ylabel, self.title = xlabel, ylabel, title
            self.cur_frames = None
            self.sel_box = None
            self._last_fit = None
            self.extra_spans = []
            self.extra_vlines = []
            self.movie_row_state(False)
            self.draw_1d()

        def set_multi(self, curves, xlabel="", title=""):
            """Overlay several 1-D curves. curves = list of (x, y, label). The
            first curve becomes the 'primary' used for selection / fitting / stats."""
            curves = [(np.asarray(cx, float).ravel(), np.asarray(cy, float).ravel(), lab)
                      for cx, cy, lab in curves if len(np.ravel(cy))]
            if not curves:
                return
            self._stop_play()
            self.view_kind = "1d"
            self.curves = curves
            cx, cy, _ = curves[0]
            n = min(len(cx), len(cy))
            self.x, self.y = cx[:n], cy[:n]        # primary
            self.cur_yerr = None
            self.xlabel = xlabel
            self.ylabel = curves[0][2] if len(curves) == 1 else ""
            self.title = title
            self.cur_frames = None
            self.sel_box = None
            self._last_fit = None
            self.extra_spans = []
            self.extra_vlines = []
            self.movie_row_state(False)
            self.draw_1d()

        def _norm_apply(self, y, x=None):
            x = self.x if x is None else x
            mode = self.norm_var.get()
            if mode == "divide by max":
                m = np.nanmax(np.abs(y))
                return y / m if m else y
            if mode == "subtract selection baseline":
                if self.sel_x0 is not None and x is not None and len(x) == len(y):
                    mask = span_mask(x, self.sel_x0, self.sel_x1)
                    base = np.nanmedian(y[mask]) if mask.any() else np.nanmedian(y)
                else:
                    base = np.nanmedian(y)
                return y - base
            return y

        def draw_1d(self, keep_lims=False):
            xl, yl = (self.ax.get_xlim(), self.ax.get_ylim()) if keep_lims else (None, None)
            self._new_axes()
            style = self.style_var.get()
            fmt = {"line": "-", "markers": "o", "line+markers": "-o"}[style]
            palette = ["tab:blue", "tab:red", "tab:green", "tab:purple",
                       "tab:orange", "tab:cyan", "tab:brown", "tab:pink"]
            show_err = self.errbar_var.get()

            def plot_curve(cx, cy, color, label, err=None):
                yv = self._norm_apply(cy, cx)
                if show_err:
                    if err is None:
                        err = poisson_error(cy)            # generic shot noise
                    if self.norm_var.get() == "divide by max":
                        m = np.nanmax(np.abs(cy))
                        err = err / m if m else err
                    self.ax.errorbar(cx, yv, yerr=err, fmt=fmt, ms=4, lw=1.3,
                                     color=color, label=label, elinewidth=0.8,
                                     capsize=2, ecolor=color, alpha=0.9)
                else:
                    self.ax.plot(cx, yv, fmt, ms=4, lw=1.3, color=color, label=label)

            self._t1d_label = ""
            if self.curves:                        # overlay of several curves
                for i, (cx, cy, lab) in enumerate(self.curves):
                    ty, _tl = self._transform_1d(cx, cy)
                    plot_curve(cx, ty, palette[i % len(palette)], lab)
                    self._t1d_label = _tl
            else:                                  # single curve
                ty, self._t1d_label = self._transform_1d(self.x, self.y)
                plot_curve(self.x, ty, "tab:blue", self.ylabel or "data",
                           err=self.cur_yerr)
            if self._last_fit is not None:
                fitd = self._last_fit
                self.ax.plot(fitd["x_dense"], fitd["y_dense"], "-k", lw=1.6,
                             label=f"fit (R\u00b2={fitd['r2']:.4f})")
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
            _tl = getattr(self, "_t1d_label", "")
            self.ax.set_title(getattr(self, "title", "")
                              + (f"   [{_tl}]" if _tl else ""), fontsize=10)
            if self.logy_var.get():
                try:
                    self.ax.set_yscale("log")
                except Exception:
                    pass
            self.ax.grid(True, alpha=0.3)
            for x0, x1, color, alpha in getattr(self, "extra_spans", []):
                self.ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0)
            if getattr(self, "bg_fit_var", None) and self.bg_fit_var.get():
                for lo, hi in self._region_ranges():   # regions excluded from BG avg
                    self.ax.axvspan(lo, hi, color="tab:green", alpha=0.12, zorder=0)
            for xv, color in getattr(self, "extra_vlines", []):
                if np.isfinite(xv):
                    self.ax.axvline(xv, color=color, ls="--", lw=1.2)
            self.ax.legend(fontsize=8)
            self.span = _make_span(self.ax, self._on_span)
            if self.sel_x0 is not None:
                self.ax.axvspan(self.sel_x0, self.sel_x1, color="tab:orange", alpha=0.15)
            if keep_lims and xl:
                self.ax.set_xlim(xl); self.ax.set_ylim(yl)
            self.fig.tight_layout()
            self.canvas.draw_idle()

        def _on_span(self, x0, x1):
            if x1 == x0:
                return
            self.sel_x0, self.sel_x1 = (min(x0, x1), max(x0, x1))
            self.sel_lbl.configure(
                text=f"x ∈ [{self.sel_x0:.6g}, {self.sel_x1:.6g}]   "
                     f"({int(span_mask(self.x, self.sel_x0, self.sel_x1).sum())} pts)")
            # while background subtraction is on, a drag defines the current BG region
            if (getattr(self, "bg_fit_var", None) and self.bg_fit_var.get()
                    and hasattr(self, "bg_region_list")):
                i = self._cur_region_index()
                if 0 <= i < len(self.bg_region_list):
                    self.bg_region_list[i] = [self.sel_x0, self.sel_x1]
                    self.edge1_entry.delete(0, "end")
                    self.edge1_entry.insert(0, f"{self.sel_x0:.6g}")
                    self.edge2_entry.delete(0, "end")
                    self.edge2_entry.insert(0, f"{self.sel_x1:.6g}")
                    self.replot_1d()

        def clear_selection(self):
            self.sel_x0 = self.sel_x1 = None
            self.sel_lbl.configure(text="(drag on the plot to select a range)")
            if self.view_kind == "1d":
                self.draw_1d()

        def zoom_selection(self):
            if self.view_kind != "1d" or self.sel_x0 is None:
                return
            self.ax.set_xlim(self.sel_x0, self.sel_x1)
            m = span_mask(self.x, self.sel_x0, self.sel_x1)
            if m.any():
                ys = self._norm_apply(self.y)[m]
                pad = 0.05 * (np.nanmax(ys) - np.nanmin(ys) + 1e-12)
                self.ax.set_ylim(np.nanmin(ys) - pad, np.nanmax(ys) + pad)
            self.canvas.draw_idle()

        def reset_view(self):
            if self.view_kind == "1d":
                self.draw_1d()
            elif self.view_kind in ("image", "stack"):
                self.redraw_image()

        def selection_stats(self):
            if self.view_kind != "1d":
                return
            y = self._norm_apply(self.y)
            s = region_stats_1d(self.x, y, self.sel_x0, self.sel_x1)
            title = "selection" if self.sel_x0 is not None else "whole trace"
            messagebox.showinfo(f"Stats — {title}", format_stats_1d(s))

        def export_selection(self):
            if self.view_kind != "1d":
                return
            x, y = self.x, self._norm_apply(self.y)
            if self.sel_x0 is not None:
                m = span_mask(x, self.sel_x0, self.sel_x1)
                x, y = x[m], y[m]
            self._save_csv(np.column_stack([x, y]),
                           header=f"{self.xlabel},{self.ylabel}", suffix="_selection")

        # ----------------------------------------------------------- fitting --
        def do_fit(self):
            if self.view_kind != "1d":
                messagebox.showinfo("Fit", "Fitting works on 1-D traces. Select or "
                                           "plot a 1-D dataset first.")
                return
            name = self.model_combo.get()
            yt, tlbl = self._transform_1d(self.x, self.y)   # bg + smoothing
            x, y = self.x, self._norm_apply(yt)             # then normalisation
            if self.fit_sel_var.get() and self.sel_x0 is not None:
                m = span_mask(x, self.sel_x0, self.sel_x1)
                x, y = x[m], y[m]
                scope = f"selection ({m.sum()} pts)"
            else:
                scope = f"full trace ({len(x)} pts)"
            res = fit_model(name, x, y)
            self.fit_text.delete("1.0", "end")
            if not res["ok"]:
                self.fit_text.insert("end", f"Fit failed on {scope}:\n{res['error']}")
                return
            if tlbl:
                res = dict(res)
                res["summary"] = res["summary"] + "\n[" + tlbl + "]"
            self._last_fit = res
            self.fit_text.insert("end", f"[{scope}]\n" + res["summary"])
            self.draw_1d(keep_lims=True)

        def clear_fit(self):
            self._last_fit = None
            self.fit_text.delete("1.0", "end")
            if self.view_kind == "1d":
                self.draw_1d()

        def fit_odmr(self):
            """Guided ODMR dip fit: polynomial background + 1 or 2 Lorentzian dips
            using the center guess(es). Blank centers -> auto; the found centers
            are written back so you can nudge them and re-fit."""
            if self.view_kind != "1d" or self.x is None:
                messagebox.showinfo("ODMR fit", "Plot a 1-D ODMR trace first (an "
                                    "ODMR channel, or a camera-sweep quantity).")
                return
            yt, tlbl = self._transform_1d(self.x, self.y)
            x, y = self.x, self._norm_apply(yt)
            if self.fit_sel_var.get() and self.sel_x0 is not None:
                mm = span_mask(x, self.sel_x0, self.sel_x1)
                x, y = x[mm], y[mm]
            try:
                n = int(self.odmr_ndip.get())
            except Exception:
                n = 1
            try:
                bgd = int(self.odmr_bgdeg.get())
            except Exception:
                bgd = 0
            txt = self.odmr_centers.get().strip()
            centers = ([float(t) for t in re.split(r"[,;]", txt) if t.strip()]
                       if txt else None)
            res = fit_odmr_dips(np.asarray(x, float), np.asarray(y, float),
                                n, bgd, centers)
            self.fit_text.delete("1.0", "end")
            if not res.get("ok"):
                self.fit_text.insert("end", "ODMR fit failed:\n"
                                     + res.get("error", "?"))
                return
            if not txt:                       # show the auto centers for nudging
                self.odmr_centers.delete(0, "end")
                self.odmr_centers.insert(0, ", ".join(f"{c:.5g}"
                                                      for c in res["centers"]))
            if tlbl:
                res = dict(res); res["summary"] += "\n[" + tlbl + "]"
            self._last_fit = res
            self.fit_text.insert("end", res["summary"])
            self.draw_1d(keep_lims=True)

        def _transform_1d(self, x, y):
            """Monomial background subtraction + smoothing for a 1-D trace, used by
            BOTH the plot and the fit so they always agree. Returns (y_out, label)."""
            x = np.asarray(x, float); y = np.asarray(y, float); label = ""
            if getattr(self, "bg_fit_var", None) and self.bg_fit_var.get():
                regs = self._region_ranges()
                txt = self.bg_fit_entry.get().strip() if hasattr(self, "bg_fit_entry") else ""
                if regs:
                    bg = background_1d(x, y, regs, "outside")   # average OUTSIDE regions
                    y = y - bg
                    label = f"bg {bg:.4g} (outside {len(regs)} reg)"
                elif txt:
                    b = eval_monomial(txt, x)
                    if b is not None:
                        y = y - b; label = f"bg={txt}"
                else:
                    bg = guess_background_1d(x, y)
                    y = y - bg; label = f"bg {bg:.4g} (auto)"
            if getattr(self, "smooth_var", None) and self.smooth_var.get():
                w = self._smooth_window()
                if w > 1:
                    y = _moving_average(y, w)
                    label += (" | " if label else "") + f"smooth {w}"
            return y, label

        def _region_ranges(self):
            """List of (lo, hi) for BG regions that have both edges set."""
            out = []
            for a, b in getattr(self, "bg_region_list", []):
                if a is not None and b is not None:
                    out.append((min(a, b), max(a, b)))
            return out

        def _cur_region_index(self):
            try:
                return max(0, self.region_sel.current())
            except Exception:
                return 0

        def _on_region_sel(self):
            i = self._cur_region_index()
            if 0 <= i < len(self.bg_region_list):
                a, b = self.bg_region_list[i]
                self.edge1_entry.delete(0, "end")
                self.edge2_entry.delete(0, "end")
                if a is not None:
                    self.edge1_entry.insert(0, f"{a:.6g}")
                if b is not None:
                    self.edge2_entry.insert(0, f"{b:.6g}")

        def _on_edge_change(self):
            i = self._cur_region_index()
            if not (0 <= i < len(self.bg_region_list)):
                return

            def val(entry):
                t = entry.get().strip()
                try:
                    return float(t)
                except Exception:
                    return None
            self.bg_region_list[i] = [val(self.edge1_entry), val(self.edge2_entry)]
            if self.view_kind == "1d":
                self.replot_1d()

        def _add_region(self):
            self.bg_region_list.append([None, None])
            names = [f"region {k + 1}" for k in range(len(self.bg_region_list))]
            self.region_sel.configure(values=names)
            self.region_sel.current(len(names) - 1)
            self._on_region_sel()

        def _del_region(self):
            i = self._cur_region_index()
            if len(self.bg_region_list) <= 1:
                self.bg_region_list = [[None, None]]
            elif 0 <= i < len(self.bg_region_list):
                self.bg_region_list.pop(i)
            names = [f"region {k + 1}" for k in range(len(self.bg_region_list))]
            self.region_sel.configure(values=names)
            self.region_sel.current(min(i, len(names) - 1))
            self._on_region_sel()
            if self.view_kind == "1d":
                self.replot_1d()

        def _smooth_window(self):
            try:
                return max(1, int(round(float(self.smooth_win.get()))))
            except Exception:
                return 1

        def _gauss_k(self):
            """2-D Gaussian circle radius as a multiple of FWHM (semi-axis = k*FWHM)."""
            try:
                return max(0.1, float(self.gauss_radius.get()))
            except Exception:
                return 1.5

        def _process_image(self, img):
            """Per-row polynomial background subtraction (systematic noise) then
            denoise (white noise: spatial average or FFT low-pass). Applied to the
            shown image and to every camera-sweep fit, so they always agree."""
            img = np.asarray(img, float)
            if img.ndim != 2:
                return img
            notes = []
            if getattr(self, "rowbg_var", None) and self.rowbg_var.get():
                try:
                    d = int(self.rowbg_deg.get())
                except Exception:
                    d = 1
                img = subtract_row_poly(img, d)
                notes.append(f"row poly deg {d}")
            if getattr(self, "denoise_var", None) and self.denoise_var.get():
                meth = (self.denoise_method.get()
                        if hasattr(self, "denoise_method") else "average")
                try:
                    p = float(self.denoise_param.get())
                except Exception:
                    p = 1.5
                if meth.startswith("FFT"):
                    img = fft_lowpass(img, p if 0 < p <= 1 else 0.3)
                    notes.append("FFT low-pass")
                else:
                    try:
                        from scipy.ndimage import gaussian_filter
                        fill = (float(np.nanmedian(img[np.isfinite(img)]))
                                if np.isfinite(img).any() else 0.0)
                        img = gaussian_filter(np.nan_to_num(img, nan=fill),
                                              sigma=max(0.1, p))
                        notes.append(f"avg \u03c3={p:g}px")
                    except Exception:
                        pass
            if notes:
                cur = getattr(self, "_bg_status", "")
                self._bg_status = ((cur + "   |   ") if cur else "") + ", ".join(notes)
            return img

        def _image_smooth(self, data, xv, yv):
            """Blur the background (outside the 2-D Gaussian spot) with the given
            sigma while keeping the spot itself sharp."""
            try:
                sigma = float(self.img_smooth_sigma.get())
            except Exception:
                sigma = 1.0
            if sigma <= 0:
                return data
            try:
                from scipy.ndimage import gaussian_filter
                fill = float(np.nanmedian(data[np.isfinite(data)])) if np.isfinite(data).any() else 0.0
                blurred = gaussian_filter(np.nan_to_num(data, nan=fill), sigma=sigma)
            except Exception:
                return data
            g = self._img_gauss
            if g and g.get("ok"):
                X, Y = np.meshgrid(np.asarray(xv, float), np.asarray(yv, float))
                k = self._gauss_k()
                rx = max(k * g["fwhm_x"], 1e-9); ry = max(k * g["fwhm_y"], 1e-9)
                inside = (((X - g["x0"]) / rx) ** 2 + ((Y - g["y0"]) / ry) ** 2) <= 1.0
                return np.where(inside, data, blurred)   # spot sharp, background smooth
            return blurred

        def _bg_for_fit(self, x, y):
            """Background for a 1-D fit -> (on, bg_at_x, bg_fn, label). bg_fn(x')
            gives the background at new x' (for the fit curve)."""
            if not (getattr(self, "bg_fit_var", None) and self.bg_fit_var.get()):
                return False, 0.0, (lambda xx: 0.0), ""
            regs = parse_regions(self.bg_regions.get()) if hasattr(self, "bg_regions") else []
            mode = self.bg_region_mode.get() if hasattr(self, "bg_region_mode") else "auto"
            if regs and mode != "auto":
                which = "outside" if mode.startswith("outside") else "inside"
                c = background_1d(x, y, regs, which)
                lbl = f"background \u2248 {c:.6g} ({which} {len(regs)} region(s))"
                return True, c, (lambda xx, cc=c: np.full_like(np.asarray(xx, float), cc)), lbl
            txt = self.bg_fit_entry.get().strip()
            if txt:
                b = eval_monomial(txt, x)
                if b is None:
                    return False, 0.0, (lambda xx: 0.0), f"(bad bg '{txt}', ignored)"
                return True, b, (lambda xx, t=txt: eval_monomial(t, xx)), f"background = {txt}"
            g = guess_background_1d(x, y)
            self.bg_fit_entry.delete(0, "end")
            self.bg_fit_entry.insert(0, f"{g:.6g}")
            return (True, g, (lambda xx, c=g: np.full_like(np.asarray(xx, float), c)),
                    f"background \u2248 {g:.6g} (auto-averaged)")

        def _guess_bg_fit(self):
            """Fill the background box with an auto-guess from the current data."""
            if self.view_kind == "1d" and self.x is not None:
                g = guess_background_1d(self.x, self._norm_apply(self.y))
            elif self.view_kind in ("image", "stack") and self._disp_data is not None:
                sub = self._disp_data
                if self.sel_box is not None:
                    c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
                    sub = sub[r0:r1, c0:c1]
                g = guess_background_2d(sub)
            else:
                messagebox.showinfo("Background", "Plot or select some data first.")
                return
            self.bg_fit_entry.delete(0, "end")
            self.bg_fit_entry.insert(0, f"{g:.6g}")
            if self.view_kind == "1d":
                self.replot_1d()

        def _guess_img_bg(self):
            """Fill the 2-D Gaussian background box with an auto-guess (average of
            the non-bright pixels) from the current ROI, or the whole image."""
            if self.view_kind not in ("image", "stack") or self._disp_data is None:
                messagebox.showinfo("Background", "Open an image first.")
                return
            sub = self._disp_data
            if self.sel_box is not None:
                c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
                sub = sub[r0:r1, c0:c1]
            g = guess_background_2d(sub)
            self.img_bg_entry.delete(0, "end")
            self.img_bg_entry.insert(0, f"{g:.6g}")
            if self.img_bg_var.get() and self.view_kind in ("image", "stack"):
                self.redraw_image()

        def _image_bg_value(self, data, xv, yv):
            """Constant background for the image, per the 'from' mode. A number in
            the box overrides. Modes: all pixels / auto (exclude bright) / outside
            2-D Gaussian (k=1.5 x FWHM) / outside ROI box."""
            txt = self.img_bg_entry.get().strip()
            if txt:
                try:
                    return float(txt)
                except ValueError:
                    pass
            mode = (self.img_bg_mode.get() if hasattr(self, "img_bg_mode")
                    else "auto (exclude bright)")
            fin = np.isfinite(data)
            if mode == "all pixels":
                v = data[fin]
                return float(np.mean(v)) if v.size else 0.0
            if (mode == "outside 2-D Gaussian" and self._img_gauss
                    and self._img_gauss.get("ok")):
                g = self._img_gauss
                X, Y = np.meshgrid(np.asarray(xv, float), np.asarray(yv, float))
                k = self._gauss_k()
                rx = max(k * g["fwhm_x"], 1e-9)     # exclusion radius = k x FWHM
                ry = max(k * g["fwhm_y"], 1e-9)
                inside = (((X - g["x0"]) / rx) ** 2
                          + ((Y - g["y0"]) / ry) ** 2) <= 1.0
                pts = data[(~inside) & fin]
                if pts.size:
                    return float(np.mean(pts))
            if mode == "outside ROI" and self.sel_box is not None:
                c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
                mask = np.ones(data.shape, bool)
                mask[r0:r1, c0:c1] = False
                pts = data[mask & fin]
                if pts.size:
                    return float(np.mean(pts))
            return guess_background_2d(data)   # auto (exclude bright) / fallback

        def _lineout_width(self):
            """Line-out averaging band width in pixels (>=1) from the entry box."""
            try:
                w = int(round(float(self.lineout_w.get())))
            except Exception:
                w = 1
            return max(1, w)

        def export_fit(self):
            if not self._last_fit:
                messagebox.showinfo("Export fit", "Run a fit first.")
                return
            d = self._last_fit
            self._save_csv(np.column_stack([d["x_dense"], d["y_dense"]]),
                           header="x_fit,y_fit", suffix="_fit")

        # ------------------------------------------------------- pulsed tab ---
        def _prepare_pulsed(self):
            """Collect signal/reference (+ per-point errors) and sequence text for
            the pulsed workflow, honoring the averaged / individual-run choice."""
            ds = self.dataset_combo.get() if hasattr(self, "dataset_combo") else "averaged"
            av = self.averaging
            sig = ref = sig_err = ref_err = None
            if av and ds.startswith("run "):
                k = int(ds.split()[1]) - 1
                rs = av["runs"]
                if rs.get("signal") and k < len(rs["signal"]):
                    sig = rs["signal"][k]; sig_err = poisson_error(sig)
                if rs.get("reference") and k < len(rs["reference"]):
                    ref = rs["reference"][k]; ref_err = poisson_error(ref)
            if sig is None or ref is None:
                _, sig = get_by_name(self.tree_data, "signal_counts", "signal",
                                     want=("1d", "image"))
                _, ref = get_by_name(self.tree_data, "reference_counts", "reference",
                                     want=("1d", "image"))
                if sig is None or ref is None:
                    self.pulsed_cache = None
                    return
                sig = np.squeeze(np.asarray(sig, float))
                ref = np.squeeze(np.asarray(ref, float))
                if sig.ndim == 2:
                    sig = sig[0]
                if ref.ndim == 2:
                    ref = ref[0]
                sr = av["runs"] if av else {}
                sig_err = (error_on_mean(sr["signal"]) if sr.get("signal")
                           and len(sr["signal"]) >= 2 else poisson_error(sig))
                ref_err = (error_on_mean(sr["reference"]) if sr.get("reference")
                           and len(sr["reference"]) >= 2 else poisson_error(ref))
            sig = np.asarray(sig, float); ref = np.asarray(ref, float)
            sig = self._bg_apply("signal_counts", sig)      # background subtraction
            ref = self._bg_apply("reference_counts", ref)
            sig_err = np.asarray(sig_err, float); ref_err = np.asarray(ref_err, float)
            n = min(len(sig), len(ref), len(sig_err), len(ref_err))
            seq = find_text(self.tree_data, "sequence:", "variable ", "type=")
            _, farr = get_by_name(self.tree_data, "frequencies", "freq_list",
                                  want=("1d",))
            self.pulsed_cache = dict(sig=sig[:n], ref=ref[:n], sig_err=sig_err[:n],
                                     ref_err=ref_err[:n], seq=seq, n=n,
                                     has_freq=(farr is not None), freq=farr, dataset=ds)
            mode = detect_pulsed_mode(seq, farr is not None)
            self.pulsed_mode = mode
            self.pmode_combo.set(mode)

        def _update_dataset_combo(self):
            """Populate the averaged/run selector. Enabled only when the file
            actually stored individual runs; otherwise parked on 'averaged'."""
            av = self.averaging
            if not av:
                self.dataset_combo.configure(values=["averaged"], state="disabled")
                self.dataset_combo.set("averaged")
                return
            vals = ["averaged"] + [f"run {i}" for i in range(1, av["navg"] + 1)]
            self.dataset_combo.configure(values=vals, state="readonly")
            self.dataset_combo.set("averaged")

        def on_dataset_change(self):
            self._prepare_pulsed()
            if self.pulsed_cache:
                self.pulsed_plot_raw()

        def _pulsed_axis(self, mode):
            c = self.pulsed_cache
            n = c["n"]
            if mode == "odmr":
                if c["freq"] is not None:
                    f = np.squeeze(c["freq"]).astype(float)[:n]
                    return f, ("frequency (GHz)" if np.nanmax(f) > 1e6 else "frequency")
                # reconstruct from a swept-frequency variable line, if present
                fax, flab = build_freq_axis(c["seq"], n)
                if flab.startswith("frequency"):
                    return fax, "frequency (GHz)"
                # else from a real frequency array (frequency_range holds the
                # actual swept frequencies; use them directly when lengths match)
                _, fr = get_by_name(self.tree_data, "frequency_range",
                                    "frequencies", want=("1d",))
                if fr is not None:
                    fr = np.squeeze(fr).astype(float)
                    if len(fr) == n:
                        return fr, "frequency (GHz)"
                    return np.linspace(fr[0], fr[-1], n), "frequency (GHz)"
                return np.arange(n, float), "scan point"
            if mode == "phase":
                return build_phase_axis(c["seq"], n)
            if mode == "rabi_amp":
                return build_amp_axis(c["seq"], n)
            return build_time_axis(c["seq"], n)

        def pulsed_plot_raw(self):
            self._rerender = self.pulsed_plot_raw
            if not self.pulsed_cache:
                self._prepare_pulsed()
            if not self.pulsed_cache:
                messagebox.showinfo("Pulsed", "No signal/reference arrays found.")
                return
            c = self.pulsed_cache
            mode = self.pmode_combo.get()
            x, xlabel = self._pulsed_axis(mode)
            if mode == "odmr" and np.nanmax(x) > 1e6:
                x = x / 1e9
            q = self.pq_combo.get()
            y, ylabel = self._pulsed_quantity(c["sig"], c["ref"])
            yerr = None
            if self.errbar_var.get():
                se = c.get("sig_err"); re_ = c.get("ref_err")
                yerr = quantity_error(q, c["sig"], c["ref"],
                                      se if se is not None else poisson_error(c["sig"]),
                                      re_ if re_ is not None else poisson_error(c["ref"]))
            self.set_1d(x, y, xlabel=xlabel, ylabel=ylabel, yerr=yerr,
                        title=f"pulsed: {mode}   [{ylabel}]   \u00b7 {c.get('dataset','averaged')}")

        def _pulsed_quantity(self, sig, ref):
            """Return (y, label) for the quantity chosen in the Pulsed tab."""
            q = self.pq_combo.get()
            with np.errstate(divide="ignore", invalid="ignore"):
                if q == "signal":
                    return sig.astype(float), "signal counts"
                if q == "reference":
                    return ref.astype(float), "reference counts"
                if q == "S-R":
                    return sig - ref, "signal − reference"
                if q == "(S-R)/R":
                    return np.where(ref != 0, (sig - ref) / ref, np.nan), "(S − R) / R"
                return np.where(ref != 0, sig / ref, np.nan), "Signal / Reference"

        def _pulsed_prep_xy(self):
            """Common prep for analyse + significance: axis, chosen quantity,
            per-point error, dead-point drop, finite mask, GHz conversion, and
            optional selection restriction. Returns
            (mode, xlabel, xplot, y_plot, x_fit, y_fit, ylabel, scope, yerr_plot)."""
            c = self.pulsed_cache
            mode = self.pmode_combo.get()
            x, xlabel = self._pulsed_axis(mode)
            sig, ref = c["sig"].copy(), c["ref"].copy()
            se = np.asarray(c.get("sig_err") if c.get("sig_err") is not None
                            else poisson_error(sig), float).copy()
            re_ = np.asarray(c.get("ref_err") if c.get("ref_err") is not None
                             else poisson_error(ref), float).copy()
            if self.drop_dead_var.get():
                med = np.median(sig[sig > 0]) if np.any(sig > 0) else 0
                live = sig > 0.2 * med
                x, sig, ref, se, re_ = x[live], sig[live], ref[live], se[live], re_[live]
            q = self.pq_combo.get()
            y, ylabel = self._pulsed_quantity(sig, ref)
            yerr = quantity_error(q, sig, ref, se, re_)
            good = np.isfinite(y)
            xg, yg, eg = x[good], y[good], yerr[good]
            xplot = xg / 1e9 if (mode == "odmr" and np.nanmax(xg) > 1e6) else xg
            xfit, yfit, scope = xg, yg, f"full trace ({len(xg)} pts)"
            if self.pulsed_sel_var.get():
                if self.sel_x0 is None:
                    scope = "no selection made \u2014 used full trace"
                else:
                    m = span_mask(xplot, self.sel_x0, self.sel_x1)
                    if int(m.sum()) >= 4:
                        xfit, yfit = xg[m], yg[m]
                        scope = f"selection ({int(m.sum())} pts)"
                    else:
                        scope = "selection <4 pts \u2014 used full trace"
            yerr_plot = eg if self.errbar_var.get() else None
            return mode, xlabel, xplot, yg, xfit, yfit, ylabel, scope, yerr_plot

        def pulsed_analyse(self):
            self._rerender = self.pulsed_analyse
            if not self.pulsed_cache:
                self._prepare_pulsed()
            if not self.pulsed_cache:
                messagebox.showinfo("Pulsed", "No signal/reference arrays found.")
                return
            mode, xlabel, xplot, yg, xfit, yfit, ylabel, scope, yerr = self._pulsed_prep_xy()
            self.pulsed_text.delete("1.0", "end")
            res = self._pulsed_fit(mode, xfit, yfit)
            ds = self.pulsed_cache.get("dataset", "averaged")
            self.set_1d(xplot, yg, yerr=yerr,
                        xlabel=("frequency (GHz)" if mode == "odmr" else xlabel),
                        ylabel=ylabel, title=f"pulsed: {mode}   [{ylabel}]   \u00b7 {ds}")
            errline = ""
            if yerr is not None and np.isfinite(yerr).any():
                errline = f"  \u00b7  median \u03c3 = {np.nanmedian(yerr):.4g}"
            head = f"[{ylabel}  \u00b7  {ds}  \u00b7  {scope}{errline}]\n\n"
            if res is not None:
                xf = res["xf"] / 1e9 if (mode == "odmr" and np.nanmax(res["xf"]) > 1e6) \
                    else res["xf"]
                self._last_fit = {"x_dense": xf, "y_dense": res["yf"],
                                  "r2": res.get("r2", float("nan"))}
                self.draw_1d()
                self.pulsed_text.insert("end", head + res["summary"])
            else:
                self.pulsed_text.insert("end", head + "fit failed for this mode.")

        def _pulsed_fit(self, mode, x, y):
            """Prefer our plot_pulsed_hdf5 fitters; fall back to the local library.
            The phase sweep has no counterpart in plot_pulsed_hdf5, so it always
            uses the built-in single-fringe cosine."""
            if mode != "phase":
                try:
                    import plot_pulsed_hdf5 as pp
                    if mode == "rabi_time":
                        d = pp.fit_rabi(x, y, amplitude=False)
                    elif mode == "rabi_amp":
                        d = pp.fit_rabi(x, y, amplitude=True)
                    elif mode == "ramsey":
                        d = pp.fit_ramsey(x, y)
                    elif mode == "echo":
                        d = pp.fit_echo(x, y)
                    elif mode == "odmr":
                        d = pp.fit_odmr(x, y)
                    else:
                        d = None
                    if d is not None:
                        r2 = r_squared(y, np.interp(x, d["xf"], d["yf"]))
                        return {"xf": d["xf"], "yf": d["yf"], "r2": r2,
                                "summary": d["title"] + "\n\n" + d["summary"] +
                                f"\n\n  R² = {r2:.5f}   (via plot_pulsed_hdf5.py)"}
                except Exception:
                    pass
            # fallback (and always, for phase): local models
            name = {"rabi_time": "Damped cosine (Rabi)",
                    "rabi_amp": "Damped cosine (Rabi)",
                    "ramsey": "Ramsey (Gaussian×cos)",
                    "echo": "Stretched exp (echo)",
                    "odmr": "Lorentzian dip (ODMR)",
                    "phase": "Cosine (phase sweep)"}[mode]
            r = fit_model(name, x, y)
            if not r["ok"]:
                return None
            note = ("\n\n  (phase-sweep cosine fit)" if mode == "phase" else
                    "\n\n  (built-in fit; keep plot_pulsed_hdf5.py next to this "
                    "file for the physics readout)")
            return {"xf": r["x_dense"], "yf": r["y_dense"], "r2": r["r2"],
                    "summary": r["summary"] + note}

        def pulsed_significance(self):
            self._rerender = self.pulsed_significance
            if not self.pulsed_cache:
                self._prepare_pulsed()
            if not self.pulsed_cache:
                messagebox.showinfo("Pulsed", "No signal/reference arrays found.")
                return
            mode, xlabel, xplot, yg, xfit, yfit, ylabel, scope, _yerr = self._pulsed_prep_xy()
            self._set_status("running permutation test …")
            self.root.update_idletasks()
            r = run_significance(mode, xfit, yfit)
            self._set_status("")
            self.pulsed_text.delete("1.0", "end")
            if not r["ok"]:
                self.pulsed_text.insert("end", r["error"]); return
            lines = [f"mode: {mode}   quantity: {ylabel}",
                     f"scope: {scope}",
                     f"points used: {len(xfit)}",
                     f"permutations: {r['n_perm']}", ""]
            if r["kind"] == "periodogram":
                lines += [f"Lomb-Scargle peak: {r['peak']:.3f} {r['unit']}",
                          f"peak power:        {r['power']:.3f}",
                          f"95% noise floor:   {r['thr']:.3f}",
                          f"Nyquist:           {r['nyquist']:.2f} {r['unit']}"]
                if mode == "rabi_time":
                    lines.append(f"→ f_Rabi ≈ {r['peak']:.2f} MHz, "
                                 f"π ≈ {500/r['peak']:.0f} ns, "
                                 f"π/2 ≈ {250/r['peak']:.0f} ns")
            else:
                lines += [f"shape-fit R²:      {r['r2']:.3f}",
                          f"95% noise R²:      {r['thr']:.3f}"]
            lines += ["", f"FALSE-ALARM PROBABILITY = {r['fap']*100:.2f}%",
                      f"VERDICT: {r['verdict']}"]
            self.pulsed_text.insert("end", "\n".join(lines))

        # -------------------------------------------------------- image tab ---
        def movie_row_state(self, on):
            # ttk widgets need .state(); plain-tk widgets need configure(state=).
            for child in self.movie_row.winfo_children():
                done = False
                try:
                    child.state(["!disabled"] if on else ["disabled"])
                    done = True
                except Exception:
                    pass
                if not done:
                    try:
                        child.configure(state="normal" if on else "disabled")
                    except Exception:
                        pass

        def show_image(self, img, title="", confocal=False, name=""):
            self._stop_play()
            self.view_kind = "image"
            self.cur_image = np.asarray(img, float)
            self.cur_image_name = name
            self.cur_frames = None
            self.frame_z = None
            self._is_confocal = bool(confocal)
            self.sel_box = None
            self.title = title
            self.movie_row_state(False)
            self.nb.select(3)
            self.redraw_image()
            self._refresh_sweep_picker()

        # ------------------------------------------- camera light-source mode --
        def _update_modality_combo(self):
            """Populate + gray the laser/white-light dropdown for the current file.
            Enabled only when both modalities are present; otherwise parked on the
            one that exists and disabled."""
            cs = self.camera_scan
            if not cs:
                self.modality_combo.configure(values=[], state="disabled")
                self.modality_combo.set("")
                return
            display = [_CAM_LABEL[p] for p in cs["present"]]
            self.modality_combo.configure(values=display)
            self.modality_combo.set(display[0])
            self.modality_combo.configure(
                state="readonly" if len(cs["present"]) > 1 else "disabled")

        def _update_odmr_combo(self):
            """Populate the CW ODMR channel selector with whichever of
            averaged / forward / reverse the file has; disable otherwise."""
            self.odmr_channels = find_odmr_channels(self.tree_data)
            if self.odmr_channels:
                vals = list(self.odmr_channels.keys())
                self.odmr_combo.configure(values=vals, state="readonly")
                self.odmr_combo.set("averaged" if "averaged" in self.odmr_channels
                                    else vals[0])
            else:
                self.odmr_combo.configure(values=[], state="disabled")
                self.odmr_combo.set("")

        def on_odmr_channel(self):
            self.show_cw_odmr(self.odmr_combo.get())

        def _current_navg(self):
            """Number of averages recorded for the current file (1 if unknown)."""
            for nm in ("n_averages", "averages", "num_averages", "navg", "n_avg"):
                v = _as_num_vec(find_node(self.tree_data, nm))
                if v is not None and v.size:
                    try:
                        nv = int(round(float(v.flat[0])))
                        if nv >= 1:
                            return nv
                    except Exception:
                        pass
            return 1

        def combine_odmr_dialog(self):
            """Pick specific runs from several ODMR files and average them."""
            win = tk.Toplevel(self.root)
            win.title("Combine ODMR runs across files")
            win.geometry("640x380")
            self._combine_rows = []
            top = ttk.Frame(win, padding=8); top.pack(fill="x")
            ttk.Label(top, text="Channel:").pack(side="left")
            chan = ttk.Combobox(top, state="readonly", width=18,
                                values=["forward", "reverse (backward)", "averaged"])
            chan.set("averaged"); chan.pack(side="left", padx=(4, 12))
            rows = ttk.Frame(win)                       # (created below the canvas)
            ttk.Button(top, text="Add file(s)\u2026",
                       command=lambda: self._combine_add_files(rows)).pack(side="left")
            ttk.Label(win, foreground="#555", padding=(8, 0),
                      text="For each file, list the run numbers to use (1-based, "
                           "comma-separated), e.g. 1  or  1,3").pack(anchor="w")
            canv = tk.Canvas(win, height=210, highlightthickness=0)
            canv.pack(fill="both", expand=True, padx=8)
            rows = ttk.Frame(canv)
            canv.create_window((0, 0), window=rows, anchor="nw")
            rows.bind("<Configure>",
                      lambda e: canv.configure(scrollregion=canv.bbox("all")))
            bot = ttk.Frame(win, padding=8); bot.pack(fill="x")
            self._combine_status = ttk.Label(bot, text="", foreground="#036")
            self._combine_status.pack(side="left")
            ttk.Button(bot, text="Combine & plot",
                       command=lambda: self._combine_run(chan.get())).pack(side="right")
            # seed with the currently-loaded file if it is an ODMR sweep
            if getattr(self, "odmr_channels", None) and self.tree_data is not None:
                self._combine_add_tree(rows, getattr(self, "loaded_path",
                                                     "current file"), self.tree_data)

        def _combine_add_files(self, rows):
            paths = filedialog.askopenfilenames(
                title="Add ODMR file(s)",
                filetypes=[("HDF5", "*.h5 *.hdf5 *.he5"), ("All", "*.*")])
            for p in paths:
                try:
                    tree = load_hdf5_tree(p)
                except Exception as e:
                    messagebox.showerror("Combine", f"Could not load\n{p}\n\n{e}")
                    continue
                self._combine_add_tree(rows, p, tree)

        def _combine_add_tree(self, rows, path, tree):
            mat = odmr_run_matrix(tree, "forward")
            if mat is None:
                mat = odmr_run_matrix(tree, "averaged")
            navg = mat.shape[0] if mat is not None else 0
            fr = ttk.Frame(rows); fr.pack(fill="x", pady=1)
            ttk.Label(fr, text=os.path.basename(str(path)), width=32,
                      anchor="w").pack(side="left")
            ttk.Label(fr, text=f"(runs 1..{navg})", foreground="#777",
                      width=12).pack(side="left", padx=(2, 6))
            ttk.Label(fr, text="use:").pack(side="left")
            ent = ttk.Entry(fr, width=10); ent.insert(0, "1")
            ent.pack(side="left", padx=(2, 6))
            row = {"path": path, "tree": tree, "entry": ent, "frame": fr, "navg": navg}
            ttk.Button(fr, text="\u2715", width=2,
                       command=lambda: self._combine_del(row)).pack(side="left")
            self._combine_rows.append(row)

        def _combine_del(self, row):
            try:
                row["frame"].destroy()
            except Exception:
                pass
            if row in self._combine_rows:
                self._combine_rows.remove(row)

        def _combine_run(self, channel):
            specs = []
            for row in self._combine_rows:
                runs = []
                for tok in row["entry"].get().replace(";", ",").split(","):
                    tok = tok.strip()
                    if tok:
                        try:
                            runs.append(int(float(tok)))
                        except Exception:
                            pass
                if runs:
                    specs.append((row["tree"], runs))
            if not specs:
                messagebox.showinfo("Combine",
                                    "Add at least one file and enter run numbers.")
                return
            try:
                f, trace, nruns, labels = combine_odmr_runs(specs, channel)
            except Exception as e:
                messagebox.showerror("Combine", str(e))
                return
            self._combine_status.configure(text=f"averaged {nruns} runs")
            self._plot_combined_odmr(f, trace, nruns, ", ".join(labels))

        def _plot_combined_odmr(self, f, trace, nruns, label):
            f = np.asarray(f, float); trace = np.asarray(trace, float)
            keep = np.isfinite(f) & (f > 0) & np.isfinite(trace) & (trace != 0)
            if keep.sum() >= 2:
                f, trace = f[keep], trace[keep]
            ghz = np.nanmax(f) > 1e6
            xf = f / 1e9 if ghz else f
            xl = "frequency (GHz)" if ghz else "frequency"
            yerr = poisson_error(trace) / np.sqrt(max(1, nruns))
            self._odmr_view = False
            self._rerender = None
            self.set_1d(xf, trace, xlabel=xl, ylabel="counts", yerr=yerr,
                        title=f"Combined ODMR ({nruns} runs: {label})  "
                              f"(err \u00f7\u221a{nruns})")
            if xf.size:
                self._odmr_view = True
                self._odmr_xrange = (float(np.min(xf)), float(np.max(xf)))
                self.ax.set_xlim(*self._odmr_xrange)
                self.canvas.draw_idle()
            self.model_combo.set("Lorentzian dip (ODMR)")
            self.nb.select(1)

        def show_camera_sweep(self):
            """Show the per-frequency camera images as a scrollable movie (one
            frame per frequency); any per-frame 2-D Gaussian fit is overlaid."""
            cs = self.camera_sweep
            if not cs:
                return
            self.show_stack(np.asarray(cs["images"], float),
                            title="ODMR camera sweep",
                            zcoords=np.asarray(cs["freqs"], float),
                            zlabel="f", zunit="GHz", name="camera_sweep")

        def fit_sweep(self):
            """Fit a 2-D Gaussian in the ROI (or whole frame) of EVERY camera-sweep
            image, store the per-frequency parameters, and overlay each frame's fit
            so you can scroll through and confirm the NV was picked."""
            cs = self.camera_sweep
            if not cs:
                messagebox.showinfo("Camera sweep",
                                    "Load an ODMR camera-sweep file first.")
                return
            imgs = cs["images"]
            box = self.sel_box
            if box is not None and self._disp_data is not None:
                c0, c1, r0, r1 = self._box_to_pixels(box)
            else:
                h, w = np.asarray(imgs[0]).shape[:2]
                c0, c1, r0, r1 = 0, w, 0, h
            fits = []
            for im in imgs:
                im = self._process_image(np.asarray(im, float))   # same cleaning as display
                r1c, c1c = min(r1, im.shape[0]), min(c1, im.shape[1])
                sub = im[r0:r1c, c0:c1c]
                res = fit_gaussian_2d(sub, np.arange(c0, c1c), np.arange(r0, r1c),
                                      circular=True)
                if res.get("ok"):
                    v = sub[np.isfinite(sub)]
                    res["roi_mean"] = float(np.mean(v)) if v.size else float("nan")
                    res["peak"] = float(np.nanmax(sub)) if sub.size else float("nan")
                    res["integral"] = float(res["amp"] * 2.0 * np.pi
                                            * res["sigma_x"] * res["sigma_y"])
                fits.append(res)
            self._sweep_fits = fits
            self._sweep_freqs = np.asarray(cs["freqs"], float)
            nok = sum(1 for f in fits if f.get("ok"))
            self.show_camera_sweep()                 # display images + overlay
            try:
                self.sweep_plot_combo.set("images (scroll & verify)")
            except Exception:
                pass
            messagebox.showinfo(
                "Camera sweep",
                f"Fitted {nok}/{len(fits)} frames "
                + (f"in ROI cols {c0}\u2013{c1}, rows {r0}\u2013{r1}."
                   if box is not None else "(whole frame \u2014 draw an ROI to target "
                   "the NV).")
                + "\n\nScroll the movie to check the overlay, then pick a quantity "
                  "in 'plot:' to see it vs frequency.")

        def _plot_sweep_quantity(self, which=None):
            """Plot a fitted parameter vs frequency, or re-show the image movie."""
            which = which or (self.sweep_plot_combo.get()
                              if hasattr(self, "sweep_plot_combo") else "sigma (px)")
            if which.startswith("images"):
                self.show_camera_sweep()
                return
            if not self._sweep_fits:
                messagebox.showinfo("Camera sweep",
                                    "Run 'Fit 2-D Gaussian: all frames' first.")
                return
            keymap = {"sigma (px)": "sigma", "center x (px)": "x0",
                      "center y (px)": "y0", "amplitude (counts)": "amp",
                      "Gaussian integral (A\u00b72\u03c0\u03c3\u00b2)": "integral",
                      "peak pixel (max)": "peak",
                      "background (counts)": "bg", "ROI mean (counts)": "roi_mean",
                      "fit R\u00b2": "r2"}
            key = keymap.get(which, "sigma")
            f = self._sweep_freqs
            vals = np.array([(fit.get(key, np.nan) if fit.get("ok") else np.nan)
                             for fit in self._sweep_fits], float)
            f = np.asarray(f, float)
            m = np.isfinite(f) & np.isfinite(vals) & (f > 0)   # drop zero-padding
            fx = f[m]
            self._rerender = lambda w=which: self._plot_sweep_quantity(w)
            self.set_1d(fx, vals[m], xlabel="frequency (GHz)",
                        ylabel=which, title=f"camera sweep: {which} vs frequency")
            if fx.size:                                        # always show min..max freq
                self._odmr_view = True
                self._odmr_xrange = (float(np.min(fx)), float(np.max(fx)))
                self.ax.set_xlim(*self._odmr_xrange)
                self.canvas.draw_idle()

        def show_cw_odmr(self, channel=None):
            """Plot a CW ODMR sweep channel (forward / reverse / averaged) vs
            frequency, with background subtraction applied to that channel."""
            chans = self.odmr_channels or find_odmr_channels(self.tree_data)
            if not chans:
                return
            if channel not in chans:
                channel = "averaged" if "averaged" in chans else next(iter(chans))
            token = chans[channel]
            _, freqs = get_by_name(self.tree_data, "frequencies", "freq_list",
                                   "freq_axis", want=("1d",))
            cpath, counts = get_by_name(self.tree_data, token, want=("1d",))
            if freqs is None or counts is None:
                return
            leaf = cpath.split("/")[-1] if cpath else token
            f = np.squeeze(freqs).astype(float)
            craw = np.squeeze(counts).astype(float)
            n = min(len(f), len(craw))
            f, craw = f[:n], craw[:n]
            keep = np.isfinite(f) & (f > 0) & np.isfinite(craw) & (craw != 0)
            if keep.sum() >= 2:              # drop padded (zero) sweep points
                f, craw = f[keep], craw[keep]
            c = self._bg_apply(leaf, craw)
            ghz = np.nanmax(f) > 1e6
            xl = "frequency (GHz)" if ghz else "frequency"
            xf = f / 1e9 if ghz else f
            try:
                self.odmr_combo.set(channel)
            except Exception:
                pass
            prev = tuple(self.ax.get_xlim()) if getattr(self, "_odmr_view", False) else None
            self._rerender = lambda ch=channel: self.show_cw_odmr(ch)
            navg = self._current_navg()
            # error on the MEAN of navg averaged runs: sqrt(counts) / sqrt(navg)
            yerr = poisson_error(craw) / np.sqrt(max(1, navg))
            self.set_1d(xf, c, xlabel=xl, ylabel="counts", yerr=yerr,
                        title=f"CW ODMR sweep \u2014 {channel}"
                              + (f"  (err \u00f7\u221a{navg})" if navg > 1 else ""))
            if xf.size:
                full = (float(np.min(xf)), float(np.max(xf)))
                span = (full[1] - full[0]) or 1.0
                self._odmr_view = True
                # keep only a genuine zoom-IN within the sweep; otherwise snap to the
                # real sweep range -> no more padded 0-to-3 zoom-out on channel switch
                if (prev is not None
                        and prev[0] >= full[0] - 0.02 * span
                        and prev[1] <= full[1] + 0.02 * span
                        and (prev[1] - prev[0]) < 0.999 * span):
                    self._odmr_xrange = prev
                else:
                    self._odmr_xrange = full
                self.ax.set_xlim(*self._odmr_xrange)
                self.canvas.draw_idle()
            self.model_combo.set("Lorentzian dip (ODMR)")
            self.nb.select(1)

        def show_camera_scan(self, modality=None):
            cs = self.camera_scan
            if not cs:
                return
            if modality is None:
                modality = cs["present"][0]        # default: laser if present
            frames = cs["modalities"][modality]
            attr = dict(_CAM_MODALITIES)[modality]
            # keep the dropdown in sync with what is shown
            try:
                self.modality_combo.set(_CAM_LABEL[modality])
            except Exception:
                pass
            self._pending_meta = cs.get("meta", {}).get(modality)
            self.show_stack(frames, title=f"camera scan — {_CAM_LABEL[modality]} "
                                          f"({attr}), {frames.shape[0]} frames")

        def on_modality_change(self):
            disp = self.modality_combo.get().lower()
            modality = "white_light" if "white" in disp else "laser"
            if self.camera_scan and modality in self.camera_scan["modalities"]:
                self.show_camera_scan(modality)

        def show_stack(self, frames, title="", confocal=False, zcoords=None, name="",
                       zlabel="z", zunit=None):
            self._stop_play()
            self.view_kind = "stack"
            self.cur_frame_meta = getattr(self, "_pending_meta", None)
            self._pending_meta = None
            self.cur_frames = np.asarray(frames, float)
            self.cur_image_name = name
            self.frame_idx = 0
            self._is_confocal = bool(confocal)
            self._frame_coord_label = zlabel
            self._frame_coord_unit = zunit
            self.frame_z = (np.asarray(zcoords, float)
                            if zcoords is not None else None)
            self.sel_box = None
            self.title = title
            self.movie_row_state(True)
            self.frame_scale.configure(from_=0, to=self.cur_frames.shape[0] - 1)
            self.frame_scale.set(0)
            self.proj_combo.set("frame")
            self.nb.select(3)
            self.redraw_image()
            self._refresh_sweep_picker()

        def _current_image_data(self):
            if self.view_kind == "stack":
                proj = self.proj_combo.get()
                img = (self.cur_frames[self.frame_idx] if proj == "frame"
                       else stack_projection(self.cur_frames, proj))
            else:
                img = self.cur_image
            if img is None:
                return None
            img = np.asarray(img, float)
            if self.bg_on and self.bg_tree is not None:      # background-FILE subtraction
                img, self._bg_status = self._bg_image(img)
            else:
                self._bg_status = ""
            return img

        def _match_frame(self, cand, a):
            """A background array reshaped to match image `a`: exact shape, or a
            single frame of a matching stack. None if not compatible (no loose
            broadcasting for images, to avoid silently-wrong subtraction)."""
            cand = np.squeeze(np.asarray(cand, float))
            if cand.shape == a.shape:
                return cand
            if cand.ndim == a.ndim + 1 and cand.shape[1:] == a.shape:
                idx = min(getattr(self, "frame_idx", 0), cand.shape[0] - 1)
                return cand[idx]
            return None

        def _bg_image(self, a):
            """(subtracted_or_original, status) for the current image/frame. Tries
            count_img and the other standard image names, then a unique same-shape
            array; reports clearly when nothing compatible is found."""
            names = [getattr(self, "cur_image_name", ""), "count_img",
                     "count_image", "raw_img", "image", "img"]
            for nm in names:
                if not nm:
                    continue
                cand = self._bg_find(str(nm).split("/")[-1])
                if cand is not None:
                    cb = self._match_frame(cand, a)
                    if cb is not None:
                        return a - cb, f"background subtracted (matched '{nm}')"
            same = []
            for _, v in flatten_tree(self.bg_tree):
                if isinstance(v, np.ndarray):
                    cb = self._match_frame(np.squeeze(np.asarray(v, float)), a)
                    if cb is not None:
                        same.append(cb)
            if len(same) == 1:
                return a - same[0], "background subtracted (unique shape match)"
            shapes = sorted({np.squeeze(np.asarray(v)).shape
                             for _, v in flatten_tree(self.bg_tree)
                             if isinstance(v, np.ndarray)}, key=str)
            if len(same) > 1:
                return a, (f"background NOT applied: several arrays match "
                           f"{a.shape} — rename the background image 'count_img'")
            return a, (f"background NOT applied: no array matching {a.shape} in the "
                       f"background file (it has {', '.join(map(str, shapes[:5]))})")

        def _image_coords(self, shape):
            """(x_vec, y_vec, xlabel, ylabel, note) for the image being drawn."""
            ny, nx = shape[0], shape[1]
            src = getattr(self, "coord_src", None)
            src = src.get() if src is not None else "auto"
            # ODMR sweep per-run map (all_counts_forward / all_counts_reverse):
            # x = frequency, y = run number, colour scale = counts
            nm = getattr(self, "cur_image_name", "") or ""
            if "all_counts" in nm and src != "pixels":
                _, freqs = get_by_name(self.tree_data, "frequencies", "freq_list",
                                       "freq_axis", want=("1d",))
                if freqs is not None:
                    f = np.squeeze(freqs).astype(float)
                    ghz = np.nanmax(f) > 1e6
                    fv = f / 1e9 if ghz else f
                    fl = "frequency (GHz)" if ghz else "frequency"
                    runs = np.arange(1, ny + 1, dtype=float)
                    if f.size == nx:  # rows = runs, cols = frequency
                        return fv, runs, fl, "run #", "runs \u00d7 frequency"
                    if f.size == ny:  # rows = frequency, cols = runs
                        return (np.arange(1, nx + 1, dtype=float), fv,
                                "run #", fl, "frequency \u00d7 runs (use transpose)")
            if self._is_confocal and self.confocal and src != "pixels":
                xv, yv, xl, yl, note = confocal_axes(self.confocal, nx, ny, src)
                return np.asarray(xv, float), np.asarray(yv, float), xl, yl, note
            return (np.arange(nx, dtype=float), np.arange(ny, dtype=float),
                    "x (pixels)", "y (pixels)", "pixel index")

        def _z_label(self):
            if (self.view_kind == "stack" and self.frame_z is not None
                    and 0 <= self.frame_idx < len(self.frame_z)):
                lab = getattr(self, "_frame_coord_label", "z") or "z"
                unit = getattr(self, "_frame_coord_unit", None)
                if unit is None:
                    unit = self.confocal.get("unit", "µm") if self.confocal else "µm"
                return f"   {lab} = {self.frame_z[self.frame_idx]:.4g} {unit}"
            return ""

        def _box_to_pixels(self, box):
            """Physical-coordinate ROI box -> integer (c0,c1,r0,r1) pixel bounds
            into self._disp_data, via the ascending shown coordinate vectors."""
            x0, x1, y0, y1 = box
            xv, yv = self.img_x, self.img_y
            if xv is None or yv is None:
                c0, c1 = sorted((int(round(x0)), int(round(x1))))
                r0, r1 = sorted((int(round(y0)), int(round(y1))))
            else:
                c0 = int(np.searchsorted(xv, min(x0, x1)))
                c1 = int(np.searchsorted(xv, max(x0, x1)))
                r0 = int(np.searchsorted(yv, min(y0, y1)))
                r1 = int(np.searchsorted(yv, max(y0, y1)))
            h, w = self._disp_data.shape[:2]
            c0 = max(0, min(c0, w - 1)); c1 = max(c0 + 1, min(c1, w))
            r0 = max(0, min(r0, h - 1)); r1 = max(r0 + 1, min(r1, h))
            return c0, c1, r0, r1

        def redraw_image(self, *_):
            data = self._current_image_data()
            if data is None:
                return
            self._kill_selectors()
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            self.ax_top = self.ax_right = None
            self.span = None; self.rect = None; self.imobj = None; self.cbar = None
            clip = float(self.clip_var.get())
            finite = data[np.isfinite(data)]
            if finite.size:
                lo = np.percentile(finite, clip)
                hi = np.percentile(finite, 100 - clip)
            else:
                lo, hi = None, None
            # physical coordinates + orientation ---------------------------
            xv, yv, xlabel, ylabel, note = self._image_coords(data.shape)
            if self.transpose_var.get():
                data = data.T
                xv, yv = yv, xv
                xlabel, ylabel = ylabel, xlabel
            # order both axes ascending so extent + ROI math stay consistent
            if xv.size > 1 and xv[0] > xv[-1]:
                data = data[:, ::-1]; xv = xv[::-1]
            if yv.size > 1 and yv[0] > yv[-1]:
                data = data[::-1, :]; yv = yv[::-1]
            self._disp_data = data
            self.img_x, self.img_y = xv, yv
            self.img_xlabel, self.img_ylabel = xlabel, ylabel
            data = self._process_image(data)     # per-row poly bg + denoise (visible)
            self._disp_data = data
            if getattr(self, "img_bg_var", None) and self.img_bg_var.get():
                _c = self._image_bg_value(data, xv, yv)
                data = data - _c
                self._disp_data = data
                self._bg_status = ((self._bg_status + "   |   ")
                                   if getattr(self, "_bg_status", "") else "") \
                    + f"bg {_c:.6g} subtracted ({self.img_bg_mode.get()})"
            if getattr(self, "img_smooth_var", None) and self.img_smooth_var.get():
                data = self._image_smooth(data, xv, yv)
                self._disp_data = data
                self._bg_status = ((self._bg_status + "   |   ")
                                   if getattr(self, "_bg_status", "") else "") \
                    + "background smoothed"
            active = not xlabel.endswith("(pixels)")
            if active:
                dx = (xv[1] - xv[0]) if xv.size > 1 else 1.0
                dy = (yv[1] - yv[0]) if yv.size > 1 else 1.0
                extent = [xv[0] - dx / 2, xv[-1] + dx / 2,
                          yv[0] - dy / 2, yv[-1] + dy / 2]
            else:
                extent = None
            self.imobj = self.ax.imshow(data, origin="lower", extent=extent,
                                        cmap=self.cmap_combo.get(),
                                        vmin=lo, vmax=hi,
                                        aspect="equal" if self.equal_var.get() else "auto",
                                        interpolation="nearest")

            cross_on = bool(getattr(self, "crosshair_var", None)
                            and self.crosshair_var.get())
            marginal = cross_on and (self._cross_xy is not None)

            # counts-vs-x strip on top, counts-vs-y strip on the right --------
            if marginal:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                div = make_axes_locatable(self.ax)
                self.ax_top = div.append_axes("top", size="32%", pad=0.08,
                                              sharex=self.ax)
                self.ax_right = div.append_axes("right", size="32%", pad=0.08,
                                                sharey=self.ax)
                cax = div.append_axes("right", size="4%", pad=0.55)
                self.cbar = self.fig.colorbar(self.imobj, cax=cax)
            else:
                self.cbar = self.fig.colorbar(self.imobj, ax=self.ax,
                                              fraction=0.046, pad=0.04)

            # user mirror toggles (view-only; shared marginals follow)
            xa, xb = self.ax.get_xlim(); ya, yb = self.ax.get_ylim()
            self.ax.set_xlim((max(xa, xb), min(xa, xb)) if self.flipx_var.get()
                             else (min(xa, xb), max(xa, xb)))
            self.ax.set_ylim((max(ya, yb), min(ya, yb)) if self.flipy_var.get()
                             else (min(ya, yb), max(ya, yb)))

            ttl = getattr(self, "title", "")
            if self.view_kind == "stack" and self.proj_combo.get() == "frame":
                n = self.cur_frames.shape[0]
                zlab = self._z_label()
                ttl += f"   frame {self.frame_idx+1}/{n}{zlab}"
                self.frame_lbl.configure(text=f"{self.frame_idx+1}/{n}{zlab}")
            meta_list = getattr(self, "cur_frame_meta", None)
            if (self.view_kind == "stack" and self.proj_combo.get() == "frame"
                    and meta_list and 0 <= self.frame_idx < len(meta_list)):
                self.meta_lbl.configure(text=format_point_meta(meta_list[self.frame_idx]))
            else:
                self.meta_lbl.configure(text="")
            self.ax.set_xlabel(xlabel); self.ax.set_ylabel(ylabel)
            if marginal:
                self.ax_top.set_title(ttl, fontsize=10)
            else:
                self.ax.set_title(ttl, fontsize=10)
            try:
                bgs = getattr(self, "_bg_status", "")
                self.coord_lbl.configure(
                    text=(note + "   |   " + bgs) if bgs else note)
            except Exception:
                pass

            if cross_on:
                self.rect = None                    # crosshair mode: no ROI drag
                if self._cross_xy is not None:
                    cx, cy = self._cross_xy
                    col = int(np.clip(np.searchsorted(xv, cx), 0, data.shape[1] - 1))
                    row = int(np.clip(np.searchsorted(yv, cy), 0, data.shape[0] - 1))
                    w = self._lineout_width()
                    r0 = max(0, row - w // 2); r1 = min(data.shape[0], row + w // 2 + 1)
                    c0b = max(0, col - w // 2); c1b = min(data.shape[1], col + w // 2 + 1)
                    xcut = np.nanmean(data[r0:r1, :], axis=0)   # avg over w rows
                    ycut = np.nanmean(data[:, c0b:c1b], axis=1)  # avg over w cols
                    self._cross_cuts = {
                        "x": (np.asarray(xv, float), np.asarray(xcut, float)),
                        "y": (np.asarray(yv, float), np.asarray(ycut, float)),
                        "row": row, "col": col, "xlabel": xlabel,
                        "ylabel": ylabel, "pos": (cx, cy)}
                    self.ax.axvline(cx, color="cyan", lw=0.9)
                    self.ax.axhline(cy, color="cyan", lw=0.9)
                    if w > 1:
                        for rr in (r0, min(r1 - 1, data.shape[0] - 1)):
                            self.ax.axhline(yv[rr], color="cyan", lw=0.5, ls=":")
                        for cc in (c0b, min(c1b - 1, data.shape[1] - 1)):
                            self.ax.axvline(xv[cc], color="cyan", lw=0.5, ls=":")
                    if self.ax_top is not None:
                        self.ax_top.plot(xv, xcut, "-", lw=1.1, color="tab:blue")
                        self.ax_top.set_ylabel("counts", fontsize=8)
                        self.ax_top.tick_params(labelbottom=False, labelsize=8)
                        self.ax_top.grid(alpha=0.3)
                        fx = self._cross_fit_x
                        if fx and fx.get("ok") and self._fit_matches(fx, xv):
                            self.ax_top.plot(fx["x_dense"], fx["y_dense"], "-k",
                                             lw=1.3, label=f"fit R\u00b2={fx['r2']:.3f}")
                            self.ax_top.legend(fontsize=7, loc="best")
                    if self.ax_right is not None:
                        self.ax_right.plot(ycut, yv, "-", lw=1.1, color="tab:red")
                        self.ax_right.set_xlabel("counts", fontsize=8)
                        self.ax_right.tick_params(labelleft=False, labelsize=8)
                        self.ax_right.grid(alpha=0.3)
                        fy = self._cross_fit_y
                        if fy and fy.get("ok") and self._fit_matches(fy, yv):
                            self.ax_right.plot(fy["y_dense"], fy["x_dense"], "-k",
                                               lw=1.3, label=f"fit R\u00b2={fy['r2']:.3f}")
                            self.ax_right.legend(fontsize=7, loc="best")
            else:
                self.rect = _make_rect(self.ax, self._on_rect)
                if self.sel_box is not None:
                    x0, x1, y0, y1 = self.sel_box
                    self.ax.add_patch(plt_rect(x0, y0, x1 - x0, y1 - y0))

            g = getattr(self, "_img_gauss", None)
            if g and g.get("ok"):
                xl = sorted(self.ax.get_xlim());
                yl = sorted(self.ax.get_ylim())
                if xl[0] <= g["x0"] <= xl[1] and yl[0] <= g["y0"] <= yl[1]:
                    th = np.linspace(0, 2 * np.pi, 120)
                    ct, st = np.cos(g["theta"]), np.sin(g["theta"])
                    hx, hy = g["fwhm_x"] / 2.0, g["fwhm_y"] / 2.0
                    ex = g["x0"] + hx * np.cos(th) * ct - hy * np.sin(th) * st
                    ey = g["y0"] + hx * np.cos(th) * st + hy * np.sin(th) * ct
                    self.ax.plot(ex, ey, "-", color="magenta", lw=1.4,
                                 label=f"2-D Gauss FWHM={g['fwhm']:.3g}")
                    k = self._gauss_k()
                    kx, ky = k * g["fwhm_x"], k * g["fwhm_y"]
                    ex2 = g["x0"] + kx * np.cos(th) * ct - ky * np.sin(th) * st
                    ey2 = g["y0"] + kx * np.cos(th) * st + ky * np.sin(th) * ct
                    self.ax.plot(ex2, ey2, "--", color="magenta", lw=1.0, alpha=0.7,
                                 label=f"bg/smooth r={k:g}\u00d7FWHM")
                    self.ax.plot([g["x0"]], [g["y0"]], "+", color="magenta",
                                 ms=12, mew=1.8)
                    self.ax.legend(fontsize=7, loc="upper right")

            sf = getattr(self, "_sweep_fits", None)   # camera-sweep per-frame fit
            if (sf and self.view_kind == "stack"
                    and self.cur_image_name == "camera_sweep"
                    and 0 <= self.frame_idx < len(sf) and sf[self.frame_idx].get("ok")):
                gg = sf[self.frame_idx]
                th = np.linspace(0, 2 * np.pi, 100)
                hx, hy = gg["fwhm_x"] / 2.0, gg["fwhm_y"] / 2.0
                self.ax.plot(gg["x0"] + hx * np.cos(th), gg["y0"] + hy * np.sin(th),
                             "-", color="magenta", lw=1.4)
                self.ax.plot([gg["x0"]], [gg["y0"]], "+", color="magenta",
                             ms=12, mew=1.8)

            if not marginal:
                self.fig.tight_layout()
            self.canvas.draw_idle()

        def _on_rect(self, eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return
            self.sel_box = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
            c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
            unit = (self.img_xlabel.split("(")[-1].rstrip(")")
                    if "(" in self.img_xlabel else "")
            self.roi_lbl.configure(
                text=f"x [{self.sel_box[0]:.4g}, {self.sel_box[1]:.4g}]  "
                     f"y [{self.sel_box[2]:.4g}, {self.sel_box[3]:.4g}] {unit}  "
                     f"({c1 - c0}×{r1 - r0} px)")

        def clear_roi(self):
            self.sel_box = None
            self.roi_lbl.configure(text="(drag a box on the image)")
            if self.view_kind in ("image", "stack"):
                self.redraw_image()

        def crop_roi(self):
            if self.view_kind not in ("image", "stack") or self.sel_box is None:
                messagebox.showinfo("Crop", "Drag a box on the image first.")
                return
            data = self._disp_data if self._disp_data is not None \
                else self._current_image_data()
            c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
            self.show_image(data[r0:r1, c0:c1],
                            title=getattr(self, "title", "") + " (cropped)")

        def roi_profile(self, axis):
            if self.view_kind not in ("image", "stack"):
                return
            data = self._disp_data if self._disp_data is not None \
                else self._current_image_data()
            if self.sel_box is not None:
                c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
            else:
                r0, r1, c0, c1 = 0, data.shape[0], 0, data.shape[1]
            sub = data[r0:r1, c0:c1]
            xv = self.img_x[c0:c1] if self.img_x is not None else np.arange(c0, c1)
            yv = self.img_y[r0:r1] if self.img_y is not None else np.arange(r0, r1)
            if axis == "x":
                self.set_1d(xv, np.nanmean(sub, axis=0), xlabel=self.img_xlabel,
                            ylabel="mean over ROI rows", title="X profile (line-out)")
            else:
                self.set_1d(yv, np.nanmean(sub, axis=1), xlabel=self.img_ylabel,
                            ylabel="mean over ROI cols", title="Y profile (line-out)")

        def roi_stats(self):
            if self.view_kind not in ("image", "stack"):
                return
            data = self._disp_data if self._disp_data is not None \
                else self._current_image_data()
            if self.sel_box is not None:
                c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
                sub = data[r0:r1, c0:c1]; scope = "ROI"
            else:
                sub = data; scope = "whole image"
            v = sub[np.isfinite(sub)]
            if v.size == 0:
                messagebox.showinfo("Stats", "ROI is empty"); return
            msg = (f"pixels     {v.size}\n"
                   f"mean ± std {np.mean(v):.6g} ± {np.std(v):.4g}\n"
                   f"min / max  {np.min(v):.6g} / {np.max(v):.6g}\n"
                   f"sum        {np.sum(v):.6g}")
            messagebox.showinfo(f"Stats — {scope}", msg)

        # -------------------------------------------------------- crosshair --
        def fit_image_gaussian(self):
            """Fit a 2-D (circular) Gaussian to the ROI, or the whole image if no
            ROI is drawn, and mark the centre + FWHM ellipse on the plot."""
            if self.view_kind not in ("image", "stack"):
                messagebox.showinfo("2-D Gaussian", "Open an image first (confocal "
                                                    "scan, camera frame, or an all_counts map).")
                return
            data = self._disp_data if self._disp_data is not None \
                else self._current_image_data()
            if data is None:
                return
            xv = self.img_x if self.img_x is not None else np.arange(data.shape[1])
            yv = self.img_y if self.img_y is not None else np.arange(data.shape[0])
            if self.sel_box is not None:
                c0, c1, r0, r1 = self._box_to_pixels(self.sel_box)
                sub, sx, sy, scope = data[r0:r1, c0:c1], xv[c0:c1], yv[r0:r1], "ROI"
            else:
                sub, sx, sy, scope = data, xv, yv, "whole image"
            # the displayed image is already monomial-subtracted (see the Image
            # tab checkbox), so just fit it directly.
            res = fit_gaussian_2d(sub, sx, sy, circular=True)  # circular=False -> elliptical

            if not res.get("ok"):
                self._img_gauss = None
                self.redraw_image()
                messagebox.showinfo("2-D Gaussian",
                                    "Fit failed: " + res.get("error", "unknown"))
                return
            self._img_gauss = res
            u = (self.img_xlabel.split("(")[-1].rstrip(")")
                 if "(" in self.img_xlabel else "px")
            self.redraw_image()
            messagebox.showinfo(
                f"2-D Gaussian fit \u2014 {scope}",
                f"centre x0 = {res['x0']:.6g} {u}\n"
                f"centre y0 = {res['y0']:.6g} {u}\n"
                f"sigma     = {res['sigma']:.6g} {u}\n"
                f"FWHM      = {res['fwhm']:.6g} {u}\n"
                f"amplitude = {res['amp']:.6g} counts\n"
                f"background = {res['bg']:.6g} counts\n"
                f"R\u00b2        = {res['r2']:.4f}")

        def _toggle_crosshair(self):
            if not self.crosshair_var.get():
                self._cross_xy = None
                self._cross_cuts = None
            if self.view_kind in ("image", "stack"):
                self.redraw_image()

        def _on_canvas_click(self, event):
            if (getattr(self, "ax", None) is None or event.inaxes is not self.ax
                    or self.view_kind not in ("image", "stack")):
                return
            if not (getattr(self, "crosshair_var", None) and self.crosshair_var.get()):
                return
            if event.xdata is None or event.ydata is None:
                return
            if event.button == 3:                   # right-click -> fit menu
                if self._cross_cuts is not None:
                    self._crosshair_menu()
                return
            self._place_crosshair(float(event.xdata), float(event.ydata))

        def _place_crosshair(self, x, y):
            data = self._disp_data
            if data is None:
                return
            xv = self.img_x if self.img_x is not None else np.arange(data.shape[1])
            yv = self.img_y if self.img_y is not None else np.arange(data.shape[0])
            col = int(np.clip(np.searchsorted(xv, x), 0, data.shape[1] - 1))
            row = int(np.clip(np.searchsorted(yv, y), 0, data.shape[0] - 1))
            self._cross_xy = (float(xv[col]), float(yv[row]))
            self._cross_fit_x = self._cross_fit_y = None   # fits are position-specific
            try:
                self.cross_fit_lbl.configure(text="")
            except Exception:
                pass
            self.redraw_image()      # cuts + marginal line-outs drawn there

        def _crosshair_menu(self):
            m = tk.Menu(self.root, tearoff=0)
            m.add_command(label="Fit counts vs x  (horizontal cut)",
                          command=lambda: self.send_cut_to_fit("x"))
            m.add_command(label="Fit counts vs y  (vertical cut)",
                          command=lambda: self.send_cut_to_fit("y"))
            try:
                px, py = self.canvas.get_tk_widget().winfo_pointerxy()
                m.tk_popup(px, py)
            finally:
                m.grab_release()

        def send_cut_to_fit(self, which):
            if not self._cross_cuts:
                return
            c = self._cross_cuts
            xarr, yarr = c[which]
            if which == "x":
                xl = c["xlabel"]; ttl = f"x line-out  (y = {c['pos'][1]:.4g})"
            else:
                xl = c["ylabel"]; ttl = f"y line-out  (x = {c['pos'][0]:.4g})"
            self.set_1d(np.asarray(xarr, float), np.asarray(yarr, float),
                        xlabel=xl, ylabel="counts", title=ttl)
            try:
                self.nb.select(1)                   # Fit tab
            except Exception:
                pass

        def _fit_matches(self, res, axis_vec):
            """True if a stored fit was computed for the axis currently shown
            (guards against stale overlays after transpose / axis-source change)."""
            try:
                xd = np.asarray(res["x_dense"], float)
                tol = 1e-6 + 0.02 * (abs(np.ptp(axis_vec)) + 1e-9)
                return (abs(xd.min() - np.min(axis_vec)) <= tol
                        and abs(xd.max() - np.max(axis_vec)) <= tol)
            except Exception:
                return False

        def _fit_cut(self, which):
            """Fit the current counts-vs-x or counts-vs-y line-out with the chosen
            model and overlay it on the matching marginal panel."""
            if not self._cross_cuts:
                messagebox.showinfo("Fit line-out",
                                    "Turn on the crosshair and click the image to "
                                    "place it first \u2014 then the line-outs can be fit.")
                return
            xarr, yarr = self._cross_cuts[which]
            xarr = np.asarray(xarr, float)
            yarr = np.asarray(yarr, float)
            name = self.cross_model.get()
            res = fit_model(name, xarr, yarr)
            if not res.get("ok"):
                self.cross_fit_lbl.configure(
                    text=f"{which}-fit failed: {res.get('error', 'no convergence')}")
                messagebox.showinfo("Fit line-out",
                                    "Fit did not succeed:\n"
                                    + res.get("error", "unknown error")
                                    + "\n\nTry a different model.")
                return
            if which == "x":
                self._cross_fit_x = res
            else:
                self._cross_fit_y = res
            parts = [f"{which}: {name}", f"R\u00b2={res['r2']:.4f}"]
            for nm, val in zip(res.get("pnames", []), res.get("params", [])):
                parts.append(f"{nm}={val:.4g}")
                if nm == "sigma":  # Gaussian width
                    parts.append(f"FWHM={2.3548 * abs(val):.4g}")
                elif nm == "hw" or (nm[:1] == "w" and nm[1:].isdigit()):
                    parts.append(f"FWHM={2 * abs(val):.4g}")  # Lorentzian HWHM -> FWHM
            self.cross_fit_lbl.configure(text="   ".join(parts))
            self.redraw_image()

        def _clear_cut_fits(self):
            self._cross_fit_x = None
            self._cross_fit_y = None
            self._img_gauss = None          # also clear the 2-D Gaussian overlay
            self.cross_fit_lbl.configure(text="")
            if self.view_kind in ("image", "stack"):
                self.redraw_image()

        # ------------------------------------------------------ movie player --
        def on_frame_scale(self, val):
            if self.view_kind != "stack" or getattr(self, "_scale_busy", False):
                return
            idx = int(round(float(val)))
            idx = max(0, min(idx, self.cur_frames.shape[0] - 1))
            if idx != self.frame_idx or self.proj_combo.get() != "frame":
                self.frame_idx = idx
                if self.proj_combo.get() != "frame":
                    self.proj_combo.set("frame")
                self.redraw_image()
                self._sync_sweep_to_frame()

        def step_frame(self, d):
            if self.view_kind != "stack":
                return
            n = self.cur_frames.shape[0]
            self.frame_idx = (self.frame_idx + d) % n
            self.proj_combo.set("frame")
            self._scale_busy = True                 # don't let .set() re-trigger
            try:
                self.frame_scale.set(self.frame_idx)
            finally:
                self._scale_busy = False
            self.redraw_image()
            self._sync_sweep_to_frame()

        def toggle_play(self):
            if self.view_kind != "stack":
                return
            self.playing = not self.playing
            self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")
            if self.playing:
                self.proj_combo.set("frame")
                self._advance()

        def _advance(self):
            if not self.playing or self.view_kind != "stack":
                return
            self.step_frame(1)
            delay = int(1000 / max(1, self.fps_var.get()))
            self.root.after(delay, self._advance)

        def _stop_play(self):
            self.playing = False
            try:
                self.play_btn.configure(text="▶ Play")
            except Exception:
                pass

        # -------------------------------------------- sweep coordinate picker --
        def _refresh_sweep_picker(self):
            """Show one dropdown per swept axis when the current stack is a
            ParameterSweep file whose points line up with the shown frames; hide
            the row otherwise. Safe to call after every view change."""
            sw = self._sweep
            active = (sw is not None and self.view_kind == "stack"
                      and self.cur_frames is not None
                      and len(sw["frame_coords"]) == self.cur_frames.shape[0])
            for child in self.sweep_row.winfo_children():
                child.destroy()
            self._sweep_combos = {}
            if not active:
                self.sweep_row.pack_forget()
                return
            ttk.Label(self.sweep_row, text="pick point:").pack(side="left")
            for ax in sw["axis_paths"]:
                ttk.Label(self.sweep_row, text="   " + ax + ":").pack(side="left")
                cb = ttk.Combobox(self.sweep_row, state="readonly", width=11,
                                  values=[sweep_fmt(v) for v in sw["values"][ax]])
                cb.pack(side="left")
                cb.bind("<<ComboboxSelected>>", self._on_sweep_pick)
                self._sweep_combos[ax] = cb
            self.sweep_pick_lbl = ttk.Label(self.sweep_row, text="",
                                            foreground="#036")
            self.sweep_pick_lbl.pack(side="left", padx=(10, 0))
            self.sweep_row.pack(fill="x", pady=(4, 0))
            self._sync_sweep_to_frame()

        def _on_sweep_pick(self, *_):
            sw = self._sweep
            if not sw or not self._sweep_combos:
                return
            try:
                sel = tuple(sw["values"][ax][self._sweep_combos[ax].current()]
                            for ax in sw["axis_paths"])
            except Exception:
                return
            idx = sw["lookup"].get(sel)
            if idx is None:                       # combination never measured
                if hasattr(self, "sweep_pick_lbl"):
                    self.sweep_pick_lbl.configure(text="(not measured)")
                return
            self.frame_idx = idx                  # drive the existing player
            self.proj_combo.set("frame")
            self._scale_busy = True               # don't let .set() re-trigger
            try:
                self.frame_scale.set(idx)
            finally:
                self._scale_busy = False
            self.redraw_image()
            self._sync_sweep_to_frame()

        def _sync_sweep_to_frame(self):
            """Point the dropdowns at whichever frame is currently shown, so they
            track the slider / play / step buttons."""
            sw = self._sweep
            if not sw or not self._sweep_combos:
                return
            i = self.frame_idx
            if not (0 <= i < len(sw["frame_coords"])):
                return
            coords = sw["frame_coords"][i]
            self._sweep_busy = True
            try:
                for ax in sw["axis_paths"]:
                    try:
                        self._sweep_combos[ax].current(sw["values"][ax].index(coords[ax]))
                    except Exception:
                        pass
                if hasattr(self, "sweep_pick_lbl"):
                    self.sweep_pick_lbl.configure(
                        text="frame %d/%d" % (i + 1, len(sw["frame_coords"])))
            finally:
                self._sweep_busy = False

        # -------------------------------------------------- sequence preview --
        def show_sequence(self, text):
            parsed = parse_sequence_text(text)
            if not parsed or not parsed.get("pulses"):
                return False
            self._stop_play()
            self.seq_parsed = parsed
            self.view_kind = "sequence"
            steps = parsed["variables"][0]["steps"] if parsed["variables"] else 1
            self.seq_scale.configure(from_=0, to=max(0, steps - 1))
            self.seq_index = steps // 2 if steps > 1 else 0
            self._seq_busy = True
            try:
                self.seq_scale.set(self.seq_index)
            finally:
                self._seq_busy = False
            self.seq_row_state(steps > 1)
            try:
                self.nb.select(self._seq_tab_index)
            except Exception:
                pass
            self.draw_sequence()
            return True

        def seq_row_state(self, on):
            for child in self.seq_row.winfo_children():
                try:
                    child.state(["!disabled"] if on else ["disabled"])
                except Exception:
                    try:
                        child.configure(state="normal" if on else "disabled")
                    except Exception:
                        pass

        def on_seq_scale(self, val):
            if self.view_kind != "sequence" or getattr(self, "_seq_busy", False):
                return
            idx = int(round(float(val)))
            if idx != self.seq_index:
                self.seq_index = idx
                self.draw_sequence()

        def step_seq(self, d):
            if self.view_kind != "sequence" or not self.seq_parsed:
                return
            steps = (self.seq_parsed["variables"][0]["steps"]
                     if self.seq_parsed["variables"] else 1)
            self.seq_index = int(np.clip(self.seq_index + d, 0, max(0, steps - 1)))
            self._seq_busy = True
            try:
                self.seq_scale.set(self.seq_index)
            finally:
                self._seq_busy = False
            self.draw_sequence()

        # a small fixed palette so each channel keeps one colour
        _SEQ_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                       "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]

        def draw_sequence(self):
            parsed = getattr(self, "seq_parsed", None)
            if not parsed:
                return
            self._new_axes()
            ax = self.ax
            geo = sequence_pulse_geometry(parsed, self.seq_index)
            channels = geo["channels"] or [0]
            lane = {ch: i for i, ch in enumerate(channels)}
            ncol = len(self._SEQ_COLORS)
            t_max = geo["t_max_ns"] or 1.0

            def draw_one(g, alpha, label=True):
                for p in g["pulses"]:
                    row = lane.get(p["channel"], 0)
                    col = self._SEQ_COLORS[row % ncol]
                    a = p["amp"] if np.isfinite(p["amp"]) else 1.0
                    h = 0.42 * (a if abs(a) <= 1 else np.sign(a))  # clip to lane
                    if abs(h) < 0.03:
                        h = 0.03 * (1 if a >= 0 else -1)
                    ax.add_patch(_rect(p["start_ns"], row, max(p["width_ns"], 0.5),
                                       h, col, alpha))
                    if label:
                        ax.text(p["start_ns"] + max(p["width_ns"], 0.5) / 2,
                                row + (0.5 if h >= 0 else -0.5), p["name"],
                                ha="center", va="bottom" if h >= 0 else "top",
                                fontsize=7, rotation=0, color=col, clip_on=True)

            if self.seq_overlay_var.get() and geo["steps"] > 1:
                # faint envelope of every scan point, then the current one solid
                for i in range(geo["steps"]):
                    draw_one(sequence_pulse_geometry(parsed, i), 0.10, label=False)
            draw_one(geo, 0.85, label=True)

            for r in range(len(channels)):
                ax.axhline(r, color="#bbb", lw=0.6, zorder=0)
            ax.set_yticks(range(len(channels)))
            ax.set_yticklabels([f"ch {c}" for c in channels])
            ax.set_ylim(-0.8, len(channels) - 0.2)
            ax.set_xlim(-0.02 * t_max, 1.03 * t_max)
            ax.set_xlabel("time (ns)")
            title = f"sequence: {parsed.get('name') or '?'}"
            if parsed.get("type"):
                title += f"  ({parsed['type']})"
            ax.set_title(title)
            # scan-point label
            if geo["var_name"] is not None:
                u = f" {geo['var_unit']}" if geo["var_unit"] else ""
                msg = (f"{geo['var_name']} = {geo['var_value']:.4g}{u}   "
                       f"(scan point {geo['scan_index'] + 1}/{geo['steps']})")
            else:
                msg = "no swept variable (single fixed sequence)"
            try:
                self.seq_lbl.configure(text=msg)
            except Exception:
                pass
            self.fig.tight_layout()
            self.canvas.draw_idle()

        # ------------------------------------------------------------ utils --
        def _save_csv(self, arr, header="", suffix=".csv"):
            base = os.path.splitext(self.filepath or "data")[0]
            default = os.path.basename(base) + suffix + ".csv"
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", initialfile=default,
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
            if not path:
                return
            np.savetxt(path, arr, delimiter=",", header=header, comments="")
            self._set_status(f"saved {os.path.basename(path)}")

        def _set_status(self, msg):
            self.status.configure(text=msg)

    # helper: matplotlib Rectangle patch (imported lazily)
    def plt_rect(x, y, w, h):
        from matplotlib.patches import Rectangle
        return Rectangle((x, y), w, h, fill=False, edgecolor="white", lw=1.3)

    # helper: filled coloured rectangle for the pulse-timing diagram
    def _rect(x, y, w, h, color, alpha):
        from matplotlib.patches import Rectangle
        return Rectangle((x, y), w, h, facecolor=color, edgecolor=color,
                         alpha=alpha, lw=0.8)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
