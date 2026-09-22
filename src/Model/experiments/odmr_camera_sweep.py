"""
ODMR Camera Sweep Experiment (widefield / camera-based ODMR)

This is the camera analog of ODMRSweepContinuousExperiment. Instead of the ADwin
counting photons on a single-mode detector, it steps the SG384 across a frequency
range and takes a CAMERA IMAGE at every frequency point. Each frame is reduced to a
single number over a configurable ROI (this scalar is the "counts" analog that drives
the ODMR spectrum, fitting and optimization). It supports:

  * stepped frequency sweep (set freq -> settle -> capture), no ADwin / no FM sync
  * forward + reverse (bidirectional) sweeps
  * term-by-term averaging over multiple full runs
  * confocal re-optimization BETWEEN runs (never mid-run), driven by the camera
    signal instead of the ADwin counter (same parabola / hill-climb optimizer and
    "retention net" as the ADwin ODMR experiment)
  * a one-time exposure auto-set before the sweep (then LOCKED, so ODMR contrast
    stays comparable across frequencies -- unlike the per-point autotune used by the
    spatial stage scan)
  * Lorentzian fitting / peak finding on the averaged spectrum
  * live pyqtgraph plotting (1D spectrum + a live camera-image pane, or a per-sweep
    2D waterfall) and HDF5 saving of the spectrum, per-sweep arrays, reference
    frames and (optionally) the full per-frequency mean-image cube

Why stepped (not phase-continuous like the ADwin version)?
    A camera exposure is milliseconds-to-seconds, so the fast FM/DAC-synchronized
    continuous sweep the ADwin needs buys us nothing here. Setting a discrete
    frequency, letting it settle, and taking a picture is simpler and correct.

Device wiring:
    The camera is looked up in self.devices under the role 'camera'. If your
    config.json registers the Roper under a different name, either (a) add an alias
    'camera' in config.json, or (b) rename the value in _DEVICES below. As a
    convenience, __init__ will also fall back to scanning self.devices for any
    Roper_Cascade_Camera instance if the 'camera' role is missing.

Adapted from ODMRSweepContinuousExperiment (Gurudev Dutt <gdutt@pitt.edu>).
License: GPL v2
"""
import datetime
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
import time
import logging

from PyQt5.QtCore import Qt

from src.core.experiment import Experiment
from src.core.parameter import Parameter
from src.core.struct_hdf5 import MyStruct, save_data, StructArray
from src.Controller import Roper_Cascade_Camera

class ODMRCameraSweepExperiment(Experiment):
    """
    Widefield / camera-based ODMR experiment.

    At each frequency the SG384 carrier is set (no modulation), the system settles,
    and one camera frame is captured. Each frame is reduced to a single ROI value
    that plays the role the ADwin photon count plays in ODMRSweepContinuousExperiment,
    so the averaging, optimization, fitting, plotting and saving all carry over.

    Parameters (see _DEFAULT_SETTINGS for the full tree):
        frequency_range: [start, stop, num_points] stepped sweep in Hz
        microwave:       enable/power/settle
        camera:          exposure/gain, capture probe, ROI, one-time auto-exposure,
                         whether to save the full per-frequency image cube
        acquisition:     averaging + between-run optimization + bidirectional

    Returns:
        frequencies, counts_forward/reverse/averaged (ROI signal vs frequency),
        per-sweep arrays, reference frames, optional mean-image cube,
        fit_parameters and resonance_frequencies.
    """

    # Reused from the spatial camera scan (positioning_stages_GUI) so the one-time
    # exposure auto-set behaves like the tool you already trust.
    AUTOTUNE_TARGET_MIN = 15000
    AUTOTUNE_TARGET_MAX = 39000
    AUTOTUNE_SATURATION = 40000
    # (inttime_ns, gain) pairs; unity gain unless you enable gain autotune.
    ROPER_EXPOSURE_SEQUENCE = [10, 50, 100, 150, 200, 300, 500,
                               1000, 2000, 5000, 10000]

    _DEFAULT_SETTINGS = [
        Parameter('frequency_range', [
            Parameter('start', 2.7e9, float, 'Start frequency in Hz', units='Hz'),
            Parameter('stop', 3.0e9, float, 'Stop frequency in Hz', units='Hz'),
            Parameter('num_points', 101, int,
                      'Number of frequency points in the stepped sweep (>= 2)'),
        ]),
        Parameter('microwave', [
            Parameter('enable', True, bool,
                      'T/F to enable MW output. DO NOT enable if the amp is not powered!'),
            Parameter('power', 12.0, float, 'Microwave power in dBm', units='dBm'),
            Parameter('settle_time', 0.02, float,
                      'Settle time after each frequency step, before the exposure starts', units='s'),
            Parameter('MW off after experiment?', True, bool,
                      'Turn the MW off after the experiment finishes'),
        ]),
        Parameter('constant_frequency', [
            Parameter('disable sweep and enable constant frequency', False, bool,
                      'Enable to hold a single frequency (e.g. for a background/reference image). '
                      'The x-axis stays the swept range even though it is not meaningful in this mode; '
                      'flag it in analysis when you use the saved file.'),
            Parameter('frequency', 2.87e9, float,
                      'Constant frequency to hold when the sweep is disabled', units='Hz'),
        ]),
        Parameter('GREEN_LASER', [
            Parameter('control_laser', True, bool,
                      'Drive the green laser (proteus + filter wheel). Turn off if you run the '
                      'laser manually or those devices are absent.'),
            Parameter('Filter Wheel OD', 0, [0, 0.5, 1, 2, 3, 4], 'Filter Wheel OD'),
            Parameter('Laser Control', 0.8, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                      'Laser Control voltage (proteus channel 4)'),
            Parameter('Laser off after experiment?', True, bool,
                      'Turn the laser off after the experiment finishes'),
        ]),
        Parameter('camera', [
            Parameter('inttime', 3000.0, float,
                      'Exposure (integration) time in milliseconds. Held FIXED for the whole '
                      'sweep so contrast is comparable across frequencies.', units='us'),
            Parameter('gain', 1.0, float, 'Camera gain (1 = no gain)'),
            Parameter('capture_probe', 'image', ['image', 'imagefast_int'],
                      "'image' = fresh blocking acquisition at the set exposure (recommended, "
                      "correct per point). 'imagefast_int' = grab the latest streamed frame "
                      "(faster but may be stale / requires live view)."),
            Parameter('auto_set_exposure_once', True, bool,
                      'Auto-tune the exposure ONCE before the sweep (on the reference frequency '
                      'below), then lock it. This keeps the brightest pixel in a good range '
                      'without rescaling per point.'),
            Parameter('autotune_reference', 'start', ['start', 'center', 'stop'],
                      'Which frequency to sit on while auto-setting the one-time exposure'),
            Parameter('roi', [
                Parameter('mode', 'full', ['full', 'center_box', 'explicit'],
                          'Region used to reduce each frame to one number for the spectrum'),
                Parameter('box_size', 100, int, 'explicit mode: ROI left edge (column)', units='px'),
                Parameter('x0', 244, int, 'explicit mode: ROI top edge (row)', units='px'),
                Parameter('y0', 230, int, 'explicit mode: ROI top edge (row)', units='px'),
                Parameter('width', 100, int, 'explicit mode: ROI width', units='px'),
                Parameter('height', 100, int, 'explicit mode: ROI height', units='px'),
                Parameter('reduce', 'mean', ['mean', 'sum'],
                          'Reduce ROI pixels to one number by mean or sum'),
            ]),
            Parameter('save_full_image_cube', False, bool,
                      'Save the per-frequency mean image as ONE 3-D array (num_points x H x W, '
                      'float32) under data/image_cube_mean. Compact + easy to slice; redundant '
                      'with save_per_frequency_images below.'),
            Parameter('save_per_frequency_images', True, bool,
                      'Save one sweep-averaged image PER FREQUENCY into the HDF5, named '
                      'image_<idx>_<freq> (e.g. image_1_2.800GHz). Each is a small sub-struct with '
                      'fields camera_image (2-D float32) + frequency_hz + frequency_ghz + index, so '
                      'your analyzer can iterate image_* and read the exact frequency. WARNING: this '
                      'stores num_points x H x W floats -- can be hundreds of MB. Off = do not save '
                      'individual images.'),
            Parameter('image_name_units', 'GHz', ['GHz', 'MHz'],
                      'Frequency units used in the per-frequency image names'),
            Parameter('image_name_decimal_char', '.', ['.', 'p'],
                      "Decimal-point character in image names: '.' -> image_1_2.800GHz (as you asked); "
                      "switch to 'p' -> image_1_2p800GHz only if your HDF5 saver rejects '.' in names"),
            Parameter('save_raw_frames', False, bool,
                      'Save EVERY captured frame (all runs, both directions), unaveraged, under '
                      'data/raw_frames/ grouped as sweep_<k>/<forward|reverse>/image_<idx>_<freq>. '
                      'This is the full dataset: averages x directions x num_points frames -- '
                      'potentially SEVERAL GB. Frames are held in RAM until the run ends, then saved '
                      'with the file (guarded by raw_frames_max_gb below).'),
            Parameter('raw_frames_dtype', 'float32', ['float32', 'uint16'],
                      "Storage dtype for raw frames. 'float32' is lossless for camera counts; "
                      "'uint16' halves the size but clips to 0..65535 (fine for a 16-bit camera)."),
            Parameter('raw_frames_max_gb', 8.0, float,
                      'Safety cap: if the estimated raw-frame size exceeds this many GB, raw-frame '
                      'saving is skipped for the run (with a warning) so it cannot exhaust memory. '
                      'Raise it if you really want a bigger dump.', units='GB'),
        ]),
        Parameter('acquisition', [
            Parameter('averaging', [
                Parameter('averages', 5, int,
                          'Number of full runs to average term-by-term (1 = single run)'),
                Parameter('optimize_between_runs', True, bool,
                          'Re-optimize confocal position between runs (never in the middle of a run)'),
                Parameter('optimization_mode', 'convergence', ['convergence', 'fit'],
                          'Re-optimization mode for the confocal position'),
                Parameter('opt_max_steps', 20, int,
                          'convergence mode only: max steps to climb per axis (safety cap so a '
                          'noisy axis cannot walk off forever)'),
                Parameter('opt_improve_frac', 0.02, float,
                          'convergence mode only: a step must raise the signal by more than this '
                          'fraction (0.02 = 2%) to be accepted, so it climbs real signal instead '
                          'of chasing shot noise'),
                Parameter('opt_xstep', 0.1, float, 'Optimization x step', units='um'),
                Parameter('opt_ystep', 0.1, float, 'Optimization y step', units='um'),
                Parameter('opt_zstep', 0.1, float, 'Optimization z step', units='um'),
                Parameter('opt_settle_time', 0.3, float, 'Settle time after each stage move', units='s'),
                Parameter('opt_damping_ratio', 0.9, float,
                          'Move to x + damping*(best - x) instead of the raw parabola vertex'),
                Parameter('min_retention_ratio', 0.5, float,
                          'If post-optimization signal < min_retention_ratio * best-known signal, '
                          'snap back toward the best-known position'),
                Parameter('acceptable_signal', 1e15, float,
                          'Absolute ceiling (in ROI signal units) above which the best-known '
                          'baseline is NOT updated (rejects spikes). Set a little above your real '
                          'maximum ROI signal; the default is effectively "no ceiling".'),
                Parameter('acceptable_signal_ratio', 1.1, float, 'See acceptable_signal'),
                Parameter('min_reoptimize_ratio', 0.5, float,
                          'reoptimize_only_if_needed=True: skip the x/y/z pass when current signal '
                          '>= min_reoptimize_ratio * best-known signal'),
                Parameter('reoptimize_only_if_needed', True, bool,
                          'If True, skip optimization before a run when the signal is still healthy '
                          '(fewer stage moves = less drift). If False, optimize before every run '
                          '2..N and rely on the retention net.'),
            ]),
            Parameter('settle_time', 0.05, float, 'Settle time between sweeps', units='s'),
            Parameter('bidirectional', True, bool,
                      'Enable bidirectional sweeps (capture on both forward and reverse passes)'),
        ]),
        Parameter('magnetic_field', [
            Parameter('enabled', False, bool, 'Enable magnetic field'),
            Parameter('strength', 0.0, float, 'Magnetic field strength in Gauss', units='G'),
            Parameter('direction', [0.0, 0.0, 1.0], list, 'Magnetic field direction [x, y, z]'),
        ]),
        Parameter('analysis', [
            Parameter('auto_fit', True, bool, 'Automatically fit resonances'),
            Parameter('smoothing', True, bool, 'Apply smoothing to data'),
            Parameter('smooth_window', 5, int, 'Smoothing window size'),
            Parameter('background_subtraction', True, bool, 'Subtract background'),
        ]),
        Parameter('2D_Plot', False, bool,
                  'Plot every individual sweep as a 2D map (no averaging) instead of the averaged '
                  '1D spectrum'),
        Parameter('filename', 'ODMR_Camera_Sweep', str, 'File name to be saved'),
        Parameter('sample', '', str, 'Sample name to be saved with the data'),
        Parameter('Magnet ON', False),
    ]

    # Roles resolved from config.json. Same rig as the ADwin ODMR experiment, minus
    # 'adwin', plus 'camera'. proteus / filter_wheel / nanodrive are used defensively
    # (see __init__) so the experiment still runs on a subset rig.
    _DEVICES = {
        'microwave': 'sg384',
        'nanodrive': 'nanodrive',
        'proteus': 'proteus',
        'filter_wheel': 'filter_wheel',
    }

    _EXPERIMENTS = {}

    def __init__(self, devices, experiments=None, name=None, settings=None,
                 log_function=None, data_path=None):
        super().__init__(name, settings, devices, experiments, log_function, data_path)
        self.logger = logging.getLogger(__name__)

        # Spectrum data (ROI signal vs frequency) -- the camera analog of the ADwin counts.
        self.frequencies = None
        self.counts_forward = None
        self.counts_reverse = None
        self.counts_averaged = None

        # Per-sweep raw scalar arrays (averages x num_points).
        self.all_forward = None
        self.all_reverse = None

        # Optional per-frequency mean-image cube + running accumulators.
        self._cube_sum = None      # float64 (num_points, H, W)
        self._cube_cnt = None      # int (num_points,)
        self._frame_shape = None   # (H, W)

        # Frames kept for display/saving (single frames; small).
        self._last_frame = None
        self._ref_frame_start = None
        self._ref_frame_onres = None
        self._ref_frame_end = None
        self._onres_min_signal = np.inf
        self._img_name_dec = None  # cached decimal places for per-frequency image names

        # Raw-frame accumulation (only when save_raw_frames is on). Each element is a dict
        # {frame, sweep, direction, freq_index, acq_index, frequency_hz}. Held in RAM until save.
        self._raw_frames = []
        self._raw_frames_enabled = None   # resolved in setup()
        self._raw_budget_checked = False

        # Analysis results.
        self.fit_parameters = None
        self.resonance_frequencies = None
        self.fit_quality = None

        # Confocal optimizer state (seeded on the first optimization of a run).
        self._opt_best_signal = None
        self._opt_best_x = None
        self._opt_best_y = None
        self._opt_best_z = None

        # --- resolve devices ---
        self.microwave = None
        if self.settings['microwave']['enable']:
            self.microwave = self.devices.get('microwave', {}).get('instance')
            if not self.microwave:
                raise ValueError("SG384 microwave generator is required (device role 'microwave').")

        self.camera = self._resolve_camera()
        if self.camera is None:
            available = list(self.devices.keys())
            raise ValueError(
                "No camera found. Add the Roper camera to config.json under the role "
                "'camera' (or update _DEVICES). Available device roles: %s" % available)

        # Optional devices -- used defensively.
        self.nanodrive = self.devices.get('nanodrive', {}).get('instance')
        self.proteus = self.devices.get('proteus', {}).get('instance')
        self.filter_wheel = self.devices.get('filter_wheel', {}).get('instance')

    def _resolve_camera(self):
        """Return the camera instance from the 'camera' role, else any Roper camera
        instance found among the loaded devices (robust to the config key name)."""
        cam = Roper_Cascade_Camera.Roper_Cascade_Camera()
        if cam is not None:
            return cam
        return None

    # ---------------------------------------------------------------- setup
    def setup(self):
        """Set up the experiment and all devices."""
        self._calculate_sweep_parameters()
        self._setup_microwave()
        self._setup_camera()
        if self.nanodrive:
            self._setup_nanodrive()
        self._initialize_data_arrays()

        # Reset raw-frame capture for this run.
        self._raw_frames = []
        self._raw_budget_checked = False
        self._raw_frames_enabled = bool(self.settings['camera']['save_raw_frames'])
        if self._raw_frames_enabled:
            averages = max(1, int(self.settings['acquisition']['averaging']['averages']))
            directions = 2 if self.settings['acquisition']['bidirectional'] else 1
            n_frames = averages * directions * self.num_points
            self.log(f"Raw-frame saving ON: expecting up to {n_frames} frames "
                     f"({averages} runs x {directions} dir x {self.num_points} pts). "
                     f"Size will be checked against the {self.settings['camera']['raw_frames_max_gb']} GB "
                     f"cap once the frame dimensions are known.")

        self.log("ODMR Camera Sweep Experiment setup complete")

    def _calculate_sweep_parameters(self):
        """Build the stepped frequency array and a rough per-run time estimate."""
        start = self.settings['frequency_range']['start']
        stop = self.settings['frequency_range']['stop']
        num_points = int(self.settings['frequency_range']['num_points'])
        if num_points < 2:
            num_points = 2
            self.log("num_points < 2; using 2.")
        self.num_points = num_points
        self.frequencies = np.linspace(start, stop, num_points)
        self._img_name_dec = None  # recompute name precision for this frequency grid

        inttime_s = float(self.settings['camera']['inttime']) * 1e-3
        mw_settle = float(self.settings['microwave']['settle_time'])
        self._per_point_s = inttime_s + mw_settle
        directions = 2 if self.settings['acquisition']['bidirectional'] else 1
        self.sweep_time = self._per_point_s * num_points * directions

        self.log(f"Frequency range: {start/1e9:.4f} - {stop/1e9:.4f} GHz, {num_points} points")
        self.log(f"Step size: {abs(stop - start)/max(1, num_points - 1)/1e6:.3f} MHz")
        self.log(f"~{self._per_point_s*1e3:.1f} ms/point, ~{self.sweep_time:.2f} s per "
                 f"{'bidirectional' if directions == 2 else 'unidirectional'} sweep")

    def _setup_microwave(self):
        """Point the SG384 at discrete frequencies with modulation OFF (a clean carrier)."""
        if not self.settings['microwave']['enable']:
            self.log("Microwave disabled -- capturing without MW.")
            return
        if not self.microwave.is_connected:
            self.microwave.connect()

        self.microwave.set_power(self.settings['microwave']['power'])
        # Stepped sweep => no FM. Make sure no leftover modulation is on.
        try:
            self.microwave.disable_modulation()
        except Exception as e:
            self.log(f"Could not disable modulation (continuing): {e}")

        if self.settings['constant_frequency']['disable sweep and enable constant frequency']:
            self._set_frequency(self.settings['constant_frequency']['frequency'])
            self.log(f"Constant-frequency mode at "
                     f"{self.settings['constant_frequency']['frequency']/1e9:.4f} GHz")
        else:
            self._set_frequency(self.frequencies[0])

        self.microwave.enable_output()
        self.log(f"SG384 output on at {self.settings['microwave']['power']} dBm")

    def _set_frequency(self, hz: float):
        """Set the SG384 carrier; when the sweep is disabled, always hold the constant freq."""
        if not self.settings['microwave']['enable']:
            return
        if self.settings['constant_frequency']['disable sweep and enable constant frequency']:
            hz = self.settings['constant_frequency']['frequency']
        self.microwave.set_frequency(float(hz))

    def _setup_camera(self):
        """Apply exposure/gain, then optionally auto-set the exposure ONCE and lock it."""
        try:
            self.camera.update({'gain': float(self.settings['camera']['gain'])})
            self.camera.update({'inttime': float(self.settings['camera']['inttime'])})
        except Exception as e:
            self.log(f"Could not set camera exposure/gain: {e}")
            raise
        if self.settings['camera']['auto_set_exposure_once']:
            self._auto_set_exposure_once()
        self.log(f"Camera exposure {self.settings['camera']['inttime']:.0f} ms, "
                 f"gain {self.settings['camera']['gain']:.0f}, "
                 f"probe '{self.settings['camera']['capture_probe']}'")

    def _auto_set_exposure_once(self):
        """Sit on the reference frequency and pick the exposure whose brightest pixel lands
        in [TARGET_MIN, TARGET_MAX] (fallback: the largest exposure that does not saturate).
        Sets and LOCKS that exposure for the whole sweep; writes it back into settings so the
        saved metadata reflects what was actually used."""
        ref = self.settings['camera']['autotune_reference']
        ref_hz = {'start': self.frequencies[0],
                  'center': 0.5 * (self.frequencies[0] + self.frequencies[-1]),
                  'stop': self.frequencies[-1]}.get(ref, self.frequencies[0])
        self._set_frequency(ref_hz)
        time.sleep(float(self.settings['microwave']['settle_time']))

        gain = float(self.settings['camera']['gain'])
        seq = list(self.ROPER_EXPOSURE_SEQUENCE)
        # Seed the search near the currently configured exposure.
        cur = float(self.settings['camera']['inttime'])
        idx = int(np.argmin([abs(s - cur) for s in seq]))

        best_it = cur
        best_max = None
        self.log("Auto-setting exposure once (locked afterwards)...")
        for _ in range(2 * len(seq)):
            it = float(seq[idx])
            try:
                self.camera.update({'inttime': it})
                self.camera.update({'gain': gain})
            except Exception as e:
                self.log(f"Exposure set failed at {it} ms: {e}")
                break
            frame = self._capture_frame()
            if frame is None:
                break
            mx = float(np.max(frame))
            best_it, best_max = it, mx
            if mx >= self.AUTOTUNE_SATURATION or mx > self.AUTOTUNE_TARGET_MAX:
                if idx > 0:
                    idx -= 1
                    continue
                break
            elif mx < self.AUTOTUNE_TARGET_MIN:
                if idx < len(seq) - 1:
                    idx += 1
                    continue
                break
            else:
                break  # in the target window

        self.settings['camera'].update({'inttime': best_it})
        try:
            self.camera.update({'inttime': best_it})
            self.camera.update({'gain': gain})
        except Exception:
            pass
        # keep the freshly-measured frame as the "start" reference (a COPY, never a bare
        # reference, so a later in-place capture can't mutate it and it can't leak between runs)
        if self._ref_frame_start is None and best_max is not None and self._last_frame is not None:
            self._ref_frame_start = self._last_frame.copy()
        self.log(f"Locked exposure at {best_it:.0f} ms "
                 f"(max pixel ~{best_max if best_max is not None else float('nan'):.0f})")

    def _setup_nanodrive(self):
        """Connect the nanodrive and log its position (no movement)."""
        try:
            if not self.nanodrive.is_connected:
                self.nanodrive.connect()
        except Exception as e:
            self.log(f"Nanodrive connect issue (continuing): {e}")
        for ax in ('x', 'y', 'z'):
            try:
                self.log(f"Nanodrive {ax} position: {self.nanodrive.get_position(ax)}")
            except Exception as e:
                self.log(f"Could not read nanodrive {ax}: {e}")

    def _initialize_data_arrays(self):
        n = self.num_points
        self.counts_forward = np.zeros(n)
        self.counts_reverse = np.zeros(n)
        self.counts_averaged = np.zeros(n)
        self.fit_parameters = None
        self.resonance_frequencies = None
        self.fit_quality = None
        # Reset reference-frame state at the START of every run. These are otherwise only
        # initialised in __init__, so re-running on the same instance (load once, toggle the
        # laser, run again) would carry a stale reference frame from the previous run into the
        # next file -- that is why laser_on and laser_off shared a byte-identical ref_frame_start.
        self._ref_frame_start = None
        self._ref_frame_onres = None
        self._ref_frame_end = None
        self._onres_min_signal = np.inf
        self._last_frame = None

    # ------------------------------------------------------------ capture / ROI
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture one frame at the currently-set exposure and return it as a 2-D float
        ndarray (raw counts). Stores it as the live/last frame for plotting."""
        probe = self.settings['camera']['capture_probe']
        key = 'imagefast_int' if probe == 'imagefast_int' else 'image'
        try:
            raw = self.camera.read_probes(key)
        except Exception as e:
            self.log(f"Camera capture failed ('{key}'): {e}")
            return None
        frame = self._frame_to_numpy(raw)
        if frame is not None:
            self._last_frame = frame
            if self._frame_shape is None:
                self._frame_shape = frame.shape
        return frame

    @staticmethod
    def _frame_to_numpy(raw) -> Optional[np.ndarray]:
        """Convert a MATLAB double (or list/ndarray) frame to a contiguous 2-D float ndarray.
        MATLAB is column-major, so we reshape with order='F' (matches camera_widget)."""
        if raw is None:
            return None
        try:
            if hasattr(raw, '_data') and hasattr(raw, 'size'):
                arr = np.array(raw._data, dtype=np.float64).reshape(raw.size, order='F')
            else:
                arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Defensive copy. np.asarray / np.ascontiguousarray can hand back the SAME
            # object a driver reuses (e.g. the 'imagefast_int' streamed buffer), which the
            # next capture then overwrites in place. Returning a fresh, contiguous array
            # guarantees every captured frame is independent, so nothing we stash later
            # (reference frames, cube inputs) can alias a buffer that changes underneath us.
            return np.ascontiguousarray(arr, dtype=np.float64).copy()
        except Exception:
            return None

    def _roi_slice(self, shape: Tuple[int, int]) -> Tuple[slice, slice]:
        """Return (row_slice, col_slice) for the configured ROI, clamped to the frame."""
        h, w = int(shape[0]), int(shape[1])
        roi = self.settings['camera']['roi']
        mode = roi['mode']
        if mode == 'full':
            return slice(0, h), slice(0, w)
        if mode == 'center_box':
            n = max(1, int(roi['box_size']))
            bh, bw = min(n, h), min(n, w)
            y0 = max(0, h // 2 - bh // 2)
            x0 = max(0, w // 2 - bw // 2)
            return slice(y0, y0 + bh), slice(x0, x0 + bw)
        # explicit
        x0 = min(max(0, int(roi['x0'])), max(0, w - 1))
        y0 = min(max(0, int(roi['y0'])), max(0, h - 1))
        bw = min(max(1, int(roi['width'])), w - x0)
        bh = min(max(1, int(roi['height'])), h - y0)
        return slice(y0, y0 + bh), slice(x0, x0 + bw)

    def _reduce_roi(self, frame: np.ndarray) -> float:
        """Reduce a frame to one number over the ROI (mean or sum)."""
        if frame is None or frame.size == 0:
            return 0.0
        ys, xs = self._roi_slice(frame.shape)
        sub = frame[ys, xs]
        if self.settings['camera']['roi']['reduce'] == 'sum':
            return float(np.sum(sub))
        return float(np.mean(sub))

    def _need_cube(self) -> bool:
        """True if any image output (the 3-D cube or the per-frequency named images) is
        requested -- both are built from the same per-frequency running mean."""
        cam = self.settings['camera']
        return bool(cam['save_full_image_cube']) or bool(cam['save_per_frequency_images'])

    def _accumulate_cube(self, freq_index: int, frame: np.ndarray):
        """Add a frame into the per-frequency running image sum (both sweep directions
        land on their forward-frequency index, so a point gets fwd+rev contributions)."""
        if not self._need_cube() or frame is None:
            return
        if self._cube_sum is None:
            h, w = frame.shape
            self._cube_sum = np.zeros((self.num_points, h, w), dtype=np.float64)
            self._cube_cnt = np.zeros(self.num_points, dtype=np.int64)
        if frame.shape != self._cube_sum.shape[1:]:
            return  # shape changed unexpectedly; skip rather than crash
        self._cube_sum[freq_index] += frame
        self._cube_cnt[freq_index] += 1

    # ----- raw-frame capture (every single frame, unaveraged) -----
    def _raw_frame_cast(self, frame: np.ndarray) -> np.ndarray:
        """Cast a frame to the configured raw-storage dtype. float32 is lossless for
        camera counts; uint16 halves the size but clips to 0..65535."""
        if self.settings['camera']['raw_frames_dtype'] == 'uint16':
            return np.clip(frame, 0, 65535).astype(np.uint16)
        return np.ascontiguousarray(frame, dtype=np.float32)

    def _maybe_check_raw_budget(self, frame: np.ndarray):
        """On the first captured frame (once H x W is known), estimate the total raw-frame
        size and DISABLE raw saving for this run if it exceeds raw_frames_max_gb -- so a big
        sensor x many points can't silently exhaust memory. Everything else still records."""
        if self._raw_budget_checked or frame is None:
            return
        self._raw_budget_checked = True
        averages = max(1, int(self.settings['acquisition']['averaging']['averages']))
        directions = 2 if self.settings['acquisition']['bidirectional'] else 1
        n_frames = averages * directions * self.num_points
        itemsize = 2 if self.settings['camera']['raw_frames_dtype'] == 'uint16' else 4
        est_gb = n_frames * int(frame.size) * itemsize / 1e9
        cap = float(self.settings['camera']['raw_frames_max_gb'])
        self.log(f"Raw frames: ~{n_frames} x {frame.shape} x {itemsize}B "
                 f"~= {est_gb:.2f} GB (cap {cap:.2f} GB)")
        if est_gb > cap:
            self.log(f"WARNING: estimated raw-frame size {est_gb:.2f} GB exceeds the "
                     f"{cap:.2f} GB cap -- DISABLING raw-frame saving for this run. "
                     f"Raise camera/raw_frames_max_gb (or narrow the ROI / reduce averages) "
                     f"and re-run to keep them.")
            self._raw_frames_enabled = False
            self._raw_frames = []  # free anything already captured

    def _record_raw(self, frame, sweep, direction, freq_index, acq_index):
        """Stash one raw frame (if enabled) with enough metadata to reconstruct it later.
        freq_index is 1-based in frequency order (image_<freq_index> in the name); acq_index
        is 1-based capture order within this pass (differs from freq_index on reverse)."""
        if not self._raw_frames_enabled or frame is None:
            return
        self._maybe_check_raw_budget(frame)
        if not self._raw_frames_enabled:   # may have just been disabled by the budget check
            return
        self._raw_frames.append({
            'frame': self._raw_frame_cast(frame),
            'sweep': int(sweep),
            'direction': direction,
            'freq_index': int(freq_index),
            'acq_index': int(acq_index),
            'frequency_hz': float(self.frequencies[freq_index - 1]),
        })

    def _build_raw_frames_struct(self):
        """Assemble the captured raw frames into a nested MyStruct for saving:
            raw_frames/sweep_<k>/<forward|reverse>/image_<idx>_<freq> -> {camera_image, ...}
        Returns None if nothing was captured. No array copies -- entries reference the frames
        already held in self._raw_frames."""
        if not self._raw_frames:
            return None
        averages = max(1, int(self.settings['acquisition']['averaging']['averages']))
        sw_width = max(1, len(str(averages)))
        root = MyStruct()
        root.n_sweeps = averages
        root.n_points = int(self.num_points)
        root.bidirectional = bool(self.settings['acquisition']['bidirectional'])
        root.dtype = str(self.settings['camera']['raw_frames_dtype'])
        root.n_frames = int(len(self._raw_frames))
        for rec in self._raw_frames:
            sweep_key = f"sweep_{rec['sweep']:0{sw_width}d}"
            if sweep_key not in vars(root):
                setattr(root, sweep_key, MyStruct())
            sweep_grp = getattr(root, sweep_key)
            dir_key = rec['direction']  # 'forward' | 'reverse'
            if dir_key not in vars(sweep_grp):
                setattr(sweep_grp, dir_key, MyStruct())
            dir_grp = getattr(sweep_grp, dir_key)
            img_name = self._image_name(rec['freq_index'] - 1)  # image_<idx>_<freq>
            setattr(dir_grp, img_name, MyStruct(
                camera_image=rec['frame'],
                frequency_hz=rec['frequency_hz'],
                frequency_ghz=rec['frequency_hz'] / 1e9,
                freq_index=rec['freq_index'],   # frequency order (matches the name index)
                acq_index=rec['acq_index'],     # capture order within the pass
                sweep=rec['sweep'],
                direction=rec['direction'],
            ))
        return root

    def _note_reference_frames(self, frame: np.ndarray, signal: float):
        """Track a few small representative frames for saving/plotting.

        Every frame is stored as an independent COPY. Bare references here were the source
        of the byte-identical reference frames: ref_end and ref_onres were assigned the
        *same* array object on the same call, and (with a reused capture buffer) all three
        could end up pointing at whichever frame was written last. The on-resonance value
        set here is only PROVISIONAL (the dimmest frame); it is replaced in
        _select_onres_reference() with the averaged frame at the fitted resonance.
        """
        if frame is None:
            return
        if self._ref_frame_start is None:
            self._ref_frame_start = frame.copy()
        self._ref_frame_end = frame.copy()
        if signal < self._onres_min_signal:   # provisional only: the dimmest frame
            self._onres_min_signal = signal
            self._ref_frame_onres = frame.copy()

    # ------------------------------------------------- confocal optimizer (camera)
    def _read_opt_signal(self) -> float:
        """Signal used by the optimizer: the ROI value of a freshly captured frame
        (the camera analog of ODMRSweepContinuousExperiment._read_opt_counts)."""
        frame = self._capture_frame()
        return self._reduce_roi(frame) if frame is not None else 0.0

    def _optimize_axis(self, axis: str, step: float) -> None:
        """3-point quadratic-fit ('fit') or hill-climb ('convergence') maximization on one
        nanodrive axis, using the camera ROI signal. Ported from the ADwin ODMR optimizer
        with _read_opt_counts -> _read_opt_signal and no ADwin binary swapping."""
        nd = self.nanodrive
        pos_key = f'{axis}_pos'
        opt = self.settings['acquisition']['averaging']
        settle_time = float(opt['opt_settle_time'])
        damping_ratio = float(opt['opt_damping_ratio'])

        def move(target) -> bool:
            try:
                nd.update({pos_key: target})
                time.sleep(settle_time)
                return True
            except Exception as e:
                if "ARGUMENT_ERROR" in str(e):
                    print(f"Skipping {axis} move (out of range): {e}")
                    return False
                raise

        def read_pos():
            return nd.read_probes(pos_key)

        if opt['optimization_mode'] == "fit":
            move(read_pos() - 2 * step)  # take out backlash
            samples = []
            for _ in range(3):
                if move(read_pos() + step):
                    samples.append((read_pos(), self._read_opt_signal()))
            if len(samples) < 3:
                print(f"Skipping {axis}-optimization: insufficient valid points.")
                return

            coords = np.array([p for p, _ in samples], dtype=float)
            counts = np.array([c for _, c in samples], dtype=float)
            a, b, _c = np.polyfit(coords, counts, 2)
            best_pos, best_counts = max(samples, key=lambda pc: pc[1])
            lo, hi = float(coords.min()), float(coords.max())

            use_vertex = False
            vertex = None
            if a < 0:
                vertex = float(-b / (2 * a))
                if lo <= vertex <= hi:
                    use_vertex = True

            if use_vertex:
                if not move(vertex):
                    return
                counts_at_vertex = self._read_opt_signal()
                if best_counts > counts_at_vertex:
                    move(vertex + damping_ratio * (best_pos - vertex))
            else:
                print(f"{axis}: unreliable parabola (a={a:.3g}); using best sampled point.")
                move(best_pos)

        elif opt['optimization_mode'] == "convergence":
            max_steps = int(opt['opt_max_steps'])
            improve_frac = float(opt['opt_improve_frac'])

            def better(new, ref):
                return new > ref * (1.0 + improve_frac)

            samples = [(read_pos(), self._read_opt_signal())]
            for _ in range(2):
                if move(read_pos() + step):
                    samples.append((read_pos(), self._read_opt_signal()))
            if len(samples) < 3:
                print(f"Skipping {axis}-optimization: insufficient valid points.")
                return
            best_pos, best_counts = max(samples, key=lambda pc: pc[1])
            c_start = samples[0][1]
            c_far = samples[-1][1]
            direction = 0
            if best_pos == samples[-1][0] and better(c_far, c_start):
                direction = +1
            elif best_pos == samples[0][0]:
                direction = -1
                move(samples[0][0])  # back to start before heading -
            steps_taken = 0
            while direction != 0 and steps_taken < max_steps:
                if not move(read_pos() + direction * step):
                    break
                c_new = self._read_opt_signal()
                if better(c_new, best_counts):
                    best_pos, best_counts = read_pos(), c_new
                    steps_taken += 1
                else:
                    break
            move(best_pos)
            print(f"{axis}: converged at {best_pos:.3f} um, ~{best_counts:.3g} (ROI) "
                  f"(+{steps_taken} step(s) past opener).")
        else:
            raise NotImplementedError

    def _optimize_position(self) -> None:
        """One-shot confocal re-optimization between runs, driven by the camera ROI signal,
        with the same best-known baseline + retention net as the ADwin ODMR experiment."""
        nd = self.nanodrive
        if nd is None:
            self.logger.warning("No nanodrive available; skipping optimization.")
            print("No nanodrive available; skipping optimization.")
            return
        opt = self.settings['acquisition']['averaging']
        settle_time = float(opt['opt_settle_time'])
        damping_ratio = float(opt['opt_damping_ratio'])
        min_retention_ratio = float(opt['min_retention_ratio'])
        acceptable_signal = float(opt['acceptable_signal'])
        acceptable_signal_ratio = float(opt['acceptable_signal_ratio'])
        min_reoptimize_ratio = float(opt['min_reoptimize_ratio'])
        reoptimize_only_if_needed = bool(opt['reoptimize_only_if_needed'])

        # Make sure the optimizer measures at the locked sweep exposure and the
        # reference (start) frequency, so the signal is stable while we climb.
        self._set_frequency(self.frequencies[0])
        time.sleep(float(self.settings['microwave']['settle_time']))

        current = self._read_opt_signal()
        if self._opt_best_signal is None:
            self._opt_best_signal = current
            self._opt_best_x = nd.read_probes('x_pos')
            self._opt_best_y = nd.read_probes('y_pos')
            self._opt_best_z = nd.read_probes('z_pos')
            print(f"Seeded best-known baseline: {current:.3g} (ROI).")

        if reoptimize_only_if_needed and current >= (min_reoptimize_ratio * self._opt_best_signal):
            print(f"Signal healthy ({current:.3g} >= {min_reoptimize_ratio:.2f} * "
                  f"{self._opt_best_signal:.3g}); skipping re-optimization.")
            return

        print("Optimizing confocal position between runs (z -> x -> y)...")
        self._optimize_axis('z', float(opt['opt_zstep']))
        self._optimize_axis('x', float(opt['opt_xstep']))
        self._optimize_axis('y', float(opt['opt_ystep']))

        optimized = self._read_opt_signal()
        if optimized > self._opt_best_signal:
            if optimized < (acceptable_signal * acceptable_signal_ratio):
                self._opt_best_signal = optimized
                self._opt_best_x = nd.read_probes('x_pos')
                self._opt_best_y = nd.read_probes('y_pos')
                self._opt_best_z = nd.read_probes('z_pos')
                print(f"New best-known position: {optimized:.3g} (ROI).")
            else:
                print(f"Ignoring implausibly high signal ({optimized:.3g}) for the baseline.")
        elif optimized < (min_retention_ratio * self._opt_best_signal):
            print(f"Post-opt signal {optimized:.3g} < {min_retention_ratio:.2f} * best "
                  f"{self._opt_best_signal:.3g}; snapping back toward best-known position.")
            cx, cy, cz = nd.read_probes('x_pos'), nd.read_probes('y_pos'), nd.read_probes('z_pos')
            try:
                nd.update({'x_pos': cx + damping_ratio * (self._opt_best_x - cx),
                           'y_pos': cy + damping_ratio * (self._opt_best_y - cy),
                           'z_pos': cz + damping_ratio * (self._opt_best_z - cz)})
                time.sleep(settle_time)
            except Exception as e:
                if "ARGUMENT_ERROR" not in str(e):
                    raise

    # ------------------------------------------------------------- main function
    def _function(self):
        """Main experiment function."""
        try:
            self.log("Starting ODMR Camera Sweep Experiment")
            # Laser + filters on (best-effort; mirrors the ADwin ODMR experiment).
            if self.settings['GREEN_LASER']['control_laser']:
                if self.filter_wheel is not None:
                    try:
                        self.filter_wheel.update({'OD': self.settings['GREEN_LASER']['Filter Wheel OD']})
                    except Exception as e:
                        self.log(f"Filter wheel set failed (continuing): {e}")
                if self.proteus is not None:
                    try:
                        self.proteus.set_channel_voltage_high(4, self.settings['GREEN_LASER']["Laser Control"])
                    except Exception as e:
                        self.log(f"Laser (proteus) enable failed (continuing): {e}")
            if self.proteus is not None:
                try:
                    self.proteus.set_channel_voltage_high(1, "MAX")
                except Exception as e:
                    self.log(f"MW (proteus) enable failed (continuing): {e}")
            self.setup()

            start_time = datetime.datetime.now()
            self.s_t = start_time.strftime("%m_%d_%Y_%H:%M:%S")
            self._log_time_estimate()

            self._run_sweep_averages()

            end_time = datetime.datetime.now()
            self.e_t = end_time.strftime("%m_%d_%Y_%H:%M:%S")

            if self.settings['GREEN_LASER']['control_laser'] and \
                    self.settings['GREEN_LASER']['Laser off after experiment?'] and \
                    self.proteus is not None:
                try:
                    self.log("turning proteus off")
                    self.proteus.driver.off()
                except Exception as e:
                    self.log(f"Could not turn laser off (continuing): {e}")

            self._analyze_data()
            self._store_results_in_data()

            self.log("ODMR Camera Sweep Experiment completed successfully")
            if self.settings['save']:
                self.save_hdf5()
            self.cleanup()
        except Exception as e:
            self.log(f"Error in ODMR camera sweep experiment: {e}")
            raise

    def _aborted(self) -> bool:
        """Cooperative stop check. Uses the base class's abort flag if present; otherwise
        this is a harmless no-op and the run stops at sweep boundaries like the ADwin ODMR."""
        return bool(getattr(self, '_abort', False))

    def _run_sweep_averages(self):
        """Run 'averages' full sweeps, optimizing between runs (before runs 2..N), and
        average the per-point ROI signal term-by-term. Mirrors the ADwin ODMR flow."""
        averages = max(1, int(self.settings['acquisition']['averaging']['averages']))
        optimize_between = bool(self.settings['acquisition']['averaging']['optimize_between_runs'])
        settle_time = float(self.settings['acquisition']['settle_time'])
        n = self.num_points

        # Reset optimizer baseline for THIS run (seeded on the first optimization).
        self._opt_best_signal = None
        self._opt_best_x = self._opt_best_y = self._opt_best_z = None

        all_forward = np.full((averages, n), np.nan, dtype=np.float64)
        all_reverse = np.full((averages, n), np.nan, dtype=np.float64)  # acquisition order

        self.log(f"Starting sweep averages: {averages} sweep(s)")
        if optimize_between:
            self.log("Confocal re-optimization ENABLED between sweeps (before sweeps 2..N)")

        for avg in range(averages):
            if self._aborted():
                self.log("Abort requested; stopping before next sweep.")
                break
            if optimize_between and avg > 0 and self.nanodrive is not None:
                try:
                    self._optimize_position()
                except Exception as e:
                    self.log(f"Optimization before sweep {avg + 1} failed (continuing): {e}")

            self.log(f"Running sweep {avg + 1}/{averages}")
            fwd_scalar, rev_scalar = self._run_single_sweep(avg, averages)
            all_forward[avg, :] = fwd_scalar
            if rev_scalar is not None:
                all_reverse[avg, :] = rev_scalar

            self.progress = 100.0 * (avg + 1) / averages
            self.updateProgress.emit(int(round(self.progress)))

            if avg < averages - 1:
                time.sleep(settle_time)

        # Term-by-term averages (nanmean tolerates an interrupted final sweep).
        self.counts_forward = np.nanmean(all_forward, axis=0)
        if self.settings['acquisition']['bidirectional'] and np.any(np.isfinite(all_reverse)):
            self.counts_reverse = np.nanmean(all_reverse, axis=0)[::-1]  # flip to align with forward
        else:
            self.counts_reverse = self.counts_forward.copy()
        self.counts_averaged = np.nanmean(
            np.vstack([self.counts_forward, self.counts_reverse]), axis=0)

        # Keep per-sweep arrays (reverse flipped so each row aligns in frequency).
        self.all_forward = all_forward
        self.all_reverse = all_reverse[:, ::-1]

        self.log("Sweep averages completed")

    def _run_single_sweep(self, avg_index: int, averages: int):
        """One full sweep. Returns (forward_scalar, reverse_scalar):
            forward_scalar : ROI signal at each freq, freq increasing
            reverse_scalar : ROI signal on the reverse pass in ACQUISITION order
                             (index 0 = stop frequency), or None if unidirectional.
        Updates the live frame, per-frequency image cube and reference frames as it goes."""
        n = self.num_points
        bidirectional = bool(self.settings['acquisition']['bidirectional'])
        mw_settle = float(self.settings['microwave']['settle_time'])

        total_points = averages * n * (2 if bidirectional else 1)
        base_done = avg_index * n * (2 if bidirectional else 1)

        fwd = np.full(n, np.nan, dtype=np.float64)
        rev = np.full(n, np.nan, dtype=np.float64) if bidirectional else None

        # ---- forward pass: freq increasing ----
        for i in range(n):
            if self._aborted():
                break
            self._set_frequency(self.frequencies[i])
            time.sleep(mw_settle)
            frame = self._capture_frame()
            s = self._reduce_roi(frame)
            fwd[i] = s
            self._accumulate_cube(i, frame)
            self._record_raw(frame, avg_index + 1, 'forward', i + 1, i + 1)
            self._note_reference_frames(frame, s)
            self._emit_point_progress(base_done + i + 1, total_points)

        # ---- reverse pass: freq decreasing (acquisition order) ----
        if bidirectional:
            for j in range(n):
                if self._aborted():
                    break
                fi = n - 1 - j                      # forward-frequency index
                self._set_frequency(self.frequencies[fi])
                time.sleep(mw_settle)
                frame = self._capture_frame()
                s = self._reduce_roi(frame)
                rev[j] = s
                self._accumulate_cube(fi, frame)    # land on the forward index
                self._record_raw(frame, avg_index + 1, 'reverse', fi + 1, j + 1)
                self._note_reference_frames(frame, s)
                self._emit_point_progress(base_done + n + j + 1, total_points)

        return fwd, rev

    def _emit_point_progress(self, done: int, total: int):
        """Fine-grained progress within a sweep (so the GUI bar/ETA advance smoothly)."""
        if total <= 0:
            return
        self.progress = 100.0 * done / total
        try:
            self.updateProgress.emit(int(round(self.progress)))
        except Exception:
            pass

    def cleanup(self):
        """Release resources."""
        if self.settings['microwave']['enable'] and self.settings['microwave']['MW off after experiment?']:
            try:
                if self.microwave and self.microwave.is_connected:
                    self.microwave.disable_modulation()
                    self.microwave.disable_output()
            except Exception as e:
                self.log(f"MW off failed (continuing): {e}")
        # Free the raw-frame buffer (it can be many GB) now that saving is done.
        self._raw_frames = []
        self.log("ODMR Camera Sweep Experiment cleanup complete")

    # ------------------------------------------------------------------ analysis
    # (peak finding / contrast / Lorentzian fit are ported verbatim from the ADwin
    #  ODMR experiment; they operate on self.counts_averaged + self.frequencies.)
    def _analyze_data(self):
        self.log("Analyzing ODMR camera sweep data...")
        if self.settings['analysis']['auto_fit']:
            self._fit_resonances()
        self.log("Data analysis completed")

    def _fit_resonances(self):
        try:
            peaks = self._find_peaks()
            contrast, _ = self._prepare_contrast()
            self.fit_parameters, self.resonance_frequencies = [], []
            for pk in peaks:
                lo, hi = max(0, pk - 15), min(len(self.frequencies), pk + 15)
                x, yc = self.frequencies[lo:hi], contrast[lo:hi]
                p0 = [-(1.0 - yc.min()), self.frequencies[pk], 5e6, 1.0]  # dip: neg amp
                try:
                    popt, _ = curve_fit(self._lorentzian_function, x, yc, p0=p0, maxfev=5000)
                    if x[0] <= popt[1] <= x[-1] and popt[0] < 0:
                        self.fit_parameters.append(popt)
                        self.resonance_frequencies.append(popt[1])
                except Exception as e:
                    self.log(f"Fit failed near {self.frequencies[pk] / 1e9:.3f} GHz: {e}")
            self.log(f"Fitted {len(self.resonance_frequencies)} resonance(s): "
                     f"{[f'{r / 1e9:.4f}' for r in self.resonance_frequencies]}")
        except Exception as e:
            self.log(f"Error in resonance fitting: {e}")

    def _find_peaks(self):
        contrast, noise = self._prepare_contrast()
        df = abs(self.frequencies[1] - self.frequencies[0]) if len(self.frequencies) > 1 else 1e6
        min_prom = max(3.0 * noise, 0.003)
        min_width = max(2, int(2e6 / df))
        min_dist = max(3, int(5e6 / df))
        idx, props = find_peaks(1.0 - contrast, prominence=min_prom, width=min_width, distance=min_dist)
        order = np.argsort(props['prominences'])[::-1]
        idx = np.sort(idx[order][:4])
        self.log(f"Found {len(idx)} dip(s) (noise={noise:.4f}, min_prom={min_prom:.4f})")
        return list(idx)

    def _prepare_contrast(self):
        """Smoothed, baseline-flattened contrast (~1.0 off-resonance, <1 at a dip)."""
        y = np.asarray(self.counts_averaged, dtype=float)
        if self.settings['analysis'].get('smoothing', True):
            w = int(self.settings['analysis'].get('smooth_window', 5))
            if 3 <= w < len(y):
                if w % 2 == 0:
                    w += 1
                y = savgol_filter(y, w, 3)
        x = np.arange(len(y))
        mask = np.ones(len(y), bool)
        for _ in range(3):
            base = np.polyval(np.polyfit(x[mask], y[mask], 2), x)
            resid = y - base
            mask = resid > -2.0 * np.std(resid[mask])
        base = np.polyval(np.polyfit(x[mask], y[mask], 2), x)
        # guard divide-by-zero for sum-reduced ROIs that can be large but never 0 here
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)
        contrast = y / base
        noise = np.std(contrast[mask])
        return contrast, noise

    def _lorentzian_function(self, x, amplitude, center, width, offset):
        return amplitude * (width / 2) ** 2 / ((x - center) ** 2 + (width / 2) ** 2) + offset

    def _smooth_data(self, data: np.ndarray) -> np.ndarray:
        window = self.settings['analysis']['smooth_window']
        if len(data) > window:
            return savgol_filter(data, window, 3)
        return data

    def _subtract_background(self, data: np.ndarray) -> np.ndarray:
        return data - np.min(data)

    # -------------------------------------------------------------------- results
    def _store_results_in_data(self):
        self.data['frequencies'] = self.frequencies
        self.data['counts_forward'] = self.counts_forward
        self.data['counts_reverse'] = self.counts_reverse
        self.data['counts_averaged'] = self.counts_averaged
        self.data['sweep_time'] = getattr(self, 'sweep_time', None)
        self.data['num_points'] = self.num_points
        self.data['fit_parameters'] = self.fit_parameters
        self.data['resonance_frequencies'] = self.resonance_frequencies
        self.data['all_counts_forward'] = self.all_forward
        self.data['all_counts_reverse'] = self.all_reverse
        self.data['image_cube_mean'] = (self._mean_cube()
                                        if self.settings['camera']['save_full_image_cube'] else None)
        self._select_onres_reference()  # pick on-res frame by resonance, not "dimmest"
        self.data['ref_frame_start'] = self._ref_frame_start
        self.data['ref_frame_onres'] = self._ref_frame_onres
        self.data['ref_frame_end'] = self._ref_frame_end

    def _select_onres_reference(self):
        """Pick the on-resonance reference frame by FREQUENCY, not by 'dimmest frame'.

        The provisional ref_frame_onres set during the sweep is just the minimum-signal
        frame; in a run with no real dip (or a monotonic intensity drift) that collapses to
        the last-acquired frame, making ref_frame_onres identical to ref_frame_end. Here we
        replace it with the per-frequency AVERAGED image closest to the resonance: the fitted
        resonance if we have one, else the held constant frequency, else the sweep centre.
        Falls back to the provisional frame if no image cube was accumulated.
        """
        cube = self._mean_cube()
        if cube is None:
            return  # no per-frequency images retained; keep the provisional frame
        target_hz = None
        if self.resonance_frequencies:
            target_hz = float(self.resonance_frequencies[0])
        elif bool(self.settings['constant_frequency']
                  ['disable sweep and enable constant frequency']):
            target_hz = float(self.settings['constant_frequency']['frequency'])
        if target_hz is None:
            target_hz = 0.5 * (float(self.frequencies[0]) + float(self.frequencies[-1]))
        freqs = np.asarray(self.frequencies, dtype=float)
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        # only accept a frame that actually has data (count > 0)
        if self._cube_cnt is not None and int(self._cube_cnt[idx]) <= 0:
            valid = np.where(self._cube_cnt > 0)[0]
            if valid.size == 0:
                return
            idx = int(valid[np.argmin(np.abs(freqs[valid] - target_hz))])
        frame = cube[idx]
        if np.all(np.isfinite(frame)):
            self._ref_frame_onres = np.ascontiguousarray(frame, dtype=np.float32).copy()
            self.log(f"On-resonance reference set from image at "
                     f"{float(freqs[idx]) / 1e9:.4f} GHz (index {idx + 1})")

    def _mean_cube(self):
        """Per-frequency mean image (float32) if the cube was accumulated, else None."""
        if self._cube_sum is None or self._cube_cnt is None:
            return None
        cnt = self._cube_cnt.astype(np.float64)
        cnt[cnt == 0] = np.nan
        mean = self._cube_sum / cnt[:, None, None]
        return mean.astype(np.float32)

    def _freq_name_decimals(self) -> int:
        """Fewest decimal places (in the chosen units) that give every frequency a UNIQUE
        name string, so no two per-frequency images collide. Floors at 3 for GHz (so you get
        2.800, 2.801, ... like your example) and 0 for MHz."""
        units = self.settings['camera']['image_name_units']
        scale = 1e-9 if units == 'GHz' else 1e-6
        floor = 3 if units == 'GHz' else 0
        vals = sorted(float(f) * scale for f in np.asarray(self.frequencies, dtype=float))
        if len(vals) < 2:
            return floor
        for dec in range(floor, 10):
            names = [f"{v:.{dec}f}" for v in vals]
            if len(set(names)) == len(names):
                return dec
        return 9

    def _image_name(self, i: int) -> str:
        """Build the per-frequency image name, e.g. image_1_2.800GHz. The index is zero-padded
        to num_points width so the names sort in frequency order (image_001..image_101); for
        <=9 points the padding is 1 digit, matching image_1_2.800GHz exactly."""
        if getattr(self, '_img_name_dec', None) is None:
            self._img_name_dec = self._freq_name_decimals()
        units = self.settings['camera']['image_name_units']
        scale = 1e-9 if units == 'GHz' else 1e-6
        s = f"{float(self.frequencies[i]) * scale:.{self._img_name_dec}f}"
        if self.settings['camera']['image_name_decimal_char'] == 'p':
            s = s.replace('.', 'p')
        width = max(1, len(str(int(self.num_points))))
        return f"image_{i + 1:0{width}d}_{s}{units}"

    # ----------------------------------------------------------------- plotting
    def _plot(self, axes_list):
        """1D spectrum in pane 0 and a live/representative camera image in pane 1;
        or, if 2D_Plot, per-sweep waterfalls in both panes (same as ADwin ODMR)."""
        if not axes_list:
            return

        def _plot_item(w):
            if hasattr(w, "plot") and hasattr(w, "clear"):
                return w
            if hasattr(w, "ci") and hasattr(w, "addPlot"):
                items = [it for it in w.ci.items if isinstance(it, pg.PlotItem)]
                return items[0] if items else w.addPlot(row=0, col=0)
            return None

        axes = [pi for pi in (_plot_item(w) for w in axes_list) if pi is not None]
        if not axes:
            return

        if bool(self.settings.get('2D_Plot', False)):
            self._plot_2d(axes)
            return

        ax = axes[0]
        ax.clear()
        if self.frequencies is None or self.counts_averaged is None:
            return
        f_ghz = self.frequencies / 1e9
        ax.plot(f_ghz, self.counts_forward, pen=None, symbol='o', symbolSize=5,
                symbolBrush='b', symbolPen=None, name='Forward')
        ax.plot(f_ghz, self.counts_reverse, pen=None, symbol='o', symbolSize=5,
                symbolBrush='g', symbolPen=None, name='Reverse')
        ax.plot(f_ghz, self.counts_averaged, pen=None, symbol='o', symbolSize=6,
                symbolBrush='r', symbolPen=None, name='Averaged')
        if self.resonance_frequencies:
            for freq in self.resonance_frequencies:
                ax.addItem(pg.InfiniteLine(pos=freq / 1e9, angle=90,
                                           pen=pg.mkPen('y', style=Qt.DashLine)))
        ax.setLabel('bottom', 'Frequency (GHz)')
        reduce_kind = self.settings['camera']['roi']['reduce']
        ax.setLabel('left', f'Camera signal (ROI {reduce_kind})')
        ax.setTitle('ODMR Camera Sweep Spectrum')
        ax.showGrid(x=True, y=True, alpha=0.3)

        if len(axes) > 1:
            # prefer the on-resonance representative frame once we have one, else the live frame
            frame = self._ref_frame_onres if self._ref_frame_onres is not None else self._last_frame
            self._draw_image(axes[1], frame, 'Camera frame (live / on-resonance)')

    def _draw_image(self, ax, frame, title):
        """Render a 2-D frame as a scaled colour image on PlotItem ax."""
        ax.clear()
        if frame is None or getattr(frame, 'size', 0) == 0 or np.ndim(frame) != 2:
            ax.setTitle(title)
            return
        vmin = float(np.nanpercentile(frame, 1))
        vmax = float(np.nanpercentile(frame, 99))
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(frame))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        cmap = self._get_colormap()
        img = pg.ImageItem()
        img.setImage(np.asarray(frame, dtype=float).T, autoLevels=False)  # T: [x, y] for pg
        img.setLevels([vmin, vmax])
        try:
            img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        h, w = frame.shape
        img.setRect(pg.QtCore.QRectF(0.0, 0.0, float(w), float(h)))
        ax.addItem(img)
        ax.setLabel('bottom', 'x (px)')
        ax.setLabel('left', 'y (px)')
        ax.setTitle(title)
        ax.showGrid(x=False, y=False)
        ax.setXRange(0.0, float(w), padding=0)
        ax.setYRange(0.0, float(h), padding=0)
        try:
            cbar = getattr(self, '_cbar_img', None)
            host_id = id(ax)
            if cbar is None or getattr(self, '_cbar_img_host', None) != host_id:
                cbar = pg.ColorBarItem(colorMap=cmap, values=(vmin, vmax))
                self._cbar_img = cbar
                self._cbar_img_host = host_id
                cbar.setImageItem(img, insert_in=ax)
            else:
                cbar.setImageItem(img)
            cbar.setLevels((vmin, vmax))
        except Exception:
            pass

    def _plot_2d(self, axes):
        """Per-sweep 'waterfall' view (no averaging): x = freq, y = sweep #, colour = ROI signal.
        Pane 0 = forward sweeps, pane 1 = reverse sweeps."""
        all_fwd = self.all_forward if self.all_forward is not None else \
            (self.data.get('all_counts_forward') if self.data else None)
        all_rev = self.all_reverse if self.all_reverse is not None else \
            (self.data.get('all_counts_reverse') if self.data else None)
        if all_fwd is None or self.frequencies is None:
            for ax in axes:
                ax.clear()
            return
        f_ghz = self.frequencies / 1e9
        self._draw_sweep_map(axes[0], np.asarray(all_fwd, dtype=float), f_ghz,
                             'ODMR camera per-sweep map -- Forward', '_cbar_fwd')
        if len(axes) > 1 and all_rev is not None:
            self._draw_sweep_map(axes[1], np.asarray(all_rev, dtype=float), f_ghz,
                                 'ODMR camera per-sweep map -- Reverse', '_cbar_rev')

    def _draw_sweep_map(self, ax, data, f_ghz, title, cbar_attr):
        """Render one (n_sweeps, n_freq) array as a scaled colour image on PlotItem ax."""
        ax.clear()
        if data.ndim != 2 or data.size == 0:
            return
        n_sweeps, n_freq = data.shape
        vmin = float(np.nanpercentile(data, 1))
        vmax = float(np.nanpercentile(data, 99))
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(data))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        cmap = self._get_colormap()
        img = pg.ImageItem()
        img.setImage(data.T, autoLevels=False)  # data is [sweep(y), freq(x)] -> transpose
        img.setLevels([vmin, vmax])
        try:
            img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        x0, x1 = float(f_ghz[0]), float(f_ghz[-1])
        width = (x1 - x0) if x1 != x0 else 1.0
        img.setRect(pg.QtCore.QRectF(x0, 0.0, width, float(n_sweeps)))
        ax.addItem(img)
        ax.setLabel('bottom', 'Frequency (GHz)')
        ax.setLabel('left', 'Sweep #  (bottom = first)')
        ax.setTitle(title)
        ax.showGrid(x=False, y=False)
        ax.setXRange(x0, x1, padding=0)
        ax.setYRange(0.0, float(n_sweeps), padding=0)
        try:
            cbar = getattr(self, cbar_attr, None)
            host_id = id(ax)
            if cbar is None or getattr(self, cbar_attr + '_host', None) != host_id:
                cbar = pg.ColorBarItem(colorMap=cmap, values=(vmin, vmax))
                setattr(self, cbar_attr, cbar)
                setattr(self, cbar_attr + '_host', host_id)
                cbar.setImageItem(img, insert_in=ax)
            else:
                cbar.setImageItem(img)
            cbar.setLevels((vmin, vmax))
        except Exception:
            pass

    def _get_colormap(self):
        """Perceptually-uniform colormap with graceful fallbacks across pyqtgraph versions."""
        for getter in (lambda: pg.colormap.get('viridis'),
                       lambda: pg.colormap.getFromMatplotlib('viridis'),
                       lambda: pg.colormap.get('CET-L9')):
            try:
                cm = getter()
                if cm is not None:
                    return cm
            except Exception:
                continue
        return pg.ColorMap([0.0, 0.5, 1.0],
                           [(68, 1, 84), (33, 145, 140), (253, 231, 37)])

    def _update(self, axes_list):
        self._plot(axes_list)

    def get_axes_layout(self, figure_list):
        """Build (or reuse) one PlotItem per GraphicsLayoutWidget (same as ADwin ODMR)."""
        axes_list = []
        if self._plot_refresh is True:
            for graph in figure_list:
                graph.clear()
                axes_list.append(graph.addPlot(row=0, col=0))
        else:
            for graph in figure_list:
                axes_list.append(graph.getItem(row=0, col=0))
        return axes_list

    # ----------------------------------------------------------------- misc / info
    @staticmethod
    def _fmt_hms(seconds: float) -> str:
        seconds = int(round(max(0.0, seconds)))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m {s}s"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _estimate_experiment_time(self) -> float:
        avg_cfg = self.settings['acquisition']['averaging']
        averages = max(1, int(avg_cfg['averages']))
        total = self.sweep_time * averages
        if bool(avg_cfg['optimize_between_runs']) and self.nanodrive is not None:
            moves_per_axis = 6
            axes = 3
            per_opt = moves_per_axis * axes * float(avg_cfg['opt_settle_time'])
            total += max(0, averages - 1) * per_opt
        return total

    def _log_time_estimate(self) -> None:
        avg_cfg = self.settings['acquisition']['averaging']
        averages = max(1, int(avg_cfg['averages']))
        total = self._estimate_experiment_time()
        note = " (incl. confocal re-optimization)" if (bool(avg_cfg['optimize_between_runs'])
                                                       and self.nanodrive is not None) else ""
        self.log(f"Estimated experiment time: ~{self._fmt_hms(total)} "
                 f"= {averages} x ~{self._fmt_hms(self.sweep_time)}/sweep{note}")

    def get_experiment_info(self) -> Dict[str, Any]:
        return {
            'name': 'ODMR Camera Sweep Experiment',
            'description': 'Widefield/camera ODMR: stepped SG384 sweep with a camera image per '
                           'frequency; forward/reverse, averaging and confocal re-optimization.',
            'devices': list(self._DEVICES.keys()),
            'frequency_range': f"{self.settings['frequency_range']['start']/1e9:.3f} - "
                               f"{self.settings['frequency_range']['stop']/1e9:.3f} GHz",
            'num_points': self.settings['frequency_range']['num_points'],
            'exposure_ms': self.settings['camera']['inttime'],
            'roi_mode': self.settings['camera']['roi']['mode'],
            'averages': self.settings['acquisition']['averaging']['averages'],
            'bidirectional': self.settings['acquisition']['bidirectional'],
        }

    # ------------------------------------------------------------------ saving
    def save_hdf5(self):
        """Save the spectrum, per-sweep arrays, reference frames and (optionally) the
        per-frequency mean-image cube, then let the parent add the external devices."""
        structure_to_save = MyStruct()

        frequencies = self.data['frequencies'] = self.frequencies
        counts_forward = self.data['counts_forward'] = self.counts_forward
        counts_reverse = self.data['counts_reverse'] = self.counts_reverse
        counts_averaged = self.data['counts_averaged'] = self.counts_averaged
        num_points = self.data['num_points'] = self.num_points
        sweep_time = self.data['sweep_time'] = getattr(self, 'sweep_time', None)
        fit_parameters = self.data['fit_parameters'] = self.fit_parameters
        resonance_frequencies = self.data['resonance_frequencies'] = self.resonance_frequencies

        all_forward = getattr(self, 'all_forward', None)
        all_reverse = getattr(self, 'all_reverse', None)

        # Both image outputs come from the same per-frequency running mean.
        save_cube = bool(self.settings['camera']['save_full_image_cube'])
        save_named = bool(self.settings['camera']['save_per_frequency_images'])
        mean_cube = self._mean_cube() if (save_cube or save_named) else None
        image_cube_mean = mean_cube if save_cube else None

        # Build the named per-frequency images: image_<idx>_<freq>, e.g. image_1_2.800GHz.
        # Each is a small sub-struct so the analyzer can open image_* and read its exact
        # frequency without parsing the (rounded) name. Indices with no captured frame
        # (e.g. an aborted final sweep) are skipped.
        images_struct = None
        image_names = []
        image_freqs_hz = []
        if save_named and mean_cube is not None:
            self._img_name_dec = None  # recompute name precision for the final grid
            images_struct = MyStruct()
            n_bytes = 0
            for i in range(self.num_points):
                if self._cube_cnt is None or int(self._cube_cnt[i]) <= 0:
                    continue
                img = np.ascontiguousarray(mean_cube[i], dtype=np.float32)
                name = self._image_name(i)
                entry = MyStruct(
                    camera_image=img,                      # 2-D float32, sweep-averaged
                    frequency_hz=float(self.frequencies[i]),
                    frequency_ghz=float(self.frequencies[i] / 1e9),
                    index=int(i + 1),
                    n_frames_averaged=int(self._cube_cnt[i]),
                )
                setattr(images_struct, name, entry)
                image_names.append(name)
                image_freqs_hz.append(float(self.frequencies[i]))
                n_bytes += img.nbytes
            self.log(f"Saving {len(image_names)} per-frequency images "
                     f"(~{n_bytes / 1e6:.1f} MB) as "
                     f"image_<idx>_<freq>{self.settings['camera']['image_name_units']} "
                     f"under data/images/")
        elif save_named and mean_cube is None:
            self.log("save_per_frequency_images is on but no image data was captured; "
                     "nothing to save under data/images/.")

        # Every captured frame, unaveraged, nested by sweep/direction (or None).
        raw_frames_struct = self._build_raw_frames_struct()

        for k in self.data:
            print("DATA KEY:", k, "->", type(self.data[k]).__name__)

        structure_to_save.data = MyStruct(
            frequencies=frequencies,
            counts_forward=counts_forward,       # averaged ROI signal
            counts_reverse=counts_reverse,       # averaged (flipped to align)
            counts_averaged=counts_averaged,     # averaged of the two
            num_points=num_points,
            sweep_time=sweep_time,
            fit_parameters=fit_parameters,
            resonance_frequencies=resonance_frequencies,
            # per-sweep raw ROI signal (averages x num_points), reverse flipped to align
            all_counts_forward=all_forward,
            all_counts_reverse=all_reverse,
            n_averages=int(self.settings['acquisition']['averaging']['averages']),
            # imaging
            roi_mode=self.settings['camera']['roi']['mode'],
            roi_reduce=self.settings['camera']['roi']['reduce'],
            exposure_ms=float(self.settings['camera']['inttime']),
            gain=float(self.settings['camera']['gain']),
            image_cube_mean=image_cube_mean,     # (num_points, H, W) float32 or None
            # per-frequency named images (image_<idx>_<freq>) + an exact-mapping index:
            images=images_struct,                # MyStruct of image_* sub-structs, or None
            image_names_json=json.dumps(image_names),          # names in frequency order
            image_frequencies_hz=np.asarray(image_freqs_hz, dtype=np.float64),  # aligned freqs
            # every raw frame, unaveraged: raw_frames/sweep_<k>/<direction>/image_<idx>_<freq>
            raw_frames=raw_frames_struct,        # nested MyStruct, or None
            n_raw_frames=int(len(self._raw_frames)),
            ref_frame_start=self._ref_frame_start,
            ref_frame_onres=self._ref_frame_onres,
            ref_frame_end=self._ref_frame_end,
        )
        structure_to_save.meta = MyStruct(
            settings=self.settings,
            end_time=self.e_t,
            start_time=self.s_t,
        )
        structure_to_save.devices = self.devices
        self.save_hdf_data(structure_to_save)
