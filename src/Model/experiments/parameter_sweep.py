# Written by <Jannet Trabelsi>
# Created for the Dutt Lab (pittqlabsys) - standalone N-dimensional parameter sweep iterator
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""
ParameterSweep
==============

A *standalone* iterator that runs one (or several) existing experiments over an
arbitrary N-dimensional grid of parameter values, saving the result of every
grid point to disk as soon as it finishes.

It exists because the older ``ExperimentIterator`` can only sweep a *single*
parameter; multi-dimensional sweeps required nesting iterators inside iterators
through a runtime class-factory + GUI dialog, which is fragile and partly
unimplemented.  ``ParameterSweep`` does the nested product directly.

Design
------
Every experiment in this codebase reads ``self.settings`` inside its
``_function`` and drives the hardware from those settings (e.g. ODMR does
``self.proteus.set_channel_voltage_high(4, self.settings["Laser Control"])``).
So to sweep a parameter we only have to:

    1. write the value into the inner experiment's ``settings``
    2. call the inner experiment's ``run()``

That is *all* this class does per grid point, plus bookkeeping (save, progress,
abort, error handling).  It never talks to a driver directly, which is what
makes it safe and generic.

Typical use (see run_overnight_sweep.py for a complete runnable script)::

    inner = <a fully-built ODMR / confocal experiment, exactly as your GUI builds it>

    sweep = ParameterSweep(
        inner_experiment=inner,
        sweep_axes=[
            ParameterSweep.step_axis('Laser Control',          0.0, 0.8, 0.1),  # outer, changes least
            ParameterSweep.axis     ('Filter Wheel OD',        [0, 0.5, 1, 2, 3, 4]),
            ParameterSweep.step_axis('frequency_range.start',  2.0e9, 3.0e9, 1.0e8),  # inner, changes most
        ],
        name='overnight_sweep',
        data_path=r'C:\\data\\2024-06-01_overnight',
    )
    sweep.run()

The axes are applied outermost-first: the *last* axis in the list varies
fastest.  Put slow/mechanical axes (filter wheel) near the top and cheap
electronic axes (MW frequency/power) near the bottom to minimise hardware wear.
"""

import copy
import datetime
import itertools
import json
import os
import pickle
import time
import traceback
from collections import OrderedDict

import numpy as np

# Same import style as experiment_iterator.py so it resolves inside the package.
from src.core import Parameter, Experiment


class ParameterSweep(Experiment):
    """N-dimensional grid sweep over one or more inner experiments."""

    # Empty because this iterator manages its inner experiment(s) itself rather
    # than through the base class' _EXPERIMENTS machinery.  These shadow the
    # NotImplementedError properties on the base class (ODMR / confocal do the
    # same thing).
    _DEVICES = {}
    _EXPERIMENTS = {}

    _DEFAULT_SETTINGS = [
        Parameter('settle_time', 0.0, float,
                  'Seconds to wait after writing parameters before running the '
                  'inner experiment, to let hardware settle (filter wheel, laser).'),
        Parameter('save_mode', 'hdf5_single',
                  ['hdf5_single', 'hdf5_only', 'full'],
                  "hdf5_single: write ONE .h5 for the whole sweep - every grid point "
                  "goes into its own group inside that single file (best for analysis). "
                  "hdf5_only: one separate .h5 per grid point via the inner experiment's "
                  "save_hdf5(). full: also make a timestamped folder with per-point "
                  "pickles, a manifest.json and combined .pkl/.mat plus the run log. "
                  "In hdf5_single / hdf5_only nothing else is written (no json, pkl, mat "
                  "or log) and files go straight into the data folder."),
        Parameter('save_inner_hdf5', True, bool,
                  "Save each grid point in the inner experiment's own hdf5 "
                  "format via its save_hdf5() method, when it implements one."),
        Parameter('abort_on_error', False, bool,
                  'If False, a single grid point that raises is logged and '
                  'skipped so the overnight run continues. If True, stop.'),
    ]

    # ------------------------------------------------------------------ #
    #  construction
    # ------------------------------------------------------------------ #
    def __init__(self, sweep_axes=None, inner_experiment=None, name=None,
                 settings=None, devices=None, sub_experiments=None,
                 log_function=None, data_path=None):
        """
        Args:
            sweep_axes: list describing the grid. Each entry is one axis and may be
                - a (path, values) tuple / list, or
                - a dict {'path': str, 'values': iterable}, or
                - a dict {'path': str, 'start': .., 'stop': .., 'step': ..}, or
                - a dict {'path': str, 'start': .., 'stop': .., 'num': ..}
                Build them easily with the .axis / .step_axis / .linspace_axis
                static helpers below.
            inner_experiment: the experiment(s) to run at each grid point. May be
                a single Experiment instance, a list of them, or an OrderedDict
                {name: experiment}. When several are given they run in order at
                every grid point.
            name, settings, log_function, data_path: passed to Experiment.
            devices, sub_experiments: accepted for signature compatibility with
                Experiment.load_and_append; not used (kept empty).
        """
        # Two ways to hold the inner experiment(s):
        #
        #  * default (headless / _INNER_AS_SUBEXPERIMENT = False): we run them ourselves
        #    and keep them OUT of the base class' sub-experiment dict. That stops the
        #    base run() from connecting their updateProgress signal and re-emitting the
        #    inner's raw 0-100% every point (which would fight our overall-progress math).
        #    We shadow _EXPERIMENTS with {} so a subclass can still advertise a non-empty
        #    class _EXPERIMENTS (letting the GUI loader build the inner + its devices)
        #    without tripping the base 'experiments' setter's key-match assertion.
        #
        #  * _INNER_AS_SUBEXPERIMENT = True: the inner(s) are ALSO real sub-experiments,
        #    so the GUI shows and lets you edit their settings tree. We keep _EXPERIMENTS
        #    intact and let the base run() wire them; progress stays correct because we
        #    override _receive_signal (below) to convert the inner's raw % into overall
        #    sweep progress, and _function skips its own progress hookup for these.
        if inner_experiment is None and sub_experiments:
            inner_experiment = sub_experiments

        if getattr(self, '_INNER_AS_SUBEXPERIMENT', False) and sub_experiments:
            Experiment.__init__(self, name=name, settings=settings,
                                devices=(devices or {}),
                                sub_experiments=sub_experiments,
                                log_function=log_function, data_path=data_path)
        else:
            self._EXPERIMENTS = {}
            Experiment.__init__(self, name=name, settings=settings,
                                devices=(devices or {}),
                                sub_experiments={}, log_function=log_function,
                                data_path=data_path)

        self.set_inner_experiments(inner_experiment)
        self.sweep_axes = self._normalize_axes(sweep_axes) if sweep_axes else []

        # runtime state ------------------------------------------------
        self._current_inner = None     # experiment currently running (for plotting)
        self._last_inner = None        # last experiment that ran (for plotting)
        self._point_idx = 0            # index of grid point currently running
        self._n_points = 0             # total number of grid points
        self._n_done = 0               # completed grid points
        self._skip = False             # skip-current-point flag
        self._output_dir = None
        self._points_dir = None
        self._inner_hdf5_dir = None
        self._manifest_path = None
        self._single_hdf5_path = None  # path of the one combined .h5 (hdf5_single mode)
        self._h5f = None               # open h5py.File handle while a single-file sweep runs
        self._h5_points_group = None   # the '/points' group inside that file
        self._h5_point_counter = 0     # numbered point-group name (0,1,2,...) -> list on read
        self._results = []             # in-memory list of per-point result dicts
        self._skippable = True         # so the GUI's "skip subexperiment" button works

    # ------------------------------------------------------------------ #
    #  axis builders  (use these when defining a sweep)
    # ------------------------------------------------------------------ #
    @staticmethod
    def axis(path, values):
        """An axis over an explicit list of values."""
        return {'path': str(path), 'values': [ParameterSweep._py(v) for v in values]}

    @staticmethod
    def step_axis(path, start, stop, step, include_stop=True):
        """An axis from start to stop in increments of `step` (stop included when
        it lands on the grid). Great for 'MW 2 GHz -> 3 GHz in 100 MHz steps'."""
        if step == 0:
            raise ValueError("step must be non-zero for axis '%s'" % path)
        # Build from an integer count instead of np.arange float accumulation,
        # then clean float dust so e.g. 0.0->0.4 step 0.1 yields exactly 0.3 and
        # not 0.30000000000000004 (which would fail a discrete-value check).
        n = int(round((stop - start) / float(step)))
        last = n + (1 if include_stop else 0)
        values = [ParameterSweep._clean(start + i * step) for i in range(last)]
        return {'path': str(path), 'values': values}

    @staticmethod
    def linspace_axis(path, start, stop, num):
        """An axis of `num` points evenly spaced from start to stop (inclusive)."""
        values = np.linspace(start, stop, int(num), endpoint=True)
        return {'path': str(path), 'values': [ParameterSweep._clean(v) for v in values]}

    @staticmethod
    def _py(v):
        """Coerce numpy scalars to native python so downstream typing is happy."""
        if isinstance(v, np.generic):
            return v.item()
        return v

    @staticmethod
    def _clean(v):
        """Native python + strip floating-point dust from computed grid values."""
        v = ParameterSweep._py(v)
        if isinstance(v, float):
            r = round(v, 12)
            # keep integer-valued floats as-is (e.g. 2.85e9), just de-dust
            return r
        return v

    def _normalize_axes(self, sweep_axes):
        axes = []
        for a in sweep_axes:
            if isinstance(a, (tuple, list)) and len(a) == 2 and not isinstance(a[0], dict):
                path, values = a
                axes.append(self.axis(path, values))
            elif isinstance(a, dict):
                path = a['path']
                if 'values' in a:
                    axes.append(self.axis(path, a['values']))
                elif 'num' in a:
                    axes.append(self.linspace_axis(path, a['start'], a['stop'], a['num']))
                elif 'step' in a:
                    axes.append(self.step_axis(path, a['start'], a['stop'], a['step']))
                else:
                    raise ValueError("axis dict for '%s' needs 'values', 'step', or 'num'" % path)
            else:
                raise TypeError('Unrecognised axis specification: %r' % (a,))
            if len(axes[-1]['values']) == 0:
                raise ValueError("axis '%s' produced zero values - check start/stop/step"
                                 % axes[-1]['path'])
        return axes

    # ------------------------------------------------------------------ #
    #  inner experiment management
    # ------------------------------------------------------------------ #
    def set_inner_experiments(self, inner_experiment):
        """(Re)set the inner experiment(s). Accepts an Experiment, list, or dict."""
        self.inners = OrderedDict()
        if inner_experiment is None:
            return
        if isinstance(inner_experiment, dict):
            for k, v in inner_experiment.items():
                self.inners[str(k)] = v
        elif isinstance(inner_experiment, (list, tuple)):
            for v in inner_experiment:
                self.inners[v.name] = v
        else:  # a single experiment
            self.inners[inner_experiment.name] = inner_experiment

    # ------------------------------------------------------------------ #
    #  dry run: validate the configuration and print the grid, run nothing
    # ------------------------------------------------------------------ #
    def preview(self):
        """Validate the sweep configuration against the inner experiment(s) and
        print the grid WITHOUT touching hardware. Call this before an overnight
        run to catch typos and out-of-range values. Returns the point count."""
        if not self.inners:
            raise RuntimeError('ParameterSweep has no inner experiment.')
        if not self.sweep_axes:
            raise RuntimeError('ParameterSweep has no sweep_axes defined.')
        self._prepare_axes()   # validates + snaps; raises on any problem
        lengths = [len(ax['values']) for ax in self.sweep_axes]
        total = int(np.prod(lengths)) if lengths else 0
        print('ParameterSweep "%s" preview: %d grid points (shape %s)'
              % (self.name, total, 'x'.join(str(n) for n in lengths)))
        for ax in self.sweep_axes:
            print('  axis %-28s %3d values : %s'
                  % (ax['path'], len(ax['values']), self._preview(ax['values'], 12)))
        print('  inner experiment(s) per point: %s' % ', '.join(self.inners.keys()))
        return total

    # ------------------------------------------------------------------ #
    #  the sweep itself
    # ------------------------------------------------------------------ #
    def _function(self):
        if not self.inners:
            raise RuntimeError('ParameterSweep has no inner experiment. Pass '
                               'inner_experiment=... or call set_inner_experiments().')
        if not self.sweep_axes:
            raise RuntimeError('ParameterSweep has no sweep_axes defined.')

        # 1) pre-flight validation + snapping -- fail now, not at 3am --
        self._prepare_axes()

        value_lists = [ax['values'] for ax in self.sweep_axes]
        axis_paths = [ax['path'] for ax in self.sweep_axes]
        self._n_points = int(np.prod([len(v) for v in value_lists]))
        self._n_done = 0
        self._results = []

        self._setup_output_dirs()
        self._write_manifest(final=False)
        self._open_single_hdf5()

        self.log('==== ParameterSweep: %d grid points over %d axes ===='
                 % (self._n_points, len(self.sweep_axes)))
        for p, vals in zip(axis_paths, value_lists):
            self.log('   axis  %-28s : %d values  [%s]'
                     % (p, len(vals), self._preview(vals)))
        for nm in self.inners:
            self.log('   inner experiment per point: %s' % nm)

        # remember originals so we can restore them at the end ---------
        originals = {nm: {'tag': e.settings['tag'],
                          'filename': e.settings['filename'] if 'filename' in e.settings else None,
                          'path': e.settings['path'] if 'path' in e.settings else None,
                          'save': e.settings['save'] if 'save' in e.settings else None}
                     for nm, e in self.inners.items()}

        # connect progress of every inner experiment. If an inner is also a base
        # sub-experiment (GUI-visible mode), the base run() already connected its
        # updateProgress to our _receive_signal, so we must NOT connect a second
        # handler or the bar would update twice.
        for e in self.inners.values():
            if e not in self.experiments.values():
                self._safe_connect(e)

        sweep_start = datetime.datetime.now()
        try:
            for idx, combo in enumerate(itertools.product(*value_lists)):
                if self._abort:
                    self.log('ParameterSweep aborted before point %d.' % (idx + 1))
                    break

                self._point_idx = idx
                coords = OrderedDict((axis_paths[k], self._py(combo[k]))
                                     for k in range(len(axis_paths)))
                coord_tag = self._coord_tag(coords)

                self.log('---- point %d/%d : %s ----'
                         % (idx + 1, self._n_points, self._coord_str(coords)))

                point_record = {
                    'index': idx,
                    'grid_index': self._unravel(idx, [len(v) for v in value_lists]),
                    'coords': dict(coords),
                    'data': OrderedDict(),
                    'status': 'ok',
                    'error': None,
                    'start_time': datetime.datetime.now().isoformat(),
                }

                self._skip = False
                try:
                    for nm, exp in self.inners.items():
                        if self._abort:
                            break

                        # (a) write every swept parameter into this experiment,
                        #     but only the ones that actually belong to it.
                        for path, value in coords.items():
                            if self._path_exists(exp, path):
                                self._set_param(exp, path, value)

                        # (b) let hardware settle if requested
                        st = float(self.settings['settle_time'])
                        if st > 0:
                            time.sleep(st)

                        # (c) tag/route this run so any files it writes are unique.
                        #     We turn the inner's *own* save off so the base class
                        #     save path (csv/mat/pyqtgraph image) does not run in a
                        #     headless overnight session; we save hdf5 ourselves.
                        upd = {'tag': '%s__%s' % (originals[nm]['tag'], coord_tag),
                               'save': False}
                        if originals[nm]['filename'] is not None:
                            upd['filename'] = '%s__%s' % (originals[nm]['filename'], coord_tag)
                        if self._inner_hdf5_dir is not None:
                            upd['path'] = str(self._inner_hdf5_dir)
                        exp.settings.update(upd)

                        # (d) RUN IT
                        self._current_inner = exp
                        exp.run()
                        self._last_inner = exp
                        self._current_inner = None

                        # (e) capture a private copy of the data
                        point_record['data'][nm] = copy.deepcopy(exp.data)

                        # (f) in hdf5_only / full, also write the inner's own per-point
                        # hdf5 (skipped in hdf5_single, where we write the combined file)
                        if self._inner_hdf5_dir is not None and self.settings['save_inner_hdf5']:
                            self._try_inner_hdf5(exp)

                        if self._skip:
                            self.log('   (skipped remaining inner experiments at this point)')
                            break

                except Exception as err:  # one bad point should not kill the night
                    point_record['status'] = 'error'
                    point_record['error'] = ''.join(
                        traceback.format_exception(type(err), err, err.__traceback__))
                    self.log('   ERROR at point %d: %s' % (idx + 1, err))
                    if self.settings['abort_on_error']:
                        self._results.append(point_record)
                        self._save_point(point_record)
                        self._append_point_to_hdf5(point_record)
                        raise

                point_record['end_time'] = datetime.datetime.now().isoformat()

                # save this point immediately -> crash-proof
                self._results.append(point_record)
                self._save_point(point_record)
                self._append_point_to_hdf5(point_record)
                self._n_done += 1

                # progress + ETA. _n_done already counts this point, so the
                # completed fraction is exactly _n_done / _n_points (frac 0.0).
                self._emit_overall_progress(inner_fraction=0.0)
                self._log_eta(sweep_start)
                self._write_manifest(final=False)

        finally:
            # restore inner experiments and disconnect signals. For base sub-experiments
            # the base run() owns the connect/disconnect, so we only undo our own.
            for nm, e in self.inners.items():
                if e not in self.experiments.values():
                    self._safe_disconnect(e)
                restore = {}
                for key in ('tag', 'filename', 'path', 'save'):
                    if key in e.settings and originals[nm][key] is not None:
                        restore[key] = originals[nm][key]
                if restore:
                    e.settings.update(restore)
            self._current_inner = None
            self._close_single_hdf5()

        # 3) consolidate everything into single files -----------------
        self._store_sweep_in_data()
        self._write_manifest(final=True)
        self._consolidate_and_save()

        done = self._n_done
        status = 'aborted' if self._abort else 'complete'
        self.log('==== ParameterSweep %s: %d/%d points done. Data in %s ===='
                 % (status, done, self._n_points, self._output_dir))

    # ------------------------------------------------------------------ #
    #  parameter writing / reading (matches experiment_iterator convention)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_path(path):
        """'frequency_range.start' -> ['frequency_range', 'start'].
        A plain key with no dots (e.g. 'Laser Control') stays a single element."""
        return path.split('.')

    @classmethod
    def _path_exists(cls, exp, path):
        node = exp.settings
        try:
            for k in cls._split_path(path):
                node = node[k]
            return True
        except (KeyError, TypeError):
            return False

    @classmethod
    def _read_param(cls, exp, path):
        node = exp.settings
        for k in cls._split_path(path):
            node = node[k]
        return node

    @classmethod
    def _set_param(cls, exp, path, value):
        """Write a (possibly nested) parameter. Builds {k0:{k1:{... value}}} and
        hands it to settings.update, exactly like the old iterator did but WITHOUT
        the lossy int()/float() coercion that truncated things like OD=0.5."""
        keys = cls._split_path(path)
        nested = value
        for k in reversed(keys):
            nested = {k: nested}
        exp.settings.update(nested)

    # ------------------------------------------------------------------ #
    #  pre-flight validation
    # ------------------------------------------------------------------ #
    def _prepare_axes(self):
        """Validate every axis against the inner experiments and, for discrete
        numeric parameters, SNAP each value onto the exact allowed entry (within
        a small tolerance) so the hardware always receives a legal value. Raises
        a single, readable error listing every problem, before anything runs."""
        problems = []
        for ax in self.sweep_axes:
            path = ax['path']
            # the path must exist in at least one inner experiment
            owners = [nm for nm, e in self.inners.items() if self._path_exists(e, path)]
            if not owners:
                problems.append("axis '%s' is not a setting of any inner experiment "
                                "(check spelling / dotted path)" % path)
                continue

            # find a discrete *numeric* allowed list among the owners, for snapping
            numeric_allowed = None
            for nm in owners:
                try:
                    valid = self._leaf_valid_values(self.inners[nm], path)
                except Exception:
                    valid = None
                if (isinstance(valid, (list, tuple)) and valid
                        and all(isinstance(a, (int, float)) and not isinstance(a, bool)
                                for a in valid)):
                    numeric_allowed = list(valid)
                    break

            if numeric_allowed is not None:
                snapped = []
                for v in ax['values']:
                    ok, near = self._match_discrete(v, numeric_allowed)
                    if ok:
                        snapped.append(near)      # exact allowed value
                    else:
                        snapped.append(v)
                        problems.append(
                            "axis '%s' value %r is not one of the allowed values "
                            "%s" % (path, v, self._preview(numeric_allowed)))
                ax['values'] = snapped
            else:
                # non-numeric discrete list, or a bare type spec: strict check
                for nm in owners:
                    try:
                        valid = self._leaf_valid_values(self.inners[nm], path)
                    except Exception:
                        valid = None
                    if valid is None:
                        continue
                    for v in ax['values']:
                        if self._value_ok(v, valid) is False:
                            problems.append(
                                "axis '%s' value %r is not allowed for experiment "
                                "'%s' (allowed: %s)" % (path, v, nm, self._preview(valid)))

        if problems:
            raise ValueError('ParameterSweep configuration problems:\n  - '
                             + '\n  - '.join(problems))

    @staticmethod
    def _match_discrete(value, allowed):
        """(ok, snapped_value). Matches `value` to an entry of `allowed`, tolerant
        of floating-point dust (np.isclose). Returns the *exact* allowed entry."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return (value in allowed), value
        for a in allowed:
            if value == a or np.isclose(value, a, rtol=1e-6, atol=1e-9):
                return True, a
        return False, value

    @classmethod
    def _leaf_valid_values(cls, exp, path):
        """Best-effort: fetch the valid_values spec for the leaf parameter. Returns
        None if it cannot be determined (then we simply skip strict checking)."""
        keys = cls._split_path(path)
        node = exp.settings
        valid = getattr(node, 'valid_values', None)
        for i, k in enumerate(keys):
            nv = None
            if isinstance(valid, dict) and k in valid:
                nv = valid[k]
            if i < len(keys) - 1:
                node = node[k]
                valid = nv if isinstance(nv, dict) else getattr(node, 'valid_values', None)
            else:
                return nv
        return None

    @staticmethod
    def _value_ok(value, valid):
        """Return True/False if we can judge, or None if we cannot."""
        # discrete list of allowed *values* e.g. [0, 0.5, 1, 2, 3, 4]
        if isinstance(valid, (list, tuple)) and valid and not isinstance(valid[0], type):
            return any(value == v for v in valid)
        # a bare type, e.g. float or int or str
        if isinstance(valid, type):
            if valid in (float, int):
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if valid is bool:
                return isinstance(value, bool)
            return isinstance(value, valid)
        # list whose first element is a type, e.g. [float] meaning "a float"
        if isinstance(valid, (list, tuple)) and valid and isinstance(valid[0], type):
            types = tuple(t for t in valid if isinstance(t, type))
            return isinstance(value, types)
        return None

    # ------------------------------------------------------------------ #
    #  abort / skip
    # ------------------------------------------------------------------ #
    def stop(self):
        """Abort the whole sweep. Already-completed points are already on disk."""
        self._abort = True
        for e in self.inners.values():
            try:
                e.stop()
            except Exception:
                pass
        print('--- stopping ParameterSweep: %s' % self.name)

    def skip_next(self):
        """Skip the rest of the current grid point and move to the next one."""
        self._skip = True
        for e in self.inners.values():
            try:
                e.stop()          # asks the inner experiment to bail out of its run
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  progress reporting
    # ------------------------------------------------------------------ #
    def _safe_connect(self, exp):
        try:
            exp.updateProgress.connect(self._on_inner_progress)
        except Exception:
            pass

    def _safe_disconnect(self, exp):
        try:
            exp.updateProgress.disconnect(self._on_inner_progress)
        except Exception:
            pass

    def _on_inner_progress(self, inner_percent):
        self._emit_overall_progress(inner_fraction=float(inner_percent) / 100.0)

    # When the inner runs as a base sub-experiment (GUI-visible mode), the base run()
    # connects inner.updateProgress -> this slot. The base default would just re-emit
    # the inner's raw 0-100%; we instead fold it into the overall sweep progress so the
    # bar climbs smoothly across the whole grid instead of resetting every point.
    def _receive_signal(self, inner_percent):
        self._on_inner_progress(inner_percent)

    def _emit_overall_progress(self, inner_fraction):
        if self._n_points <= 0:
            return
        frac = (self._n_done + max(0.0, min(1.0, inner_fraction))) / self._n_points
        self.progress = max(0.0, min(100.0, 100.0 * frac))
        try:
            self.updateProgress.emit(int(self.progress))
        except Exception:
            pass

    def _log_eta(self, sweep_start):
        if self._n_done <= 0:
            return
        elapsed = (datetime.datetime.now() - sweep_start).total_seconds()
        per_point = elapsed / self._n_done
        remaining = per_point * (self._n_points - self._n_done)
        self.log('   progress %d/%d (%.1f%%)  elapsed %s  eta %s'
                 % (self._n_done, self._n_points, self.progress,
                    self._fmt(elapsed), self._fmt(remaining)))

    # ------------------------------------------------------------------ #
    #  saving
    # ------------------------------------------------------------------ #
    def _setup_output_dirs(self):
        # Prefer an explicit settings['path']; else data_path; else cwd.
        base = self.settings['path'] if self.settings['path'] else None
        if base and not os.path.isabs(base) and self.data_path:
            base = os.path.join(self.data_path, base)
        if not base:
            base = self.data_path or os.getcwd()

        mode = str(self.settings['save_mode'])
        self._single_hdf5_path = None

        if mode == 'full':
            # timestamped folder with per-point pickles, manifest and combined files
            stamp = datetime.datetime.now().strftime('%y%m%d-%H_%M_%S')
            self._output_dir = os.path.join(base, '%s_%s' % (stamp, self.settings['tag']))
            self._points_dir = os.path.join(self._output_dir, 'points')
            self._inner_hdf5_dir = os.path.join(self._output_dir, 'inner_hdf5')
            for d in (self._output_dir, self._points_dir, self._inner_hdf5_dir):
                os.makedirs(d, exist_ok=True)
            self._manifest_path = os.path.join(self._output_dir, 'sweep_manifest.json')
            return

        # hdf5_single / hdf5_only: no scaffolding, write straight into `base`.
        self._output_dir = base
        self._points_dir = None
        self._manifest_path = None
        if base:
            os.makedirs(base, exist_ok=True)
        if mode == 'hdf5_single':
            self._inner_hdf5_dir = None   # we write the one combined file ourselves
            stamp = datetime.datetime.now().strftime('%y%m%d-%H_%M_%S')
            self._single_hdf5_path = os.path.join(base, '%s_%s.h5' % (self.settings['tag'], stamp))
        else:  # hdf5_only: one .h5 per point via the inner's own saver
            self._inner_hdf5_dir = base

    def _save_point(self, record):
        """Write one grid point to its own pickle right away (crash-proof). Skipped in
        hdf5_only mode (the .h5 is the only artifact then)."""
        if self._points_dir is None:
            return
        fname = os.path.join(self._points_dir, 'point_%06d.pkl' % record['index'])
        try:
            with open(fname, 'wb') as fh:
                pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
            record['file'] = os.path.relpath(fname, self._output_dir)
        except Exception as err:
            self.log('   WARNING could not pickle point %d: %s' % (record['index'], err))
            record['file'] = None

    def _try_inner_hdf5(self, exp):
        try:
            exp.save_hdf5()
        except NotImplementedError:
            pass  # this experiment has no hdf5 saver; the pickle above is enough
        except Exception as err:
            self.log('   WARNING inner save_hdf5 failed: %s' % err)

    # ---- one combined .h5 for the whole sweep (save_mode == 'hdf5_single') -------
    #
    # We write the layout your data analyzer already understands (struct_hdf5 /
    # nv_data_gui conventions): a group whose children are numbered 0,1,2,... which
    # the reader turns into a LIST of points. Each point carries its 2-D image
    # (count_img) DIRECTLY plus its swept coordinates as scalar attributes; the
    # companion 1-D/aux arrays go under an 'aux' subgroup so only the image forms a
    # stack. find_image_stacks() then detects the images across points and the GUI
    # opens on a frame-slider "movie" you can step through. We flush after every
    # point, so a crash still leaves a valid, readable file.
    _PRIMARY_IMAGE_KEYS = ('count_img', 'count_image', 'image', 'raw_img')

    def _open_single_hdf5(self):
        if self._single_hdf5_path is None:
            return
        try:
            import h5py
        except Exception as err:
            self.log('   WARNING h5py unavailable (%s); falling back to per-point '
                     'pickles in the data folder.' % err)
            self._h5f = None
            self._points_dir = self._output_dir      # so _save_point still persists data
            return
        try:
            self._h5f = h5py.File(self._single_hdf5_path, 'w', libver='latest')
            self._h5_point_counter = 0
            # points/ -> numbered subgroups -> read back as a LIST of points
            self._h5_points_group = self._h5f.create_group('points')
            # meta/ -> a normal (named) subgroup the reader treats as a dict.
            # Everything here is a scalar attr or a numeric/string dataset - never a
            # list-valued attribute (those crash the analyzer's scalar cleaner).
            meta = self._h5f.create_group('meta')
            meta.attrs['sweep_name'] = str(self.name)
            meta.attrs['tag'] = str(self.settings['tag'])
            meta.attrs['created'] = datetime.datetime.now().isoformat()
            meta.attrs['n_points_total'] = int(self._n_points)
            meta.attrs['n_axes'] = int(len(self.sweep_axes))
            meta.attrs['axis_paths'] = ' | '.join(str(ax['path']) for ax in self.sweep_axes)
            try:
                meta.create_dataset('shape',
                                    data=np.array([len(ax['values']) for ax in self.sweep_axes],
                                                  dtype=int))
            except Exception:
                pass
            for i, ax in enumerate(self.sweep_axes):
                try:
                    ds = meta.create_dataset('axis_%d_values' % i, data=np.asarray(ax['values']))
                    ds.attrs['path'] = str(ax['path'])
                except Exception:
                    pass
            try:
                meta.attrs['settings_json'] = json.dumps(self._settings_to_plain())
            except Exception:
                pass
            self._h5f.flush()
            self.log('   writing ONE combined HDF5 (your struct_hdf5 layout): %s'
                     % self._single_hdf5_path)
        except Exception as err:
            self.log('   WARNING could not open combined HDF5 (%s); falling back to '
                     'per-point pickles.' % err)
            self._h5f = None
            self._h5_points_group = None
            self._points_dir = self._output_dir

    def _append_point_to_hdf5(self, record):
        if self._h5f is None or self._h5_points_group is None:
            return
        try:
            grp = self._h5_points_group.create_group(str(self._h5_point_counter))
            self._h5_point_counter += 1
            # swept coordinates + bookkeeping as scalar leaves (attributes)
            grp.attrs['index'] = int(record['index'])
            grp.attrs['status'] = str(record.get('status'))
            for k, v in (record.get('coords') or {}).items():
                self._h5_write_value(grp, str(k), v)
            # inner-experiment data. First inner: primary image(s) go straight on the
            # point group so they stack; everything else under 'aux'. Extra inners
            # (rare) go under aux/<name>.
            for pos, (inner_name, data) in enumerate(sorted((record.get('data') or {}).items())):
                if pos == 0:
                    self._write_inner_data(grp, data)
                else:
                    self._write_inner_data(grp.require_group('aux').require_group(str(inner_name)),
                                           data, all_to_this_group=True)
            self._h5f.flush()
        except Exception as err:
            self.log('   WARNING could not write point %s to HDF5: %s'
                     % (record.get('index'), err))

    def _write_inner_data(self, grp, data, all_to_this_group=False):
        if not isinstance(data, dict):
            self._h5_write_value(grp, 'data', data)
            return
        aux = None
        for key, val in data.items():
            primary = (not all_to_this_group
                       and str(key) in self._PRIMARY_IMAGE_KEYS
                       and self._is_2d(val))
            if primary:
                self._h5_write_value(grp, str(key), val)
            else:
                if all_to_this_group:
                    self._h5_write_value(grp, str(key), val)
                else:
                    if aux is None:
                        aux = grp.require_group('aux')
                    self._h5_write_value(aux, str(key), val)

    @staticmethod
    def _is_2d(v):
        try:
            return np.asarray(v).squeeze().ndim == 2
        except Exception:
            return False

    def _h5_write_value(self, grp, name, value):
        """Mirror struct_hdf5._write_value: scalars/str -> attribute, arrays ->
        dataset. Never writes a list/array as an attribute (that is what tripped the
        analyzer's reader)."""
        name = str(name).replace('/', '_')
        if value is None:
            return
        if isinstance(value, str) or np.isscalar(value):
            try:
                grp.attrs[name] = value
            except Exception:
                grp.attrs[name] = str(value)
            return
        try:
            arr = np.asarray(value)
            if arr.dtype == object:
                grp.attrs[name] = str(value)
                return
            if name in grp:
                del grp[name]
            grp.create_dataset(name, data=arr, chunks=True)
        except Exception:
            try:
                grp.attrs[name] = str(value)
            except Exception:
                pass

    def _settings_to_plain(self):
        """self.settings (Parameter) -> a JSON-safe plain dict for the meta block."""
        def conv(v):
            if isinstance(v, dict):
                return {str(k): conv(v[k]) for k in v.keys()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, bool) or v is None:
                return v
            if isinstance(v, (int, float, str)):
                return v
            try:
                return float(v)
            except Exception:
                return str(v)
        try:
            return conv(dict(self.settings))
        except Exception:
            return {}

    def _close_single_hdf5(self):
        if self._h5f is None:
            return
        try:
            if 'meta' in self._h5f:
                self._h5f['meta'].attrs['n_points_done'] = int(self._n_done)
                self._h5f['meta'].attrs['aborted'] = bool(self._abort)
            self._h5f.flush()
            self._h5f.close()
        except Exception:
            pass
        finally:
            self._h5f = None
            self._h5_points_group = None

    def _write_manifest(self, final):
        if self._manifest_path is None:
            return
        manifest = {
            'name': self.name,
            'tag': self.settings['tag'],
            'created': datetime.datetime.now().isoformat(),
            'final': bool(final),
            'n_points_total': self._n_points,
            'n_points_done': self._n_done,
            'aborted': bool(self._abort),
            'axes': [{'path': ax['path'], 'values': ax['values']} for ax in self.sweep_axes],
            'shape': [len(ax['values']) for ax in self.sweep_axes],
            'inner_experiments': list(self.inners.keys()),
            'points': [{'index': r['index'],
                        'grid_index': r['grid_index'],
                        'coords': r['coords'],
                        'status': r['status'],
                        'file': r.get('file')} for r in self._results],
        }
        try:
            with open(self._manifest_path, 'w') as fh:
                json.dump(manifest, fh, indent=2, default=str)
        except Exception as err:
            self.log('   WARNING could not write manifest: %s' % err)

    def _store_sweep_in_data(self):
        """Put a structured summary into self.data so the rest of the framework
        (and any analysis) can find everything in one place."""
        self.data = {
            'axis_paths': [ax['path'] for ax in self.sweep_axes],
            'axis_values': {ax['path']: list(ax['values']) for ax in self.sweep_axes},
            'shape': tuple(len(ax['values']) for ax in self.sweep_axes),
            'inner_experiments': list(self.inners.keys()),
            'points': self._results,          # full per-point records incl. data
            'output_dir': self._output_dir,
        }

    def _consolidate_and_save(self):
        """Authoritative combined pickle + best-effort .mat. Never raises. Skipped in
        hdf5_only mode."""
        if str(self.settings['save_mode']) != 'full':
            return
        combined_pkl = os.path.join(self._output_dir, 'sweep_combined.pkl')
        try:
            with open(combined_pkl, 'wb') as fh:
                pickle.dump(self.data, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self.log('   wrote %s' % combined_pkl)
        except Exception as err:
            self.log('   WARNING could not write combined pickle: %s' % err)

        # best-effort MATLAB export -- a failure here must not lose data
        try:
            from scipy.io import savemat
            mat = {
                'axis_paths': np.array([self._matlab_key(p) for p in self.data['axis_paths']],
                                       dtype=object),
                'shape': np.array(self.data['shape']),
                'axis_values': {self._matlab_key(k): np.array(v)
                                for k, v in self.data['axis_values'].items()},
            }
            points_struct = {}
            for r in self._results:
                key = 'point_%06d' % r['index']
                points_struct[key] = {
                    'coords': {self._matlab_key(k): self._matlab_sanitize(v)
                               for k, v in r['coords'].items()},
                    'status': r['status'],
                    'data': {self._matlab_key(nm): self._matlab_sanitize(d)
                             for nm, d in r['data'].items()},
                }
            mat['points'] = points_struct
            mat_path = os.path.join(self._output_dir, 'sweep_combined.mat')
            savemat(mat_path, mat, long_field_names=True, do_compression=True)
            self.log('   wrote %s' % mat_path)
        except Exception as err:
            self.log('   (MATLAB export skipped: %s -- your data is safe in the .pkl)' % err)

    # The sweep writes its own per-point + combined files inside _function, so the
    # base run()'s end-of-run save hooks (which fire when settings['save'] is on in
    # the GUI) must not fight that or choke on our structured summary dict. Route
    # them to the consolidated writer / make them harmless.
    def save_data_to_matlab(self, *args, **kwargs):
        self._consolidate_and_save()

    def save_data(self, *args, **kwargs):
        self._consolidate_and_save()

    def save_image_to_disk(self, *args, **kwargs):
        pass

    def save_log(self, *args, **kwargs):
        # only write the run's .log text file in full mode
        if str(self.settings['save_mode']) == 'full':
            try:
                Experiment.save_log(self, *args, **kwargs)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  plotting -- delegate to the current / last inner experiment so the
    #  GUI shows the live scan while running and the last scan afterwards.
    # ------------------------------------------------------------------ #
    def _plot_target(self):
        return self._current_inner or self._last_inner

    def plot(self, figure_list):
        tgt = self._plot_target()
        if tgt is not None:
            tgt.plot(figure_list)

    def _plot(self, axes_list):
        tgt = self._plot_target()
        if tgt is not None:
            tgt._plot(axes_list)

    def get_axes_layout(self, figure_list):
        tgt = self._plot_target()
        if tgt is not None:
            return tgt.get_axes_layout(figure_list)
        return Experiment.get_axes_layout(self, figure_list)

    def to_dict(self):
        dictator = Experiment.to_dict(self)
        # use the *actual* class name so GUI-loadable subclasses
        # (e.g. ParameterSweepExperiment) save/reload as themselves.
        dictator[self.name]['class'] = type(self).__name__
        return dictator

    # ------------------------------------------------------------------ #
    #  small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unravel(flat_index, shape):
        return [int(i) for i in np.unravel_index(flat_index, shape)] if shape else [0]

    @staticmethod
    def _short(path):
        return path.split('.')[-1].replace(' ', '_')

    def _coord_tag(self, coords):
        parts = []
        for path, value in coords.items():
            parts.append('%s-%s' % (self._short(path), self._num(value)))
        tag = '_'.join(parts)
        # keep filenames sane
        return tag.replace('+', 'P').replace(' ', '_')[:120]

    def _coord_str(self, coords):
        return ', '.join('%s=%s' % (self._short(p), self._num(v)) for p, v in coords.items())

    @staticmethod
    def _num(v):
        if isinstance(v, float):
            return ('%g' % v)
        return str(v)

    @staticmethod
    def _preview(values, n=6):
        vals = list(values)
        shown = ', '.join(ParameterSweep._num(v) if isinstance(v, (int, float)) else str(v)
                          for v in vals[:n])
        return shown + (' ...' if len(vals) > n else '')

    @staticmethod
    def _fmt(seconds):
        return str(datetime.timedelta(seconds=int(max(0, seconds))))

    @staticmethod
    def _matlab_key(name):
        key = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in str(name))
        if key and key[0].isdigit():
            key = 'v_' + key
        return key or 'field'

    @classmethod
    def _matlab_sanitize(cls, obj):
        """Make an object savemat-friendly. None -> [], lists/dicts recursed."""
        if obj is None:
            return np.array([])
        if isinstance(obj, dict):
            return {cls._matlab_key(k): cls._matlab_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            try:
                arr = np.array(obj)
                if arr.dtype == object:
                    raise ValueError
                return arr
            except Exception:
                return np.array([cls._matlab_sanitize(v) for v in obj], dtype=object)
        return obj
