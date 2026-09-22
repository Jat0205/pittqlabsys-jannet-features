# Written by <Jannet Trabelsi>
# Created for the Dutt Lab (pittqlabsys) - generic GUI-loadable parameter sweep
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
Generic single-experiment, multi-parameter sweep
=================================================

This module gives you a **generic** parameter sweep you run from the GUI: you pick
which experiment to sweep, then pick which of *that experiment's* parameters to sweep
and over what range - all from the settings tree, no code editing per run.

How it works (and why it's built this way)
------------------------------------------
The framework decides an experiment's sub-experiment(s) from a **static** class
attribute ``_EXPERIMENTS`` that the loader reads *before* the object exists, and a JSON
import can only edit the settings of sub-experiments that attribute already names - it
can't introduce a new inner experiment. So "choose the inner experiment purely by
clicking" would require the runtime class-factory the old ExperimentIterator used (the
fragile path). Instead, this file **auto-generates one loadable sweep experiment per
entry** in :data:`SWEEPABLE_EXPERIMENTS` below. Each generated class shows up in the GUI
just like any experiment; selecting e.g. ``ConfocalFastSweep`` *is* choosing the inner.

To make another experiment sweepable, add ONE line to ``SWEEPABLE_EXPERIMENTS`` (display
name -> its module + class). That is the only "programming" involved, it's done once, and
it lives in one place. Everything else - which parameters, what ranges - is pure GUI.

Choosing what to sweep, in the GUI
----------------------------------
The settings tree carries several identical **axis slots** (``sweep_1`` .. ``sweep_N``).
For each axis you want, tick ``enable`` and fill in:

* ``parameter`` - the dotted path into the chosen experiment's settings, exactly as it
  appears in that experiment (e.g. ``Laser Control``, ``Filter Wheel OD``,
  ``frequency_range.start`` for ODMR, or ``MICROWAVE.frequency`` for a confocal scan).
* ``mode`` - ``range_step`` (start..stop by step), ``range_count`` (N points start..stop),
  or ``list`` (explicit comma-separated values in the ``list`` field, e.g.
  ``0,0.5,1,2,3,4`` for the filter wheel).

Axis order = slot order: ``sweep_1`` is the OUTERMOST (slowest / changes least), the
highest-numbered enabled slot varies fastest. Put slow/mechanical knobs (filter wheel) in
low-numbered slots. Non-swept ("fixed") settings go in ``inner_overrides`` as
``path=value`` pairs, e.g. ``acquisition.integration_time=0.002, acquisition.averages=50``.

Every grid point is saved as it finishes (crash-proof), plus a combined ``.pkl`` / ``.mat``
and a manifest - see :mod:`parameter_sweep` for the details. The live plot mirrors whichever
inner scan is running; progress runs smoothly 0->100% across the whole grid.
"""

import warnings
from importlib import import_module

from src.core import Parameter, Experiment
from .parameter_sweep import ParameterSweep


# ===========================================================================
#  REGISTRY - add one line here to make another experiment sweepable.
#  display name (the class name shown in the GUI)  ->  (module path, class name)
# ===========================================================================
SWEEPABLE_EXPERIMENTS = {
    'ODMRSweep':         ('src.Model.experiments.odmr_sweep_continuous',           'ODMRSweepContinuousExperiment'),
    'ConfocalSlowSweep': ('src.Model.experiments.nanodrive_adwin_confocal_scan_slow', 'NanodriveAdwinConfocalScanSlow'),
    'ConfocalFastSweep': ('src.Model.experiments.nanodrive_adwin_confocal_scan_fast', 'NanodriveAdwinConfocalScanFast'),
}

# How many sweep-axis slots to expose in the settings tree.
N_AXES = 4


def _coerce(token):
    """Turn a GUI string token into a bool / int / float, else leave it a string."""
    tok = str(token).strip()
    low = tok.lower()
    if low in ('true', 'false'):
        return low == 'true'
    try:
        if any(c in tok for c in '.eE'):
            return float(tok)
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def _axis_slot(idx, enable, parameter, mode, start, stop, step, count, listvals):
    """One reusable sweep-axis slot for the settings tree."""
    return Parameter('sweep_%d' % idx, [
        Parameter('enable', enable, bool, 'include this axis in the sweep'),
        Parameter('parameter', parameter, str,
                  'dotted path into the chosen experiment settings, e.g. '
                  'frequency_range.start, Laser Control, MICROWAVE.frequency'),
        Parameter('mode', mode, ['range_step', 'range_count', 'list'],
                  'range_step: start..stop by step | range_count: N points start..stop '
                  '| list: explicit comma-separated values in the field below'),
        Parameter('start', start, float, 'range start (range_* modes)'),
        Parameter('stop', stop, float, 'range stop, inclusive (range_* modes)'),
        Parameter('step', step, float, 'range step (range_step mode)'),
        Parameter('count', count, int, 'number of points (range_count mode)'),
        Parameter('list', listvals, str,
                  'explicit values, comma-separated (list mode), e.g. 0,0.5,1,2,3,4'),
    ])


class _ConfigurableSweep(ParameterSweep):
    """A ParameterSweep whose axes come from the settings tree instead of code, so it
    works with ANY inner experiment. Not loaded directly - the concrete, GUI-selectable
    subclasses are generated from SWEEPABLE_EXPERIMENTS at the bottom of this file."""

    _DEVICES = {}
    _EXPERIMENTS = {}          # overridden per generated subclass with the real inner
    N_AXES = N_AXES
    # Show the inner experiment as an editable sub-experiment in the GUI, so its own
    # settings (point_a, point_b, resolution, MICROWAVE, ...) appear and can be changed
    # there. Progress stays correct (see ParameterSweep._receive_signal).
    _INNER_AS_SUBEXPERIMENT = True

    # Defaults pre-filled with the classic Laser x Filter-OD x Frequency example so a
    # freshly-loaded ODMRSweep already does something sensible. Edit freely in the GUI.
    _DEFAULT_SETTINGS = ParameterSweep._DEFAULT_SETTINGS + [
        _axis_slot(1, True, 'frequency_range.start', 'range_step', 2.7e9, 3.0e9, 1.0e8, 4, ''),
        _axis_slot(2, False, 'Laser Control', 'range_step', 0.0, 0.8, 0.1, 9, ''),
        _axis_slot(3, False,  'Filter Wheel OD',       'list',        0.0,   0.0,   0.0,  5, '0,0.5,1,2,3,4'),
        _axis_slot(4, False, '',                      'range_step',  0.0,   1.0,   0.1,  5, ''),
        Parameter('inner_overrides', '', str,
                  'Fixed (non-swept) inner settings applied before the sweep, as '
                  'path=value pairs, comma-separated. '
                  'e.g. acquisition.integration_time=0.002, acquisition.averages=50'),
    ]

    # ------------------------------------------------------------------ #
    #  construction
    # ------------------------------------------------------------------ #
    def __init__(self, name=None, settings=None, devices=None, sub_experiments=None,
                 log_function=None, data_path=None):
        # ParameterSweep.__init__ pulls the loader-built inner out of sub_experiments and
        # shadows _EXPERIMENTS so the base class runs it inline (no double progress bar).
        ParameterSweep.__init__(self, sweep_axes=None, inner_experiment=None,
                                name=name, settings=settings, devices=devices,
                                sub_experiments=sub_experiments,
                                log_function=log_function, data_path=data_path)
        # translate the settings tree into a concrete grid, and push fixed settings down.
        self.sweep_axes = self._axes_from_slots()
        self._apply_inner_overrides()

    # ------------------------------------------------------------------ #
    #  settings-tree  ->  concrete sweep
    # ------------------------------------------------------------------ #
    def refresh_from_settings(self):
        """Rebuild the grid + re-apply fixed inner settings from the current settings."""
        self.sweep_axes = self._axes_from_slots()
        self._apply_inner_overrides()
        return self.sweep_axes

    def _function(self):
        # honour any edits made in the GUI right up until Run is pressed
        self.sweep_axes = self._axes_from_slots()
        self._apply_inner_overrides()
        ParameterSweep._function(self)

    def preview(self):
        self.sweep_axes = self._axes_from_slots()
        self._apply_inner_overrides()
        return ParameterSweep.preview(self)

    def _axes_from_slots(self):
        axes = []
        for i in range(1, self.N_AXES + 1):
            slot = self.settings['sweep_%d' % i]
            if not bool(slot['enable']):
                continue
            path = str(slot['parameter']).strip()
            if not path:
                continue
            mode = str(slot['mode'])
            if mode == 'list':
                vals = self._parse_list(slot['list'])
                if not vals:
                    raise ValueError("axis %d (%s) is in 'list' mode but no values were "
                                     "given in its 'list' field." % (i, path))
                axes.append(ParameterSweep.axis(path, vals))
            elif mode == 'range_count':
                axes.append(ParameterSweep.linspace_axis(
                    path, float(slot['start']), float(slot['stop']), int(slot['count'])))
            else:  # range_step
                axes.append(ParameterSweep.step_axis(
                    path, float(slot['start']), float(slot['stop']), float(slot['step'])))
        return axes

    @staticmethod
    def _parse_list(text):
        out = []
        for tok in str(text).replace(';', ',').split(','):
            tok = tok.strip()
            if tok:
                out.append(_coerce(tok))
        return out

    def _apply_inner_overrides(self):
        """Apply the fixed (non-swept) settings from the inner_overrides text field - but
        only paths the inner actually has, so it stays valid across experiment types."""
        if not self.inners:
            return
        text = str(self.settings['inner_overrides']).strip()
        if not text:
            return
        inner = next(iter(self.inners.values()))
        for chunk in text.replace(';', ',').split(','):
            chunk = chunk.strip()
            if not chunk or '=' not in chunk:
                continue
            path, _, raw = chunk.partition('=')
            path, raw = path.strip(), raw.strip()
            if not path:
                continue
            if self._path_exists(inner, path):
                self._set_param(inner, path, _coerce(raw))
            else:
                self.log("inner_overrides: '%s' is not a setting of this experiment; "
                         "skipped." % path)


# ===========================================================================
#  Generate one concrete, GUI-selectable sweep class per registry entry.
#  Each is a real class in THIS module's namespace, so your GUI's experiment
#  discovery / JSON import (filepath = this file, class = the name) finds it.
# ===========================================================================
def _make_sweep_class(display_name, module_path, class_name):
    module = import_module(module_path)
    inner_cls = getattr(module, class_name)
    doc = ("Parameter sweep of %s.\n\nTick the axes you want in the settings tree "
           "(sweep_1 is outermost), set each parameter's dotted path + range, put any "
           "fixed settings in inner_overrides, then Run." % class_name)
    return type(display_name, (_ConfigurableSweep,), {
        '_DEVICES': {},
        '_EXPERIMENTS': {'inner': inner_cls},   # <-- makes the GUI build this inner + its devices
        '__doc__': doc,
        '__module__': __name__,
    })


SWEEP_CLASSES = {}
for _disp, (_mod, _cls) in SWEEPABLE_EXPERIMENTS.items():
    try:
        _generated = _make_sweep_class(_disp, _mod, _cls)
    except Exception as _err:  # a missing hardware dep for ONE experiment shouldn't break the rest
        warnings.warn("parameter_sweep_experiment: could not build sweep '%s' for %s.%s: %s"
                      % (_disp, _mod, _cls, _err))
        continue
    globals()[_disp] = _generated
    SWEEP_CLASSES[_disp] = _generated

# Back-compat alias: the ODMR sweep also answers to the old class name if it was built.
if 'ODMRSweep' in SWEEP_CLASSES:
    ParameterSweepExperiment = SWEEP_CLASSES['ODMRSweep']
