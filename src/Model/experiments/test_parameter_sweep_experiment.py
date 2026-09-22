#!/usr/bin/env python3
# Written by <Jannet Trabelsi>
"""
Offline test for the generic, registry-driven parameter_sweep_experiment.

No hardware, no PyQt, no real `src` package: it stubs src.core (Parameter/Experiment)
and the three inner experiment modules, then loads parameter_sweep_experiment.py so its
registry generates the sweep classes, and exercises them exactly as the GUI's loader
would (handing the inner in through sub_experiments).

    python test_parameter_sweep_experiment.py
"""
import sys, os, types, importlib.util, tempfile, json, glob, datetime, shutil

OUT = os.path.dirname(os.path.abspath(__file__))
STUB_CORE = os.path.join(OUT, '_selftest_stub', 'src', 'core', '__init__.py')


# ---- faithful in-memory stand-in for h5py (real h5py isn't installed in CI) --------
class _FakeAttrs(dict):
    pass
class _FakeGroup:
    def __init__(self, name):
        self.name = name; self.attrs = _FakeAttrs(); self._groups = {}; self._datasets = {}
    def create_group(self, n):
        g = _FakeGroup(n); self._groups[n] = g; return g
    def require_group(self, n):
        return self._groups.get(n) or self.create_group(n)
    def create_dataset(self, n, data=None, **kw):
        d = _FakeDataset(n, data); self._datasets[n] = d; return d
    def __contains__(self, n):
        return n in self._groups or n in self._datasets
    def __getitem__(self, n):
        if n in self._groups: return self._groups[n]
        if n in self._datasets: return self._datasets[n]
        raise KeyError(n)
    def __delitem__(self, n):
        self._groups.pop(n, None); self._datasets.pop(n, None)
class _FakeDataset:
    def __init__(self, name, data):
        self.name = name; self.data = data; self.attrs = _FakeAttrs()
class _FakeH5File(_FakeGroup):
    instances = []
    def __init__(self, path, mode='r', **kw):
        _FakeGroup.__init__(self, '/'); self.path = path; self.mode = mode
        self.closed = False; self.flushes = 0; _FakeH5File.instances.append(self)
    def flush(self): self.flushes += 1
    def close(self): self.closed = True
    def __enter__(self): return self
    def __exit__(self, *a): self.close(); return False


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec); sys.modules[modname] = m
    spec.loader.exec_module(m); return m


def _build_env():
    src = types.ModuleType('src'); src.__path__ = []; sys.modules['src'] = src
    core = _load('src.core', STUB_CORE); src.core = core
    _load('src.core.parameter_sweep', os.path.join(OUT, 'parameter_sweep.py'))
    Parameter, Experiment = core.Parameter, core.Experiment
    fake = types.ModuleType('h5py'); fake.File = _FakeH5File
    sys.modules['h5py'] = fake

    model = types.ModuleType('src.Model'); model.__path__ = []
    sys.modules['src.Model'] = model; src.Model = model
    exps = types.ModuleType('src.Model.experiments'); exps.__path__ = []
    sys.modules['src.Model.experiments'] = exps; model.experiments = exps

    def reg(modname, clsobj, clsname):
        mod = types.ModuleType(modname); setattr(mod, clsname, clsobj)
        sys.modules[modname] = mod; setattr(exps, modname.split('.')[-1], mod)

    class _Base(Experiment):
        _DEVICES = {}; _EXPERIMENTS = {}
        def __init__(self, **kw):
            Experiment.__init__(self, **kw); self.runs = []
        def run(self):
            self.is_running = True; self._abort = False
            self.updateProgress.emit(50)
            snap = {'Laser Control': self.settings['Laser Control'],
                    'Filter Wheel OD': self.settings['Filter Wheel OD'],
                    'tag': self.settings['tag'], 'filename': self.settings['filename']}
            snap.update(self._extra())
            self.runs.append(snap); self.data = {'y': [1, 2, 3]}
            self.updateProgress.emit(100); self.is_running = False
        def _extra(self): return {}
        def save_hdf5(self):
            base = self.settings['path']
            if base:
                os.makedirs(base, exist_ok=True)
                t = datetime.datetime.now().strftime('%H_%M_%S_%f')
                open(os.path.join(base, '%s_%s.h5' % (os.path.splitext(self.settings['filename'])[0], t)), 'w').write('x')
        def plot(self, fl): pass
        def _plot(self, al): pass
        def get_axes_layout(self, fl): return []

    class MockODMR(_Base):
        _DEFAULT_SETTINGS = [
            Parameter('frequency_range', [Parameter('start', 2.7e9, float, 's'), Parameter('stop', 3.0e9, float, 's')]),
            Parameter('acquisition', [Parameter('integration_time', 0.001, float, 'i'), Parameter('averages', 10, int, 'a')]),
            Parameter('Laser Control', 0.8, [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'l'),
            Parameter('Filter Wheel OD', 0, [0,0.5,1,2,3,4], 'od'),
            Parameter('filename', 'ODMR', str, 'f'),
        ]
        def _extra(self):
            return {'freq_start': self.settings['frequency_range']['start'],
                    'integration_time': self.settings['acquisition']['integration_time'],
                    'averages': self.settings['acquisition']['averages']}

    class MockConfocal(_Base):
        _DEFAULT_SETTINGS = [
            Parameter('MICROWAVE', [Parameter('enable', False, [True, False], 'e'),
                                    Parameter('frequency', 2.0e9, float, 'fr'),
                                    Parameter('power', -10.0, float, 'p')]),
            Parameter('time_per_pt', 2.0, float, 't'),
            Parameter('Laser Control', 0.8, [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], 'l'),
            Parameter('Filter Wheel OD', 0, [0,0.5,1,2,3,4], 'od'),
            Parameter('filename', 'confocal', str, 'f'),
        ]
        def _extra(self):
            return {'mw_freq': self.settings['MICROWAVE']['frequency']}

    reg('src.Model.experiments.odmr_sweep_continuous', MockODMR, 'ODMRSweepContinuousExperiment')
    reg('src.Model.experiments.nanodrive_adwin_confocal_scan_slow', MockConfocal, 'NanodriveAdwinConfocalScanSlow')
    reg('src.Model.experiments.nanodrive_adwin_confocal_scan_fast', MockConfocal, 'NanodriveAdwinConfocalScanFast')
    return MockODMR, MockConfocal


def main():
    MockODMR, MockConfocal = _build_env()
    pse = _load('src.Model.experiments.parameter_sweep_experiment',
                os.path.join(OUT, 'parameter_sweep_experiment.py'))

    fails = []
    def check(c, m):
        print(("  PASS: " if c else "  FAIL: ") + m)
        if not c: fails.append(m)

    check(set(pse.SWEEP_CLASSES) == {'ODMRSweep', 'ConfocalSlowSweep', 'ConfocalFastSweep'}, "registry generated all 3 sweep classes")
    check(issubclass(pse.ConfocalFastSweep, pse.ParameterSweep), "generated class is a ParameterSweep subclass")
    check(pse.ODMRSweep._EXPERIMENTS == {'inner': MockODMR}, "generated _EXPERIMENTS points at the real inner")
    check(pse.ConfocalSlowSweep.__module__ == 'src.Model.experiments.parameter_sweep_experiment', "generated __module__ is this file")

    tmp = tempfile.mkdtemp(prefix='gen_')
    try:
        inner = MockODMR(name='inner')
        sw = pse.ODMRSweep(name='odmr_sweep', sub_experiments={'inner': inner}, data_path=tmp)
        check(sw.experiments.get('inner') is inner, "inner IS a visible/editable sub-experiment (shows its settings in the GUI)")
        check([a['path'] for a in sw.sweep_axes] == ['Laser Control', 'Filter Wheel OD', 'frequency_range.start'], "default axes from settings tree")
        check([len(a['values']) for a in sw.sweep_axes] == [9, 6, 11], "default axis sizes 9x6x11")
        check(sw.sweep_axes[1]['values'] == [0, 0.5, 1, 2, 3, 4], "list-mode OD parsed")

        inner2 = MockODMR(name='inner')
        sw2 = pse.ODMRSweep(name='odmr2', sub_experiments={'inner': inner2}, data_path=tmp, settings={
            'save_mode': 'full',
            'sweep_1': {'start': 0.0, 'stop': 0.2, 'step': 0.1},
            'sweep_2': {'mode': 'list', 'list': '0, 1'},
            'sweep_3': {'mode': 'range_count', 'start': 2.0e9, 'stop': 2.2e9, 'count': 3},
            'inner_overrides': 'acquisition.integration_time=0.002, acquisition.averages=50',
        })
        check([len(a['values']) for a in sw2.sweep_axes] == [3, 2, 3], "edited axes 3x2x3 (range_count)")
        check(inner2.settings['acquisition']['averages'] == 50, "inner_overrides applied")
        seen = []; fin = {'v': False}
        sw2.updateProgress.connect(lambda p: seen.append(p)); sw2.finished.connect(lambda: fin.__setitem__('v', True))
        check(sw2.preview() == 18, "preview() == 18")
        sw2.run()
        check(len(inner2.runs) == 18 and inner2.runs[1]['freq_start'] == 2.1e9, "ran 18x, last slot fastest")
        check(inner2.runs[0]['averages'] == 50, "fixed averages seen during the run")
        man = json.load(open(os.path.join(sw2._output_dir, 'sweep_manifest.json')))
        check(man['final'] and man['shape'] == [3, 2, 3], "manifest final 3x2x3")
        check(len(glob.glob(os.path.join(sw2._output_dir, 'points', '*.pkl'))) == 18, "18 point pickles")
        check(seen and seen[-1] == 100 and fin['v'], "progress hit 100% + finished emitted")

        cinner = MockConfocal(name='inner')
        cs = pse.ConfocalSlowSweep(name='conf_sweep', sub_experiments={'inner': cinner}, data_path=tmp, settings={
            'sweep_1': {'enable': True, 'parameter': 'Laser Control', 'mode': 'list', 'list': '0.2, 0.5'},
            'sweep_2': {'enable': False},
            'sweep_3': {'enable': True, 'parameter': 'MICROWAVE.frequency', 'mode': 'range_step',
                        'start': 2.80e9, 'stop': 2.90e9, 'step': 0.05e9},
            'sweep_4': {'enable': False},
        })
        check([a['path'] for a in cs.sweep_axes] == ['Laser Control', 'MICROWAVE.frequency'], "confocal: GUI-picked paths")
        cs.run()
        check(len(cinner.runs) == 6 and cinner.runs[1]['mw_freq'] == 2.85e9, "confocal ran 6x, MW freq fastest")

        check(sw.to_dict()['odmr_sweep']['class'] == 'ODMRSweep', "to_dict labels ODMRSweep")
        check(cs.to_dict()['conf_sweep']['class'] == 'ConfocalSlowSweep', "to_dict labels ConfocalSlowSweep")

        # GUI-edit visibility: the inner is editable; a change to a NON-swept inner
        # setting (as a user would make in the GUI tree) is what the sweep actually uses.
        gi = MockODMR(name='inner')
        gsw = pse.ODMRSweep(name='odmr_edit', sub_experiments={'inner': gi}, data_path=tmp, settings={
            'sweep_1': {'start': 0.0, 'stop': 0.1, 'step': 0.1},
            'sweep_2': {'enable': False}, 'sweep_3': {'enable': False},
        })
        check('inner' in gsw.experiments, "inner exposed for GUI editing")
        gsw.experiments['inner'].settings.update({'acquisition': {'integration_time': 0.005}})  # user edit
        gsw.run()
        check(abs(gi.runs[0]['integration_time'] - 0.005) < 1e-12, "edit to inner's non-swept setting is used at run time (not guessed)")

        # save behaviour. Default is hdf5_single: ONE .h5 with all iterations inside.
        import glob as _g
        hi = MockODMR(name='inner')
        hsw = pse.ODMRSweep(name='odmr_single', sub_experiments={'inner': hi}, data_path=tmp, settings={
            'sweep_1': {'start': 0.0, 'stop': 0.1, 'step': 0.1},  # 2 points
            'sweep_2': {'enable': False}, 'sweep_3': {'enable': False},
        })
        check(str(hsw.settings['save_mode']) == 'hdf5_single', "save_mode defaults to hdf5_single")
        _FakeH5File.instances = []
        hsw.run()
        check(len(_FakeH5File.instances) == 1, "one combined .h5 opened")
        f = _FakeH5File.instances[-1]
        pts = f._groups['points']._groups
        check(set(pts.keys()) == {'0', '1'}, "points are numbered groups (read back as a list): %s" % sorted(pts))
        check('Laser Control' in pts['0'].attrs, "each point stores its swept coordinate(s) as attributes")
        check(int(f._groups['meta'].attrs['n_points_done']) == 2 and f.closed, "meta records n_points_done; file closed")

        # hdf5_only: one separate .h5 per point (the inner writes its own file)
        h5dir = tempfile.mkdtemp(prefix='perpoint_')
        pi = MockODMR(name='inner')
        psw = pse.ODMRSweep(name='odmr_perpoint', sub_experiments={'inner': pi}, data_path=h5dir, settings={
            'save_mode': 'hdf5_only',
            'sweep_1': {'start': 0.0, 'stop': 0.1, 'step': 0.1},
            'sweep_2': {'enable': False}, 'sweep_3': {'enable': False},
        })
        psw.run()
        check(len(_g.glob(os.path.join(h5dir, '*.h5'))) == 2, "hdf5_only: one .h5 per point straight in the folder")
        check(_g.glob(os.path.join(h5dir, '**', '*.json'), recursive=True) == [], "hdf5_only: no json")
        check(not os.path.isdir(os.path.join(h5dir, 'points')), "hdf5_only: no points/ subfolder")
        shutil.rmtree(h5dir, ignore_errors=True)

        print("\n" + "=" * 60)
        if fails:
            print("RESULT: %d CHECK(S) FAILED" % len(fails))
            for f in fails: print("   - " + f)
            return 1
        print("RESULT: ALL CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
