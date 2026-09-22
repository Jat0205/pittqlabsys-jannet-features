# Written by <Jannet Trabelsi>
import sys, os, glob, json, pickle, time, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '_selftest_stub'))   # provides src.core stub
sys.path.insert(0, HERE)          # provides parameter_sweep

import numpy as np
from src.core import Parameter, Experiment

# ---- faithful in-memory stand-in for h5py (real h5py isn't installed in CI) --------
import types as _types
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
_fake_h5py = _types.ModuleType('h5py'); _fake_h5py.File = _FakeH5File
sys.modules['h5py'] = _fake_h5py


# --- emulate nv_data_gui.load_hdf5_tree reading what our writer produced ----------
def _clean_scalar_like(v):
    if isinstance(v, bytes):
        return v.decode('utf-8', 'ignore')
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(-1)[0].item()
    return v

def _clean_dataset_like(v):
    arr = np.asarray(v)
    return arr.item() if arr.ndim == 0 else arr

def emulate_read_group(g):
    """Mirror nv_data_gui.load_hdf5_tree: attrs -> scalars, datasets -> arrays,
    an all-numeric-keyed subgroup -> list, any other subgroup -> nested dict."""
    node = {}
    for k, val in g.attrs.items():
        node[k] = _clean_scalar_like(val)
    for name, ds in g._datasets.items():
        node[name] = _clean_dataset_like(ds.data)
    for name, sub in g._groups.items():
        childnames = list(sub._groups.keys()) + list(sub._datasets.keys())
        if childnames and all(str(k).isdigit() for k in childnames):
            node[name] = [emulate_read_group(sub._groups[k])
                          for k in sorted(sub._groups.keys(), key=int)]
        else:
            node[name] = emulate_read_group(sub)
    return node

def analyzer_would_stack(tree):
    """Faithful mirror of nv_data_gui.find_image_stacks case 3: a LIST of dicts that
    share a common 2-D field becomes a steppable image stack."""
    from collections import Counter
    found = []
    def scan(node):
        if isinstance(node, dict):
            for v in node.values():
                scan(v)
        elif isinstance(node, list):
            fields = {}
            for item in node:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2:
                            fields.setdefault(k, []).append(np.squeeze(v))
            for k, imgs in fields.items():
                if len(imgs) >= 3:
                    shp = Counter(im.shape for im in imgs).most_common(1)[0][0]
                    cohort = [im for im in imgs if im.shape == shp]
                    found.append((k, np.stack(cohort, 0)))
            for item in node:
                scan(item)
    scan(tree)
    return found


from parameter_sweep import ParameterSweep

# The checks below exercise FULL-mode artifacts (per-point pickles, manifest.json,
# combined .pkl/.mat). The library default is now 'hdf5_single', so make these builds
# use 'full'. TEST 9/10 override the mode to check hdf5_only / hdf5_single.
_orig_ps_init = ParameterSweep.__init__
def _full_mode_init(self, *a, **k):
    _orig_ps_init(self, *a, **k)
    try:
        self.settings.update({'save_mode': 'full'})
    except Exception:
        pass
ParameterSweep.__init__ = _full_mode_init

FAILS = []
def check(cond, msg):
    if cond:
        print("  PASS:", msg)
    else:
        print("  FAIL:", msg)
        FAILS.append(msg)


# ---------------------------------------------------------------------------
# Mock experiments that mimic the real ODMR / confocal: they read self.settings
# and produce data, emit progress, respect self._abort, and save_hdf5().
# ---------------------------------------------------------------------------
class MockODMR(Experiment):
    _DEVICES = {}
    _EXPERIMENTS = {}
    _DEFAULT_SETTINGS = [
        Parameter('frequency_range', [
            Parameter('start', 2.7e9, float, 'start'),
            Parameter('stop', 3.0e9, float, 'stop')]),
        Parameter('microwave', [Parameter('power', -10.0, float, 'power')]),
        Parameter('Laser Control', 0.8, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'laser'),
        Parameter('Filter Wheel OD', 0, [0, 0.5, 1, 2, 3, 4], 'od'),
        Parameter('filename', 'ODMR', str, 'fname'),
        Parameter('sample', '', str, 'sample'),
    ]
    hdf5_calls = []

    def __init__(self, devices=None, experiments=None, name=None, settings=None,
                 log_function=None, data_path=None):
        super().__init__(name, settings, devices, experiments, log_function, data_path)
        self.raise_on_od = None

    def _function(self):
        for i in range(4):
            if self._abort:
                return
            time.sleep(0.0005)
            self.updateProgress.emit(int(100 * (i + 1) / 4))
        lc = self.settings['Laser Control']
        od = self.settings['Filter Wheel OD']
        f0 = self.settings['frequency_range']['start']
        f1 = self.settings['frequency_range']['stop']
        if self.raise_on_od is not None and od == self.raise_on_od:
            raise RuntimeError("simulated hardware glitch at OD=%s" % od)
        freqs = np.linspace(f0, f1, 21)
        sig = (1 - 0.3 * np.exp(-((freqs - 2.87e9) / 5e6) ** 2)) * (lc + 0.1) / (1 + od)
        self.data = {
            'frequencies': freqs,
            'counts_averaged': sig,
            'laser_used': lc,
            'od_used': od,
            'mw_start_used': f0,
            'a_none_field': None,          # exercise None-sanitising for .mat
            'nested': {'k': [1, 2, 3]},
        }

    def save_hdf5(self):
        MockODMR.hdf5_calls.append(
            {'path': self.settings['path'], 'filename': self.settings['filename'],
             'tag': self.settings['tag']})


class MockConfocal(Experiment):
    """Has Laser Control + OD but NO frequency_range (like the real confocal)."""
    _DEVICES = {}
    _EXPERIMENTS = {}
    _DEFAULT_SETTINGS = [
        Parameter('Laser Control', 0.8, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'laser'),
        Parameter('Filter Wheel OD', 0, [0, 0.5, 1, 2, 3, 4], 'od'),
        Parameter('resolution', 1.0, [2.0, 1.0, 0.5], 'res'),
        Parameter('filename', 'confocal', str, 'fname'),
    ]

    def _function(self):
        self.updateProgress.emit(100)
        self.data = {'image': np.ones((4, 4)) * self.settings['Laser Control'],
                     'od_used': self.settings['Filter Wheel OD']}

    def save_hdf5(self):
        raise NotImplementedError  # like an experiment without hdf5 support


def new_dir():
    d = tempfile.mkdtemp(prefix='sweep_test_')
    return d


# ===========================================================================
print("\n[TEST 1] axis builders: inclusive endpoints / discrete values")
ax = ParameterSweep.step_axis('frequency_range.start', 2.0e9, 3.0e9, 1.0e8)
check(len(ax['values']) == 11, "2e9->3e9 step 1e8 gives 11 points (got %d)" % len(ax['values']))
check(abs(ax['values'][-1] - 3.0e9) < 1, "last value is 3.0e9 (got %g)" % ax['values'][-1])
check(all(isinstance(v, float) for v in ax['values']), "values are native python floats")
ax2 = ParameterSweep.linspace_axis('microwave.power', -20, 0, 5)
check(ax2['values'] == [-20.0, -15.0, -10.0, -5.0, 0.0], "linspace endpoints correct")
ax3 = ParameterSweep.axis('Filter Wheel OD', [0, 0.5, 1, 2, 3, 4])
check(ax3['values'] == [0, 0.5, 1, 2, 3, 4], "explicit axis preserved (incl 0.5, no int truncation)")


# ===========================================================================
print("\n[TEST 2] full 3-D sweep end-to-end (the user's exact scenario, small)")
MockODMR.hdf5_calls = []
d = new_dir()
odmr = MockODMR(name='ODMR')
sweep = ParameterSweep(
    inner_experiment=odmr,
    sweep_axes=[
        ParameterSweep.step_axis('Laser Control', 0.0, 0.4, 0.1),       # 5 (0,.1,.2,.3,.4)
        ParameterSweep.axis('Filter Wheel OD', [0, 0.5, 1]),            # 3
        ParameterSweep.step_axis('frequency_range.start', 2.85e9, 2.89e9, 0.02e9),  # 3
    ],
    name='overnight', data_path=d,
)
progress_seen = []
sweep.updateProgress.connect(lambda p: progress_seen.append(p))
sweep.run()

out = sweep._output_dir
check(sweep._n_points == 45, "total grid points = 5*3*3 = 45 (got %d)" % sweep._n_points)
check(sweep._n_done == 45, "all 45 points completed (got %d)" % sweep._n_done)
pkls = glob.glob(os.path.join(out, 'points', 'point_*.pkl'))
check(len(pkls) == 45, "45 per-point pickles on disk (got %d)" % len(pkls))
check(os.path.exists(os.path.join(out, 'sweep_combined.pkl')), "combined pickle written")
check(os.path.exists(os.path.join(out, 'sweep_combined.mat')), "combined .mat written")
check(os.path.exists(os.path.join(out, 'sweep_manifest.json')), "manifest written")
check(max(progress_seen) == 100, "overall progress reached 100 (got %s)" % max(progress_seen))
check(progress_seen == sorted(progress_seen), "progress is monotonically non-decreasing")

# manifest correctness
man = json.load(open(os.path.join(out, 'sweep_manifest.json')))
check(man['final'] is True and man['n_points_done'] == 45, "manifest marked final w/ 45 done")
check(man['shape'] == [5, 3, 3], "manifest shape [5,3,3] (got %s)" % man['shape'])

# spot-check a point's coordinates actually reached the inner experiment's data
first = pickle.load(open(pkls[0], 'rb'))
data0 = first['data']['ODMR']
check(data0['laser_used'] == first['coords']['Laser Control'],
      "Laser Control coord propagated into inner data")
check(data0['od_used'] == first['coords']['Filter Wheel OD'],
      "Filter Wheel OD coord propagated into inner data")
check(data0['mw_start_used'] == first['coords']['frequency_range.start'],
      "frequency_range.start coord propagated into inner data")

# every (laser, od, mw) combo present exactly once
combos = set()
for f in pkls:
    r = pickle.load(open(f, 'rb'))
    combos.add((round(r['coords']['Laser Control'], 6),
                r['coords']['Filter Wheel OD'],
                round(r['coords']['frequency_range.start'], 3)))
check(len(combos) == 45, "45 unique coordinate combinations (got %d)" % len(combos))

# hdf5 saved per point, unique tags, routed into inner_hdf5 dir
check(len(MockODMR.hdf5_calls) == 45, "inner save_hdf5 called 45x (got %d)" % len(MockODMR.hdf5_calls))
check(len(set(c['tag'] for c in MockODMR.hdf5_calls)) == 45, "45 unique inner tags")
check(all(c['path'] == sweep._inner_hdf5_dir for c in MockODMR.hdf5_calls),
      "inner hdf5 routed into the sweep's inner_hdf5 folder")

# inner experiment restored afterwards
check(odmr.settings['tag'] == 'odmr', "inner tag restored to original (got %r)" % odmr.settings['tag'])
check(odmr.settings['filename'] == 'ODMR', "inner filename restored (got %r)" % odmr.settings['filename'])
check(odmr.settings['save'] is False, "inner save flag restored")


# ===========================================================================
print("\n[TEST 3] pre-flight validation catches typos & bad values BEFORE running")
odmr = MockODMR(name='ODMR')
try:
    ParameterSweep(inner_experiment=odmr,
                   sweep_axes=[ParameterSweep.axis('Filter Wheel OD', [0, 0.5, 0.7])],  # 0.7 illegal
                   name='bad', data_path=new_dir()).run()
    check(False, "illegal OD value should raise")
except ValueError as e:
    check('0.7' in str(e) and 'Filter Wheel OD' in str(e), "illegal OD value reported clearly")

odmr = MockODMR(name='ODMR')
try:
    ParameterSweep(inner_experiment=odmr,
                   sweep_axes=[ParameterSweep.axis('nonexistent.param', [1, 2])],
                   name='bad2', data_path=new_dir()).run()
    check(False, "nonexistent path should raise")
except ValueError as e:
    check('nonexistent.param' in str(e), "nonexistent path reported clearly")


# ===========================================================================
print("\n[TEST 4] error handling: one bad point does NOT kill the run")
d = new_dir()
odmr = MockODMR(name='ODMR')
odmr.raise_on_od = 1        # every point with OD==1 will raise
sweep = ParameterSweep(
    inner_experiment=odmr,
    sweep_axes=[ParameterSweep.axis('Filter Wheel OD', [0, 0.5, 1, 2])],  # 4 points, 1 bad
    name='resilient', data_path=d,
    settings={'abort_on_error': False},
)
sweep.run()
statuses = [pickle.load(open(f, 'rb'))['status'] for f in
            sorted(glob.glob(os.path.join(sweep._output_dir, 'points', '*.pkl')))]
check(sweep._n_done == 4, "all 4 points recorded despite 1 failure (got %d)" % sweep._n_done)
check(statuses.count('error') == 1 and statuses.count('ok') == 3,
      "exactly 1 error + 3 ok (got %s)" % statuses)
err_rec = [pickle.load(open(f, 'rb')) for f in glob.glob(os.path.join(sweep._output_dir, 'points', '*.pkl'))
           if pickle.load(open(f, 'rb'))['status'] == 'error'][0]
check('simulated hardware glitch' in (err_rec['error'] or ''), "error traceback captured in the record")

# abort_on_error=True should raise
odmr = MockODMR(name='ODMR'); odmr.raise_on_od = 0
try:
    ParameterSweep(inner_experiment=odmr,
                   sweep_axes=[ParameterSweep.axis('Filter Wheel OD', [0, 0.5])],
                   name='strict', data_path=new_dir(),
                   settings={'abort_on_error': True}).run()
    check(False, "abort_on_error=True should propagate the exception")
except RuntimeError:
    check(True, "abort_on_error=True propagates the exception")


# ===========================================================================
print("\n[TEST 5] abort mid-run keeps completed points, marks manifest aborted")
d = new_dir()
odmr = MockODMR(name='ODMR')
sweep = ParameterSweep(
    inner_experiment=odmr,
    sweep_axes=[ParameterSweep.step_axis('Laser Control', 0.0, 0.8, 0.1)],  # 9 points
    name='abortme', data_path=d,
)
# stop the sweep once 3 points are done
def maybe_stop(_p):
    if sweep._n_done >= 3:
        sweep.stop()
sweep.updateProgress.connect(maybe_stop)
sweep.run()
check(3 <= sweep._n_done < 9, "aborted after ~3 points, not all 9 (got %d)" % sweep._n_done)
man = json.load(open(os.path.join(sweep._output_dir, 'sweep_manifest.json')))
check(man['aborted'] is True, "manifest records aborted=True")
n_pkl = len(glob.glob(os.path.join(sweep._output_dir, 'points', '*.pkl')))
check(n_pkl == sweep._n_done, "every completed point is on disk (%d pkls == %d done)" % (n_pkl, sweep._n_done))


# ===========================================================================
print("\n[TEST 6] multiple inner experiments per point; params routed correctly")
d = new_dir()
odmr = MockODMR(name='ODMR')
conf = MockConfocal(name='Confocal')
sweep = ParameterSweep(
    inner_experiment=[conf, odmr],   # run confocal then odmr at each point
    sweep_axes=[
        ParameterSweep.axis('Filter Wheel OD', [0, 1]),                 # both have this
        ParameterSweep.step_axis('frequency_range.start', 2.86e9, 2.88e9, 0.02e9),  # only ODMR
    ],
    name='multi', data_path=d,
    settings={'save_inner_hdf5': True},
)
sweep.run()
rec = pickle.load(open(sorted(glob.glob(os.path.join(sweep._output_dir, 'points', '*.pkl')))[0], 'rb'))
check(set(rec['data'].keys()) == {'Confocal', 'ODMR'}, "both inner experiments produced data")
check('image' in rec['data']['Confocal'], "confocal data captured")
check(rec['data']['ODMR']['mw_start_used'] == rec['coords']['frequency_range.start'],
      "MW axis only affected the experiment that has it (ODMR)")
check(rec['data']['Confocal']['od_used'] == rec['coords']['Filter Wheel OD'],
      "OD axis affected the confocal too")
check(sweep._n_points == 4, "multi-inner grid is 2*2=4 points")


# ===========================================================================
print("\n[TEST 7] the .mat file loads back and has the expected structure")
from scipy.io import loadmat
d = new_dir()
odmr = MockODMR(name='ODMR')
sweep = ParameterSweep(
    inner_experiment=odmr,
    sweep_axes=[ParameterSweep.axis('Filter Wheel OD', [0, 0.5]),
                ParameterSweep.axis('Laser Control', [0.1, 0.2])],
    name='matcheck', data_path=d)
sweep.run()
m = loadmat(os.path.join(sweep._output_dir, 'sweep_combined.mat'))
check('points' in m and 'axis_values' in m and 'shape' in m, ".mat has points/axis_values/shape")
check(list(m['shape'].ravel()) == [2, 2], ".mat shape == [2,2] (got %s)" % list(m['shape'].ravel()))
print("  (mat top-level keys: %s)" % [k for k in m.keys() if not k.startswith('__')])


# ===========================================================================
print("\n[TEST 8] plotting delegates to inner without crashing")
odmr = MockODMR(name='ODMR')
sweep = ParameterSweep(inner_experiment=odmr,
                       sweep_axes=[ParameterSweep.axis('Filter Wheel OD', [0])],
                       name='plotcheck', data_path=new_dir())
sweep.run()
try:
    sweep.plot(['fig1', 'fig2'])          # after run: should delegate to last inner
    sweep._plot([])
    layout = sweep.get_axes_layout(['fig'])
    check(True, "plot/_plot/get_axes_layout run without error after a sweep")
except Exception as e:
    check(False, "plot delegation raised: %s" % e)


# ===========================================================================
print("\n[TEST 9] hdf5_only mode: only .h5 saved, no folders / json / pkl / mat")
MockODMR.hdf5_calls = []
d9 = new_dir()
odmr = MockODMR(name='ODMR')
sweep = ParameterSweep(inner_experiment=odmr,
                       sweep_axes=[ParameterSweep.axis('Filter Wheel OD', [0, 1]),
                                   ParameterSweep.step_axis('frequency_range.start', 2.85e9, 2.87e9, 0.02e9)],
                       name='h5only', data_path=d9)
sweep.settings.update({'save_mode': 'hdf5_only'})   # library default; set explicitly to bypass the test shim
sweep.run()
check(sweep._output_dir == d9, "writes into the data folder itself (no timestamped subfolder)")
check(sweep._points_dir is None and sweep._manifest_path is None, "no points dir / no manifest path")
check(len(MockODMR.hdf5_calls) == 4, "inner save_hdf5 still called per point (got %d)" % len(MockODMR.hdf5_calls))
check(all(c['path'] == d9 for c in MockODMR.hdf5_calls), "each .h5 routed straight into the data folder")
check(glob.glob(os.path.join(d9, '**', '*.pkl'), recursive=True) == [], "no .pkl anywhere")
check(glob.glob(os.path.join(d9, '**', '*.mat'), recursive=True) == [], "no .mat anywhere")
check(glob.glob(os.path.join(d9, '**', '*.json'), recursive=True) == [], "no .json anywhere")
check(glob.glob(os.path.join(d9, '**', '*.log'), recursive=True) == [], "no .log anywhere")
check(not os.path.isdir(os.path.join(d9, 'points')) and not os.path.isdir(os.path.join(d9, 'inner_hdf5')),
      "no points/ or inner_hdf5/ subfolders")


# ===========================================================================
print("\n[TEST 10] hdf5_single: ONE .h5 in the struct_hdf5 layout your analyzer reads")
_FakeH5File.instances = []
d10 = new_dir()
conf = MockConfocal(name='confocal')          # emits a 2-D 'image' like the real scan
sweep = ParameterSweep(inner_experiment=conf,
                       sweep_axes=[ParameterSweep.axis('Filter Wheel OD', [0, 1]),
                                   ParameterSweep.step_axis('Laser Control', 0.1, 0.3, 0.1)],  # 2x3 = 6
                       name='onefile', data_path=d10)
sweep.settings.update({'save_mode': 'hdf5_single'})
sweep.run()

check(len(_FakeH5File.instances) == 1, "exactly ONE HDF5 file opened (got %d)" % len(_FakeH5File.instances))
f = _FakeH5File.instances[-1]
check(f.closed is True, "file closed cleanly")
tree = emulate_read_group(f)                  # what your analyzer's loader would build
check(isinstance(tree.get('points'), list), "'points' reads back as a LIST of points")
check(len(tree['points']) == 6, "all 6 grid points present (got %d)" % len(tree.get('points', [])))
p0 = tree['points'][0]
check(isinstance(p0.get('image'), np.ndarray) and np.squeeze(p0['image']).ndim == 2,
      "each point holds its 2-D image directly")
check('Filter Wheel OD' in p0 and 'Laser Control' in p0, "swept coordinates stored as point scalars")
check('aux' in p0 and isinstance(p0['aux'], dict), "companion (non-image) data tucked under aux")
stacks = analyzer_would_stack(tree)
names = [k for k, _ in stacks]
check(names == ['image'], "analyzer finds exactly one image stack ('image'), no competing stacks")
check(dict(stacks)['image'].shape[0] == 6, "stack has one frame per grid point (got %d)"
      % dict(stacks)['image'].shape[0])
check(isinstance(tree.get('meta'), dict) and tree['meta'].get('n_points_done') == 6,
      "meta reads back as a dict recording n_points_done")
check(glob.glob(os.path.join(d10, '**', '*.json'), recursive=True) == [], "no .json written")
check(glob.glob(os.path.join(d10, '**', '*.mat'), recursive=True) == [], "no .mat written")


# ===========================================================================
print("\n" + "=" * 70)
if FAILS:
    print("RESULT: %d CHECK(S) FAILED" % len(FAILS))
    for f in FAILS:
        print("   - " + f)
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED")
