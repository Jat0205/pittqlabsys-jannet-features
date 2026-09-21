# AQuISS Code Conventions
# For use by Claude Code when converting MATLAB scripts to Python

## Repo overview

AQuISS (Advanced Quantum Information Science System) is an MVC-structured laboratory
automation framework for NV-center quantum sensing experiments. The architecture is
inspired by pylabcontrol (BBN-Q) and b26-toolkit (LISE-B26).

```
src/
├── Controller/          # Hardware device drivers  ← new devices go here
├── Model/
│   └── experiments/     # Experiment definitions   ← new experiments go here
├── View/                # PyQt5 GUI (do not modify)
└── core/                # Base classes and Parameter system (do not modify)
```

Config lives in `config.json` (not tracked in git). Templates are in
`config.sample.json` and `src/config.template.json`.

---

## Exemplar files (Claude Code: read all of these before writing any code)

- Device example: `src/Controller/sg384.py`   (SG384 microwave generator — SCPI/IP)
- Device example: `src/Controller/awg520.py`  (AWG520 — waveform generator)
- Experiment example: `src/Model/experiments/odmr.py`       (ODMR sweep)
- Experiment example: `src/Model/experiments/confocal_scan.py` (2D confocal scan)
- Core base classes:   `src/core/instrument.py`  (Device + Parameter)
- Core base classes:   `src/core/experiment.py`  (Experiment base)
- `__init__.py` exports: `src/Controller/__init__.py`, `src/Model/experiments/__init__.py`

---

## The Parameter system

Settings are declared as a class-level list of `Parameter` objects, NOT as `__init__`
arguments. Import from `src.core`:

```python
from src.core import Parameter

Parameter(name, default_value, type, description_string)
# Examples:
Parameter('frequency', 2.87e9, float, 'ESR frequency in Hz')
Parameter('num_points', 100,   int,   'Number of frequency points in sweep')
Parameter('use_mw',     True,  bool,  'Enable microwave output')
Parameter('label',      'run', str,   'Label for saved data')
```

- `type` must be a Python built-in (`float`, `int`, `bool`, `str`) or a list of
  `Parameter` objects for nested parameter groups.
- Units go in the description string, never in the parameter name
  (use `'frequency'` not `'frequency_GHz'`).
- Bare SI units throughout (Hz, s, T, V — not GHz, µs, mT).

---

## Device class conventions

```python
from src.core import Device, Parameter

class MyNewDevice(Device):
    """One-line summary of what this device is.

    Longer description if needed. Mention the hardware model, interface
    (SCPI/GPIB/USB/serial), and what it does in the experiment.

    Attributes:
        _DEFAULT_SETTINGS: Class-level list of Parameter objects.
        _PROBES: Dict mapping probe key → human-readable description.
    """

    _DEFAULT_SETTINGS = [
        Parameter('ip_address', '192.168.1.100', str,   'IP address of the device'),
        Parameter('port',       4000,             int,   'SCPI port number'),
        Parameter('timeout',    5.0,              float, 'Connection timeout in s'),
        # ... add instrument-specific parameters here
    ]

    _PROBES = {
        'output_status': 'Whether RF output is currently enabled (bool)',
        'frequency':     'Current output frequency in Hz (float)',
    }

    def __init__(self, name='MyNewDevice', settings=None):
        """Initialize device with optional settings override.

        Args:
            name: Human-readable identifier for this device instance.
            settings: Dict of settings to override defaults.
        """
        super().__init__(name, settings)
        self._instrument = None  # actual hardware handle, set in update()

    def update(self, settings):
        """Apply new settings to the hardware.

        Args:
            settings: Dict mapping parameter name → new value.
        """
        # Call super first so self.settings is updated
        super().update(settings)
        # Then push to hardware
        for key, value in settings.items():
            if key == 'frequency':
                self._set_frequency(value)
            # etc.

    def read_probes(self, key):
        """Query a live value from the hardware.

        Args:
            key: One of the keys defined in _PROBES.

        Returns:
            The queried value in SI units.

        Raises:
            KeyError: If key is not in _PROBES.
        """
        if key == 'output_status':
            return self._query_output_status()
        elif key == 'frequency':
            return float(self._query('FREQ?'))
        else:
            raise KeyError(f'{key} is not a valid probe for {self.name}')

    @property
    def is_connected(self):
        """Return True if the hardware connection is active."""
        return self._instrument is not None

    # ------------------------------------------------------------------
    # Private helpers — prefix with single underscore
    # ------------------------------------------------------------------
    def _connect(self):
        """Open the hardware connection. Called lazily from update()."""
        import pyvisa  # import hardware driver INSIDE the method, not at module top
        rm = pyvisa.ResourceManager()
        self._instrument = rm.open_resource(
            f"TCPIP::{self.settings['ip_address']}::GPIB0::INSTR"
        )

    def _query(self, command):
        """Send a SCPI query and return the stripped response string."""
        return self._instrument.query(command).strip()
```

### Device rules
- **Import hardware drivers inside `_connect()` or `update()`**, never at module
  level. The repo must import cleanly on machines without the driver installed.
- `_DEFAULT_SETTINGS` and `_PROBES` are **class attributes**, not instance attributes.
- `update()` must call `super().update(settings)` before touching hardware.
- Private methods are prefixed with `_`. No double underscore (`__`) for lab code.
- Connection/teardown is handled by the base class context manager; do not add a
  separate `close()` unless the hardware requires explicit teardown.

---

## Experiment class conventions

```python
from src.core import Experiment, Parameter
from src.Controller.my_new_device import MyNewDevice

class MyNewExperiment(Experiment):
    """One-line summary of the experiment.

    What physical quantity is being measured, how, and why.
    Reference the relevant lab notebook section or paper if applicable.

    Attributes:
        _DEFAULT_SETTINGS: Experiment parameters (sweep ranges, counts, etc.).
        _DEVICES: Devices required to run this experiment.
        _EXPERIMENTS: Sub-experiments (usually empty dict for leaf experiments).
    """

    _DEFAULT_SETTINGS = [
        Parameter('start_freq',  2.8e9, float, 'Start frequency in Hz'),
        Parameter('end_freq',    2.9e9, float, 'End frequency in Hz'),
        Parameter('num_points',  100,   int,   'Number of frequency steps'),
        Parameter('mw_power',   -20.0,  float, 'Microwave power in dBm'),
        Parameter('averages',    10,    int,   'Number of averages per point'),
        Parameter('save_data',   True,  bool,  'Save data to disk after run'),
        Parameter('tag',         '',    str,   'Optional tag appended to filename'),
    ]

    _DEVICES = {'mw_generator': MyNewDevice(), 'daq': SomeDAQDevice()}
    _EXPERIMENTS = {}

    def _function(self):
        """Core experiment logic. Called by the base class run() method.

        Iterates over parameter space, acquires data, stores in self.data.
        Should update self._current_subscript_stage for GUI progress display.
        Must check self._abort flag regularly to support mid-run cancellation.
        """
        frequencies = np.linspace(
            self.settings['start_freq'],
            self.settings['end_freq'],
            self.settings['num_points'],
        )
        counts = np.zeros(len(frequencies))

        for i, freq in enumerate(frequencies):
            if self._abort:
                break
            self.instruments['mw_generator']['instance'].update({'frequency': freq})
            counts[i] = self._acquire_counts()
            self._current_subscript_stage = {'name': 'sweep', 'index': i}

        self.data = {'frequencies': frequencies, 'counts': counts}

    def _plot(self, axes_list):
        """Update the live plot. Called by the GUI during and after the run.

        Args:
            axes_list: List of pyqtgraph or matplotlib axes provided by the GUI.
                       axes_list[0] is the primary axes for most experiments.
        """
        if self.data is not None and 'frequencies' in self.data:
            axes_list[0].plot(
                self.data['frequencies'] / 1e9,
                self.data['counts'],
                clear=True,
            )

    def _update(self, axes_list):
        """Incremental plot update during a live run (same signature as _plot)."""
        self._plot(axes_list)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _acquire_counts(self):
        """Read photon counts from DAQ, averaged over self.settings['averages']."""
        pass  # implement with actual DAQ calls
```

### Experiment rules
- The experiment logic lives entirely in `_function()`. Do not override `run()`.
- Check `self._abort` in the inner loop to allow GUI-triggered cancellation.
- Store all results in `self.data` as a dict of NumPy arrays. Never store in
  instance attributes outside `self.data`.
- `_plot()` and `_update()` must handle the case where `self.data` is `None`
  (experiment hasn't run yet).
- Device access inside `_function()` goes through
  `self.instruments['device_key']['instance']`, not a direct attribute.

---

## Naming conventions

| Thing | Convention | Examples |
|---|---|---|
| Classes | PascalCase | `SG384Device`, `PulseBlaster`, `ODMRExperiment` |
| Files | snake_case | `sg384.py`, `odmr.py`, `confocal_scan.py` |
| Methods | snake_case verbs | `update()`, `read_probes()`, `_acquire_counts()` |
| Parameters | snake_case nouns | `start_freq`, `num_points`, `mw_power` |
| Private | leading underscore | `_connect()`, `_query()` |
| Constants | UPPER_SNAKE | `DEFAULT_TIMEOUT = 5.0` (rare; use Parameter instead) |

Experiment class names use either:
- `NounVerb` pattern: `ConfocalScan`, `ODMRSweep`, `RabiOscillation`
- Or underscored variant when there are fast/slow variants: `ConfocalScan_Fast`

---

## Config registration (config.json)

Every new device should be registerable via `config.json`:

```json
{
  "devices": {
    "my_new_device": {
      "class": "MyNewDevice",
      "filepath": "src/Controller/my_new_device.py",
      "settings": {
        "ip_address": "192.168.1.105",
        "port": 4000,
        "timeout": 5.0
      }
    }
  }
}
```

---

## `__init__.py` exports

After adding a new file, update the relevant `__init__.py`:

```python
# src/Controller/__init__.py
from .my_new_device import MyNewDevice

# src/Model/experiments/__init__.py
from .my_new_experiment import MyNewExperiment
```

---

## Code style

- **Formatter**: `black`, line length 88. Run `black src/` before committing.
- **Linter**: `flake8`, max line 88, ignoring E203 and W503.
- **Type hints**: required on all public method signatures (mypy strict).
- **Docstrings**: Google style. Every class and every public method gets one.
- **Python version**: 3.8+ compatible. Use `Optional[X]` not `X | None`.

---

## What NOT to do

- Do not import hardware drivers at module level (lazy import inside `_connect()`).
- Do not use global variables or module-level mutable state.
- Do not store experiment results anywhere except `self.data`.
- Do not override `run()` in experiment subclasses.
- Do not add GUI code to Controller or Model files.
- Do not commit `config.json` or `src/View/gui_config.json`.
- Do not use 1-indexed arrays (MATLAB habit); Python/NumPy is 0-indexed throughout.
- Do not use `print()` for status messages in device/experiment classes; use
  Python `logging` module (`import logging; logger = logging.getLogger(__name__)`).
