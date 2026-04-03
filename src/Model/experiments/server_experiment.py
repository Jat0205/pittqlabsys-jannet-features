#config can tell us which devices the server can host
from src.core import Parameter, Experiment
from fastapi import FastAPI
import uvicorn
import asyncio

class server_experiment(Experiment):
    _DEVICES = {
        'spectrum_analyzer': 'spectrum_analyzer',
        'nanodrive': 'nanodrive',
        'adwin': 'adwin',
    }

    _DEFAULT_SETTINGS = [
        Parameter('selected_device','spectrum_analyzer', list(_DEVICES.keys()), 'Choose which device to use')
    ]

    _EXPERIMENTS = {}

    def __init__(self, devices, experiments=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Initializes and connects to devices
        Args:
            name (optional): name of experiment, if empty same as class name
            settings (optional): settings for this experiment, if empty same as default settings
        """
        super().__init__(name, settings=settings, sub_experiments=experiments, devices=devices,
                         log_function=log_function, data_path=data_path)
        # Determine which device the user selected in the settings
        selected_device_name = self.settings.get('selected_device')
        if selected_device_name not in self._DEVICES:
            raise ValueError(f"Selected device '{selected_device_name}' not found in _DEVICES")

        # Get the instance of the selected device
        self.selected_device = self.devices[selected_device_name]['instance']

        # Create FastAPI app
        self.app = FastAPI(title=f"PittQLabSys Server - {selected_device_name}")
        self._lock = asyncio.Lock()  # Prevent concurrent device access
        self._register_routes()


    def _register_routes(self):
        app = self.app

        from fastapi import Body

        @app.post("/update")
        def update(payload: dict = Body(...)):
            """
            Expects JSON body like:
            {
                "x_pos": 5.0,
                "y_pos": 10.0
            }
            """
            self.selected_device.update(payload)
            #return {"status": "ok", "data": payload}
            return payload

        @app.get("/read/{key}")
        def read_probes(key: str):
            return self.selected_device.read_probes(key)

        @app.post("/write/{message}")
        def write(message: str):
            return self.selected_device.write(message)

        @app.get("/probes")
        def list_probes():
            return self.selected_device._PROBES

        @app.get("/settings")
        def list_settings():
            """Return the device's default settings (safe for all Parameter formats)."""
            defaults = getattr(self.selected_device, "_DEFAULT_SETTINGS", None)
            if defaults is None:
                return []
            # Case 1: _DEFAULT_SETTINGS is a Parameter container
            if isinstance(defaults, Parameter):
                settings = []
                for key, value in defaults.items():
                    # Use .get() everywhere — fall back to None or empty string if missing
                    valid_vals = defaults.valid_values.get(key, None)
                    info = defaults.info.get(key, "")
                    units = getattr(defaults, "_units", {}).get(key, "")
                    # Handle if valid_vals is not a list or type (for display)
                    if isinstance(valid_vals, type):
                        valid_vals_str = valid_vals.__name__
                        choices = None
                    elif isinstance(valid_vals, list):
                        valid_vals_str = "list"
                        choices = valid_vals
                    else:
                        valid_vals_str = str(type(valid_vals))
                        choices = None
                    settings.append({
                        "name": key,
                        "default": value,
                        "type": valid_vals_str,
                        "description": info,
                        "choices": choices,
                        "units": units or None,
                    })
                return settings
            # Case 2: _DEFAULT_SETTINGS is a list or tuple of Parameter objects
            elif isinstance(defaults, (list, tuple)):
                settings = []
                for param in defaults:
                    if not isinstance(param, Parameter):
                        continue
                    key = getattr(param, "name", None)
                    value = list(param.values())[0] if param else None
                    valid_vals = list(param.valid_values.values())[0] if param.valid_values else None
                    info = list(param.info.values())[0] if param.info else ""
                    units = list(param.units.values())[0] if param.units else ""
                    settings.append({
                        "name": key,
                        "default": value,
                        "type": str(valid_vals),
                        "description": info,
                        "choices": valid_vals if isinstance(valid_vals, list) else None,
                        "units": units or None,
                    })
                return settings
            # Case 3: Fallback (unknown type)
            else:
                raise ValueError(f"Unexpected _DEFAULT_SETTINGS type: {type(defaults)}")

    def _function(self, host="0.0.0.0", port=5000):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        self.host = host
        self.port = port
        uvicorn.run(self.app, host=self.host, port=self.port)

    def _update(self):
        pass