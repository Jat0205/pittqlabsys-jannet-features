#config can tell us which devices the server can host
from src.core import Parameter, Experiment
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import asyncio

class multiple_device_server_experiment(Experiment):
    _DEVICES = {
            'spectrum_analyzer': 'spectrum_analyzer',
            'microdrive': 'microdrive',
            'nanodrive': 'nanodrive',
            'adwin': 'adwin',
            'amscope_camera': 'amscope_camera',
            'sg384': 'sg384'
    }
    _DEFAULT_SETTINGS = [
        Parameter('spectrum_analyzer_enable', False, bool, 'Enable Spectrum Analyzer'),
        Parameter('microdrive_enable', False, bool, 'Enable microdrive'),
        Parameter('nanodrive_enable', False, bool, 'Enable Nanodrive'),
        Parameter('adwin_enable', False, bool, 'Enable Adwin'),
        Parameter('camera_enable', False, bool, 'Enable camera'),
        Parameter('sg384_enable', False, bool, 'Enable Microwave'),
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
        # Server state management
        self.server_running = True
        self.device_instance = {}
        self._uvicorn_server = None
        # Create FastAPI app with lifespan
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            print("Server starting...")
            yield
            # Shutdown
            print("Server shutting down...")
            self.server_running = False
        self.app = FastAPI(lifespan=lifespan, title=f"PittQLabSys Server - Multiple devices experiment")
        self._lock = asyncio.Lock()  # Prevent concurrent device access
    def _register_routes(self):
        app = self.app
        from fastapi import Body
        @app.post("/shutdown")
        async def shutdown_server(background_tasks: BackgroundTasks):
            """Endpoint for GUI to shutdown server"""
            background_tasks.add_task(self.graceful_shutdown)
            return {"message": "Server shutdown initiated"}
        @app.post("/update/{dev_name}")
        def update(dev_name, payload: dict = Body(...)):
            """
            Expects JSON body like:
            {
                "x_pos": 5.0,
                "y_pos": 10.0
            }
            """
            self.device_instance[dev_name].update(payload)
            #return {"status": "ok", "data": payload}
            return payload
        @app.get("/read/{dev_name}/{key}")
        def read_probes(dev_name, key: str):
            return self.device_instance[dev_name].read_probes(key)
        @app.post("/write/{dev_name}/{message}")
        def write(dev_name, message: str):
            return self.device_instance[dev_name].write(message)
        @app.get("/probes/{dev_name}")
        def list_probes(dev_name):
            return self.device_instance[dev_name]._PROBES
        @app.get("/settings/{dev_name}")
        def list_settings(dev_name):
            """Return the device's default settings (safe for all Parameter formats)."""
            defaults = getattr(self.device_instance[dev_name], "_DEFAULT_SETTINGS", None)
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
    async def graceful_shutdown(self):
        """Perform graceful shutdown - used by both /shutdown endpoint and stop() method"""
        print("Initiating graceful shutdown...")
        self.server_running = False
        # Close all device connections
        for device_id, device in self.device_instance.items():
            if hasattr(device, 'close'):
                await device.close()
            print(f"Closed device: {device_id}")
        await asyncio.sleep(1)
        # Stop the uvicorn server if it exists
        if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
            self._uvicorn_server.should_exit = True
    def _function(self, host="0.0.0.0", port=5000):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        self.active_devices = [
            name for name in self._DEVICES
            if self.settings.get(f"{name}_enable", False)
        ]
        for dev_name in self.active_devices:
            self.device_instance[dev_name] = self.devices[dev_name]['instance']
        self._register_routes()
        self.host = host
        self.port = port
        # Store the uvicorn server instance for later shutdown
        config = uvicorn.Config(self.app, host=self.host, port=self.port)
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()
    def stop(self):
        print("[INFO] Stopping all device servers...")
        for subexperiment in list(self.experiments.values()):
            subexperiment.stop()
        print(('--- stopping: ', self.name))
        self._abort = True
        # Trigger graceful shutdown
        if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
            print(f"[INFO] Requesting server shutdown...")
            self._uvicorn_server.should_exit = True
        # Also set the running flag
        self.server_running = False
        print("[INFO] Server stop requested.")
    def _update(self):
        pass