#python -m src.Controller.multiple_device_client

from src.core import Device
import requests

class Multiple_Device_Client(Device):
    SERVER_URL = "http://192.168.2.4:5000"

    def __init__(self, dev_name, *args, **kwargs):
        # Create the cache *before* calling super().__init__
        # preventing Device __getattr__ and __setattr__ from breaking.
        object.__setattr__(self, "_probes_cache", {})

        super().__init__(dev_name, *args, **kwargs)

        # Now safely fetch data from the server
        self.dev_name = dev_name
        self._DEFAULT_SETTINGS = self.get_settings()
        self._probes_cache = self.get_probes()

    def update(self, settings: dict):
        """Send updated settings to the remote device server."""
        response = requests.post(f"{self.SERVER_URL}/update/{self.dev_name}", json=settings)
        response.raise_for_status()
        return response.json()

    def read_probes(self, key=None):
        """Fetch one or more probes from the server."""
        if key is None:
            all_probes = {}
            for k in self._PROBES.keys():
                all_probes[k] = self.read_probes(k)
            return all_probes

        response = requests.get(f"{self.SERVER_URL}/read/{self.dev_name}/{key}")
        response.raise_for_status()
        return response.json()

    def get_settings(self):
        """Fetch available settings from the server."""
        response = requests.get(f"{self.SERVER_URL}/settings/{self.dev_name}")
        response.raise_for_status()
        return response.json()

    def get_probes(self):
        """Fetch available probes from the server."""
        response = requests.get(f"{self.SERVER_URL}/probes/{self.dev_name}")
        response.raise_for_status()
        return response.json()

    @property
    def _PROBES(self):
        """Expose probes like the local Device subclasses do."""
        # use object.__getattribute__ to avoid triggering __getattr__
        return object.__getattribute__(self, "_probes_cache")

    def write(self, message):
        """Future programmers: This is a dangerous one, please only use in a GUI that checks the inputs (the way I am using it in spectrum analyzer tab)"""
        response = requests.post(f"{self.SERVER_URL}/write/{self.dev_name}/{message}")
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    #in this test, I worked with spectrum_analyzer and sg384
    dev = Multiple_Device_Client('spectrum_analyzer')
    print("Updating center frequency...")
    print(dev.update({"center frequency": 400}))
    print("Reading probe...")
    print(dev.read_probes("center frequency"))
    print("Updating center frequency...")
    print(dev.write("CF 300MZ;"))
    print("Reading probe...")
    print(dev.read_probes("center frequency"))
    dev2 = Multiple_Device_Client('sg384')
    print("Updating sweep cf...")
    print(dev2.update({"frequency": 2.87e9}))
    print("Reading probe...")
    print(dev2.read_probes("frequency"))
    print("Updating sweep_center_frequency...")
    print(dev2.update({"frequency": 2.67e9}))
    print("Reading probe...")
    print(dev2.read_probes("frequency"))