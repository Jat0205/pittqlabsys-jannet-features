#python -m src.Controller.device_client

from src.core import Device, Parameter
import requests


class Device_Client(Device):
    SERVER_URL = "http://192.168.2.4:5001"

    def __init__(self, *args, **kwargs):
        # Create the cache *before* calling super().__init__
        # preventing Device __getattr__ and __setattr__ from breaking.
        object.__setattr__(self, "_probes_cache", {})

        super().__init__(*args, **kwargs)

        # Now safely fetch data from the server
        self._DEFAULT_SETTINGS = self.get_settings()
        self._probes_cache = self.get_probes()

    def update(self, settings: dict):
        """Send updated settings to the remote device server."""
        response = requests.post(f"{self.SERVER_URL}/update", json=settings)
        response.raise_for_status()
        return response.json()

    def read_probes(self, key=None):
        """Fetch one or more probes from the server."""
        if key is None:
            all_probes = {}
            for k in self._PROBES.keys():
                all_probes[k] = self.read_probes(k)
            return all_probes

        response = requests.get(f"{self.SERVER_URL}/read/{key}")
        response.raise_for_status()
        return response.json()

    def get_settings(self):
        """Fetch available settings from the server."""
        response = requests.get(f"{self.SERVER_URL}/settings")
        response.raise_for_status()
        return response.json()

    def get_probes(self):
        """Fetch available probes from the server."""
        response = requests.get(f"{self.SERVER_URL}/probes")
        response.raise_for_status()
        return response.json()

    @property
    def _PROBES(self):
        """Expose probes like the local Device subclasses do."""
        # use object.__getattribute__ to avoid triggering __getattr__
        return object.__getattribute__(self, "_probes_cache")

    def write(self, message):
        """Future programmers: This is a dangerous one, please only use in a GUI that checks the inputs (the way I am using it in spectrum analyzer tab)"""
        response = requests.post(f"{self.SERVER_URL}/write/{message}")
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    dev = Device_Client()
    print("Updating center frequency...")
    print(dev.update({"center frequency": 400}))
    print("Reading probe...")
    print(dev.read_probes("center frequency"))
    print("Updating center frequency...")
    print(dev.write("CF 300MZ;"))
    print("Reading probe...")
    print(dev.read_probes("center frequency"))
