from pylablib.devices import Thorlabs
from src.core import Device, Parameter

# Find the specific COM port assigned to the FW102C in your Device Manager
port = "COM18"  # Change this to your specific COM port
_server_port = 5001
# OD <-> filter-wheel position map
_OD_TO_POS = {0: 2, 0.5: 3, 1.0: 4, 2.0: 5, 3.0: 6, 4.0: 1}
_POS_TO_OD = {pos: od for od, pos in _OD_TO_POS.items()}

class Thorlabs_FW102C(Device):
    _DEFAULT_SETTINGS = Parameter([
        Parameter('get_data', True, [False, True], 'choose whether you need to get data from this device or not'),
        Parameter('connection_type', 'RS232', ['RS232'], 'type of connection to open to controller'),
        Parameter('port', port, ["COM18"],'COM port on which to connect'),
        Parameter('OD', 4.0, [0, 1.0, 0.5, 2.0, 3.0, 4.0],'Filter_OD'),
        Parameter('position', 1, [1,2,3,4,5,6],'Filter_position'),
        Parameter('server_port', _server_port, int, 'server_port'),
    ])

    def __init__(self, name=None, settings=None):
        try:
            super(Thorlabs_FW102C, self).__init__(name, settings)
        except Exception:
            self.close()  # if the open succeeded but something later threw, don't leak the port
            raise

    def close(self):
        wheel = getattr(self, 'wheel', None)
        if wheel is not None:
            try:
                wheel.close()
            finally:
                self.wheel = None

    def _connect(self):
        if getattr(self, 'wheel', None) is not None:
            return 0
        self.wheel = Thorlabs.FW(self.settings['port'])
        return 0

    @property
    def is_connected(self):
        if self.wheel is not None:
            return True
        else:
            return False

    @property
    def _PROBES(self):
        return {
            'get_data': 'choose whether you need to get data from this device or not',
            "OD": 'Filter OD',
            "position": 'Filter position',
        }

    def read_probes(self, key):
        assert (
            self._settings_initialized)  # will cause read_probes to fail if settings (and thus also connection) not yet initialized
        assert key in list(self._PROBES.keys())
        if key == 'get_data':
            return self.settings['get_data']
        if key == "OD":
            return self._position_to_OD(self.wheel.get_position())
        elif key == "Position":
            return self.wheel.get_position()
        else:
            raise KeyError(key)

    def update(self, settings):
        super(Thorlabs_FW102C, self).update(settings)
        for key, value in settings.items():
            if not (key == 'get_data' or key == 'server_port'):
                if key == 'connection_type' or key == 'port':
                    self._connect()
                elif key == 'OD':
                    self.wheel.set_position(self._od_to_pos(value))
                elif key == 'position':
                    if value not in _POS_TO_OD:
                        raise ValueError(f"Invalid position {value}; valid values are {sorted(_POS_TO_OD)}")
                    self.wheel.set_position(value)
                else:
                    raise KeyError(key)

    def _od_to_pos(self, od):
        # OD [0, 0.5, 2.0, 3.0, 4.0] are positions [2, 3, 4, 5, 1]
        if od not in _OD_TO_POS:
            raise ValueError(f"Invalid OD {od}; valid values are {sorted(_OD_TO_POS)}")
        return _OD_TO_POS[od]

    def _position_to_OD(self, position):
        if position not in _POS_TO_OD:
            raise ValueError(f"Invalid position {position}; valid values are {sorted(_POS_TO_OD)}")
        return _POS_TO_OD[position]