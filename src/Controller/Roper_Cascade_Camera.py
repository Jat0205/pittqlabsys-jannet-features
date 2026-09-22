from src.core import Device, Parameter
import matlab.engine
eng = matlab.engine.start_matlab()
_DEFAULT_AUTO_EXPOSURE_TARGET  = 120
_DEFAULT_LEVEL_RANGE = 125
_DEFAULT_GAIN = 100
_DEFAULT_EXPOSURE_TIME_US = 10000
_MIN_EXPOSURE_TIME = 244
_MAX_EXPOSURE_TIME = 2000000 # or 350000 what i got from get_MaxAutoExpoTimeAGain
_EXPOGAIN_MIN              = 100      # exposure gain, minimum value
_EXPOGAIN_MAX              = 500      # exposure gain, max value
_DEFAULT_RESOLUTION = "low"
_server_port = 5005

class Roper_Cascade_Camera(Device):
    _DEFAULT_SETTINGS = Parameter(Device._get_base_settings() + [
        Parameter('gain', _DEFAULT_GAIN, float, 'camera gain', min_value=_EXPOGAIN_MIN,
                  max_value=_EXPOGAIN_MAX),
        Parameter('inttime', _DEFAULT_EXPOSURE_TIME_US, float, 'camera exposure time in us',
                  min_value=_MIN_EXPOSURE_TIME, max_value=_MAX_EXPOSURE_TIME),
        Parameter('resolution', _DEFAULT_RESOLUTION, str, 'camera saturation'),
        Parameter('server_port', _server_port, int, 'server_port'),
        # _DEFAULT_AUTO_EXPOSURE_TARGET = 120
        # _DEFAULT_LEVEL_RANGE = 125
    ])

    def __init__(self, name=None, settings=None):
        super(Roper_Cascade_Camera, self).__init__(name, settings)
        self.cam = None
        self.eng = None
        try:
            self._connect()
            print(list(self.cam.keys()))
        except Exception as e:
            raise e

    def update(self, settings: dict):
        super(Roper_Cascade_Camera, self).update(settings)
        for key, value in settings.items():
            if self.settings.valid_values[
                key] == bool:  # converts booleans, which are more natural to store for on/off, to
                value = int(value)  # the integers used internally in the laser
            key = self._param_to_internal(key)
            # only send update to Device if connection to Device has been established
            if self._settings_initialized:
                if key == "gain":
                    self.eng.feval(self.cam['setgain'], float(value))
                elif key == "inttime":
                    self.eng.feval(self.cam['setinttime'], float(value))
                elif key == "start":
                    self.eng.feval(self.cam['startlive'], int(value))
                elif key == "stop":
                    self.eng.feval(self.cam['stoplive'], float(value))
                else:
                    raise ValueError("Unknown key '%s'" % key)

    def _param_to_internal(self, param):
        return param

    def read_probes(self, key=None):
        assert (
            self._settings_initialized)  # will cause read_probes to fail if settings (and thus also connection) not yet initialized
        assert key in list(self._PROBES.keys())
        key_internal = self._param_to_internal(key)
        if key_internal == "gain":
            value = self.eng.feval(self.cam['getgain'])
        elif key == 'get_data':
            return self.settings['get_data']
        elif key_internal == "gainlimits":
            value = self.eng.feval(self.cam['getgainlimits'])
        elif key_internal == "image":
            value = self.eng.feval(self.cam['getimage'], float(self.read_probes("inttime")))
        elif key_internal == "imagefast_int":
            value = self.eng.feval(self.cam['getimagefast'])
        elif key_internal == "inttime":
            value = self.eng.feval(self.cam['getinttime'])
        elif key_internal == "maxpixelvalue":
            value = self.eng.feval(self.cam['getmaxpixelvalue'])
        elif key_internal == "pixelwarninglevel":
            value = self.eng.feval(self.cam['getpixelwarninglevel'])
        elif key_internal == "resolution":
            value = self.eng.feval(self.cam['getresolution'])
        else:
            raise NotImplementedError
        return value

    @property
    def _PROBES(self):
        return {
            'get_data': 'choose whether you need to get data from this device or not',
            'gain': 'gain',
            'gainlimits': 'gainlimits',
            'image': 'image_int',
            'imagefast_int': 'imagefast_int',
            'inttime': 'inttime',
            'maxpixelvalue':'maxpixelvalue',
            'pixelwarninglevel': 'pixelwarninglevel',
            'resolution': 'resolution'
        }

    def _connect(self):
        self.eng = matlab.engine.start_matlab()
        # Add folder containing the client functions
        s = self.eng.genpath(r"D:\software_by_our_lab\working")
        self.eng.addpath(s, nargout=0)

        # Call the function directly
        try:
            ret = self.eng.DLRC1_camera_findinstrument()
            print("Server response:", ret)
            self.cam = self.eng.DLRC1_camera_create_device_class()
        except matlab.engine.MatlabExecutionError as e:
            self.cam = None
            raise RuntimeError(f"Failed to connect to camera server: {e}") from e
        return 0

    def close_instrument(self):
        print(self.eng.DLRC1_sendmsgtoserver('roper_ccd_32bit_closeinstrument'))
        # Quit MATLAB engine
        self.eng.quit()
        print('Camera closed')

    def stop_server(self):
        self.eng.DLRC1_stop_server()

    def close(self):
        pass

if __name__ == "__main__":
    s=Roper_Cascade_Camera()
    print(f"integration time: {s.read_probes("inttime")}")
    s.update({'inttime': 99.0})
    print(f"integration time: {s.read_probes("inttime")}")
    """print(f"integration time: {s.read_probes("inttime")}")
    print(f"gain: {s.read_probes("gain")}")
    print(f"resolution: {s.read_probes("resolution")}")
    s.update({'inttime': 100})
    print(f"integration time: {s.read_probes("inttime")}")
    s.update({'inttime': 99.0, 'gain': 0.9})
    print(f"integration time: {s.read_probes("inttime")}")
    print(f"gain: {s.read_probes("gain")}")
    print(f"gainlimits: {s.read_probes("gainlimits")}")
    print(f"maxpixelvalue: {s.read_probes("maxpixelvalue")}")
    print(f"pixelwarninglevel: {s.read_probes("pixelwarninglevel")}")"""
    ##print(f"imagefast_int: {s.read_probes("imagefast_int")}")
    ##print(f"image: {s.read_probes("image")}")