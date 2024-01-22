import serial
from systole import serialSim, ppg_peaks
from systole.recording import Oximeter


class OximeterTask(Oximeter):

    def __init__(self, serial_port, **systole_kw):
        if serial_port < 0:
            port = serialSim()
        else:
            port = serial.Serial(serial_port)
        super().__init__(serial=port, sfreq=75, add_channels=1, **systole_kw)

    def get_peaks(self, duration=5.0):
        signal = (
            self.read(duration=duration).recording[-75 * 6:]  # noqa
        )
        return ppg_peaks(signal, sfreq=75, new_sfreq=1000, clipping=True)
