import sounddevice as sd
import numpy as np
from typing import Optional


class AudioCapture:
    def __init__(self, sr: int = 48000, frame_size: int = 960, device: Optional[int] = None):
        self.sr = sr
        self.frame_size = frame_size
        self.device = device
        self.stream: Optional[sd.InputStream] = None

    def start(self, callback):
        self.stream = sd.InputStream(
            samplerate=self.sr,
            blocksize=self.frame_size,
            device=self.device,
            channels=1,
            dtype=np.float32,
            callback=callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    @staticmethod
    def list_devices():
        return sd.query_devices()

    @staticmethod
    def list_input_devices():
        devices = sd.query_devices()
        return [d for d in devices if d["max_input_channels"] > 0]
