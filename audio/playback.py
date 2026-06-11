import sounddevice as sd
import numpy as np
from typing import Optional


class AudioPlayback:
    def __init__(self, sr: int = 48000, frame_size: int = 960, device: Optional[int] = None):
        self.sr = sr
        self.frame_size = frame_size
        self.device = device
        self.stream: Optional[sd.OutputStream] = None

    def start(self, callback):
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.frame_size,
            device=self.device,
            channels=1,
            dtype=np.float32,
            callback=callback,
        )
        self.stream.start()

    def play(self, audio: np.ndarray):
        sd.play(audio, self.sr, device=self.device)

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        sd.stop()

    @staticmethod
    def list_output_devices():
        devices = sd.query_devices()
        return [d for d in devices if d["max_output_channels"] > 0]
