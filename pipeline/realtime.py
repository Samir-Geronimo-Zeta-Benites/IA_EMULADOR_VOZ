import json
import threading
import time
import gc
import numpy as np
from queue import Queue, Empty
import soundfile as sf

from audio import AudioCapture, AudioPlayback
from core.vc_engine import VoiceConverter


class RealtimePipeline:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.running = False
        self.threads = []
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.processing_times = []

        audio_cfg = self.cfg["audio"]
        self.sr = audio_cfg["sample_rate"]
        self.frame_size = audio_cfg["frame_size"]

        self.converter = VoiceConverter(config_path)
        self.input_device = audio_cfg.get("input_device")
        self.output_device = audio_cfg.get("output_device")
        self._last_input = np.zeros(960, dtype=np.float32)
        self._last_output = np.zeros(960, dtype=np.float32)
        self._accum = np.array([], dtype=np.float32)
        self._accum_min = int(self.sr * 0.3)

    def start(self):
        self.running = True
        self._accum = np.array([], dtype=np.float32)

        self.threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._process_loop, daemon=True),
        ]

        for t in self.threads:
            t.start()

        print("Pipeline en tiempo real iniciado")

    def stop(self):
        self.running = False

        for q in [self.input_queue]:
            try:
                while True: q.get_nowait()
            except Empty: pass
            q.put(None)

        for t in self.threads:
            t.join(timeout=1.0)

        self.threads.clear()
        gc.collect()
        print("Pipeline detenido")

    def _capture_loop(self):
        capture = AudioCapture(sr=self.sr, frame_size=self.frame_size,
                               device=self.input_device)

        def callback(indata, frames, time_info, status):
            if self.running:
                try:
                    self.input_queue.put_nowait(indata.copy())
                except:
                    pass

        capture.start(callback)
        while self.running:
            time.sleep(0.05)
        capture.stop()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.05)
            except Empty:
                continue
            if frame is None:
                break

            frame = frame.squeeze()
            self._last_input = frame[-960:]

            self._accum = np.concatenate([self._accum, frame])
            if len(self._accum) < self._accum_min:
                continue

            chunk = self._accum.copy()
            self._accum = self._accum[-self._accum_min:]

            t0 = time.perf_counter()
            try:
                processed = self.converter.convert(chunk, self.sr)
                if not np.all(np.isfinite(processed)):
                    processed = chunk
            except Exception:
                processed = chunk

            self.processing_times.append(time.perf_counter() - t0)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)

            self._last_output = processed[-960:]
            try:
                self.output_queue.put_nowait(processed)
            except:
                pass

            self._play(processed)

    def _play(self, audio):
        if audio is None or len(audio) < 10:
            return
        try:
            import sounddevice as sd
            sd.play(audio.astype(np.float32), self.sr,
                    device=self.output_device, blocking=False)
        except Exception:
            pass

    def get_latency_ms(self) -> float:
        if not self.processing_times:
            return 0.0
        return float(np.mean(self.processing_times[-20:]) * 1000)
