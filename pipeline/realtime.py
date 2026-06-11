import json
import threading
import time
import gc
import numpy as np
from queue import Queue, Empty

from audio import AudioCapture, AudioPlayback
from core.vc_engine import VoiceConverter


class RealtimePipeline:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.running = False
        self.threads = []
        self.input_queue = Queue(maxsize=20)
        self.output_queue = Queue(maxsize=20)
        self.processing_times = []

        audio_cfg = self.cfg["audio"]
        self.sr = audio_cfg["sample_rate"]
        self.frame_size = audio_cfg["frame_size"]

        self.converter = VoiceConverter(config_path)
        self._convert_ct = 0
        self.input_device = audio_cfg.get("input_device")
        self.output_device = audio_cfg.get("output_device")
        self._last_input = np.zeros(960, dtype=np.float32)
        self._last_output = np.zeros(960, dtype=np.float32)
        self._accum = np.array([], dtype=np.float32)

    def start(self):
        self.running = True
        self._accum = np.array([], dtype=np.float32)

        self.threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._process_loop, daemon=True),
            threading.Thread(target=self._playback_loop, daemon=True),
        ]

        for t in self.threads:
            t.start()

        print("Pipeline iniciado")

    def stop(self):
        self.running = False
        for q in [self.input_queue, self.output_queue]:
            try:
                while True: q.get_nowait()
            except Empty: pass
        self.input_queue.put(None)
        for t in self.threads:
            t.join(timeout=1.5)
        self.threads.clear()
        gc.collect()
        print("Pipeline detenido")

    def _capture_loop(self):
        cap = AudioCapture(sr=self.sr, frame_size=self.frame_size, device=self.input_device)

        def cb(indata, frames, ti, st):
            if self.running:
                try:
                    self.input_queue.put_nowait(indata.copy())
                except: pass

        cap.start(cb)
        while self.running:
            time.sleep(0.05)
        cap.stop()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.05)
            except Empty:
                continue
            if frame is None:
                break

            frame = frame.squeeze()
            if frame.ndim == 0:
                continue

            self._last_input = frame[-960:]

            self._accum = np.concatenate([self._accum, frame])

            if len(self._accum) >= int(self.sr * 0.3):
                chunk = self._accum.copy()
                self._accum = self._accum[-int(self.sr * 0.1):]
                processed = self._convert(chunk)
                if processed is not None and len(processed) > 0:
                    self._last_output = processed[-960:]
                    try:
                        self.output_queue.put_nowait(processed)
                    except: pass

    def _convert(self, audio):
        t0 = time.perf_counter()
        try:
            out = self.converter.convert(audio, self.sr)
            if self._convert_ct % 5 == 0:
                diff = np.max(np.abs(audio[:len(out)] - out[:len(audio)]))
                print(f"  [pipe] convert {self._convert_ct}: diff={diff:.4f}")
            self._convert_ct += 1
            if out is None or len(out) == 0 or not np.all(np.isfinite(out)):
                return None
            elapsed = time.perf_counter() - t0
            self.processing_times.append(elapsed)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)
            return out
        except Exception:
            return None

    def _playback_loop(self):
        pb = AudioPlayback(sr=self.sr, frame_size=self.frame_size, device=self.output_device)
        buffer = np.array([], dtype=np.float32)

        def cb(outdata, frames, ti, st):
            nonlocal buffer
            while len(buffer) < frames:
                try:
                    chunk = self.output_queue.get_nowait()
                    if chunk is not None:
                        buffer = np.concatenate([buffer, chunk])
                except Empty:
                    break

            if len(buffer) >= frames:
                outdata[:] = buffer[:frames].reshape(-1, 1)
                buffer = buffer[frames:]
            else:
                outdata[:len(buffer)] = buffer.reshape(-1, 1)
                outdata[len(buffer):] = 0
                buffer = np.array([], dtype=np.float32)

        pb.start(cb)
        while self.running:
            time.sleep(0.05)
        pb.stop()

    def get_latency_ms(self) -> float:
        if not self.processing_times:
            return 0.0
        return float(np.mean(self.processing_times[-20:]) * 1000)
