import json, threading, time, gc, numpy as np
from queue import Queue, Empty

from core.vc_engine import VoiceConverter


class RealtimePipeline:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.running = False
        self.threads = []
        self.output_queue = Queue(maxsize=10)
        self.processing_times = []
        ac = self.cfg["audio"]
        self.sr = ac["sample_rate"]
        self.frame_size = ac["frame_size"]
        self.converter = VoiceConverter(config_path)
        self.id_in = ac.get("input_device")
        self.id_out = ac.get("output_device")
        self._last_input = np.zeros(960, dtype=np.float32)
        self._last_output = np.zeros(960, dtype=np.float32)
        self._accum = np.array([], dtype=np.float32)
        self._silent_frames = 0

    def start(self):
        self.running = True
        self._accum = np.array([], dtype=np.float32)
        self._silent_frames = 0
        self.threads = [
            threading.Thread(target=self._capture, daemon=True),
            threading.Thread(target=self._process, daemon=True),
        ]
        for t in self.threads:
            t.start()
        print("Pipeline iniciado")

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
        self.threads.clear()
        gc.collect()
        print("Pipeline detenido")

    def _capture(self):
        import sounddevice as sd

        def cb(indata, frames, ti, st):
            if self.running:
                try:
                    self.output_queue.put_nowait(indata.copy())
                except:
                    pass

        try:
            stream = sd.InputStream(samplerate=self.sr, blocksize=self.frame_size,
                                    device=self.id_in, channels=1,
                                    dtype=np.float32, callback=cb)
            stream.start()
            while self.running:
                time.sleep(0.05)
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"Capture error: {e}")

    def _process(self):
        MIN_CHUNK = int(self.sr * 0.3)

        while self.running:
            try:
                frame = self.output_queue.get(timeout=0.05)
            except Empty:
                continue

            frame = frame.squeeze()
            self._last_input = frame[-960:]

            energy = float(np.sqrt(np.mean(frame ** 2)))

            if energy < 0.003:
                self._silent_frames += 1
                if self._silent_frames > 15 and len(self._accum) > 0:
                    self._accum = np.array([], dtype=np.float32)
                continue
            self._silent_frames = 0

            self._accum = np.concatenate([self._accum, frame])

            if len(self._accum) < MIN_CHUNK:
                continue

            chunk = self._accum.copy()
            self._accum = np.array([], dtype=np.float32)

            t0 = time.perf_counter()
            try:
                processed = self.converter.convert(chunk, self.sr)
                if not np.all(np.isfinite(processed)):
                    processed = chunk
            except Exception:
                processed = chunk

            dt = time.perf_counter() - t0
            self.processing_times.append(dt)
            if len(self.processing_times) > 30:
                self.processing_times.pop(0)

            self._last_output = processed[-960:]

            try:
                import sounddevice as sd
                sd.play(processed.astype(np.float32), self.sr,
                        device=self.id_out, blocking=False)
            except Exception as e:
                print(f"Play error: {e}")
                try:
                    sd.play(processed.astype(np.float32), self.sr, blocking=False)
                except:
                    pass

    def get_latency_ms(self) -> float:
        if not self.processing_times:
            return 0.0
        return float(np.mean(self.processing_times[-20:]) * 1000)
