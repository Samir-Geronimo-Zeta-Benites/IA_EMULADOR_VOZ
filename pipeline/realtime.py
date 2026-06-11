import json, threading, time, gc, numpy as np
from queue import Queue, Empty

from core.vc_engine import VoiceConverter


class RealtimePipeline:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.running = False
        self.threads = []
        self.processing_times = []
        ac = self.cfg["audio"]
        self.sr = ac["sample_rate"]
        self.frame_size = ac["frame_size"]
        self.converter = VoiceConverter(config_path)
        self.id_in = ac.get("input_device")
        self.id_out = ac.get("output_device")
        self.passthrough = bool(self.cfg.get("pipeline", {}).get("passthrough", False))
        self._last_input = np.zeros(960, dtype=np.float32)
        self._last_output = np.zeros(960, dtype=np.float32)
        self._accum = np.array([], dtype=np.float32)
        self._speaking = False

    def start(self):
        self.running = True
        self._accum = np.array([], dtype=np.float32)
        self._speaking = False
        self.threads = [
            threading.Thread(target=self._run, daemon=True),
        ]
        for t in self.threads:
            t.start()
        print("Pipeline iniciado")

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.5)
        self.threads.clear()
        gc.collect()
        print("Pipeline detenido")

    def _run(self):
        import sounddevice as sd

        def cb(indata, frames, ti, st):
            nonlocal accum, speaking
            if not self.running:
                return
            frame = indata.squeeze().copy()
            self._last_input = frame[-960:]
            energy = float(np.sqrt(np.mean(frame ** 2)))

            if energy < 0.002:
                if speaking and len(accum) > int(self.sr * 0.1):
                    chunk = accum.copy()
                    self._process_and_play(chunk)
                accum = np.array([], dtype=np.float32)
                speaking = False
                return

            speaking = True
            accum = np.concatenate([accum, frame])

            if len(accum) >= int(self.sr * 0.4):
                chunk = accum.copy()
                accum = accum[-int(self.sr * 0.1):]
                self._process_and_play(chunk)

        accum = np.array([], dtype=np.float32)
        speaking = False

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

    def _process_and_play(self, chunk):
        t0 = time.perf_counter()
        try:
            processed = chunk if self.passthrough else self.converter.convert(chunk, self.sr)
            if processed is None or not np.all(np.isfinite(processed)):
                return
        except Exception as e:
            print(f"Convert error: {e}")
            return

        self.processing_times.append(time.perf_counter() - t0)
        if len(self.processing_times) > 20:
            self.processing_times.pop(0)

        self._last_output = processed[-960:]
        try:
            import sounddevice as sd
            sd.play(processed.astype(np.float32), self.sr,
                    device=self.id_out, blocking=False)
        except Exception:
            try:
                sd.play(processed.astype(np.float32), self.sr, blocking=False)
            except:
                pass

    def get_latency_ms(self) -> float:
        if not self.processing_times:
            return 0.0
        return float(np.mean(self.processing_times[-10:]) * 1000)
