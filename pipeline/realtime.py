import json
import threading
import time
import numpy as np
from queue import Queue, Empty

from audio import AudioCapture, AudioPlayback, VADBuffer, VoiceActivityDetector
from core.rvc import RVCInference
from .buffer import CrossFader


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

        self.vad_buffer = VADBuffer(
            VoiceActivityDetector(
                mode=self.cfg["vad"]["mode"],
                frame_ms=self.cfg["vad"]["frame_ms"],
                padding_ms=self.cfg["vad"]["padding_ms"],
            )
        )

        self.crossfader = CrossFader(
            fade_len=self.cfg["pipeline"]["crossfade_len"]
        )

        self.rvc = RVCInference(config_path)

        self.input_device = audio_cfg.get("input_device")
        self.output_device = audio_cfg.get("output_device")
        self._last_input = np.array([], dtype=np.float32)
        self._last_output = np.array([], dtype=np.float32)

    def start(self):
        self.running = True

        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        process_thread = threading.Thread(target=self._process_loop, daemon=True)
        playback_thread = threading.Thread(target=self._playback_loop, daemon=True)

        self.threads = [capture_thread, process_thread, playback_thread]

        for t in self.threads:
            t.start()

        print("Pipeline en tiempo real iniciado")

    def stop(self):
        self.running = False
        self.input_queue.put(None)
        self.output_queue.put(None)
        for t in self.threads:
            t.join(timeout=2.0)
        print("Pipeline detenido")

    def _capture_loop(self):
        capture = AudioCapture(
            sr=self.sr,
            frame_size=self.frame_size,
            device=self.input_device,
        )

        def callback(indata, frames, time_info, status):
            if self.running:
                self.input_queue.put(indata.copy())

        capture.start(callback)

        while self.running:
            time.sleep(0.01)

        capture.stop()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            if frame is None:
                break

            is_active = self.vad_buffer.add_frame(frame, self.sr)
            self._last_input = frame[-960:]

            if not is_active and not self.vad_buffer.is_speaking:
                continue

            if not is_active and self.vad_buffer.silence_counter > 0:
                continue

            chunk = self._process_chunk(frame)

            if chunk is not None and len(chunk) > 0:
                self.output_queue.put(chunk)
                self._last_output = chunk[-960:]

    def _process_chunk(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()

        frame = frame.squeeze()
        if frame.ndim == 0:
            frame = np.array([frame.item()])

        try:
            processed = self.rvc.infer(frame, self.sr)
        except Exception as e:
            processed = frame

        processed = self._normalize(processed)

        elapsed = time.perf_counter() - t0
        self.processing_times.append(elapsed)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

        return processed

    def _playback_loop(self):
        playback = AudioPlayback(
            sr=self.sr,
            frame_size=self.frame_size,
            device=self.output_device,
        )

        def callback(outdata, frames, time_info, status):
            try:
                chunk = self.output_queue.get_nowait()
                if len(chunk) >= len(outdata):
                    outdata[:] = chunk[:len(outdata)].reshape(-1, 1)
                else:
                    outdata[:len(chunk)] = chunk.reshape(-1, 1)
                    outdata[len(chunk):] = 0
            except Empty:
                outdata.fill(0)

        playback.start(callback)

        while self.running:
            time.sleep(0.01)

        playback.stop()

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        return audio

    def get_latency_ms(self) -> float:
        if not self.processing_times:
            return 0.0
        return float(np.mean(self.processing_times[-50:]) * 1000)
