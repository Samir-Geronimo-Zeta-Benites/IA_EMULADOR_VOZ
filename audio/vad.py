import numpy as np
from collections import deque


class VoiceActivityDetector:
    def __init__(self, mode: int = 1, frame_ms: int = 30, padding_ms: int = 300):
        self.mode = mode
        self.frame_ms = frame_ms
        self.padding_ms = padding_ms
        self._vad = self._init_vad()
        self._energy_threshold = None
        self._energy_history = deque(maxlen=20)

    def _init_vad(self):
        try:
            import torch
            silero_vad, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
                trust_repo=True,
            )
            return silero_vad
        except Exception as e:
            print(f"silero-vad no disponible: {e}")
            print("Usando VAD basado en energía")
            return None

    def is_speech(self, audio_frame: bytes, sr: int = 16000) -> bool:
        if self._vad is not None:
            return self._vad_is_speech(audio_frame, sr)
        return self._energy_based_vad(audio_frame)

    def _vad_is_speech(self, audio_frame: bytes, sr: int = 16000) -> bool:
        try:
            import torch
            audio_int16 = np.frombuffer(audio_frame, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32)
            speech_prob = self._vad(audio_tensor, sr).item()
            return speech_prob > 0.5
        except Exception:
            return self._energy_based_vad(audio_frame)

    def _energy_based_vad(self, audio_frame: bytes) -> bool:
        audio = np.frombuffer(audio_frame, dtype=np.int16)
        energy = np.sqrt(np.mean(audio.astype(float) ** 2))
        self._energy_history.append(energy)

        if self._energy_threshold is None and len(self._energy_history) > 5:
            self._energy_threshold = np.median(self._energy_history) * 2.5

        if self._energy_threshold is None:
            return False

        return energy > self._energy_threshold

    def process_frame(self, frame: np.ndarray, sr: int = 48000) -> bool:
        if sr != 16000:
            frame = self._resample(frame, sr, 16000)
        frame_int16 = (frame * 32767).astype(np.int16)
        frame_bytes = frame_int16.tobytes()
        return self.is_speech(frame_bytes, 16000)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int):
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_len = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        )


class VADBuffer:
    def __init__(self, vad: VoiceActivityDetector, padding_frames: int = 10):
        self.vad = vad
        self.padding_frames = padding_frames
        self.speech_buffer = deque()
        self.silence_counter = 0
        self.is_speaking = False

    def add_frame(self, frame: np.ndarray, sr: int = 48000) -> bool:
        has_speech = self.vad.process_frame(frame, sr)

        if has_speech:
            self.speech_buffer.append(frame)
            self.silence_counter = 0
            self.is_speaking = True
        elif self.is_speaking:
            self.speech_buffer.append(frame)
            self.silence_counter += 1
            if self.silence_counter >= self.padding_frames:
                self.is_speaking = False

        return self.is_speaking

    def get_buffer(self) -> np.ndarray:
        if not self.speech_buffer:
            return np.array([], dtype=np.float32)
        result = np.concatenate(list(self.speech_buffer))
        self.speech_buffer.clear()
        self.silence_counter = 0
        return result

    def clear(self):
        self.speech_buffer.clear()
        self.silence_counter = 0
        self.is_speaking = False
