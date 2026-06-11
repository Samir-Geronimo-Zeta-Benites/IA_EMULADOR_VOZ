import numpy as np


class PitchExtractor:
    def __init__(self, method: str = "parselmouth"):
        self.method = method

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.method == "parselmouth":
            return self._parselmouth(audio, sr)
        elif self.method == "simple":
            return self._simple(audio, sr)
        else:
            return self._parselmouth(audio, sr)

    def _parselmouth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import parselmouth
            snd = parselmouth.Sound(audio.astype(float), sampling_frequency=sr)
            pitch = snd.to_pitch(
                time_step=0.01,
                pitch_floor=50.0,
                pitch_ceiling=1100.0,
            )
            f0 = pitch.selected_array["frequency"]
            return np.nan_to_num(f0, nan=0.0).astype(np.float32)
        except Exception:
            return self._simple(audio, sr)

    def _simple(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from scipy.signal import correlate

        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        hop = sr // 100
        frames = np.arange(0, len(audio) - hop, hop)
        f0 = np.zeros(len(frames), dtype=np.float32)
        for i, start in enumerate(frames):
            frame = audio[start:start + hop]
            corr = correlate(frame, frame, mode="full")
            center = len(corr) // 2
            peaks = corr[center + 50:center + sr // 50]
            if len(peaks) > 0 and np.max(peaks) > 0:
                lag = np.argmax(peaks) + 50
                f0[i] = sr / lag if lag > 0 else 0.0
        return f0

    def interpolate(self, f0: np.ndarray) -> np.ndarray:
        f0 = f0.copy()
        mask = f0 == 0
        if np.all(mask):
            return f0
        indices = np.arange(len(f0))
        valid = np.where(~mask)[0]
        if len(valid) == 0:
            return f0
        f0[mask] = np.interp(indices[mask], indices[valid], f0[valid])
        return f0
