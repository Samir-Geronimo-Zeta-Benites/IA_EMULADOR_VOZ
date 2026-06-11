import numpy as np
import pyworld as pw
import librosa
import json
from pathlib import Path


class VoiceConverter:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.sr = 16000
        self.f0_up_key = self.cfg["rvc"]["f0_up_key"]
        self.target_stats = None
        self._load_target()

    def _load_target(self):
        model_path = Path(self.cfg["rvc"]["model_path"])
        stats_path = model_path.with_suffix(".stats.npy")
        if stats_path.exists():
            self.target_stats = np.load(str(stats_path), allow_pickle=True).item()
            print(f"Estadisticas cargadas")
        else:
            print("Sin estadisticas - entrena el modelo")

    def train(self, audio_path: str):
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        max_len = 90 * self.sr
        if len(audio) > max_len:
            audio = audio[:max_len]
        audio = (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float64)

        f0, t = pw.dio(audio, self.sr, f0_floor=50.0, f0_ceil=1100.0)
        f0 = pw.stonemask(audio, f0, t, self.sr)
        f0_valid = f0[f0 > 0]

        stats = {
            "f0_mean": float(np.mean(f0_valid)) if len(f0_valid) > 0 else 150.0,
            "f0_std": float(np.std(f0_valid)) if len(f0_valid) > 0 else 50.0,
            "sr": self.sr,
        }

        model_path = Path(self.cfg["rvc"]["model_path"])
        np.save(str(model_path.with_suffix(".stats.npy")), stats)
        self.target_stats = stats
        print(f"F0 target: {stats['f0_mean']:.0f}Hz std={stats['f0_std']:.0f}Hz")
        return stats

    def convert(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.target_stats is None:
            return audio

        audio = audio.squeeze().astype(np.float64)
        if audio.ndim == 0 or len(audio) < sr * 0.1:
            return np.zeros(sr, dtype=np.float32)

        orig_sr = sr
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        mx = np.max(np.abs(audio))
        if mx < 1e-8:
            return audio
        audio = audio / mx * 0.8

        target_hz = self.target_stats["f0_mean"]
        default_hz = 120.0
        semitones = 12.0 * np.log2(target_hz / default_hz) + self.f0_up_key

        shifted = librosa.effects.pitch_shift(
            audio, sr=self.sr, n_steps=semitones,
            bins_per_octave=24
        )

        shifted = np.nan_to_num(shifted, nan=0.0)
        smax = np.max(np.abs(shifted))
        if smax > 1e-8:
            shifted = shifted / smax * 0.9

        if orig_sr != self.sr:
            shifted = librosa.resample(shifted, orig_sr=self.sr, target_sr=orig_sr)

        shifted = np.clip(shifted, -0.95, 0.95)
        return shifted.astype(np.float32)
