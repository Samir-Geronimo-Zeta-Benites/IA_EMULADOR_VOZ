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
            print(f"Estadisticas target cargadas")
        else:
            print("Sin estadisticas target - entrena el modelo primero")

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
        stats_path = model_path.with_suffix(".stats.npy")
        np.save(str(stats_path), stats)
        self.target_stats = stats
        print(f"F0 target: mean={stats['f0_mean']:.0f}Hz, std={stats['f0_std']:.0f}Hz")
        return stats

    def convert(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.target_stats is None:
            return audio

        audio = audio.squeeze().astype(np.float32)
        if audio.ndim == 0 or len(audio) < sr * 0.1:
            return np.zeros(sr, dtype=np.float32)

        orig_sr = sr
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        max_val = np.max(np.abs(audio))
        if max_val < 1e-8:
            return audio
        audio_norm = audio / max_val

        f0, _ = pw.dio(audio_norm.astype(np.float64), self.sr,
                       f0_floor=50.0, f0_ceil=1100.0)
        f0 = pw.stonemask(audio_norm.astype(np.float64), f0, _, self.sr)

        f0_src = f0[f0 > 0]
        if len(f0_src) > 3:
            src_mean = float(np.mean(f0_src))
            target_mean = self.target_stats["f0_mean"]
            ratio = target_mean / (src_mean + 1e-3)
            ratio = max(0.5, min(2.0, ratio))

            semitones = 12.0 * np.log2(ratio) + self.f0_up_key
            semitones = max(-12.0, min(12.0, semitones))

            audio_norm = audio_norm.astype(np.float64)
            shifted = librosa.effects.pitch_shift(
                audio_norm, sr=self.sr, n_steps=semitones,
                bins_per_octave=24
            )
        else:
            shifted = audio_norm

        shifted = np.nan_to_num(shifted, nan=0.0)
        out_max = np.max(np.abs(shifted))
        if out_max > 1e-8:
            shifted = shifted / out_max * 0.95

        if shifted.max() < 1e-6:
            shifted = audio

        if orig_sr != self.sr:
            shifted = librosa.resample(shifted, orig_sr=self.sr, target_sr=orig_sr)

        return shifted.astype(np.float32)
