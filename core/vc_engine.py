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

        max_len = 60 * self.sr
        if len(audio) > max_len:
            print(f"  Audio largo ({len(audio)/sr:.0f}s), truncando a 60s")
            audio = audio[:max_len]

        audio = (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float64)

        print("  Extrayendo F0 (dio)...")
        f0, t = pw.dio(audio, self.sr, f0_floor=50.0, f0_ceil=1100.0)
        print("  Extrayendo envolvente espectral (cheaptrick)...")
        f0 = pw.stonemask(audio, f0, t, self.sr)
        sp = pw.cheaptrick(audio, f0, t, self.sr)
        print("  Extrayendo aperiodicidad (d4c)...")
        ap = pw.d4c(audio, f0, t, self.sr)

        f0_valid = f0[f0 > 0]
        stats = {
            "f0_mean": float(np.mean(f0_valid)) if len(f0_valid) > 0 else 150.0,
            "f0_std": float(np.std(f0_valid)) if len(f0_valid) > 0 else 50.0,
            "sp_mean": np.mean(sp, axis=0).astype(np.float32),
            "sp_std": np.std(sp, axis=0).astype(np.float32),
            "sr": self.sr,
        }

        model_path = Path(self.cfg["rvc"]["model_path"])
        stats_path = model_path.with_suffix(".stats.npy")
        np.save(str(stats_path), stats)
        self.target_stats = stats
        print(f"Estadisticas guardadas en {stats_path}")
        return stats

    def convert(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.target_stats is None:
            return audio

        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        audio = audio.astype(np.float64)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        f0, t = pw.dio(audio, self.sr, f0_floor=50.0, f0_ceil=1100.0)
        f0 = pw.stonemask(audio, f0, t, self.sr)
        sp = pw.cheaptrick(audio, f0, t, self.sr)
        ap = pw.d4c(audio, f0, t, self.sr)

        f0_src_valid = f0[f0 > 0]
        if len(f0_src_valid) > 0:
            src_mean = np.mean(f0_src_valid)
            src_std = np.std(f0_src_valid) + 1e-8
            f0_shifted = f0.copy()
            mask = f0 > 0
            f0_shifted[mask] = (
                (f0[mask] - src_mean) / src_std
                * self.target_stats["f0_std"]
                + self.target_stats["f0_mean"]
            )
            f0 = f0_shifted

        sp = sp.astype(np.float32)
        target_sp_mean = self.target_stats["sp_mean"]
        sp_src_mean = np.mean(sp, axis=0)
        sp_shifted = sp - sp_src_mean + target_sp_mean
        sp_shifted = np.maximum(sp_shifted, 1e-8)

        y = pw.synthesize(
            f0.astype(np.float64),
            sp_shifted.astype(np.float64),
            ap.astype(np.float64),
            self.sr,
        )

        y = np.nan_to_num(y, nan=0.0)
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val * 0.95

        if sr != self.sr:
            y = librosa.resample(y, orig_sr=self.sr, target_sr=sr)

        return y.astype(np.float32)
