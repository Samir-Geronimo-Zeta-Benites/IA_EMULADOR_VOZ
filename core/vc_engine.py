import numpy as np
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
        sp = model_path.with_suffix(".stats.npy")
        if sp.exists():
            self.target_stats = np.load(str(sp), allow_pickle=True).item()
            print(f"Estadisticas cargadas")

    def train(self, audio_path: str):
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        audio = audio[:90 * self.sr]
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=160))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=1100,
                                       sr=self.sr, fill_na=0)
        f0_v = f0[f0 > 0]

        stats = {
            "f0_mean": float(np.mean(f0_v)) if len(f0_v) > 0 else 120.0,
            "f0_std": float(np.std(f0_v)) if len(f0_v) > 0 else 30.0,
            "spec_mean": np.mean(S_db, axis=1).astype(np.float32),
            "sr": self.sr,
        }

        model_path = Path(self.cfg["rvc"]["model_path"])
        np.save(str(model_path.with_suffix(".stats.npy")), stats)
        self.target_stats = stats
        print(f"F0: {stats['f0_mean']:.0f}Hz, espec: {stats['spec_mean'].shape}")
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

        S = librosa.stft(audio, n_fft=2048, hop_length=160)
        mag = np.abs(S)
        phase = np.angle(S)

        # === PITCH SHIFT via F0 ===
        target_hz = self.target_stats["f0_mean"]
        try:
            f0_src, _, _ = librosa.pyin(audio, fmin=50, fmax=1100,
                                         sr=self.sr, fill_na=0)
            f0_v = f0_src[f0_src > 0]
            if len(f0_v) > 3:
                src_hz = float(np.mean(f0_v))
            else:
                src_hz = 120.0
        except Exception:
            src_hz = 120.0

        pitch_ratio = target_hz / max(src_hz, 1e-3)
        pitch_ratio = max(0.6, min(1.8, pitch_ratio))
        pitch_ratio = pitch_ratio * (2 ** (self.f0_up_key / 12))

        n_freqs = mag.shape[0]
        mag_shifted = np.zeros_like(mag)
        for f in range(n_freqs):
            src_f = int(f / pitch_ratio)
            if 0 <= src_f < n_freqs:
                mag_shifted[f] = mag[src_f]
            else:
                mag_shifted[f] = mag[0] * (n_freqs - f) / n_freqs

        # === SPECTRAL SHAPING (EQ to match target timbre) ===
        target_spec = self.target_stats["spec_mean"]
        src_spec = 20 * np.log10(np.mean(mag_shifted, axis=1) + 1e-10)

        min_len = min(len(src_spec), len(target_spec))
        target_spec = target_spec[:min_len]
        src_spec = src_spec[:min_len]

        eq = target_spec - src_spec
        eq_smooth = np.convolve(eq, np.hanning(21), mode='same') / np.sum(np.hanning(21))
        eq_smooth = np.clip(eq_smooth, -15, 15)

        for f in range(min_len):
            mag_shifted[f] *= 10 ** (eq_smooth[f] / 20)

        # === RECONSTRUCT ===
        mag_shifted = np.maximum(mag_shifted, 1e-10)
        S_out = mag_shifted * np.exp(1j * phase[:mag_shifted.shape[0]])
        y = librosa.istft(S_out, hop_length=160, length=len(audio))

        y = np.nan_to_num(y, nan=0.0)
        ym = np.max(np.abs(y))
        if ym > 1e-8:
            y = y / ym * 0.9

        if orig_sr != self.sr:
            y = librosa.resample(y, orig_sr=self.sr, target_sr=orig_sr)

        y = np.clip(y, -0.95, 0.95)
        return y.astype(np.float32)
