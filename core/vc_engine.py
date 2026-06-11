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
            print(f"Estadisticas cargadas: F0={self.target_stats.get('f0_mean',0):.0f}Hz")

    def train(self, audio_path: str):
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        audio = audio[:90 * self.sr]
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        f0, voiced, _ = librosa.pyin(audio, fmin=50, fmax=1100, sr=self.sr, fill_na=0)
        f0_v = f0[f0 > 0]

        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=160))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        spec_mean = np.mean(S_db, axis=1).astype(np.float32)

        # Find first formant: search between 100-900 Hz
        low_bin = int(100 * 2048 / self.sr)
        high_bin = int(900 * 2048 / self.sr)
        peak_idx = low_bin + np.argmax(spec_mean[low_bin:high_bin])
        f1_hz = peak_idx * self.sr / 2048

        stats = {
            "f0_mean": float(np.mean(f0_v)) if len(f0_v) > 0 else 120.0,
            "f0_std": float(np.std(f0_v)) if len(f0_v) > 0 else 30.0,
            "spec_mean": spec_mean,
            "f1_hz": float(f1_hz),
            "sr": self.sr,
        }

        model_path = Path(self.cfg["rvc"]["model_path"])
        np.save(str(model_path.with_suffix(".stats.npy")), stats)
        self.target_stats = stats
        print(f"Target: F0={stats['f0_mean']:.0f}Hz F1={stats['f1_hz']:.0f}Hz")
        return stats

    def convert(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.target_stats is None:
            return audio

        audio = audio.squeeze().astype(np.float64)
        if audio.ndim == 0 or len(audio) < sr * 0.15:
            return np.zeros(max(len(audio), 1), dtype=np.float32)

        orig_sr = sr
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        mx = np.max(np.abs(audio))
        if mx < 1e-8:
            return audio

        S = librosa.stft(audio / mx * 0.8, n_fft=2048, hop_length=160)
        mag, phase = np.abs(S), np.angle(S)

        n_freqs = mag.shape[0]
        target_mag = np.zeros_like(mag)

        target_hz = self.target_stats["f0_mean"]
        tgt_f1 = self.target_stats.get("f1_hz", 500.0)

        try:
            f0_src, _, _ = librosa.pyin(audio / mx * 0.8, fmin=50, fmax=1100,
                                         sr=self.sr, fill_na=0)
            f0_v = f0_src[f0_src > 0]
            src_hz = float(np.mean(f0_v)) if len(f0_v) > 3 else 120.0
        except Exception:
            src_hz = 120.0

        try:
            src_spec = 20 * np.log10(np.mean(mag, axis=1) + 1e-10)
            src_f1 = np.argmax(src_spec[:800]) * self.sr / 2048
        except Exception:
            src_f1 = 500.0

        pitch_ratio = target_hz / max(src_hz, 1e-3)
        pitch_ratio *= 2 ** (self.f0_up_key / 12)
        pitch_ratio = max(0.5, min(2.0, pitch_ratio))

        formant_ratio = tgt_f1 / max(src_f1, 1e-3)
        formant_ratio = max(0.7, min(1.4, formant_ratio))

        warp_ratio = formant_ratio / pitch_ratio
        warp_ratio = max(0.5, min(2.0, warp_ratio))

        for f in range(n_freqs):
            src_idx = int(f * warp_ratio)
            if 0 <= src_idx < n_freqs - 1:
                frac = f * warp_ratio - src_idx
                target_mag[f] = (1 - frac) * mag[src_idx] + frac * mag[src_idx + 1]
            elif src_idx < 0:
                target_mag[f] = mag[0]
            else:
                target_mag[f] = mag[-1] * 0.01

        tgt_db = self.target_stats["spec_mean"]
        cur_db = 20 * np.log10(np.maximum(np.mean(target_mag, axis=1), 1e-10))
        ml = min(len(cur_db), len(tgt_db))
        eq = tgt_db[:ml] - cur_db[:ml]
        eq = np.nan_to_num(eq, nan=0.0)
        eq = np.clip(eq, -15, 15)
        w = np.hanning(61)
        eq = np.convolve(eq, w / w.sum(), mode='same')

        for f in range(ml):
            target_mag[f] *= 10 ** (eq[f] / 20)

        target_mag = np.maximum(target_mag, 1e-10)
        S_out = target_mag * np.exp(1j * phase)
        y = librosa.istft(S_out, hop_length=160, length=len(audio))
        y = np.nan_to_num(y, nan=0.0)
        ym = np.max(np.abs(y))
        if ym > 1e-8:
            y = y / ym * 0.9

        if orig_sr != self.sr:
            y = librosa.resample(y, orig_sr=self.sr, target_sr=orig_sr)
        y = np.clip(y, -0.95, 0.95)
        return y.astype(np.float32)
