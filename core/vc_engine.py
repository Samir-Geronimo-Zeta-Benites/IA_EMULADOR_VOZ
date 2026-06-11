import numpy as np
import torch
import json
import pyworld as pw
import librosa
from pathlib import Path


class VoiceConverter:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.sr = 40000
        self.device = torch.device("cpu")
        self.model = None
        self._hubert = None
        self._load_model()

    def _load_model(self):
        try:
            from core.rvc_model.models import SynthesizerTrnMs256NSF

            model_path = Path(self.cfg["rvc"]["model_path"])
            base_path = Path("models/base/f0G40k.pth")

            if base_path.exists():
                ckpt = torch.load(str(base_path), map_location="cpu")
                sd = ckpt["model"]
                self.model = SynthesizerTrnMs256NSF(
                    spec_channels=1025, segment_size=12800,
                    inter_channels=192, hidden_channels=192,
                    filter_channels=768, n_heads=2, n_layers=6,
                    kernel_size=3, p_dropout=0, resblock="1",
                    resblock_kernel_sizes=[3, 7, 11],
                    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    upsample_rates=[10, 10, 2, 2],
                    upsample_initial_channel=512,
                    upsample_kernel_sizes=[16, 16, 4, 4],
                    spk_embed_dim=109, gin_channels=256,
                    sr=self.sr, is_half=False, phone_dim=768,
                )
                self.model.load_state_dict(sd, strict=False)
                self.model.eval().to(self.device)
                print(f"RVC model loaded: f0G40k")
            else:
                print("f0G40k.pth not found in models/base/")
        except Exception as e:
            print(f"Error loading RVC: {e}")
            self.model = None

    def train(self, audio_path: str):
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        audio = audio[:90 * self.sr]
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio = audio.astype(np.float64)

        f0, t = pw.dio(audio, self.sr, f0_floor=50, f0_ceil=1100)
        f0 = pw.stonemask(audio, f0, t, self.sr)
        f0_valid = f0[f0 > 0]

        stats = {
            "f0_mean": float(np.mean(f0_valid)) if len(f0_valid) > 0 else 150,
            "f0_std": float(np.std(f0_valid)) if len(f0_valid) > 0 else 50,
        }

        model_path = Path(self.cfg["rvc"]["model_path"])
        np.save(str(model_path.with_suffix(".stats.npy")), stats)
        print(f"F0: {stats['f0_mean']:.0f}Hz")
        return stats

    def convert(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.model is None:
            return audio

        orig_sr = sr
        audio = audio.squeeze().astype(np.float64)
        if audio.ndim == 0 or len(audio) < orig_sr * 0.1:
            return np.zeros(max(len(audio), 1), dtype=np.float32)

        if orig_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)

        mx = np.max(np.abs(audio))
        if mx < 1e-8:
            return audio
        audio = audio / mx * 0.8

        feats = self._extract_hubert(audio)
        if feats is None:
            return audio

        f0, t = pw.dio(audio.astype(np.float64), self.sr,
                       f0_floor=50, f0_ceil=1100)
        f0 = pw.stonemask(audio.astype(np.float64), f0, t, self.sr)

        f0_t = torch.from_numpy(f0).float().unsqueeze(0)
        feats_t = torch.from_numpy(feats).float()

        min_len = min(feats_t.size(-1), f0_t.size(-1))
        feats_t = feats_t[:, :, :min_len]
        f0_t = f0_t[:, :min_len]

        lengths = torch.tensor([min_len])
        spk_id = torch.tensor([0])

        with torch.no_grad():
            try:
                output, _, _ = self.model.infer(
                    feats_t, lengths, f0_t, f0_t, spk_id, max_len=min_len
                )
                output = output.squeeze().cpu().numpy()
            except Exception as e:
                print(f"RVC infer error: {e}")
                return audio

        output = np.nan_to_num(output, nan=0.0)
        om = np.max(np.abs(output))
        if om > 1e-8:
            output = output / om * 0.9

        if orig_sr != self.sr:
            output = librosa.resample(output, orig_sr=self.sr, target_sr=orig_sr)

        output = np.clip(output, -0.95, 0.95)
        return output.astype(np.float32)

    def _extract_hubert(self, audio):
        if self._hubert is None:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            self._hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/hubert-base-ls960"
            )
            self._hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self._hubert.eval().to(self.device)

        sr16 = 16000
        if len(audio) > 0 and sr16 != self.sr:
            audio16 = librosa.resample(audio.astype(np.float64),
                                       orig_sr=self.sr, target_sr=sr16)
        else:
            audio16 = audio
        audio16 = audio16.astype(np.float32)
        inputs = self._hubert_fe(audio16, sampling_rate=sr16,
                                 return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._hubert(**inputs.to(self.device))
            feats = outputs.last_hidden_state
        return feats.cpu().numpy()  # (1, T, 768)
