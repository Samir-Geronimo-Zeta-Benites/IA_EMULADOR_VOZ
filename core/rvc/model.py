import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class ContentEncoder(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.proj(x.transpose(1, 2))
        x = self.norm(x)
        return x.transpose(1, 2)


class Generator(nn.Module):
    def __init__(self, in_dim: int = 768, content_dim: int = 128, f0_dim: int = 1):
        super().__init__()
        self.content_encoder = ContentEncoder(in_dim, content_dim)
        self.input_proj = nn.Conv1d(content_dim + f0_dim, 256, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(256, 256, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(256, 256, 3, padding=1),
                nn.LeakyReLU(0.2),
            ) for _ in range(2)
        ])
        self.output_proj = nn.Conv1d(256, 80, 1)

    def forward(self, content_feats, f0):
        content_feats = self.content_encoder(content_feats)
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        if f0.size(-1) != content_feats.size(-1):
            f0 = nn.functional.interpolate(f0, size=content_feats.size(-1))
        x = torch.cat([content_feats, f0], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            skip = x
            x = block(x)
            x = x + skip
        mel = self.output_proj(x)
        return mel


class RVCInference:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.rvc_cfg = self.cfg["rvc"]
        self.f0_up_key = self.rvc_cfg["f0_up_key"]
        self.device = torch.device("cpu")

        self.generator = None
        self._load_models()

    def _load_models(self):
        model_path = Path(self.rvc_cfg["model_path"])
        if model_path.exists():
            try:
                ckpt = torch.load(str(model_path), map_location="cpu")
                self.generator = Generator()
                if "generator" in ckpt:
                    self.generator.load_state_dict(ckpt["generator"])
                else:
                    self.generator.load_state_dict(ckpt)
                self.generator.eval().to(self.device)
                print(f"Generator cargado: {model_path}")
            except Exception as e:
                print(f"Error cargando generator: {e}")
                self.generator = None
        else:
            print(f"Modelo no encontrado: {model_path}")
            print("Usa el pipeline de entrenamiento primero.")

    def infer(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        if self.generator is None:
            return audio

        from core.extractors import HubertExtractor, PitchExtractor

        hubert = HubertExtractor()
        pitch = PitchExtractor()

        feats = hubert.extract(audio, sr)
        f0 = pitch.extract(audio, sr)
        f0 = pitch.interpolate(f0)

        feats_t = torch.from_numpy(feats).float()
        f0_t = torch.from_numpy(f0).float().unsqueeze(0)

        if f0_t.size(-1) != feats_t.size(-1):
            f0_t = nn.functional.interpolate(
                f0_t.unsqueeze(1), size=feats_t.size(-1)
            ).squeeze(1)

        with torch.no_grad():
            mel = self.generator(feats_t, f0_t)

        from core.vocoder import Vocoder
        vocoder = Vocoder()
        output = vocoder.decode(mel.cpu().numpy())
        return output

    def extract_features(self, audio, sr):
        from core.extractors import HubertExtractor
        hubert = HubertExtractor()
        return hubert.extract(audio, sr)

    def set_f0_shift(self, semitones: int):
        self.f0_up_key = semitones

    def set_device(self, use_gpu: bool):
        self.device = torch.device("cuda" if use_gpu else "cpu")
        if self.generator is not None:
            self.generator.to(self.device)
