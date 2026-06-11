import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, config_path="config/settings.json"):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.exp_dir = Path("models/trained")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.sr = int(self.cfg.get("training", {}).get("sr", 40000))
        torch.set_num_threads(int(self.cfg.get("training", {}).get("cpu_threads", 4)))
        self.device = torch.device(
            "cuda"
            if self.cfg.get("pipeline", {}).get("use_gpu", False)
            and torch.cuda.is_available()
            else "cpu"
        )
        self.use_half = self.device.type == "cuda"

    def run_pipeline(self, audio_path: str, model_name: str = None) -> str:
        if model_name is None:
            stem = Path(audio_path).stem.replace(" ", "_").replace("-", "_")
            model_name = "voz_jimmy" if "jimmy" in stem.lower() else stem

        output_path = self.exp_dir / f"{model_name}.pth"
        print("=== ENTRENAMIENTO: RVC Fine-tune ===\n")
        print(f"Audio: {audio_path}")
        print(f"Modelo destino: {output_path}")
        print(f"Device: {self.device}\n")

        base_path = Path("models/base/f0G40k.pth")
        if not base_path.exists():
            raise FileNotFoundError(
                "Falta models/base/f0G40k.pth. Ejecuta scripts/setup_gpu.bat "
                "o python scripts/download_models.py antes de entrenar."
            )

        model = self._load_rvc_model(base_path)
        feats_all, f0_all, f0f_all, spec_all = self._extract_training_tensors(audio_path)
        self._fine_tune(model, feats_all, f0_all, f0f_all, spec_all, output_path)

        self.cfg["rvc"]["model_path"] = str(output_path).replace("\\", "/")
        with open(self.config_path, "w") as f:
            json.dump(self.cfg, f, indent=2)

        print(f"\nEntrenamiento completado. Modelo listo en: {output_path}")
        return str(output_path)

    def _load_rvc_model(self, base_path: Path):
        from core.rvc_model.models import SynthesizerTrnMs256NSF

        ckpt = torch.load(str(base_path), map_location="cpu")
        model = SynthesizerTrnMs256NSF(
            spec_channels=1025,
            segment_size=12800,
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            spk_embed_dim=109,
            gin_channels=256,
            sr=self.sr,
            is_half=self.use_half,
            phone_dim=768,
        )
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(self.device)
        if self.use_half:
            model.half()
        model.train()
        del ckpt
        gc.collect()
        return model

    def _extract_training_tensors(self, audio_path: str):
        import librosa
        import pyworld as pw
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        train_cfg = self.cfg.get("training", {})
        max_seconds = int(train_cfg.get("max_seconds", 360))
        chunk_seconds = int(train_cfg.get("feature_chunk_seconds", 5))

        print("Cargando audio...")
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        audio = audio[: max_seconds * self.sr]
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        print(f"Duracion usada: {len(audio) / self.sr:.1f}s")

        print("Extrayendo HuBERT, F0 y espectrogramas...")
        hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)
        hubert.eval()
        if self.use_half:
            hubert.half()

        all_feats, all_f0, all_f0f, all_spec = [], [], [], []
        chunk_len = chunk_seconds * self.sr
        for start in range(0, len(audio), chunk_len):
            end = min(start + chunk_len, len(audio))
            seg = audio[start:end].astype(np.float64)
            if len(seg) < self.sr:
                continue

            a16 = librosa.resample(seg, orig_sr=self.sr, target_sr=16000)
            inp = hubert_fe(
                a16.astype(np.float32),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            if self.use_half:
                inp["input_values"] = inp["input_values"].half()
            with torch.no_grad():
                feats = hubert(**inp).last_hidden_state.float().cpu()

            f0, t = pw.dio(seg, self.sr, f0_floor=50, f0_ceil=1100)
            f0 = pw.stonemask(seg, f0, t, self.sr)
            spec = np.abs(librosa.stft(seg, n_fft=2048, hop_length=400))

            f0_i = np.clip((f0 / 1100.0 * 256.0).astype(np.int64), 0, 255)
            min_t = min(feats.shape[1], len(f0_i), spec.shape[1])
            if min_t < 8:
                continue

            all_feats.append(feats[:, :min_t, :])
            all_f0.append(torch.from_numpy(f0_i[:min_t]).long())
            all_f0f.append(torch.from_numpy(f0[:min_t].astype(np.float32)))
            all_spec.append(torch.from_numpy(spec[:, :min_t]).unsqueeze(0).float())
            print(f"  {start // self.sr:>4}-{end // self.sr:<4}s -> {min_t} frames")

        del hubert, hubert_fe
        gc.collect()
        if not all_feats:
            raise RuntimeError("No se pudo extraer material de entrenamiento del audio.")

        feats_all = torch.cat(all_feats, dim=1)
        f0_all = torch.cat(all_f0).unsqueeze(0)
        f0f_all = torch.cat(all_f0f).unsqueeze(0)
        spec_all = torch.cat(all_spec, dim=2)
        print(f"Tensores: feats={tuple(feats_all.shape)} f0={tuple(f0_all.shape)} spec={tuple(spec_all.shape)}")
        return feats_all, f0_all, f0f_all, spec_all

    def _fine_tune(self, model, feats_all, f0_all, f0f_all, spec_all, output_path: Path):
        train_cfg = self.cfg.get("training", {})
        epochs = int(train_cfg.get("epochs", 30))
        save_every = int(train_cfg.get("save_every_epoch", 5))
        lr = float(train_cfg.get("learning_rate", 5e-5))
        train_frames = int(train_cfg.get("train_frames", 320))

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        total_frames = feats_all.size(1)
        ds = torch.tensor([0], device=self.device)

        print(f"Fine-tune: epochs={epochs}, frames/step={train_frames}, lr={lr}")
        for ep in range(1, epochs + 1):
            span = min(train_frames, total_frames)
            start = 0 if total_frames <= span else np.random.randint(0, total_frames - span)
            end = start + span

            feats = feats_all[:, start:end, :].to(self.device)
            f0 = f0_all[:, start:end].to(self.device)
            f0f = f0f_all[:, start:end].to(self.device)
            spec = spec_all[:, :, start:end].to(self.device)
            if self.use_half:
                feats = feats.half()
                f0f = f0f.half()
                spec = spec.half()

            lengths = torch.tensor([feats.size(1)], device=self.device)
            spec_lengths = torch.tensor([spec.size(2)], device=self.device)

            opt.zero_grad(set_to_none=True)
            output, ids_slice, _, _, posterior = model(
                feats, lengths, f0, f0f, spec, spec_lengths, ds
            )
            z, z_p, m_p, logs_p, m_q, logs_q = posterior

            output_spec = torch.stft(
                output.float().squeeze(0),
                n_fft=2048,
                hop_length=400,
                return_complex=True,
            ).abs().unsqueeze(0)
            target_start = int(ids_slice[0].item())
            target = spec.float()[:, :, target_start : target_start + output_spec.size(2)]
            min_f = min(output_spec.size(1), target.size(1))
            min_t = min(output_spec.size(2), target.size(2))

            kl = logs_p - logs_q + 0.5 * (
                (m_q - m_p).pow(2) * (-2 * logs_p).exp()
                + (2 * (logs_q - logs_p)).exp()
                - 1
            )
            loss = F.l1_loss(output_spec[:, :min_f, :min_t], target[:, :min_f, :min_t])
            loss = loss + 0.1 * kl.float().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if ep == 1 or ep % save_every == 0 or ep == epochs:
                print(f"  Epoca {ep}/{epochs}: loss={loss.item():.4f}")
                self._save_model(model, output_path)
                gc.collect()

    def _save_model(self, model, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
        torch.save({"model": state, "sr": self.sr, "version": "v2"}, str(path))
