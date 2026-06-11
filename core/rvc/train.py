import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class VoiceDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.files = sorted(Path(dataset_dir).glob("*.wav"))
        self.metadata = None
        meta_path = Path(dataset_dir) / "metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import soundfile as sf
        import librosa

        audio, sr = sf.read(str(self.files[idx]))
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return torch.from_numpy(audio).float()


class RVCTrainer:
    def __init__(self, config_path="config/settings.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.train_cfg = self.cfg["training"]
        self.exp_dir = Path("models/trained")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(self, audio_path: str, output_dir: str = None):
        from train.preprocess import AudioPreprocessor

        preprocessor = AudioPreprocessor()
        output = output_dir or str(self.exp_dir / "dataset")
        preprocessor.run(audio_path, output)
        return output

    def train(self, dataset_dir: str = None, model_name: str = "target"):
        if dataset_dir is None:
            dataset_dir = str(self.exp_dir / "dataset")

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(
                f"Dataset no encontrado en {dataset_dir}. "
                "Ejecuta preprocess primero."
            )

        from core.extractors import HubertExtractor, PitchExtractor
        from core.rvc.model import Generator

        print("=== Entrenamiento del Modelo RVC ===\n")
        print(f"Dataset: {dataset_dir}")

        dataset = VoiceDataset(dataset_dir)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        device = torch.device("cpu")
        generator = Generator().to(device)
        optimizer = optim.AdamW(
            generator.parameters(),
            lr=self.train_cfg["learning_rate"],
        )
        loss_fn = nn.MSELoss()

        hubert = HubertExtractor()
        pitch = PitchExtractor()

        epochs = self.train_cfg["epochs"]
        save_every = self.train_cfg["save_every_epoch"]

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                audio = batch.numpy().squeeze()
                if len(audio) < 16000:
                    continue

                with torch.no_grad():
                    feats = hubert.extract(audio, 16000)
                    f0 = pitch.extract(audio, 16000)
                    f0 = pitch.interpolate(f0)

                    target_mel = self._compute_mel(audio, 16000)

                    feats_t = torch.from_numpy(feats).float()
                    f0_t = torch.from_numpy(f0).float().unsqueeze(0).unsqueeze(0)
                    target_t = torch.from_numpy(target_mel).float().unsqueeze(0)

                    if f0_t.size(-1) != feats_t.size(-1):
                        f0_t = nn.functional.interpolate(
                            f0_t, size=feats_t.size(-1)
                        )
                    if target_t.size(-1) != feats_t.size(-1):
                        target_t = nn.functional.interpolate(
                            target_t, size=feats_t.size(-1)
                        )

                pred = generator(feats_t, f0_t)
                loss = loss_fn(pred, target_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Época [{epoch}/{epochs}] loss: {avg_loss:.6f}")

            if epoch % save_every == 0:
                self._save_checkpoint(generator, epoch, avg_loss)

        output_path = str(self.exp_dir / f"{model_name}.pth")
        self._save_model(generator, output_path)
        print(f"\nModelo guardado: {output_path}")
        return output_path

    def _compute_mel(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=1024, hop_length=320,
            n_mels=80, fmin=0, fmax=sr // 2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return (mel_db + 80) / 80

    def _save_checkpoint(self, generator, epoch, loss):
        path = self.exp_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            "generator": generator.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }, str(path))

    def _save_model(self, generator, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator": generator.state_dict()}, path)

    def export_onnx(self, pth_path: str = None):
        if pth_path is None:
            pth_path = str(self.exp_dir / "target.pth")

        from core.rvc.model import Generator

        ckpt = torch.load(pth_path, map_location="cpu")
        generator = Generator()
        generator.load_state_dict(ckpt["generator"])
        generator.eval()

        dummy_feats = torch.randn(1, 256, 100)
        dummy_f0 = torch.randn(1, 1, 100)

        onnx_path = str(self.exp_dir / "target.onnx")
        torch.onnx.export(
            generator,
            (dummy_feats, dummy_f0),
            onnx_path,
            input_names=["feats", "f0"],
            output_names=["mel"],
            dynamic_axes={
                "feats": {2: "time"},
                "f0": {2: "time"},
                "mel": {2: "time"},
            },
            opset_version=17,
        )
        print(f"ONNX exportado: {onnx_path}")
        return onnx_path
