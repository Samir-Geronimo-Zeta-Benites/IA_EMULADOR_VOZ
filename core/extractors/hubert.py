import numpy as np
import torch


class HubertExtractor:
    def __init__(self, model_name: str = "facebook/hubert-base-ls960"):
        self.device = torch.device("cpu")
        self.model = None
        self.sample_rate = 16000
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor

            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name
            )
            self.model = HubertModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            print(f"HuBERT cargado: {model_name}")
        except Exception as e:
            print(f"Error cargando HuBERT: {e}")
            self.feature_extractor = None

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.model is None or self.feature_extractor is None:
            return np.zeros((1, 768, max(1, len(audio) // 320)), dtype=np.float32)

        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)

        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs.to(self.device))
            features = outputs.last_hidden_state

        return features.cpu().numpy().transpose(0, 2, 1)

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int):
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        new_len = int(len(audio) * ratio)
        return np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        )
