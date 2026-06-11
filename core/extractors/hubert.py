import numpy as np
import torch
import gc


class HubertExtractor:
    def __init__(self, model_name: str = "facebook/hubert-base-ls960"):
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.sample_rate = 16000
        self._model = None
        self._feature_extractor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name
            )
            self._model = HubertModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
            print(f"HuBERT cargado: {self.model_name}")
        except Exception as e:
            print(f"Error cargando HuBERT: {e}")

    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        self._load()
        if self._model is None or self._feature_extractor is None:
            return np.zeros((1, 768, max(1, len(audio) // 320)), dtype=np.float32)

        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)

        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        inputs = self._feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            outputs = self._model(**inputs.to(self.device))
            features = outputs.last_hidden_state

        result = features.cpu().numpy().transpose(0, 2, 1)
        return result

    def unload(self):
        self._model = None
        self._feature_extractor = None
        gc.collect()

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
