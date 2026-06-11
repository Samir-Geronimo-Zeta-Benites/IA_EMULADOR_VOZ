import json
from pathlib import Path


class Trainer:
    def __init__(self, config_path="config/settings.json"):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.exp_dir = Path("models/trained")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, audio_path: str, model_name: str = None) -> str:
        if model_name is None:
            model_name = Path(audio_path).stem.replace(" ", "_").replace("-", "_")

        print("=== ENTRENAMIENTO: WORLD Voice Conversion ===\n")
        print(f"Audio: {audio_path}")
        print(f"Nombre: {model_name}\n")

        from core.vc_engine import VoiceConverter

        vc = VoiceConverter(self.config_path)
        stats = vc.train(audio_path)

        model_path = str(self.exp_dir / f"{model_name}.pth")
        import numpy as np
        np.save(str(Path(model_path).with_suffix(".stats.npy")), stats)

        self.cfg["rvc"]["model_path"] = model_path
        with open(self.config_path, "w") as f:
            json.dump(self.cfg, f, indent=2)

        print(f"\nEntrenamiento completado. Modelo listo en: {model_path}")
        return model_path
