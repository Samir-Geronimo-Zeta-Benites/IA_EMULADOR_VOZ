import json
import os
import subprocess
import sys
from pathlib import Path


class Trainer:
    def __init__(self, config_path="config/settings.json"):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = json.load(f)
        self.train_cfg = self.cfg["training"]
        self.exp_dir = Path("models/trained")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, audio_path: str, model_name: str = None) -> str:
        if model_name is None:
            model_name = Path(audio_path).stem.replace(" ", "_").replace("-", "_")

        from train.preprocess import AudioPreprocessor

        print("=== FASE 1: Preprocesamiento ===")
        preprocessor = AudioPreprocessor()
        dataset_dir = preprocessor.run(
            audio_path,
            str(self.exp_dir / f"dataset_{model_name}"),
        )

        print("\n=== FASE 2: Entrenamiento RVC ===")
        from core.rvc.train import RVCTrainer

        trainer = RVCTrainer()
        model_path = trainer.train(dataset_dir, model_name=model_name)

        print(f"\n=== FASE 3: Modelo listo ===")

        self.cfg["rvc"]["model_path"] = model_path
        with open(self.config_path if hasattr(self, 'config_path') else "config/settings.json", "w") as f:
            json.dump(self.cfg, f, indent=2)

        print(f"\nEntrenamiento completado. Modelo listo en: {model_path}")
        return model_path
