#!/usr/bin/env python3
import sys
import os
from pathlib import Path


def main():
    print()
    print("  ╔═══════════════════════════════════════╗")
    print("  ║        VoiceMod - RVC Cloner          ║")
    print("  ║  Clonador de Voz en Tiempo Real       ║")
    print("  ╚═══════════════════════════════════════╝")
    print()

    args = sys.argv[1:]

    if "--terminal" in args or "-t" in args:
        _run_terminal()
    elif "--train" in args or "-T" in args:
        _run_training()
    else:
        _run_gui()


def _run_gui():
    from ui import VoiceModGUI

    gui = VoiceModGUI()
    gui.run()


def _run_terminal():
    from ui import VoiceModGUI

    gui = VoiceModGUI()
    gui._run_terminal()


def _run_training():
    if len(sys.argv) < 3:
        print("Uso: python run.py --train <ruta_del_audio>")
        return

    audio_path = sys.argv[2]
    if not os.path.exists(audio_path):
        print(f"Archivo no encontrado: {audio_path}")
        return

    from train import Trainer

    trainer = Trainer()
    trainer.run_pipeline(audio_path)


def _check_models():
    models_dir = Path("models/base")
    if not models_dir.exists():
        print("Ejecuta setup.bat primero para descargar los modelos base.")
        return False

    required = ["contentvec_hubert_base.onnx", "hifigan.onnx"]
    for model in required:
        if not (models_dir / model).exists():
            print(f"Modelo faltante: {model}")
            print("Ejecuta: python scripts/download_models.py")
            return False
    return True


if __name__ == "__main__":
    main()
