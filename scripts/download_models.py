import subprocess
import sys
from pathlib import Path


def main():
    print("\n=== Descarga de Modelos Base ===\n")

    models_dir = Path(__file__).parent.parent / "models" / "base"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Los modelos base se descargarán automáticamente\n"
          "desde HuggingFace la primera vez que ejecutes\n"
          "el programa.\n")

    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        print("Verificando descarga de HuBERT...")
        HubertModel.from_pretrained("facebook/hubert-base-ls960")
        Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        print("  HuBERT: OK")
    except Exception as e:
        print(f"  Error descargando HuBERT: {e}")
        print("  Se descargará automáticamente al ejecutar el programa.")

    print("\nDescarga completada.")


if __name__ == "__main__":
    main()
