from pathlib import Path
from urllib.request import urlretrieve


def main():
    print("\n=== Descarga de Modelos Base ===\n")

    models_dir = Path(__file__).parent.parent / "models" / "base"
    models_dir.mkdir(parents=True, exist_ok=True)

    rvc_base = models_dir / "f0G40k.pth"
    if not rvc_base.exists():
        print("Descargando checkpoint base RVC f0G40k.pth...")
        url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth"
        urlretrieve(url, str(rvc_base))
        print("  RVC base: OK")
    else:
        print("  RVC base: OK")

    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        print("Verificando descarga de HuBERT...")
        HubertModel.from_pretrained("facebook/hubert-base-ls960")
        Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        print("  HuBERT: OK")
    except Exception as e:
        print(f"  Error descargando HuBERT: {e}")
        print("  Se descargara automaticamente al ejecutar entrenamiento o conversion.")

    print("\nDescarga completada.")


if __name__ == "__main__":
    main()
