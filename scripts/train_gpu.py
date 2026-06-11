import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import Trainer


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "voces_audios_crudos/audio_jimmy.m4a"
    trainer = Trainer()
    trainer.run_pipeline(audio_path, "voz_jimmy")


if __name__ == "__main__":
    main()
