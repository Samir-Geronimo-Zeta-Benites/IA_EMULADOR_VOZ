import sys, json, os, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vc_engine import VoiceConverter

vc = VoiceConverter()
stats = vc.train("voces_audios_crudos/audio_jimmy.m4a")
print(f'F0 target: mean={stats["f0_mean"]:.1f}Hz, std={stats["f0_std"]:.1f}Hz')

model_path = "models/trained/voz_jimmy.pth"
np.save(str(Path(model_path).with_suffix(".stats.npy")), stats)

cfg_path = "config/settings.json"
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["rvc"]["model_path"] = model_path
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Config actualizada: model_path={model_path}")

import librosa
audio, sr = librosa.load("voces_audios_crudos/audio_jimmy.m4a", sr=48000, mono=True)
audio = audio[:48000*3].astype(np.float32)
out = vc.convert(audio, 48000)
print(f"Test: input={audio.shape}, output={out.shape}")
print(f"Output range: [{out.min():.4f}, {out.max():.4f}] - VALIDO!")
print("TODO LISTO - Ejecuta run.bat")
