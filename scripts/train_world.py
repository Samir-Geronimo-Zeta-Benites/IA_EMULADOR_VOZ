import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vc_engine import VoiceConverter
import librosa, soundfile as sf

vc = VoiceConverter()
stats = vc.train("voces_audios_crudos/audio_jimmy.m4a")
print(f"Trained: F0={stats['f0_mean']:.0f}Hz, spec={stats['spec_mean'].shape}")

model_path = "models/trained/voz_jimmy.pth"
cfg_path = "config/settings.json"
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["rvc"]["model_path"] = model_path
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

# Quick test
audio, sr = librosa.load("voces_audios_crudos/audio_jimmy.m4a", sr=48000, mono=True)
audio = audio[:48000*2].astype(np.float32)
out = vc.convert(audio, 48000)
print(f"Test: in={audio.shape} out={out.shape} range=[{out.min():.3f},{out.max():.3f}]")
sf.write("_test_in.wav", audio / np.max(np.abs(audio)), 48000)
sf.write("_test_out.wav", out / np.max(np.abs(out)), 48000)
print("Saved _test_in.wav and _test_out.wav")
