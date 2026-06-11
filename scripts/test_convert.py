import sys, numpy as np, soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.vc_engine import VoiceConverter

print("=== Test VoiceConverter ===")

vc = VoiceConverter()
print(f"Stats loaded: {vc.target_stats is not None}")
if vc.target_stats:
    print(f"Target F0: mean={vc.target_stats['f0_mean']:.1f}, std={vc.target_stats['f0_std']:.1f}")

sr = 48000
duration = 2.0
t = np.linspace(0, duration, int(sr * duration), False)
audio = (np.sin(2*np.pi*150*t) * 0.4 + np.sin(2*np.pi*300*t) * 0.2 + np.sin(2*np.pi*450*t) * 0.1).astype(np.float32)

print(f"Input: shape={audio.shape}, range=[{audio.min():.4f}, {audio.max():.4f}]")

out = vc.convert(audio, sr)
print(f"Output: shape={out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")

sf.write("_test_input.wav", audio, sr)
sf.write("_test_output.wav", out, sr)
print(f"\nSaved _test_input.wav ({len(audio)/sr:.2f}s) and _test_output.wav ({len(out)/sr:.2f}s)")
print("Listen to compare - if output is noise, converter is broken")
