import sys, time, numpy as np, torch
from pathlib import Path
torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).parent.parent))

def log(msg):
    print(msg, flush=True)

log("Loading model...")
t0 = time.time()
from core.vc_engine import VoiceConverter
vc = VoiceConverter()
log(f"Loaded in {time.time()-t0:.1f}s, model: {vc.model is not None}")

log("Loading HuBERT separately...")
t0 = time.time()
feats = vc._extract_hubert(np.random.randn(16000).astype(np.float32))
log(f"HuBERT ok in {time.time()-t0:.1f}s, shape: {feats.shape}")

log("Testing full inference...")
t0 = time.time()
audio = np.random.randn(40000).astype(np.float32) * 0.01
out = vc.convert(audio, 40000)
log(f"Inference done in {time.time()-t0:.1f}s")
log(f"Input: {audio.shape}, Output: {out.shape}")
log(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
log("SUCCESS!")
