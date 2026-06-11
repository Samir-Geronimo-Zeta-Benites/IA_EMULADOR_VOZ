import sys, gc, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
torch.set_num_threads(2)

OUT_DIR = Path("models/trained/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import librosa
audio, sr = librosa.load("voces_audios_crudos/audio_jimmy.m4a", sr=40000, mono=True)
print(f"Audio: {len(audio)/sr:.1f}s")
audio = audio[:90 * sr]
audio = audio / (np.max(np.abs(audio)) + 1e-8)

print("Loading HuBERT...")
from transformers import HubertModel, Wav2Vec2FeatureExtractor
hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

import pyworld as pw

CHUNK = sr * 5
feat_idx = 0
for start in range(0, len(audio), CHUNK):
    end = min(start + CHUNK, len(audio))
    seg = audio[start:end].astype(np.float64)
    if len(seg) < sr:
        break
    print(f"  Chunk {start//sr}-{end//sr}s")

    audio16 = librosa.resample(seg, orig_sr=40000, target_sr=16000)
    inputs = hubert_fe(audio16.astype(np.float32), sampling_rate=16000,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = hubert(**inputs).last_hidden_state.cpu().numpy()

    f0, t = pw.dio(seg, sr, f0_floor=50, f0_ceil=1100)
    f0 = pw.stonemask(seg, f0, t, sr)

    S = librosa.stft(seg, n_fft=2048, hop_length=400)
    mel = np.abs(S)

    f0_int = np.clip((f0 / 1100.0 * 256.0).astype(np.int64), 0, 255)
    min_t = min(len(f0_int), feats.shape[1], mel.shape[1])
    feats = feats[:, :min_t, :]
    f0_int = f0_int[:min_t]
    f0_float = np.where(f0[:min_t] > 0, f0[:min_t], 0).astype(np.float32)
    mel = mel[:, :min_t]

    np.save(OUT_DIR / f"feats_{feat_idx}.npy", feats.astype(np.float32))
    np.save(OUT_DIR / f"f0_{feat_idx}.npy", f0_int.astype(np.int64))
    np.save(OUT_DIR / f"f0f_{feat_idx}.npy", f0_float)
    np.save(OUT_DIR / f"mel_{feat_idx}.npy", mel.astype(np.float32))
    feat_idx += 1
    gc.collect()

print(f"Saved {feat_idx} chunks to {OUT_DIR}")
del hubert, hubert_fe
gc.collect()
print("DONE")
