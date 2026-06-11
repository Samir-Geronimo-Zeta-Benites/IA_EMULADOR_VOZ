import sys, os, json, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
torch.set_num_threads(2)

AUDIO_PATH = "voces_audios_crudos/audio_jimmy.m4a"
OUTPUT = "models/trained/voz_jimmy.pth"
EPOCHS = 30
SR = 40000

def log(msg):
    print(msg, flush=True)

def load_rvc():
    from core.rvc_model.models import SynthesizerTrnMs256NSF
    ckpt = torch.load("models/base/f0G40k.pth", map_location="cpu")
    model = SynthesizerTrnMs256NSF(
        spec_channels=1025, segment_size=12800, inter_channels=192,
        hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
        kernel_size=3, p_dropout=0, resblock="1",
        resblock_kernel_sizes=[3,7,11],
        resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
        upsample_rates=[10,10,2,2], upsample_initial_channel=512,
        upsample_kernel_sizes=[16,16,4,4],
        spk_embed_dim=109, gin_channels=256, sr=SR,
        is_half=False, phone_dim=768,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model

log("=== RVC Fine-Tuning ===")
log(f"Audio: {AUDIO_PATH}")
log(f"Output: {OUTPUT}")

import librosa
audio, orig_sr = librosa.load(AUDIO_PATH, sr=SR, mono=True)
log(f"Loaded audio: {len(audio)/SR:.1f}s, sr={orig_sr}")
audio = audio[:90 * SR]
audio = audio / (np.max(np.abs(audio)) + 1e-8)

log("Loading HuBERT for feature extraction...")
from transformers import HubertModel, Wav2Vec2FeatureExtractor
hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
hubert.eval()

log("Extracting content features from audio...")
CHUNK = SR * 5
all_feats = []
all_f0 = []
all_mel = []

import pyworld as pw
for start in range(0, len(audio), CHUNK):
    end = min(start + CHUNK, len(audio))
    seg = audio[start:end].astype(np.float64)
    if len(seg) < SR:
        break

    log(f"  Segment {start//SR}-{end//SR}s")

    audio16 = librosa.resample(seg, orig_sr=SR, target_sr=16000)
    inputs = hubert_fe(audio16.astype(np.float32), sampling_rate=16000,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = hubert(**inputs).last_hidden_state.cpu().numpy()

    f0, t = pw.dio(seg, SR, f0_floor=50, f0_ceil=1100)
    f0 = pw.stonemask(seg, f0, t, SR)

    S = librosa.stft(seg, n_fft=2048, hop_length=400)
    mel = np.abs(S)

    f0_t = torch.from_numpy(f0).float().unsqueeze(0)
    f0_t = f0_t[:, :feats.shape[1]]
    feats_t_full = torch.from_numpy(feats).float().transpose(1, 2)
    feats_t_full = feats_t_full[:, :, :f0_t.size(-1)]

    all_feats.append(feats_t_full)
    all_f0.append(f0_t)
    all_mel.append(torch.from_numpy(mel).float().unsqueeze(0)[:, :, :f0_t.size(-1)])

feats_all = torch.cat(all_feats, dim=2)
f0_all = torch.cat(all_f0, dim=1)
mel_all = torch.cat(all_mel, dim=2)
log(f"Features: feats={feats_all.shape}, f0={f0_all.shape}, mel={mel_all.shape}")

del hubert, hubert_fe
log("HuBERT unloaded (memory freed)")

log("Loading RVC model...")
model = load_rvc()
model.train()

log("Starting training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()

TRAIN_LEN = SR * 3
for epoch in range(1, EPOCHS + 1):
    start = np.random.randint(0, max(1, feats_all.size(2) - TRAIN_LEN // 400))
    end = start + TRAIN_LEN // 400

    feats_b = feats_all[:, :, start:end]
    f0_b = f0_all[:, start:end]
    mel_b = mel_all[:, :1025, start:end]

    lengths = torch.tensor([feats_b.size(2)])
    spk_id = torch.tensor([0])

    optimizer.zero_grad()
    output, _, _, _, _ = model(feats_b, lengths, f0_b, f0_b,
                                mel_b, mel_b, spk_id)
    loss = loss_fn(output, mel_b)
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 5 == 0:
        log(f"  Epoch {epoch}/{EPOCHS}: loss={loss.item():.6f}")

torch.save({"model": model.state_dict()}, OUTPUT)
log(f"\nFine-tuned model saved: {OUTPUT}")
log("DONE!")
