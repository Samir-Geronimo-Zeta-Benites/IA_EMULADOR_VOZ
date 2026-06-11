import sys, gc, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 40000
AUDIO_PATH = "voces_audios_crudos/audio_jimmy.m4a"
OUTPUT = "models/trained/voz_jimmy.pth"
EPOCHS = 30

def log(msg): print(msg, flush=True)

log(f"Device: {DEVICE}")
log("Loading audio...")
import librosa
audio, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True)
audio = audio[:90 * SR]
audio = audio / (np.max(np.abs(audio)) + 1e-8)

log("Extracting HuBERT features (GPU)...")
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import pyworld as pw
hubert_fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
hubert.eval()

all_feats, all_f0, all_f0f, all_mel = [], [], [], []
for start in range(0, len(audio), SR * 5):
    end = min(start + SR * 5, len(audio))
    seg = audio[start:end].astype(np.float64)
    if len(seg) < SR: break

    a16 = librosa.resample(seg, orig_sr=SR, target_sr=16000)
    inp = hubert_fe(a16.astype(np.float32), sampling_rate=16000,
                    return_tensors="pt", padding=True)
    with torch.no_grad():
        f = hubert(inp.input_values.to(DEVICE)).last_hidden_state.cpu().numpy()

    f0, t = pw.dio(seg, SR, f0_floor=50, f0_ceil=1100)
    f0 = pw.stonemask(seg, f0, t, SR)
    S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=400))

    f0i = np.clip((f0 / 1100 * 256).astype(np.int64), 0, 255)
    min_t = min(len(f0i), f.shape[1], S.shape[1])
    all_feats.append(torch.from_numpy(f[:, :min_t, :]).float())
    all_f0.append(torch.from_numpy(f0i[:min_t]).long())
    all_f0f.append(torch.from_numpy(np.where(f0[:min_t] > 0, f0[:min_t], 0).astype(np.float32)))
    all_mel.append(torch.from_numpy(S[:, :min_t]).unsqueeze(0).float())
    log(f"  {start // SR}-{end // SR}s: {min_t} frames")

feats_all = torch.cat(all_feats, dim=1); f0_all = torch.cat(all_f0).unsqueeze(0)
f0f_all = torch.cat(all_f0f).unsqueeze(0); mel_all = torch.cat(all_mel, dim=2)
del hubert, hubert_fe, all_feats, all_f0, all_f0f, all_mel; gc.collect()
log(f"Features: feats={feats_all.shape} f0={f0_all.shape} mel={mel_all.shape}")

log("Loading RVC model...")
from core.rvc_model.models import SynthesizerTrnMs256NSF
ckpt = torch.load("models/base/f0G40k.pth", map_location="cpu")
model = SynthesizerTrnMs256NSF(
    spec_channels=1025, segment_size=12800, inter_channels=192,
    hidden_channels=192, filter_channels=768, n_heads=2, n_layers=6,
    kernel_size=3, p_dropout=0, resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 10, 2, 2], upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    spk_embed_dim=109, gin_channels=256, sr=SR, is_half=True, phone_dim=768,
).to(DEVICE)
model.load_state_dict(ckpt["model"], strict=False)
model.train()
del ckpt; gc.collect()

log(f"Training {EPOCHS} epochs with MEL LOSS on GPU...")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
TF = SR // 400 * 3

for ep in range(1, EPOCHS + 1):
    t = feats_all.size(1)
    s = np.random.randint(0, max(1, t - TF))
    e = s + TF

    fb = feats_all[:, s:e, :].to(DEVICE)
    f0b = f0_all[:, s:e].to(DEVICE)
    ffb = f0f_all[:, s:e].to(DEVICE)
    mb = mel_all[:, :1025, s:e].to(DEVICE)

    lengths = torch.tensor([fb.size(1)])
    ml = torch.tensor([mb.size(2)])

    opt.zero_grad()
    o, ids, _, _, (z, z_p, m_p, logs_p, m_q, logs_q) = model(
        fb.half(), lengths, f0b, ffb.half(), mb.half(), ml,
        torch.tensor([0]).to(DEVICE)
    )

    # STFT on GPU (finally works!)
    o_spec = torch.stft(o.squeeze(0), n_fft=2048, hop_length=400,
                        return_complex=True).abs().unsqueeze(0)
    tgt = mb[:, :, ids.cpu()]

    min_f = min(o_spec.size(1), tgt.size(1))
    min_t = min(o_spec.size(2), tgt.size(2))

    kl = logs_p - logs_q + 0.5 * ((m_q - m_p).pow(2) * (-2 * logs_p).exp()
                                    + (2 * (logs_q - logs_p)).exp() - 1)
    kl = kl.mean()

    mel_loss = F.l1_loss(o_spec[:, :min_f, :min_t], tgt[:, :min_f, :min_t])
    loss = mel_loss + 0.1 * kl
    loss.backward()
    opt.step()

    if ep == 1 or ep % 5 == 0:
        log(f"  Ep {ep}/{EPOCHS}: mel_loss={mel_loss.item():.4f} kl={kl.item():.4f}")
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, OUTPUT)
        gc.collect()

torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, OUTPUT)
log(f"SAVED: {OUTPUT}")
log("DONE - Copy voz_jimmy.pth to VoiceMod's models/trained/")
