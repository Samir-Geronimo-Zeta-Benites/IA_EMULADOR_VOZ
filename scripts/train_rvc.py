import sys, gc, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
torch.set_num_threads(2)

OUTPUT = "models/trained/voz_jimmy.pth"
FEAT_DIR = Path("models/trained/features")
EPOCHS = 30
SR = 40000

def log(msg): print(msg, flush=True)

log("Loading features + raw audio...")
chunks = sorted(FEAT_DIR.glob("feats_*.npy"))
max_frames = 10 * SR // 400
loaded = 0
all_feats, all_f0, all_f0f, all_mel, all_audio = [], [], [], [], []
for chunk in chunks:
    idx = chunk.stem.split("_")[1]
    f = torch.from_numpy(np.load(FEAT_DIR / f"feats_{idx}.npy")).float()
    f0i = torch.from_numpy(np.load(FEAT_DIR / f"f0_{idx}.npy")).long()
    f0fv = torch.from_numpy(np.load(FEAT_DIR / f"f0f_{idx}.npy"))
    m = np.load(FEAT_DIR / f"mel_{idx}.npy")
    a = np.load(FEAT_DIR / f"audio_{idx}.npy")
    take = min(f.size(1), m.shape[1], len(a) // 400, max_frames - loaded)
    if take > 0:
        all_feats.append(f[:, :take, :])
        all_f0.append(f0i[:take]); all_f0f.append(f0fv[:take])
        all_mel.append(torch.from_numpy(m).unsqueeze(0).float()[:, :, :take])
        all_audio.append(torch.from_numpy(a).float()[:take * 400])
        loaded += take; log(f"  {chunk.stem}: {take}")
    if loaded >= max_frames: break

feats_all = torch.cat(all_feats, dim=1); f0_all = torch.cat(all_f0).unsqueeze(0)
f0f_all = torch.cat(all_f0f).unsqueeze(0); mel_all = torch.cat(all_mel, dim=2)
audio_all = torch.cat(all_audio)
del all_feats, all_f0, all_f0f, all_mel, all_audio; gc.collect()
log(f"Total: feats={feats_all.shape} f0={f0_all.shape} mel={mel_all.shape} audio={audio_all.shape}")

log("Loading RVC...")
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
    spk_embed_dim=109, gin_channels=256, sr=SR, is_half=False, phone_dim=768,
)
model.load_state_dict(ckpt["model"], strict=False); model.train(); del ckpt; gc.collect()

log(f"Training with KL divergence + waveform L1...")
opt = torch.optim.SGD(model.parameters(), lr=1e-4)
TF = SR // 400 * 2

for ep in range(1, EPOCHS + 1):
    t = feats_all.size(1)
    s = np.random.randint(0, max(1, t - TF))
    e = s + TF

    fb = feats_all[:, s:e, :]; f0b = f0_all[:, s:e]; ffb = f0f_all[:, s:e]; mb = mel_all[:, :1025, s:e]
    audio_start = s * 400; audio_end = e * 400
    audio_target = audio_all[audio_start:audio_end]

    lengths = torch.tensor([fb.size(1)]); ml = torch.tensor([mb.size(2)])

    opt.zero_grad()
    o, _, _, _, (z, z_p, m_p, logs_p, m_q, logs_q) = model(fb, lengths, f0b, ffb, mb, ml, torch.tensor([0]))

    # KL divergence loss: KL(q||p) where q=N(m_q, logs_q), p=N(m_p, logs_p)
    kl = logs_p - logs_q + 0.5 * ((m_q - m_p).pow(2) * (-2 * logs_p).exp() + (2 * (logs_q - logs_p)).exp() - 1)
    kl = kl.mean()

    # Waveform L1 loss
    min_samples = min(o.numel(), audio_target.numel())
    l1 = torch.nn.functional.l1_loss(o.flatten()[:min_samples],
                                      audio_target.flatten()[:min_samples])

    loss = l1 + 0.1 * kl
    loss.backward()
    opt.step()

    if ep == 1 or ep % 5 == 0:
        log(f"  Ep {ep}/{EPOCHS}: loss={loss.item():.6f}")
        torch.save({"model": model.state_dict()}, OUTPUT); gc.collect()

torch.save({"model": model.state_dict()}, OUTPUT)
log(f"DONE: {OUTPUT}")
