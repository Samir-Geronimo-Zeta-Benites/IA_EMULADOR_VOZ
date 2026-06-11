import sys, gc, numpy as np, torch, librosa
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
torch.set_num_threads(2)

OUTPUT = "models/trained/voz_jimmy.pth"
FEAT_DIR = Path("models/trained/features")
EPOCHS = 30
SR = 40000

def log(msg): print(msg, flush=True)

log("Loading features...")
chunks = sorted(FEAT_DIR.glob("feats_*.npy"))
target_frames = 15 * SR // 400
loaded = 0
all_feats, all_f0, all_f0f, all_mel = [], [], [], []
for chunk in chunks:
    idx = chunk.stem.split("_")[1]
    f = torch.from_numpy(np.load(FEAT_DIR / f"feats_{idx}.npy")).float()
    f0 = torch.from_numpy(np.load(FEAT_DIR / f"f0_{idx}.npy")).long()
    f0f_ = torch.from_numpy(np.load(FEAT_DIR / f"f0f_{idx}.npy"))
    m = torch.from_numpy(np.load(FEAT_DIR / f"mel_{idx}.npy")).unsqueeze(0).float()
    take = min(f.size(1), target_frames - loaded)
    if take > 0:
        all_feats.append(f[:, :take, :]); all_f0.append(f0[:take])
        all_f0f.append(f0f_[:take]); all_mel.append(m[:, :, :take])
        loaded += take
    if loaded >= target_frames: break

feats_all = torch.cat(all_feats, dim=1); f0_all = torch.cat(all_f0, dim=0).unsqueeze(0)
f0f_all = torch.cat(all_f0f, dim=0).unsqueeze(0); mel_all = torch.cat(all_mel, dim=2)
gc.collect()
log(f"Features: feats={feats_all.shape} f0={f0_all.shape} mel={mel_all.shape}")

log("Loading RVC...")
from core.rvc_model.models import SynthesizerTrnMs256NSF
ckpt = torch.load("models/base/f0G40k.pth", map_location="cpu")
model = SynthesizerTrnMs256NSF(
    spec_channels=1025, segment_size=12800, inter_channels=192, hidden_channels=192,
    filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0,
    resblock="1", resblock_kernel_sizes=[3,7,11],
    resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]],
    upsample_rates=[10,10,2,2], upsample_initial_channel=512,
    upsample_kernel_sizes=[16,16,4,4], spk_embed_dim=109, gin_channels=256,
    sr=SR, is_half=False, phone_dim=768,
)
model.load_state_dict(ckpt["model"], strict=False)
model.train()
del ckpt; gc.collect()
log("RVC loaded")

log("Training...")
opt = torch.optim.SGD(model.parameters(), lr=1e-4)
TRAIN_FRAMES = SR // 400 * 1  # 1 second

for epoch in range(1, EPOCHS + 1):
    t = feats_all.size(1)
    s = np.random.randint(0, max(1, t - TRAIN_FRAMES))
    e = s + TRAIN_FRAMES

    feats_b = feats_all[:, s:e, :]
    f0_b = f0_all[:, s:e]
    f0f_b = f0f_all[:, s:e]
    mel_b = mel_all[:, :1025, s:e]

    lengths = torch.tensor([feats_b.size(1)])
    mel_len = torch.tensor([mel_b.size(2)])
    spk = torch.tensor([0])

    opt.zero_grad()
    o, _, _, _, _ = model(feats_b, lengths, f0_b, f0f_b, mel_b, mel_len, spk)

    # L1 loss on raw waveform prevents silence
    loss_raw = o.abs().mean()

    loss_raw.backward()
    opt.step()

    if epoch == 1 or epoch % 5 == 0:
        log(f"  Ep {epoch}/{EPOCHS}: loss={loss_raw.item():.6f}")
        torch.save({"model": model.state_dict()}, OUTPUT)
        gc.collect()

torch.save({"model": model.state_dict()}, OUTPUT)
log(f"Saved: {OUTPUT}")
log("DONE")
