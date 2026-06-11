import sys, gc, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
torch.set_num_threads(2)

OUTPUT = "models/trained/voz_jimmy.pth"
FEAT_DIR = Path("models/trained/features")
EPOCHS = 30
SR = 40000

def log(msg): print(msg, flush=True)

log("Loading ALL features...")
chunks = sorted(FEAT_DIR.glob("feats_*.npy"))
all_feats, all_f0, all_f0f, all_mel = [], [], [], []
for chunk in chunks:
    idx = chunk.stem.split("_")[1]
    all_feats.append(torch.from_numpy(np.load(FEAT_DIR / f"feats_{idx}.npy")).float())
    all_f0.append(torch.from_numpy(np.load(FEAT_DIR / f"f0_{idx}.npy")).long())
    all_f0f.append(torch.from_numpy(np.load(FEAT_DIR / f"f0f_{idx}.npy")))
    m = np.load(FEAT_DIR / f"mel_{idx}.npy")
    all_mel.append(torch.from_numpy(m).unsqueeze(0).float())
    log(f"  {chunk.stem}: feats={all_feats[-1].shape} mel={m.shape}")

feats_all = torch.cat(all_feats, dim=1)
f0_all = torch.cat(all_f0).unsqueeze(0)
f0f_all = torch.cat(all_f0f).unsqueeze(0)
mel_all = torch.cat(all_mel, dim=2)
gc.collect()
log(f"Total: feats={feats_all.shape} f0={f0_all.shape} mel={mel_all.shape}")

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
    spk_embed_dim=109, gin_channels=256, sr=SR,
    is_half=False, phone_dim=768,
)
model.load_state_dict(ckpt["model"], strict=False)
model.train()
del ckpt; gc.collect()

log("Training with MEL LOSS (proper voice cloning)...")
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
TRAIN_FRAMES = SR // 400 * 2

for ep in range(1, EPOCHS + 1):
    t_total = feats_all.size(1)
    s = np.random.randint(0, max(1, t_total - TRAIN_FRAMES))
    e = s + TRAIN_FRAMES

    feats_b = feats_all[:, s:e, :]
    f0_b = f0_all[:, s:e]
    f0f_b = f0f_all[:, s:e]
    mel_b = mel_all[:, :1025, s:e]

    lengths = torch.tensor([feats_b.size(1)])
    mel_len = torch.tensor([mel_b.size(2)])
    spk = torch.tensor([0])

    opt.zero_grad()

    o, ids_slice, x_mask, y_mask, _ = model(
        feats_b, lengths, f0_b, f0f_b, mel_b, mel_len, spk
    )

    import torchaudio
    o_spec = torch.stft(o.squeeze(0), n_fft=2048, hop_length=400,
                        return_complex=True).abs().unsqueeze(0)

    tgt_mel = mel_b[:, :, ids_slice.squeeze(0)]

    min_f = min(o_spec.size(1), tgt_mel.size(1))
    min_t = min(o_spec.size(2), tgt_mel.size(2))
    o_spec = o_spec[:, :min_f, :min_t]
    tgt_mel = tgt_mel[:, :min_f, :min_t]

    loss = F.l1_loss(o_spec, tgt_mel)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if ep == 1 or ep % 5 == 0:
        log(f"  Ep {ep}/{EPOCHS}: loss={loss.item():.6f}")
        torch.save({"model": model.state_dict()}, OUTPUT)
        gc.collect()

torch.save({"model": model.state_dict()}, OUTPUT)
log(f"DONE: {OUTPUT}")
