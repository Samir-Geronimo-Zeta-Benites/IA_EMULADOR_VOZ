import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from core.rvc_model.models import SynthesizerTrnMs256NSF

print("Loading checkpoint...")
ckpt = torch.load("models/base/f0G40k.pth", map_location="cpu")
sd = ckpt["model"]

print("Creating model...")
model = SynthesizerTrnMs256NSF(
    spec_channels=1025,
    segment_size=12800,
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0,
    resblock="1",
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_rates=[10, 10, 2, 2],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 4, 4],
    spk_embed_dim=109,
    gin_channels=256,
    sr=40000,
    is_half=False,
    phone_dim=768,
)

print("Loading weights...")
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if len(missing) < 30:
    print("MODELO RVC CARGADO!")
    if missing:
        print(f"  Missing: {missing}")
    if unexpected:
        print(f"  Unexpected: {unexpected[:5]}...")
else:
    for m in missing[:10]:
        print(f"  missing: {m}")
