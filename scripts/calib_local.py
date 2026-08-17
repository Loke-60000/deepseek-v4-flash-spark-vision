"""Calibrate the projector's output scale using local images. No network.

Same reasoning as the COCO version: DSV4 applies no scaling in
embed_input_ids, so text tokens enter the residual stream at L2 norm ~7.29
while the step-4800 projector emits 187-328. Scaling W2/b2 of the final
Linear scales the output exactly; the separator already sits at text scale
and is left alone.
"""
import glob, sys, torch, torch.nn as nn
from safetensors.torch import load_file
from PIL import Image

sys.path.insert(0, "/home/lokman/ai/vision-train/vision-plugin/src")
from dsv4_vision_vllm.deepencoderv2 import (build_sam_vit_b,
                                            build_qwen2_decoder_as_encoder)

A, DEV = "/home/lokman/ai/vision-train", "cuda"
IMAGE_SIZE, ENC, HID = 1024, 896, 4096


def preprocess(img):
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    a = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
    a = a.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float().div_(255.0)
    return a.sub_(0.5).div_(0.5)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(ENC, HID), nn.GELU(), nn.Linear(HID, HID))
        self.view_seperator = nn.Parameter(torch.zeros(HID))

    def forward(self, x):
        return self.proj(x)


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam_model = build_sam_vit_b()
        self.qwen2_model = build_qwen2_decoder_as_encoder()

    def forward(self, px):
        return self.qwen2_model(self.sam_model(px))


emb = load_file(f"{A}/vision-train/embed_u.safetensors")["embed.weight"].float()
en = emb.norm(dim=-1)
target = en.mean().item()
print("  text-token norm: mean %.4f  median %.4f" % (target, en.median()))

tower = Tower()
sd = load_file(f"{A}/tower/deepencoder_v2_tower.safetensors")
tower.load_state_dict({k[6:]: v for k, v in sd.items()
                       if k.startswith(("model.sam_model.", "model.qwen2_model."))},
                      strict=False)
tower = tower.to(DEV, torch.bfloat16).eval()

ck = torch.load(f"{A}/adapter/merged-004800-5af0c5.pt", map_location="cpu",
                weights_only=False)
ad = Adapter()
ad.load_state_dict(ck.get("adapter", ck), strict=True)
ad = ad.to(DEV, torch.bfloat16).eval()

files = []
for pat in ("/home/lokman/Pictures/**/*.png", "/home/lokman/Pictures/**/*.jpg",
            "/home/lokman/Pictures/**/*.jpeg", "/tmp/gyaru.jpg", "/tmp/shapes.png"):
    files += glob.glob(pat, recursive=True)
files = sorted(set(files))[:48]
print("  calibrating on %d local images" % len(files))

norms = []
with torch.no_grad():
    for f in files:
        try:
            px = preprocess(Image.open(f)).unsqueeze(0).to(DEV, torch.bfloat16)
            p = ad(tower(px)).float().reshape(-1, HID)
            norms.append(p.norm(dim=-1).mean().item())
        except Exception:
            continue

cur = sum(norms) / len(norms)
s = target / cur
print("  n=%d  mean proj norm %.3f  (min %.1f max %.1f)  ->  scale %.6f"
      % (len(norms), cur, min(norms), max(norms), s))

ad32 = Adapter()
ad32.load_state_dict(ck.get("adapter", ck), strict=True)
with torch.no_grad():
    ad32.proj[2].weight.mul_(s)
    ad32.proj[2].bias.mul_(s)

out = dict(ck)
out["adapter"] = dict(ad32.state_dict())
out["scale_calibration"] = {"source_mean_norm": cur, "target_mean_norm": target,
                            "scale": s, "n_images": len(norms),
                            "basis": "local image set, mean L2 over projected tokens"}
torch.save(out, f"{A}/adapter/calibrated.pt")
print("  wrote %s/adapter/calibrated.pt  (scale %.6f)" % (A, s))
