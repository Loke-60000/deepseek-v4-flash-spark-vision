import json
import torch
from safetensors import safe_open

palette = json.load(open("/p/palette.json"))
print("  palette size: %d   ids %d..%d" % (len(palette), min(palette), max(palette)))

with safe_open("/m/carried-001.safetensors", framework="pt") as f:
    tabs = {L: f.get_tensor("layers.%d.ffn.gate.tid2eid" % L) for L in (0, 1, 2)}

n_exp = 216
g = torch.Generator().manual_seed(0)

for L, t in tabs.items():
    pal = t[torch.tensor(palette, dtype=torch.long)]          # (64, 6)
    experts = pal.flatten()
    cov = int(experts.unique().numel())
    counts = torch.bincount(experts, minlength=n_exp).float()
    used = counts[counts > 0]
    # load imbalance: max/mean over experts actually hit
    imb = float(used.max() / used.mean())

    # baseline: 64 random token ids
    rnd = torch.randint(0, t.shape[0], (64,), generator=g)
    rex = t[rnd].flatten()
    rcov = int(rex.unique().numel())
    rc = torch.bincount(rex, minlength=n_exp).float()
    rused = rc[rc > 0]
    rimb = float(rused.max() / rused.mean())

    print("  layer%d: palette covers %3d/%d experts (imbalance %.2f)  |  random64 %3d/%d (%.2f)"
          % (L, cov, n_exp, imb, rcov, n_exp, rimb))

# how good could a chosen palette be? greedy search for max coverage
t = tabs[0]
best, seen = [], set()
cand = torch.randperm(t.shape[0], generator=g)[:20000]
for tid in cand.tolist():
    e = set(t[tid].tolist())
    if len(e - seen) >= 4:
        best.append(tid)
        seen |= e
    if len(best) == 64:
        break
print("  greedy-selected 64 ids cover %d/%d experts on layer0" % (len(seen), n_exp))
