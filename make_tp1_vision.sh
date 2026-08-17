#!/usr/bin/env bash
# Build data/tp1-vision: symlinks to the backbone shards plus a config.json
# whose architectures field selects the vision wrapper. Costs a few kilobytes;
# the 98.8 GiB of weights are not copied.
set -euo pipefail
SRC=${1:-data/tp1}
DST=${2:-data/tp1-vision}
mkdir -p "$DST"
for f in "$SRC"/*; do
  b=$(basename "$f")
  [ "$b" = "config.json" ] && continue
  ln -sf "$(realpath "$f")" "$DST/$b"
done
python3 - "$SRC/config.json" "$DST/config.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
c = json.load(open(src))
c["architectures"] = ["DeepseekV4VisionForCausalLM"]
json.dump(c, open(dst, "w"), indent=2)
print("wrote", dst, "architectures =", c["architectures"])
PY
