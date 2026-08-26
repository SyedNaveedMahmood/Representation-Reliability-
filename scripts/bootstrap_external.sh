#!/usr/bin/env bash
set -euo pipefail
GROUP="${1:-core}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/repos"
mkdir -p "$DEST"

clone_if_missing () {
  local name="$1"; local url="$2"
  if [ -d "$DEST/$name/.git" ]; then
    echo "[exists] $name"
  else
    echo "[clone] $name"
    git clone --depth 1 "$url" "$DEST/$name"
  fi
}

clone_core () {
  clone_if_missing nnsight https://github.com/ndif-team/nnsight.git
  clone_if_missing pyvene https://github.com/frankaging/pyvene.git
  clone_if_missing axbench https://github.com/stanfordnlp/axbench.git
  clone_if_missing internal_probing https://github.com/zazamrykh/internal_probing.git
  clone_if_missing Reasoning-Flow https://github.com/MasterZhou1/Reasoning-Flow.git
  clone_if_missing multi-component-causal-tracing https://github.com/ZiruiYan/multi-component-causal-tracing.git
  clone_if_missing activation-steering https://github.com/IBM/activation-steering.git
}

clone_optional () {
  clone_if_missing SAELens https://github.com/decoderesearch/SAELens.git
  clone_if_missing sae-feature-consistency https://github.com/xiangchensong/sae-feature-consistency.git
  clone_if_missing ICR_Probe https://github.com/XavierZhang2002/ICR_Probe.git
  clone_if_missing interpretability https://github.com/PAIR-code/interpretability.git
  clone_if_missing honest_llama https://github.com/likenneth/honest_llama.git
  clone_if_missing pyreft https://github.com/stanfordnlp/pyreft.git
  clone_if_missing StateBridge https://github.com/YanwenPneg/StateBridge.git
  clone_if_missing selective-steering https://github.com/knoveleng/steering.git
}

case "$GROUP" in
  core) clone_core ;;
  optional) clone_optional ;;
  all) clone_core; clone_optional ;;
  *) echo "Usage: $0 [core|optional|all]"; exit 2 ;;
esac

echo "Cloned only. Do NOT install all repos into one environment."
echo "Read docs/EXTERNAL_MODULES.md."
