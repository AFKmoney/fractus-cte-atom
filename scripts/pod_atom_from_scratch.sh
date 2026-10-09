#!/bin/bash
# Does not start a pod. Refuses unless ATOM_POD=1 is set on the machine
# that already has the GPU. x8 checkpoints are not an input.
set -euo pipefail
if [[ "${ATOM_POD:-}" != "1" ]]; then
  echo "refusing: set ATOM_POD=1 on the GPU box. this script does not open a pod." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
python scripts/train_atom_from_scratch.py \
  --scale 1b \
  --corpus data/atom_corpus.i16 \
  --steps "${STEPS:-100}" \
  --seq-len 32 \
  --ss-rate 0.2 \
  --out checkpoints/atom_scratch
