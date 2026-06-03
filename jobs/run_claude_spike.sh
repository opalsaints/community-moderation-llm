#!/usr/bin/env bash
# Claude Sonnet 4.6 spike: 2 subs x 2 templates x 2 thinking modes x n=100 = 800 requests
# Auth: OAuth (Max plan) via $CLAUDE_CODE_OAUTH_TOKEN
# Model locked: claude-sonnet-4-6. No fallback.
set -euo pipefail

module load 2025 Python/3.13.1-GCCcore-14.2.0
source $HOME/capstone_env_2025/bin/activate

export PATH=$HOME/.local/bin:$PATH
export CLAUDE_CODE_OAUTH_TOKEN=$(cat $HOME/.claude_oauth_token)

OUT_DIR=$HOME/data/results/claude_spike
mkdir -p $OUT_DIR

SCRIPT=$HOME/capstone/scripts/claude_eval.py

SUBS=(changemyview AmItheAsshole)
TEMPLATES=(slm_mod enriched)
THINKING=(off on)

for sub in "${SUBS[@]}"; do
  for template in "${TEMPLATES[@]}"; do
    for thinking in "${THINKING[@]}"; do
      out=$OUT_DIR/${sub}_${template}_${thinking}.json
      echo
      echo "============================================================"
      echo "  sub=$sub template=$template thinking=$thinking"
      echo "  out=$out"
      echo "============================================================"
      python $SCRIPT \
        --sub "$sub" \
        --template "$template" \
        --thinking "$thinking" \
        --n 100 \
        --out "$out"
    done
  done
done

echo
echo "DONE. Spike results in $OUT_DIR"
ls -la $OUT_DIR
