#!/bin/bash
# Run Gemini eval over the 2026 main-experiment subreddit set.
#
# Usage:
#   bash scripts/run_gemini_2026.sh                          # all 15 subs
#   bash scripts/run_gemini_2026.sh --subs CMV,politics      # specific subs
#   bash scripts/run_gemini_2026.sh --skip antiai            # skip some
#   bash scripts/run_gemini_2026.sh --first AskHistorians    # put first
#
# Default: gemini-2.5-flash, with-rules only, no-thinking, n=2000.
# Idempotent: skips a sub if metrics file already exists.

set -e
cd ~/capstone

source ~/capstone_env_2025/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.gcloud_adc.json
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_GENAI_USE_VERTEXAI=True

MODEL="gemini-2.5-flash"
DATA_ROOT="$HOME/data"
OUT_DIR="$HOME/data/results/gemini_2026"
RPM=500
MAX_SAMPLES=2000
SKIP=()
FIRST=()
OVERRIDE_SUBS=()

DEFAULT_SUBS=(AskHistorians askscience science legaladvice personalfinance relationships AmItheAsshole changemyview explainlikeimfive Games news TwoXChromosomes politics antiai aiwars)

while [[ $# -gt 0 ]]; do
  case $1 in
    --subs) IFS="," read -ra OVERRIDE_SUBS <<< "$2"; shift 2 ;;
    --skip) SKIP+=("$2"); shift 2 ;;
    --first) FIRST+=("$2"); shift 2 ;;
    --max-test-samples) MAX_SAMPLES=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --rpm) RPM=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [ ${#OVERRIDE_SUBS[@]} -gt 0 ]; then
  SUBS=("${OVERRIDE_SUBS[@]}")
else
  SUBS=("${DEFAULT_SUBS[@]}")
fi

mkdir -p "$OUT_DIR"

# Apply --first reordering
if [ ${#FIRST[@]} -gt 0 ]; then
  REORDERED=("${FIRST[@]}")
  for s in "${SUBS[@]}"; do
    skip_this=0
    for f in "${FIRST[@]}"; do [ "$f" = "$s" ] && skip_this=1; done
    [ "$skip_this" = "0" ] && REORDERED+=("$s")
  done
  SUBS=("${REORDERED[@]}")
fi

echo "============================================="
echo "  Gemini 2026 eval - $(date)"
echo "  Model:    $MODEL"
echo "  Subs:     ${#SUBS[@]} (${SUBS[*]})"
echo "  Skip:     ${SKIP[*]:-none}"
echo "  Max n:    $MAX_SAMPLES"
echo "  Output:   $OUT_DIR"
echo "============================================="

for sub in "${SUBS[@]}"; do
  for s in "${SKIP[@]}"; do
    if [ "$s" = "$sub" ]; then
      echo "  SKIP: $sub (--skip)"
      continue 2
    fi
  done

  DS_DIR="$DATA_ROOT/dataset_2026/$sub/random_split"
  RULES_FILE="$DATA_ROOT/rules/$sub/rules.txt"
  if [ ! -f "$DS_DIR/test.jsonl" ]; then echo "  SKIP: $sub (no test.jsonl)"; continue; fi
  if [ ! -f "$RULES_FILE" ]; then echo "  SKIP: $sub (no rules.txt)"; continue; fi

  EXPECTED="$OUT_DIR/${MODEL}_${sub}_with_rules.json"
  if [ -f "$EXPECTED" ]; then
    echo "  SKIP: $sub (already done)"
    continue
  fi

  echo ""
  echo "=== r/$sub === $(date +%H:%M:%S)"
  python3 -u scripts/gemini_eval.py \
    --model "$MODEL" \
    --subreddit "$sub" \
    --dataset-dir "$DS_DIR" \
    --output-dir "$OUT_DIR" \
    --with-rules --rules-file "$RULES_FILE" \
    --no-thinking \
    --rpm $RPM \
    --max-test-samples $MAX_SAMPLES
done

echo ""
echo "Batch complete: $(date)"
