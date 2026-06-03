#!/bin/bash
# Phase 2.1b main rerun orchestrator: per-sub Qwen 3 14B + LoRA + enriched
# fine-tunes with R4_stacked config on the 2026 rebuild data, plus dependent
# eval per sub.
#
# Config locked 2026-04-18 after 7 spike rounds on 3 test subs (relationships,
# changemyview, politics). Winning config: enriched template + 2 epochs +
# --completion-only-loss + 7 LoRA target modules (r=alpha=16, dropout=0.05).
# See memory/project_r4_stacked_config.md for full context.
#
# Spike-first supported via --spike <sub>. For the full 12-sub launch (with 3
# spike subs already done) use --skip relationships --skip changemyview
# --skip politics. Use --first <sub> to move a sub to the head of the queue
# so it lands in the first singleton lane (good for canary monitoring).
#
# Concurrency is capped at 2 GPU jobs at a time across the array via
# --dependency=singleton on two round-robin job names (p21b_a, p21b_b).
# Eval jobs use unique names so they don't share the singleton lock.
#
# Usage:
#   bash scripts/slurm_phase2_main.sh --spike AskHistorians         # spike one sub
#   bash scripts/slurm_phase2_main.sh --dry-run                     # print, don't submit
#   bash scripts/slurm_phase2_main.sh                               # launch all 15
#   bash scripts/slurm_phase2_main.sh \
#       --skip relationships --skip changemyview --skip politics \
#       --first AskHistorians --first antiai                        # launch remaining 12
#
# After completion, per-sub artifacts land in:
#   ~/data/results/finetuned_2026/r4_stacked_<sub>_adapter/
#   ~/data/results/finetuned_2026/r4_stacked_<sub>_metrics.json
#   ~/data/results/finetuned_2026/r4_stacked_<sub>_predictions.jsonl
#   ~/data/results/finetuned_2026/r4_stacked_<sub>_train_meta.json

set -e
cd ~/capstone
mkdir -p logs

SUB_LIST="${HOME}/data/dataset_2026/final_subs_2026.txt"
DATASET_ROOT="${HOME}/data/dataset_2026"
RULES_ROOT="${HOME}/data/rules_2026"
OUTPUT_DIR="${HOME}/data/results/finetuned_2026"
TRAIN_SCRIPT="scripts/slurm_v3_train.sh"
EVAL_SCRIPT="scripts/slurm_v3_eval.sh"
MODEL="Qwen/Qwen3-14B"
TEMPLATE="enriched"
PARTITION="gpu_a100"

DRY_RUN=false
SPIKE_SUB=""
SKIP_SUBS=()
FIRST_SUBS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --spike) SPIKE_SUB="$2"; shift 2 ;;
        --skip) SKIP_SUBS+=("$2"); shift 2 ;;
        --first) FIRST_SUBS+=("$2"); shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$SUB_LIST" ]]; then
    echo "ERROR: sub list not found: $SUB_LIST" >&2
    exit 1
fi

# Strip comments and blanks from the sub list
mapfile -t ALL_SUBS < <(grep -vE '^\s*(#|$)' "$SUB_LIST")

if [[ -n "$SPIKE_SUB" ]]; then
    SUBS=("$SPIKE_SUB")
    echo "=== SPIKE MODE: only $SPIKE_SUB ==="
else
    FILTERED=()
    for s in "${ALL_SUBS[@]}"; do
        skip=false
        for k in "${SKIP_SUBS[@]}"; do
            if [[ "$s" == "$k" ]]; then skip=true; break; fi
        done
        $skip || FILTERED+=("$s")
    done

    # Reorder: --first subs move to the head of the queue (preserves --first
    # argument order). Any --first sub not found in FILTERED prints a warning
    # and is silently dropped (could be a typo or already --skip'd).
    SUBS=()
    REMAINING=("${FILTERED[@]}")
    for f in "${FIRST_SUBS[@]}"; do
        found=false
        NEW_REMAINING=()
        for s in "${REMAINING[@]}"; do
            if [[ "$s" == "$f" && $found == false ]]; then
                SUBS+=("$s")
                found=true
            else
                NEW_REMAINING+=("$s")
            fi
        done
        REMAINING=("${NEW_REMAINING[@]}")
        if ! $found; then
            echo "WARN: --first $f not found in filtered list (typo or already --skip'd?)"
        fi
    done
    SUBS+=("${REMAINING[@]}")

    if [[ ${#FIRST_SUBS[@]} -gt 0 ]]; then
        echo "=== FULL FAN-OUT: ${#SUBS[@]} subs (first: ${FIRST_SUBS[*]}) ==="
    else
        echo "=== FULL FAN-OUT: ${#SUBS[@]} subs ==="
    fi
fi

mkdir -p "$OUTPUT_DIR"

LANE_NAMES=(p21b_a p21b_b)
i=0

submit_sub() {
    local SUB="$1"
    local LANE="${LANE_NAMES[$((i % 2))]}"
    local TAG="r4_stacked_${SUB}"
    local DDIR="${DATASET_ROOT}/${SUB}/enriched_v2"
    local RFILE="${RULES_ROOT}/${SUB}/rules.txt"

    if [[ ! -f "$DDIR/train.jsonl" ]]; then
        echo "[$SUB] SKIP: no train.jsonl at $DDIR"
        return
    fi
    if [[ ! -f "$RFILE" ]]; then
        echo "[$SUB] SKIP: no rules.txt at $RFILE"
        return
    fi

    local COMMON_ARGS=(
        --model "$MODEL"
        --dataset-dir "$DDIR"
        --output-dir "$OUTPUT_DIR"
        --subreddit "$SUB"
        --rules-file "$RFILE"
        --template "$TEMPLATE"
        --run-tag "$TAG"
        --epochs 2
        --completion-only-loss
        --target-modules all
    )

    if $DRY_RUN; then
        echo "[$SUB] LANE=$LANE"
        echo "  TRAIN: sbatch --partition=$PARTITION --job-name=$LANE --dependency=singleton $TRAIN_SCRIPT ${COMMON_ARGS[*]}"
        echo "  EVAL:  sbatch --partition=$PARTITION --job-name=v3_${TAG}_eval --dependency=afterok:\$TRAIN $EVAL_SCRIPT ${COMMON_ARGS[*]} --skip-train"
        echo ""
        i=$((i + 1))
        return
    fi

    echo "--- Submitting $SUB (lane $LANE) ---"

    TRAIN_JOB=$(sbatch --parsable --partition="$PARTITION" \
        --job-name="$LANE" \
        --dependency=singleton \
        "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}")
    echo "  Train job: $TRAIN_JOB"

    EVAL_JOB=$(sbatch --parsable --partition="$PARTITION" \
        --dependency="afterok:$TRAIN_JOB" \
        --job-name="v3_${TAG}_eval" \
        "$EVAL_SCRIPT" "${COMMON_ARGS[@]}" --skip-train)
    echo "  Eval job:  $EVAL_JOB (depends on $TRAIN_JOB)"
    echo ""

    i=$((i + 1))
}

for SUB in "${SUBS[@]}"; do
    submit_sub "$SUB"
done

echo "=== Submitted ${#SUBS[@]} sub(s). squeue -u $USER to monitor. ==="
