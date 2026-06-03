#!/bin/bash
# SLM-Mod 2017 replication launcher for thesis Section 4.5 (Path B-2 locked 2026-05-04).
#
# Trains Qwen 3 14B on Agam Goyal's 2017 SLM-Mod data (changemyview, politics)
# under two recipes per sub:
#   - R4-style:   alpha=16, dropout=0.05, cosine, warmup=50, label_smoothing=0.1,
#                 7 modules, 1 epoch, --completion-only-loss
#   - Paper-exact: alpha=32, dropout=0, linear, warmup=5, label_smoothing=0,
#                 7 modules, 1 epoch, no --completion-only-loss
#
# Both recipes use SLM-Mod template (rules + parent_body + comment body), since
# Agam's data has no metadata for the enriched template. 7 modules (q/k/v/o +
# gate/up/down) deviates from SLM-Mod paper's 4 (attention-only); deviation is
# documented in Section 4.5 methodology.
#
# Concurrency capped at 2 GPU jobs via --dependency=singleton on round-robin
# lane names (slm17_a, slm17_b). Eval jobs use unique names.
#
# Usage:
#   bash scripts/slurm_slm_mod_2017.sh --smoke                       # CMV R4-style only (Phase 3)
#   bash scripts/slurm_slm_mod_2017.sh --dry-run                     # print, don't submit
#   bash scripts/slurm_slm_mod_2017.sh                               # all 4 runs (Phase 4)
#   bash scripts/slurm_slm_mod_2017.sh --subs changemyview --recipes paper
#
# Model-axis extension (2026-05-13, Stage 0.2):
#   --model <HF-ID>          HuggingFace model id (default Qwen/Qwen3-14B).
#                            Tag suffix derived automatically (qwen/mistral/llama/gemma).
#   --first <sub>            Push this sub to the head of SUBS so it lands in lane A.
#                            Repeatable for multiple first-landers (one per lane).
#   --skip <sub>/<recipe>    Skip a specific cell. Repeatable.
#
# Unsloth pivot (2026-05-13, Stage U-0.6):
#   --framework {v3,unsloth} Which training script to dispatch (default v3).
#                            'unsloth' uses scripts/slurm_unsloth_train.sh +
#                            scripts/finetune_unsloth.py and appends '_unsloth'
#                            to the model-short tag suffix. Eval always uses
#                            scripts/slurm_v3_eval.sh (vLLM is framework-
#                            independent; PEFT adapters load on the vanilla
#                            base).
#
# Example (2-cell Mistral spike):
#   bash scripts/slurm_slm_mod_2017.sh \
#       --model mistralai/Mistral-Nemo-Instruct-2407 \
#       --subs changemyview,science --recipes r4,paper \
#       --skip changemyview/paper --skip science/r4
#
# Example (Unsloth Mistral spike, 2026-05-13):
#   bash scripts/slurm_slm_mod_2017.sh \
#       --framework unsloth \
#       --model mistralai/Mistral-Nemo-Instruct-2407 \
#       --subs changemyview,science --recipes r4,paper \
#       --skip changemyview/paper --skip science/r4
#
# Artifacts (tag suffix tracks --model):
#   ~/data/results/finetuned_2017/slm_mod_<recipe>_<sub>_2017_<model-short>_adapter/
#   ~/data/results/finetuned_2017/slm_mod_<recipe>_<sub>_2017_<model-short>_metrics.json
#   ~/data/results/finetuned_2017/slm_mod_<recipe>_<sub>_2017_<model-short>_predictions.jsonl
#   ~/data/results/finetuned_2017/slm_mod_<recipe>_<sub>_2017_<model-short>_train_meta.json
#
# Existing Phase-4 (2026-05-04) cube tags `slm_mod_<recipe>_<sub>_2017` (no model suffix)
# remain untouched on Snellius -- the new suffixed scheme applies to all new submissions.

set -e
cd ~/capstone
mkdir -p logs

DATASET_ROOT="${HOME}/data/dataset_2017"
RULES_FILE="${HOME}/data/agam_2017/subreddit_rules_15.json"
OUTPUT_DIR="${HOME}/data/results/finetuned_2017"
V3_TRAIN_SCRIPT="scripts/slurm_v3_train.sh"
UNSLOTH_TRAIN_SCRIPT="scripts/slurm_unsloth_train.sh"
EVAL_SCRIPT="scripts/slurm_v3_eval.sh"
MODEL="Qwen/Qwen3-14B"
TEMPLATE="slm-mod"
PARTITION="gpu_a100"

DRY_RUN=false
SMOKE=false
FRAMEWORK="v3"
SUBS=(changemyview politics)
RECIPES=(r4 paper)
FIRST_SUBS=()
SKIP_PAIRS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --smoke) SMOKE=true; shift ;;
        --subs) IFS=',' read -ra SUBS <<< "$2"; shift 2 ;;
        --recipes) IFS=',' read -ra RECIPES <<< "$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --framework) FRAMEWORK="$2"; shift 2 ;;
        --first) FIRST_SUBS+=("$2"); shift 2 ;;
        --skip) SKIP_PAIRS+=("$2"); shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate FRAMEWORK and pick the train script + tag suffix.
case "$FRAMEWORK" in
    v3)
        TRAIN_SCRIPT="$V3_TRAIN_SCRIPT"
        FRAMEWORK_SUFFIX=""
        ;;
    unsloth)
        TRAIN_SCRIPT="$UNSLOTH_TRAIN_SCRIPT"
        FRAMEWORK_SUFFIX="_unsloth"
        ;;
    *)
        echo "ERROR: --framework must be 'v3' or 'unsloth' (got '$FRAMEWORK')" >&2
        exit 1
        ;;
esac

# Derive a short model tag (qwen / mistral / llama / gemma / fallback) and
# append the framework suffix (empty for v3, "_unsloth" for unsloth).
case "$MODEL" in
    *Qwen*)    MODEL_SHORT="qwen" ;;
    *Mistral*|*mistral*) MODEL_SHORT="mistral" ;;
    *Llama*|*llama*)     MODEL_SHORT="llama" ;;
    *Gemma*|*gemma*)     MODEL_SHORT="gemma" ;;
    *) MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | cut -d'-' -f1) ;;
esac
MODEL_SHORT="${MODEL_SHORT}${FRAMEWORK_SUFFIX}"

# Reorder SUBS so any --first entries land at the head (in the order given).
if [[ ${#FIRST_SUBS[@]} -gt 0 ]]; then
    REORDERED=()
    for f in "${FIRST_SUBS[@]}"; do
        for s in "${SUBS[@]}"; do
            if [[ "$s" == "$f" ]]; then
                REORDERED+=("$s")
                break
            fi
        done
    done
    for s in "${SUBS[@]}"; do
        keep=true
        for f in "${FIRST_SUBS[@]}"; do
            if [[ "$s" == "$f" ]]; then
                keep=false
                break
            fi
        done
        if $keep; then
            REORDERED+=("$s")
        fi
    done
    SUBS=("${REORDERED[@]}")
fi

if $SMOKE; then
    SUBS=(changemyview)
    RECIPES=(r4)
    echo "=== SMOKE MODE: changemyview R4-style only (Phase 3 gate) ==="
fi

mkdir -p "$OUTPUT_DIR"

LANE_NAMES=("slm17_${MODEL_SHORT}_a" "slm17_${MODEL_SHORT}_b")
i=0

# Per-sub rules text file (extracted from the JSON for finetune_v3.py)
ensure_rules_txt() {
    local SUB="$1"
    local RTXT="${HOME}/data/agam_2017/rules_txt/${SUB}.txt"
    mkdir -p "$(dirname "$RTXT")"
    if [[ ! -f "$RTXT" ]]; then
        python3 -c "
import json, sys
with open('$RULES_FILE') as f:
    r = json.load(f)
sub = '$SUB'
if sub not in r:
    sys.exit(f'rules JSON missing entry for {sub}')
print('\n'.join(f'- {rule}' for rule in r[sub]))
" > "$RTXT"
    fi
    echo "$RTXT"
}

submit_run() {
    local SUB="$1"
    local RECIPE="$2"

    # Honor --skip <sub>/<recipe> entries.
    for pair in "${SKIP_PAIRS[@]}"; do
        if [[ "$pair" == "${SUB}/${RECIPE}" ]]; then
            echo "[$SUB/$RECIPE] SKIP: matched --skip $pair"
            return
        fi
    done

    local LANE="${LANE_NAMES[$((i % 2))]}"
    local TAG="slm_mod_${RECIPE}_${SUB}_2017_${MODEL_SHORT}"
    local DDIR="${DATASET_ROOT}/${SUB}/slm_mod"

    if [[ ! -f "$DDIR/train.jsonl" ]]; then
        echo "[$SUB/$RECIPE] SKIP: no train.jsonl at $DDIR"
        return
    fi

    local RTXT
    RTXT=$(ensure_rules_txt "$SUB")

    # Common args used by both recipes.
    local COMMON_ARGS=(
        --model "$MODEL"
        --dataset-dir "$DDIR"
        --output-dir "$OUTPUT_DIR"
        --subreddit "$SUB"
        --rules-file "$RTXT"
        --template "$TEMPLATE"
        --run-tag "$TAG"
        --target-modules all     # 7 modules: q/k/v/o + gate/up/down
        --epochs 1
    )

    # Recipe-specific args.
    local RECIPE_ARGS=()
    case "$RECIPE" in
        r4)
            RECIPE_ARGS=(
                --lora-alpha 16
                --lora-dropout 0.05
                --lr-scheduler cosine
                --warmup-steps 50
                --label-smoothing 0.1
                --completion-only-loss
            )
            ;;
        paper)
            RECIPE_ARGS=(
                --lora-alpha 32
                --lora-dropout 0
                --lr-scheduler linear
                --warmup-steps 5
                --label-smoothing 0
                # NO --completion-only-loss: matches SLM-Mod paper objective
            )
            ;;
        *)
            echo "Unknown recipe: $RECIPE" >&2
            exit 1
            ;;
    esac

    if $DRY_RUN; then
        echo "[$SUB/$RECIPE] LANE=$LANE TAG=$TAG"
        echo "  TRAIN: sbatch --partition=$PARTITION --job-name=$LANE --dependency=singleton $TRAIN_SCRIPT ${COMMON_ARGS[*]} ${RECIPE_ARGS[*]}"
        echo "  EVAL:  sbatch --partition=$PARTITION --job-name=v3_${TAG}_eval --dependency=afterok:\$TRAIN $EVAL_SCRIPT ${COMMON_ARGS[*]} ${RECIPE_ARGS[*]} --skip-train"
        echo ""
        i=$((i + 1))
        return
    fi

    echo "--- Submitting $SUB / $RECIPE (lane $LANE) ---"

    TRAIN_JOB=$(sbatch --parsable --partition="$PARTITION" \
        --job-name="$LANE" \
        --dependency=singleton \
        "$TRAIN_SCRIPT" "${COMMON_ARGS[@]}" "${RECIPE_ARGS[@]}")
    echo "  Train job: $TRAIN_JOB"

    EVAL_JOB=$(sbatch --parsable --partition="$PARTITION" \
        --dependency="afterok:$TRAIN_JOB" \
        --job-name="v3_${TAG}_eval" \
        "$EVAL_SCRIPT" "${COMMON_ARGS[@]}" "${RECIPE_ARGS[@]}" --skip-train)
    echo "  Eval job:  $EVAL_JOB (depends on $TRAIN_JOB)"
    echo ""

    i=$((i + 1))
}

# Order matters: CMV first so it lands in the first lane (smoke / canary slot).
for SUB in "${SUBS[@]}"; do
    for RECIPE in "${RECIPES[@]}"; do
        submit_run "$SUB" "$RECIPE"
    done
done

echo "=== Submitted $i run(s). squeue -u $USER to monitor. ==="
