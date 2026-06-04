# Reproduction guide

This walks through the full pipeline, from raw Arctic Shift data to the numbers in `results/`. Training and vLLM evaluation need an A100-class GPU; everything else runs on a laptop. Subreddit names below use `changemyview` as the running example; swap in any of the 15 from `configs/subreddits.json`.

## 1. Environment

Local (data preparation, the API baselines, analysis):

```bash
pip install -r requirements.txt
gcloud auth application-default login     # Gemini, via Vertex AI
export ANTHROPIC_API_KEY=...              # Claude baseline
```

GPU node (Snellius shown; adjust modules and partitions for other clusters):

```bash
module load 2025 Python/3.13.1-GCCcore-14.2.0
python -m venv ~/capstone_env && source ~/capstone_env/bin/activate
pip install -r requirements-snellius.txt
```

## 2. Data

The dataset is built from the [Arctic Shift](https://arctic-shift.photon-reddit.com/) archive, window 2025-10-01 to 2026-04-01. A comment is labelled `removed` only when its text was captured and its metadata marks a moderator removal (`removal_type == "removed"`); user self-deletions and text-less `[removed]`/`[deleted]` stubs are dropped. The test split is balanced 50/50 and drawn with seed 42, so the same comment IDs land in the test set on every rebuild.

```bash
python src/data/download_subreddit.py changemyview --after 2025-10-01 --before 2026-04-01 --output-dir data/raw/
python src/data/build_parent_index.py --subs changemyview --input-dir data/raw --output-dir data/extracted
python src/data/build_dataset.py --subs changemyview --input-dir data/extracted --output-dir data/dataset
python src/data/collect_rules.py --subs changemyview --output-dir data/rules

# enriched_v2 sidecars (post titles, author features), then the enriched split
python src/data/fetch_post_titles.py --dataset-dir data/dataset/changemyview --output data/dataset/changemyview/post_titles.json
python src/data/compute_author_features.py --input data/dataset/changemyview --output data/dataset/changemyview/author_features.json
python src/data/build_enriched_dataset.py --dataset-dir data/dataset/changemyview \
  --post-titles data/dataset/changemyview/post_titles.json \
  --author-features data/dataset/changemyview/author_features.json \
  --rules-file data/rules/changemyview/rules.txt \
  --output-dir data/dataset/changemyview/enriched_v2
```

`build_enriched_dataset.py` (with `enrich_v2.py`, `fetch_post_titles.py`, `fetch_account_ages.py`, `compute_author_features.py`) produces the `enriched_v2` split: the comment, its parent, the subreddit rules, and a set of lightweight post-level and author-level features (post title, thread position, account age, and so on).

To rebuild the exact test sets used here without re-downloading entire subreddits, start from `data/comment_ids/<subreddit>_test_ids.txt` and fetch those IDs through Arctic Shift. Those ID lists are the seed-42 balanced split, so rebuilding from them reproduces the split used for every number in `results/`.

## 3. Per-community fine-tuning

Training and evaluation are two separate jobs because bitsandbytes and vLLM cannot share GPU memory in one process. Train first, then load the saved adapter for evaluation with `--skip-train`.

```bash
# train
python src/train/finetune_v3.py --model Qwen/Qwen3-14B \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/ \
  --subreddit changemyview --rules-file data/rules/changemyview/rules.txt \
  --template enriched --run-tag r4_stacked_changemyview \
  --epochs 2 --completion-only-loss --target-modules all

# evaluate (vLLM loads the adapter saved above)
python src/train/finetune_v3.py --model Qwen/Qwen3-14B \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/ \
  --subreddit changemyview --rules-file data/rules/changemyview/rules.txt \
  --template enriched --run-tag r4_stacked_changemyview \
  --epochs 2 --completion-only-loss --target-modules all --skip-train
```

The full 15-subreddit fan-out is `jobs/slurm_phase2_main.sh` (two GPU lanes, train then dependent eval per subreddit). Hyperparameters are fixed in `configs/training_config.json`.

The trained adapters are on Hugging Face under [opalitestudios](https://huggingface.co/opalitestudios), named `qwen3-14b-reddit-moderation-<subreddit>` and `qwen3-14b-reddit-moderation-pooled`; load one on top of the base model with PEFT rather than retraining:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B")
model = PeftModel.from_pretrained(base, "opalitestudios/qwen3-14b-reddit-moderation-changemyview")
```

Once the adapter repositories carry version tags, you can pin an exact adapter revision by passing `revision=` to `from_pretrained` (for example `revision="v1.0.0"`), so a rebuild loads the same weights regardless of later updates.

## 4. Pooled adapter

One adapter trained on the union of all 15 training sets, with the subreddit name included in the prompt. It matches the per-community adapters (macro kappa 0.581 vs 0.573). Train it with the same recipe as the per-community adapters but the `pooled` template, which reads each example's own subreddit and rules from the record:

```bash
python src/train/finetune_v3.py --model Qwen/Qwen3-14B \
  --dataset-dir data/dataset/pooled/enriched_v2 --output-dir results/ \
  --subreddit pooled --rules-file data/rules/changemyview/rules.txt \
  --template pooled --run-tag pooled_all \
  --epochs 2 --completion-only-loss --target-modules all
```

The dataset directory is the union of the 15 `enriched_v2` train sets, each record carrying its own `subreddit` and `rules_text`. `--rules-file` is a required fallback only; the `pooled` template pulls each example's rules from the record, and keeping the `Subreddit: r/<name>` prefix (i.e. not passing `--no-sub-prefix`) matches the released adapter. The exact recorded configuration is in `results/main_2026/pooled_all_train_meta.json`; metrics are in `results/main_2026/pooled_all_*_metrics.json`.

## 5. Baselines

```bash
python src/eval/prompt_eval.py --model Qwen/Qwen3-14B --subreddit changemyview \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/      # zero-shot
python src/eval/gemini_eval.py --model gemini-2.5-flash --subreddit changemyview \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/ \
  --with-rules --rules-file data/rules/changemyview/rules.txt                     # Gemini 2.5 Flash
python src/eval/claude_eval.py --sub changemyview --template slm_mod --thinking on \
  --prompt-variant cot --n 500 --out results/claude_changemyview.json --rules-root data/rules   # Claude Sonnet 4.6
```

The Gemini baseline runs the full balanced test set; the Claude baseline uses 500 examples per subreddit (rate limits) with the SLM-Mod chain-of-thought prompt (`--template slm_mod --thinking on --prompt-variant cot`).

## 6. Metrics, natural-rate analysis, figures

```bash
python src/analysis/compute_metrics.py --results-dir results/
python src/analysis/imbalanced_resample.py \
  --predictions-dir results/ --output-dir results/main_2026/natural_rate/   # natural-rate bootstraps
python src/analysis/length_baseline.py \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/
python src/analysis/generate_figures.py        # regenerates the thesis figures from results/
```

`compute_metrics.py` uses only the standard library (accuracy, macro-F1, Cohen's kappa, bootstrap CIs, McNemar's test). The natural-rate ("imbalanced") analysis resamples the balanced test set down to each community's real removal rate; the per-rate kappa, F1, and precision with bootstrap intervals are in `results/main_2026/natural_rate/`.

## 7. SLM-Mod replication

`src/replication/` rebuilds the 2017 Chandrasekharan splits and runs the matched configuration, for comparison with the 2026 study (see the thesis Section 5).

## 8. 2024 pilot

`results/pilot_2024/all_metrics_summary.json` holds the per-subreddit metrics from the model-selection pilot: 30 subreddits on 2024 data across Gemma 2 9B, Llama 3.1 8B, Mistral NeMo 12B, and Qwen 2.5 7B (zero-shot with and without rules, and fine-tuned), plus Gemini. That pilot is what selected Qwen 3 for the main study.
