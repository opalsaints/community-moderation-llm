# Evaluating Open-Source LLMs for Community-Specific Content Moderation on Reddit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21130401.svg)](https://doi.org/10.5281/zenodo.21130401)

Code, configuration, and aggregate results for my BSc Capstone thesis (Amsterdam University College / University of Amsterdam, 2026).

The project asks a single question: can current open-weight LLMs stand in for human moderators on individual subreddits, each with its own rules and norms? I fine-tune one small adapter per community and measure how closely its remove/keep decisions agree with the moderators who actually made them.

## What is here

- The data pipeline (Arctic Shift download, dataset building, the `enriched_v2` prompt features).
- Training (`Qwen/Qwen3-14B` + LoRA, one adapter per subreddit and one pooled adapter).
- Evaluation: the fine-tuned models, a zero-shot baseline, and two prompted commercial APIs (Gemini 2.5 Flash, Claude Sonnet 4.6).
- The 2024 model-selection pilot across four open-source families.
- Aggregate metrics for every run. Moderator-removed comment text and per-comment predictions are not redistributed (see [Data and ethics](#data-and-ethics)); trained adapters live on Hugging Face.

## Main findings

Fifteen subreddits, drawn from the 2026 Arctic Shift archive, evaluated on a balanced (50/50 remove/keep) test set.

- Per-community fine-tuned `Qwen3-14B` reaches **macro Cohen's kappa 0.573** (79% balanced agreement), approaching the agreement ceiling implied by moderator disagreement.
- A **single pooled adapter** trained on all 15 communities does just as well (macro kappa 0.581), so community-specific weights are not strictly necessary.
- The fine-tuned models beat both prompted APIs on every subreddit: macro kappa 0.573 against **0.209** (Gemini 2.5 Flash) and **0.267** (Claude Sonnet 4.6), a 0.31 to 0.36 gap.
- Balanced numbers overstate real-world utility. At each community's natural removal rate (under 1% to 28%), macro kappa falls to about 0.16 and removed-class precision to about 0.15 while recall holds. The practical conclusion is no for autonomous removal, but useful as a human-in-the-loop triage layer.

Per-subreddit numbers (`results/main_2026/summary.csv`):

| Subreddit | n (test) | Fine-tuned kappa | Pooled kappa | Gemini kappa | Claude kappa |
|---|--:|--:|--:|--:|--:|
| antiai | 2000 | 0.715 | 0.644 | 0.409 | 0.516 |
| explainlikeimfive | 2000 | 0.701 | 0.688 | 0.264 | 0.400 |
| changemyview | 2000 | 0.622 | 0.634 | 0.410 | 0.528 |
| AmItheAsshole | 2000 | 0.606 | 0.670 | 0.127 | 0.181 |
| AskHistorians | 2000 | 0.589 | 0.578 | 0.237 | 0.246 |
| TwoXChromosomes | 2000 | 0.582 | 0.562 | 0.250 | 0.233 |
| aiwars | 1519 | 0.581 | 0.590 | 0.221 | 0.358 |
| relationships | 2000 | 0.578 | 0.612 | 0.088 | 0.207 |
| politics | 2000 | 0.549 | 0.588 | 0.226 | 0.209 |
| science | 2000 | 0.547 | 0.580 | 0.096 | 0.123 |
| Games | 2000 | 0.545 | 0.502 | 0.199 | 0.254 |
| personalfinance | 2000 | 0.511 | 0.534 | 0.172 | 0.204 |
| askscience | 1060 | 0.502 | 0.510 | 0.136 | 0.177 |
| news | 2000 | 0.496 | 0.532 | 0.112 | 0.123 |
| legaladvice | 2000 | 0.477 | 0.486 | 0.192 | 0.249 |
| **macro mean** | | **0.573** | **0.581** | **0.209** | **0.267** |

The `n (test)` column is the fine-tuned and Gemini test size. The pooled adapter is evaluated on 1,000 examples per subreddit and the Claude baseline on 500 (rate limits), so those two columns are computed on a subset of the listed n.

## Models

| Model | Size | Role | Engine |
|---|---|---|---|
| Qwen 3 14B | 14B | Per-subreddit + pooled LoRA fine-tuning (main study) | vLLM |
| Gemini 2.5 Flash | - | Prompted baseline | Vertex AI |
| Claude Sonnet 4.6 | - | Prompted baseline | Anthropic API |
| Gemma 2 9B, Llama 3.1 8B, Mistral NeMo 12B, Qwen 2.5 7B | 8-12B | 2024 model-selection pilot only | vLLM |

The fine-tuning recipe (`configs/training_config.json`): LoRA rank 16, alpha 16, all seven projection modules, two epochs, completion-only loss, the `enriched_v2` prompt template (post title, thread position, account age, and other lightweight metadata on top of the subreddit rules). The adapters are on Hugging Face under [opalitestudios](https://huggingface.co/opalitestudios), named `qwen3-14b-reddit-moderation-<subreddit>` (plus a `qwen3-14b-reddit-moderation-pooled` adapter). Load one with PEFT on top of the base model rather than retraining (see `docs/reproduction.md` for the snippet); once the adapter repositories carry version tags, pass `revision=` to `PeftModel.from_pretrained` to pin an exact adapter version.

## Subreddits

`AskHistorians, askscience, science, legaladvice, personalfinance, relationships, AmItheAsshole, changemyview, explainlikeimfive, Games, news, TwoXChromosomes, politics, antiai, aiwars` (`configs/subreddits.json`).

## Repository layout

```
configs/        subreddit list and the locked training configuration
src/data/       Arctic Shift download, dataset building, enriched_v2 features, rule collection
src/train/      finetune_v3.py (LoRA training + vLLM eval, per-subreddit and pooled)
src/eval/       zero-shot, Gemini, and Claude evaluation
src/analysis/   metrics, figures, length baseline, natural-rate (imbalanced) analysis
src/replication/ SLM-Mod 2017 replication
jobs/           SLURM scripts for Snellius (A100)
data/rules/     subreddit moderation rules (public)
data/comment_ids/ test-set comment IDs, to rebuild the exact splits from Arctic Shift
data/samples/   small anonymized demo
results/main_2026/  aggregate metrics for the 15-subreddit study (fine-tuned, pooled, baselines, natural-rate, length baseline)
results/pilot_2024/ the 2024 model-selection pilot metrics (30 subreddits, four families)
appendix/       rendered example prompts (one removed, one approved each) for the 12 subreddits not shown in the thesis appendix, which covers AskHistorians, changemyview, antiai
figures/        figures from the thesis
docs/           detailed reproduction notes
```

## Setup

Local machine (data prep, the API baselines, analysis):

```bash
pip install -r requirements.txt
gcloud auth application-default login        # Gemini, via Vertex AI
export ANTHROPIC_API_KEY=...                 # Claude baseline
```

Snellius HPC (training and vLLM evaluation):

```bash
module load 2025 Python/3.13.1-GCCcore-14.2.0
python -m venv ~/capstone_env && source ~/capstone_env/bin/activate
pip install -r requirements-snellius.txt
```

## Reproduction

The full run is driven by the SLURM scripts in `jobs/`. The per-subreddit steps, using `changemyview` as the example:

```bash
# 1. Data: download, extract parent context, build the balanced split, collect rules
python src/data/download_subreddit.py changemyview --after 2025-10-01 --before 2026-04-01 --output-dir data/raw/
python src/data/build_parent_index.py --subs changemyview --input-dir data/raw --output-dir data/extracted
python src/data/build_dataset.py --subs changemyview --input-dir data/extracted --output-dir data/dataset
python src/data/collect_rules.py --subs changemyview --output-dir data/rules

# 2. enriched_v2 template: post-title + author-feature sidecars, then the enriched split
python src/data/fetch_post_titles.py --dataset-dir data/dataset/changemyview --output data/dataset/changemyview/post_titles.json
python src/data/compute_author_features.py --input data/dataset/changemyview --output data/dataset/changemyview/author_features.json
python src/data/build_enriched_dataset.py --dataset-dir data/dataset/changemyview \
  --post-titles data/dataset/changemyview/post_titles.json \
  --author-features data/dataset/changemyview/author_features.json \
  --rules-file data/rules/changemyview/rules.txt \
  --output-dir data/dataset/changemyview/enriched_v2

# 3. Train, then evaluate, the per-community adapter (eval reuses the saved adapter)
python src/train/finetune_v3.py --model Qwen/Qwen3-14B \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/ \
  --subreddit changemyview --rules-file data/rules/changemyview/rules.txt \
  --template enriched --run-tag r4_stacked_changemyview \
  --epochs 2 --completion-only-loss --target-modules all
python src/train/finetune_v3.py ... --skip-train      # eval pass loads the adapter via vLLM

# 4. Baselines (zero-shot Qwen, Gemini, Claude)
python src/eval/prompt_eval.py --model Qwen/Qwen3-14B --subreddit changemyview \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/ \
  --with-rules --rules-file data/rules/changemyview/rules.txt
python src/eval/gemini_eval.py --model gemini-2.5-flash --subreddit changemyview \
  --dataset-dir data/dataset/changemyview/enriched_v2 --output-dir results/ \
  --with-rules --rules-file data/rules/changemyview/rules.txt
python src/eval/claude_eval.py --sub changemyview --template slm_mod --thinking on \
  --prompt-variant cot --n 500 --out results/claude_changemyview.json --rules-root data/rules

# 5. Metrics and figures
python src/analysis/compute_metrics.py --results-dir results/
python src/analysis/generate_figures.py        # regenerates the thesis figures from results/
```

`data/comment_ids/` lets you reconstruct the exact test sets from Arctic Shift without re-downloading whole subreddits. See `docs/reproduction.md` for the end-to-end walkthrough and the pooled-adapter and SLM-Mod replication steps.

## Data and ethics

Comments come from the [Arctic Shift](https://arctic-shift.photon-reddit.com/) archive (2025-10-01 to 2026-04-01). Usernames are treated as personal data and are not published. The released artifacts contain the subreddit rules, the test-set comment IDs, aggregate metrics, and a small demo of approved comments. Moderator-removed comment text and per-comment predictions are not redistributed, in line with Reddit's content policy: the worked examples in `appendix/` show approved comments verbatim but give removed comments only as anonymized paraphrases, alongside the original comment IDs so the source can be retrieved from Arctic Shift. To rebuild the full datasets, run the pipeline against the provided comment IDs.

An approved-comments-only corpus (no moderator-removed text) is published separately for convenience: https://www.kaggle.com/datasets/jonathancowley/reddit-approved-comments-15-communities-2026

## Citation

If you use the code, the released adapters, or the dataset, please cite the thesis:

```bibtex
@thesis{cowley2026moderation,
  title  = {Evaluating Open-Source LLMs for Community-Specific Content Moderation on Reddit},
  author = {Cowley, Jonathan},
  year   = {2026},
  school = {Amsterdam University College / University of Amsterdam},
  type   = {BSc Thesis}
}
```

The repository itself is archived on Zenodo: concept DOI [10.5281/zenodo.21130401](https://doi.org/10.5281/zenodo.21130401) always resolves to the latest release, and each release carries its own version DOI (v1.0.0 is [10.5281/zenodo.21130402](https://doi.org/10.5281/zenodo.21130402)).

### How to cite the dataset or the adapters

The thesis is the primary reference for every artifact in this project. The three
artifact homes cross-link back here and to one another:

- This repository: https://github.com/opalsaints/community-moderation-llm (code, configs, aggregate results, `CITATION.cff`), archived at [doi:10.5281/zenodo.21130401](https://doi.org/10.5281/zenodo.21130401).
- The LoRA adapters on Hugging Face: https://huggingface.co/opalitestudios (named `qwen3-14b-reddit-moderation-<subreddit>` and `qwen3-14b-reddit-moderation-pooled`).
- The approved-comments dataset on Kaggle: https://www.kaggle.com/datasets/jonathancowley/reddit-approved-comments-15-communities-2026

When citing the **adapters** or the **dataset** specifically, cite the thesis above and
add a parenthetical pointing at the artifact you used (the Hugging Face adapter name, or
the Kaggle dataset URL). `CITATION.cff` carries the machine-readable version of this
citation; GitHub renders a "Cite this repository" widget from it.

## License

Code is released under the MIT License (`LICENSE`). Subreddit rules are the property of their respective communities.
