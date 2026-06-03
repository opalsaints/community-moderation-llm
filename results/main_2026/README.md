# 2026 main study results (15 subreddits)

Aggregate metrics only. Per-comment predictions and comment text are not included.

- `r4_stacked_<sub>_metrics.json` / `_train_meta.json` - per-community fine-tuned Qwen 3 14B, the headline result (macro Cohen's kappa 0.573).
- `pooled_all_<sub>_metrics.json` - the single pooled adapter evaluated on each subreddit (macro kappa 0.581); `pooled_all_train_meta.json` is its training record.
- `baselines/` - prompted Gemini 2.5 Flash and Claude Sonnet 4.6 (with rules), predictions stripped.
- `natural_rate/<sub>.json` - bootstrapped accuracy, F1, kappa, and precision at each community's real removal rate, with confidence intervals. This is the deployment-realistic degradation.
- `length_baseline/<sub>.json` - comment-length baseline classifier.
- `summary.csv` - the per-subreddit headline table.

Adapter weights are on Hugging Face; see the top-level README.
