# 2024 model-selection pilot

`all_metrics_summary.json` holds the per-subreddit metrics from the pilot that selected the model for the main study: 30 subreddits on 2024 Arctic Shift data, across Gemma 2 9B, Llama 3.1 8B, Mistral NeMo 12B, and Qwen 2.5 7B (zero-shot with and without rules, and LoRA fine-tuned), plus Gemini 2.5 Flash.

Keys are `<category>/<model>_<subreddit>_<condition>`; each value reports accuracy, F1, precision, recall, Cohen's kappa, AUROC, and the confusion-matrix counts. Qwen 3 was chosen from this pilot and is evaluated in `../main_2026/`.

This pilot covered a larger subreddit set than the 15-subreddit main study; not all of these communities appear in the final work.
