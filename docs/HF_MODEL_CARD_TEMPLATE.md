# Hugging Face model-card template (per-adapter)

One template for the LoRA adapters published under
[opalitestudios](https://huggingface.co/opalitestudios). Fill the `{sub}`
placeholder with a subreddit name to produce that adapter's card (for the pooled
adapter use `pooled`; see the note at the end). Keep the HF repos PRIVATE for
now; this file is a template for review, not a push.

Replace `{sub}` everywhere, then set `{kappa}` from the per-subreddit table at
the bottom. The block between the `---` lines is YAML frontmatter and must stay
at the very top of the rendered `README.md`.

---

```markdown
---
license: mit
base_model: Qwen/Qwen3-14B
library_name: peft
tags:
  - lora
  - peft
  - content-moderation
  - reddit
  - text-classification
  - qwen3
language:
  - en
pipeline_tag: text-classification
---

# qwen3-14b-reddit-moderation-{sub}

A LoRA adapter that fine-tunes [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B)
to predict whether a comment on the **r/{sub}** subreddit would be removed or kept
by that community's moderators. It is one of a set of per-community adapters (plus
one pooled adapter) released with a BSc Capstone study on community-specific
content moderation.

The adapter takes a comment, its thread context, and r/{sub}'s moderation rules,
and outputs a remove/keep decision. The decisions are modeled after the moderators
who actually labeled the data; the moderator labels are the object of study, not a
ground truth of what "should" be removed.

## Intended use and limitations

**Intended use:** a human-in-the-loop triage layer for r/{sub}, surfacing comments
that may violate community norms for a human moderator to review.

**Not intended for autonomous moderation.** On a balanced (50/50) test set the
adapters agree well with moderators, but at each community's real removal rate
(often well under 5%) removed-class precision is low, so used to auto-remove it
would sweep up legitimate comments to catch a few violations. Always keep a human
in the loop. The model pattern-matches a community's encoded norms; it does not
"understand" the underlying topics, and its confidence scores are weak.

## Model details

- **Base model:** `Qwen/Qwen3-14B` (Apache-2.0).
- **Adapter type:** LoRA (PEFT).
- **LoRA config:** rank 16, alpha 16, all seven projection modules
  (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Training:** 2 epochs, completion-only loss, `enriched_v2` prompt template
  (subreddit rules plus lightweight metadata: post title, thread position, account
  age, and similar features).
- **Task:** binary remove/keep classification of Reddit comments for r/{sub}.

## Training data

Comments from the [Arctic Shift](https://arctic-shift.photon-reddit.com/) Reddit
archive, sampled over 2025-10-01 to 2026-04-01, restricted to r/{sub} and to
English. The training split is balanced 50/50 remove/keep. Moderator-removed
comment text and per-comment predictions are not redistributed; the project
releases the subreddit rules, the test-set comment IDs (to rebuild the splits via
Arctic Shift), aggregate metrics, and an approved-comments-only demo. See the
project repository's data statement for provenance, anonymization, and Reddit
content-policy details.

## Evaluation

On a balanced (50/50 remove/keep) held-out test set for r/{sub}, this adapter
reaches **macro Cohen's kappa {kappa}**. Across all 15 communities the
per-community adapters average macro kappa 0.573, beating prompted commercial API
baselines (Gemini 2.5 Flash 0.209, Claude Sonnet 4.6 0.267) on every subreddit. A
single pooled adapter does about as well (macro kappa 0.581). Balanced numbers
overstate real-world utility: at natural removal rates macro kappa falls to about
0.16 and removed-class precision to about 0.15 while recall holds. Full per-run
metrics are in the project repository.

## How to use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
model = PeftModel.from_pretrained(base, "opalitestudios/qwen3-14b-reddit-moderation-{sub}")
```

Once release tags exist you can pin a specific adapter version with
`PeftModel.from_pretrained(..., revision="v1.0.0")`.

The adapter expects the `enriched_v2` prompt format (comment, thread context, and
r/{sub}'s rules); see the project repository for the exact template and an
end-to-end evaluation script.

## Links

- Project code and aggregate results: https://github.com/opalsaints/community-moderation-llm
- Approved-comments corpus: https://www.kaggle.com/datasets/jonathancowley/reddit-approved-comments-15-communities-2026
- All adapters: https://huggingface.co/opalitestudios

## License

This adapter is released under the **MIT License**. The base model
`Qwen/Qwen3-14B` is licensed under Apache-2.0, which permits redistribution of
derivative adapters; only the LoRA weights are distributed here, not the base
model.

## Citation

```bibtex
@thesis{cowley2026moderation,
  title  = {Evaluating Open-Source LLMs for Community-Specific Content Moderation on Reddit},
  author = {Cowley, Jonathan},
  year   = {2026},
  school = {Amsterdam University College / University of Amsterdam},
  type   = {BSc Thesis}
}
```
```

---

## Per-subreddit kappa (fill `{kappa}` from this table)

Balanced-test fine-tuned macro Cohen's kappa, from `results/main_2026/summary.csv`:

| `{sub}` | `{kappa}` |
|---|--:|
| antiai | 0.715 |
| explainlikeimfive | 0.701 |
| changemyview | 0.622 |
| AmItheAsshole | 0.606 |
| AskHistorians | 0.589 |
| TwoXChromosomes | 0.582 |
| aiwars | 0.581 |
| relationships | 0.578 |
| politics | 0.549 |
| science | 0.547 |
| Games | 0.545 |
| personalfinance | 0.511 |
| askscience | 0.502 |
| news | 0.496 |
| legaladvice | 0.477 |

## Pooled adapter note

For `qwen3-14b-reddit-moderation-pooled`, replace `{sub}` with `pooled` and adjust
the prose: it is trained on the union of all 15 communities (each example carries
its own subreddit and rules in the prompt) and reaches macro kappa **0.581**
across the 15 test sets, so set `{kappa}` to that value and drop the single-r/{sub}
framing in favor of "all 15 communities."
