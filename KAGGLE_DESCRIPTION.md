# Kaggle dataset description (paste-ready)

Paste-ready text for the Kaggle dataset
`jonathancowley/reddit-approved-comments-15-communities-2026`
(approved-comments-only corpus). Keep the dataset PRIVATE for now; this file is
just the text to paste into the currently-empty description, subtitle, and tags.

---

## Subtitle (one line, max ~80 chars)

Approved Reddit comments from 15 communities (2026), for moderation-norm research

---

## Tags / keywords

```
reddit, content-moderation, nlp, llm, arctic-shift, text-classification, social-media, moderation, lora, qwen
```

---

## Description (paste into the Kaggle "Description" box)

### Reddit approved comments: 15 communities (2026)

A small, anonymized corpus of moderator-approved Reddit comments drawn from 15
communities, released as a companion to a BSc Capstone study on community-specific
content moderation. The study asks whether open-weight LLMs can stand in for human
moderators on individual subreddits, each with its own rules and norms, by
fine-tuning one small LoRA adapter per community and measuring how closely its
remove/keep decisions agree with the moderators who actually made them.

This Kaggle dataset is the convenience release of the **approved (kept) comments
only**. It contains no moderator-removed comment text and no per-comment model
predictions. The full release (code, configuration, aggregate metrics, the trained
adapters, and the comment IDs needed to rebuild the exact test splits) lives in
the project repository and on Hugging Face (links below).

### What is in it

Each row is a single moderator-approved comment with its thread context:

| Field | Type | Description |
|---|---|---|
| `body` | string | The approved comment text. |
| `subreddit` | string | Source community (one of the 15 below). |
| `context` | string | Preceding thread context (parent/ancestor comment text), or empty for a top-level reply. |

Every comment in this dataset is moderator-approved: there is no removed-comment
text and no per-comment model predictions. A small number of rows mention
Reddit's own `[deleted]` or `[ Removed by moderator ]` placeholder inside `body`
or `context` (for example, an approved comment that discusses a removal). That is
Reddit's placeholder string, not leaked removed-comment content; the comment in
`body` is itself approved.

### Communities (15)

`AskHistorians`, `askscience`, `science`, `legaladvice`, `personalfinance`,
`relationships`, `AmItheAsshole`, `changemyview`, `explainlikeimfive`, `Games`,
`news`, `TwoXChromosomes`, `politics`, `antiai`, `aiwars`.

### Provenance

Comments come from the [Arctic Shift](https://arctic-shift.photon-reddit.com/)
Reddit archive, sampled over the window 2025-10-01 to 2026-04-01. The data is
English-language. Usernames are treated as personal data and are not published;
only the per-comment features above are released.

### What this is for

- Studying how moderation norms differ across communities (what gets approved,
  and how that varies by subreddit).
- Building and probing text classifiers for content moderation.
- Reproducing or extending the Capstone study's `enriched_v2` feature setup.

This is a research artifact. The study's own conclusion is that fine-tuned
open-weight models are useful as a human-in-the-loop triage layer, not as an
autonomous remove/keep authority: at each community's real (natural) removal
rate, removed-class precision is low. Treat any model trained on this data
accordingly, and do not deploy it to remove content without human review.

### Licensing and Reddit content policy

The released features and compilation are shared for research and may be used
under a CC-BY-style attribution arrangement; the underlying comments remain
subject to Reddit's content policy and the rights of their authors. Do not use
this dataset to deanonymize, profile, or target individuals.

### Companions and citation

- Project code and aggregate results: https://github.com/opalsaints/community-moderation-llm
- Trained LoRA adapters (per-community + pooled): https://huggingface.co/opalitestudios
- Approved-comments corpus (this dataset): https://www.kaggle.com/datasets/jonathancowley/reddit-approved-comments-15-communities-2026

If you use this data, please cite the thesis:

```bibtex
@thesis{cowley2026moderation,
  title  = {Evaluating Open-Source LLMs for Community-Specific Content Moderation on Reddit},
  author = {Cowley, Jonathan},
  year   = {2026},
  school = {Amsterdam University College / University of Amsterdam},
  type   = {BSc Thesis}
}
```
