# Data statement

This statement describes the data used and released by the study *Evaluating Open-Source LLMs for Community-Specific Content Moderation on Reddit* (Cowley, 2026, BSc thesis, Amsterdam University College / University of Amsterdam). It follows the spirit of a short data statement: it records why the data was collected, where it came from, what is and is not released, and how a third party can rebuild the full datasets responsibly.

## Curation rationale

The study asks whether open-weight LLMs can reproduce the remove/keep decisions of human moderators on individual subreddits, each with its own rules. Studying that requires real moderation outcomes from communities with distinct, documented norms, so the corpus is fifteen subreddits chosen to span rule-bound communities (for example AskHistorians, legaladvice) and more discretionary ones (for example news, politics), and to include a built-in contrast in contested norms (antiai versus aiwars). Moderator labels are treated as the object of study, not as ground-truth correctness.

## Language

All released text is English. Non-English comments were excluded during dataset construction.

## Provenance

Comments come from the [Arctic Shift](https://arctic-shift.photon-reddit.com/) public archive of Reddit, restricted to the window 2025-10-01 to 2026-04-01. The balanced (50/50 remove/keep) test splits were drawn with a fixed seed (42).

## What is released

- The subreddit moderation rules (`data/rules/`), reproduced to support reproduction of the experiments.
- The test-set comment IDs (`data/comment_ids/`), one Reddit comment ID per line per subreddit, sufficient to rebuild the exact evaluation splits.
- Aggregate metrics for every run (`results/`): kappa, precision, recall, and related figures, computed over communities, never per-comment outputs.
- A 30-row, approved-comments-only demo (`data/samples/demo_sample.jsonl`) illustrating the record schema.
- Worked moderation examples in the thesis appendix (`appendix/`): approved comments are shown verbatim, and moderator-removed comments are shown only as anonymized paraphrases, with the original Reddit comment IDs retained so the source can be retrieved from the archive.
- The trained LoRA adapters, hosted on Hugging Face under [opalitestudios](https://huggingface.co/opalitestudios).
- A separately published, approved-comments-only corpus on [Kaggle](https://www.kaggle.com/datasets/jonathancowley/reddit-approved-comments-15-communities-2026) for convenience.

## What is not released

- Raw moderator-removed comment text. It is never published in any form, including the appendix, where removed comments appear only as paraphrases.
- Per-comment model predictions. Only aggregate metrics are released, so no individual user's comment can be paired with a model decision.
- Usernames. Author identities are treated as personal data and are not published.

## Anonymization method

Usernames are stripped from every released artifact. In the demo records, only the comment metadata and text needed to illustrate the schema are kept, with no author handle. In the appendix examples, authors are referred to by neutral placeholders (for example `user_anon_001`) and removed comments are replaced with paraphrases rather than verbatim text. Comment IDs are retained because they are opaque identifiers that do not themselves reveal an author, and because they are the mechanism by which a reader can re-fetch the original record from the public archive. The task-overview figure (`figures/moderation_task_figure.png`) is an illustrative mock-up: the usernames and comment text shown in it are fabricated for illustration and do not correspond to any real Reddit user or comment.

## Legal and policy rationale

Two regimes shape what is released. Under the GDPR, Reddit usernames and the comment text attached to them are personal data, so authors' identities are withheld and removed-comment text is not redistributed. Under [Reddit's content policy](https://redditinc.com/policies/content-policy), redistributing moderator-removed content would conflict with the moderators' removal decision, so the release ships identifiers and paraphrases rather than the removed text itself. Aggregate metrics and the rules listing fall outside both concerns and are released openly. A reader who needs the full data reconstructs it from the comment IDs against the live archive, which keeps the release in step with any later deletions or removals at the source.

## How to reconstruct the data

Run [`src/data/hydrate_from_ids.py`](src/data/hydrate_from_ids.py) against `data/comment_ids/`. It reads each `<subreddit>_test_ids.txt`, fetches the corresponding comments from the Arctic Shift API, and writes them out in the project's dataset schema, rebuilding the exact balanced test sets used in the paper. The script documents its network and rate-limit requirements; see also [`docs/reproduction.md`](docs/reproduction.md) for the end-to-end pipeline.

## Intended use and misuse caveat

The released artifacts are intended for research on community-specific content moderation and for reproducing the study. The headline result is that the fine-tuned models are useful as a human-in-the-loop triage layer, not as autonomous moderators: at each community's natural removal rate, removed-class precision collapses, and it collapses most in low-base-rate political and identity communities. Deploying these models to remove content automatically would sweep up a large amount of legitimate speech to catch a few violations, and would do so most aggressively exactly where free-expression stakes are highest. The data and models should not be used to build or justify autonomous removal systems, to profile or deanonymize Reddit users, or to make claims about individuals.

## Base model and licensing

The adapters are LoRA weights trained on top of [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), which is released under the Apache License 2.0. Apache 2.0 permits redistribution and derivative works, so the adapters in this project are distributed under the MIT License (see [`LICENSE`](LICENSE)). The base model weights themselves are not redistributed and remain under their own Apache 2.0 terms.

## Alternative DOI route

A DOI for the public release is available in one click through the GitHub-Zenodo integration (see `.zenodo.json`). The institution-blessed alternative is to mint a DOI through the UvA / AUAS figshare instance with the help of a faculty data steward; either route is acceptable for archival citation.
