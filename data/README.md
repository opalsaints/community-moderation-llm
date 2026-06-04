# Data

This directory holds the released data artifacts for the study: the subreddit moderation rules, the test-set comment IDs needed to rebuild the exact evaluation splits, and a small approved-comments-only demo. Moderator-removed comment text and per-comment model predictions are not redistributed. See [`DATA_STATEMENT.md`](../DATA_STATEMENT.md) for the full curation and ethics rationale.

## Provenance

All comments come from the [Arctic Shift](https://arctic-shift.photon-reddit.com/) public archive of Reddit, restricted to the window 2025-10-01 to 2026-04-01. The fifteen subreddits are listed in [`configs/subreddits.json`](../configs/subreddits.json). The balanced (50/50 remove/keep) test splits were drawn with a fixed seed (42) so that the reconstruction from comment IDs is deterministic.

## What is in each subdirectory

### `rules/`

One folder per subreddit, each containing `rules.txt` (a flat, human-readable listing) and `rules.json` (the structured rule objects as returned by Reddit, with short name, description, violation reason, and creation time). These are the community moderation rules fed into the `enriched_v2` prompt template. The rules are the property of their respective communities and are reproduced here only to support reproduction of the experiments.

### `comment_ids/`

One `<subreddit>_test_ids.txt` per subreddit, one Reddit comment ID per line, listing every comment in that subreddit's balanced test split. These are identifiers only, not content. Running [`src/data/hydrate_from_ids.py`](../src/data/hydrate_from_ids.py) against these files re-fetches the comments from Arctic Shift and rebuilds the exact test set used in the paper. Split sizes range from 1,060 (askscience) to 2,000 comments; aiwars has 1,519 and askscience 1,060 because those communities had fewer eligible removed comments in the window.

### `samples/`

`demo_sample.jsonl` is a 30-row, approved-comments-only demo (two or three example rows per subreddit) so that a reader can see the exact record shape without downloading anything. Every row has `label` equal to `approved`; no moderator-removed comment text appears in this file. Usernames have been stripped. The demo exists only to illustrate the schema and is not a training or evaluation set.

## `demo_sample.jsonl` schema

Each line is a JSON object with the following fields.

| Field | Type | Description |
|---|---|---|
| `subreddit` | string | The community the comment was posted in. |
| `label` | string | The moderator outcome. In this demo every value is `approved`; in the full dataset it is `approved` or `removed`. |
| `body` | string | The comment text. |
| `parent_body` | string | The text of the parent comment, or the post body when the comment is a top-level reply. |
| `post_title` | string | The title of the submission the comment belongs to. |
| `is_top_level` | bool | True when the comment is a direct reply to the post rather than to another comment. |
| `account_age_days` | float | The comment author's account age in days at the time of the comment, used as a lightweight metadata feature in `enriched_v2`. |

### A note on the `[ Removed by moderator ]` post title

Two demo rows carry the literal `post_title` value `[ Removed by moderator ]` (one in science, one in AmItheAsshole). This is Reddit's own placeholder for a parent submission whose title was removed by a moderator, captured as-is in the archive. It is not moderator-removed comment text leaking into the release: the comments themselves are approved and shown in full, and only the surrounding submission title was unavailable.

## License

The compilation in this directory (the rules listing and the demo records) is offered under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the purpose of reproducing and building on this research. The underlying Reddit comments remain subject to [Reddit's content policy](https://redditinc.com/policies/content-policy) and the original authors' rights; the subreddit rules remain the property of their respective communities. Code in this repository is MIT (see [`LICENSE`](../LICENSE)).
