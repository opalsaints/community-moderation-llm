#!/usr/bin/env python3
"""
Rebuild the balanced test set by hydrating Reddit comment IDs from Arctic Shift.

The release ships the test split as bare comment IDs (one per line) in
data/comment_ids/<sub>_test_ids.txt rather than as raw comment text. This script
reads those IDs and fetches each comment back from the Arctic Shift API, writing
one NDJSON file per subreddit in the same raw format produced by
download_subreddit.py (one JSON object per line, full metadata including _meta).

That means the output of this script can be fed straight back into the rest of
the pipeline:

    1. hydrate_from_ids.py  --subs <subs> --output-dir data/raw_test/
    2. build_parent_index.py --input-dir data/raw_test/ --output-dir data/extracted_test/
    3. build_dataset.py      --input-dir data/extracted_test/ --output-dir data/dataset_test/ --subs <subs>

Network + rate-limit requirements:
    - Requires outbound HTTPS access to https://arctic-shift.photon-reddit.com/
      (no API key or authentication needed).
    - The API is community-run; be polite. This script batches up to 100 IDs per
      request, sleeps REQUEST_DELAY seconds between requests, and backs off on
      rate-limit / network errors (same handling as download_subreddit.py).
    - Hydration is best-effort: comments that have since been deleted at the
      source, or that the API can no longer return, will be missing. The script
      reports how many of the requested IDs were recovered per subreddit so the
      reader can judge coverage.

Usage:
    python hydrate_from_ids.py --subs AskHistorians
    python hydrate_from_ids.py --subs AskHistorians,changemyview,antiai
    python hydrate_from_ids.py --subs all --ids-dir data/comment_ids --output-dir data/raw_test

Output: data/raw_test/<sub>.ndjson per subreddit (one JSON object per line),
same schema as download_subreddit.py / the original .zst dumps.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Arctic Shift comments lookup-by-ID endpoint. Accepts a comma-separated list of
# fullnames (t1_ prefix for comments), mirroring the posts/ids endpoint used in
# fetch_post_titles.py.
API_IDS = "https://arctic-shift.photon-reddit.com/api/comments/ids"
BATCH_SIZE = 100        # max IDs per request
REQUEST_DELAY = 0.5     # seconds between requests (be polite)
ID_SUFFIX = "_test_ids.txt"


def api_get_ids(comment_ids, max_retries=5):
    """Fetch a batch of comments by ID from Arctic Shift.

    comment_ids: list of bare comment IDs (no t1_ prefix).
    Returns (list_of_comments, error_or_None). Uses the same retry / backoff /
    rate-limit handling as download_subreddit.py.
    """
    ids_str = ",".join(f"t1_{cid}" for cid in comment_ids)
    url = f"{API_IDS}?ids={ids_str}"
    # API_IDS is a fixed https constant; only the comment IDs vary. Guard the
    # scheme so the request can never be redirected to a local file:// path.
    if not url.startswith("https://"):
        raise ValueError(f"Refusing non-https Arctic Shift URL: {url}")
    req = Request(url, headers={"User-Agent": "capstone-research/1.0"})

    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                if data.get("error"):
                    err = str(data["error"])
                    if "Timeout" in err or "slow down" in err:
                        wait = 15 * (attempt + 1)
                        print(f"\n    Rate limited, waiting {wait}s...", end="", flush=True)
                        time.sleep(wait)
                        continue
                    return [], err
                return data.get("data", []), None
        except (URLError, HTTPError, TimeoutError, OSError, ConnectionError) as e:
            wait = 10 * (2 ** attempt)  # exponential backoff: 10, 20, 40, 80, 160s
            if attempt < max_retries - 1:
                print(f"\n    Network error (attempt {attempt+1}/{max_retries}): {e}. "
                      f"Retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            return [], str(e)
    return [], "Max retries exceeded"


def read_ids(ids_file):
    """Read bare comment IDs (one per line) from a *_test_ids.txt file."""
    ids = []
    with open(ids_file) as f:
        for line in f:
            cid = line.strip()
            if cid:
                ids.append(cid)
    return ids


def hydrate_subreddit(subreddit, ids_dir, output_dir):
    """Hydrate one subreddit's test IDs into an NDJSON file.

    Returns (requested, recovered) counts.
    """
    ids_file = os.path.join(ids_dir, f"{subreddit}{ID_SUFFIX}")
    if not os.path.exists(ids_file):
        print(f"  r/{subreddit}: no ID file at {ids_file}, skipping")
        return 0, 0

    comment_ids = read_ids(ids_file)
    requested = len(comment_ids)
    if requested == 0:
        print(f"  r/{subreddit}: ID file is empty, skipping")
        return 0, 0

    output_file = os.path.join(output_dir, f"{subreddit}.ndjson")
    recovered = 0

    with open(output_file, "w") as f:
        for i in range(0, requested, BATCH_SIZE):
            batch = comment_ids[i:i + BATCH_SIZE]
            comments, error = api_get_ids(batch)

            if error:
                print(f"\n  r/{subreddit}: error at batch starting {i}: {error}")
                f.flush()
                continue

            for comment in comments:
                f.write(json.dumps(comment, ensure_ascii=False) + "\n")
            recovered += len(comments)

            f.flush()
            print(f"\r  r/{subreddit}: {recovered:,}/{requested:,} comments recovered",
                  end="", flush=True)

            time.sleep(REQUEST_DELAY)

    coverage = recovered / requested if requested else 0
    print(f"\r  r/{subreddit}: {recovered:,}/{requested:,} comments recovered "
          f"({coverage:.1%}) -> {output_file}")
    return requested, recovered


def resolve_subreddits(args):
    """Determine which subreddits to hydrate from --subs and the ids-dir."""
    ids_dir = Path(args.ids_dir)
    if args.subs.strip().lower() == "all":
        subs = sorted(
            p.name[: -len(ID_SUFFIX)]
            for p in ids_dir.glob(f"*{ID_SUFFIX}")
        )
        if not subs:
            print(f"No {ID_SUFFIX} files found in {ids_dir}")
            sys.exit(1)
        return subs
    return [s.strip() for s in args.subs.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the test set by hydrating comment IDs via Arctic Shift")
    parser.add_argument("--subs", required=True,
                        help="Comma-separated subreddits, or 'all' to hydrate every "
                             "*_test_ids.txt file in --ids-dir")
    parser.add_argument("--ids-dir", default="data/comment_ids",
                        help="Directory with <sub>_test_ids.txt files "
                             "(default: data/comment_ids)")
    parser.add_argument("--output-dir", default="data/raw_test",
                        help="Output directory for hydrated NDJSON (default: data/raw_test)")
    args = parser.parse_args()

    subreddits = resolve_subreddits(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Hydrating {len(subreddits)} subreddit(s) from comment IDs")
    print(f"IDs:    {args.ids_dir}/")
    print(f"Output: {args.output_dir}/")
    print(f"API:    {API_IDS}")
    print()

    total_requested = 0
    total_recovered = 0
    failed = []

    for sub in subreddits:
        try:
            requested, recovered = hydrate_subreddit(sub, args.ids_dir, args.output_dir)
            total_requested += requested
            total_recovered += recovered
        except Exception as e:
            print(f"\n  FAILED r/{sub}: {e}")
            failed.append(sub)

    print()
    coverage = total_recovered / total_requested if total_requested else 0
    print(f"Done: {total_recovered:,}/{total_requested:,} comments recovered "
          f"({coverage:.1%}) across {len(subreddits)} subreddit(s)")

    if total_requested and recovered_is_partial(total_recovered, total_requested):
        print("Note: comments deleted at the source cannot be recovered; "
              "see the per-subreddit coverage above.")

    if failed:
        print(f"\nFailed subreddits ({len(failed)}): {', '.join(failed)}")
        print("Re-run to retry these (output files are overwritten per run).")


def recovered_is_partial(recovered, requested):
    """True if some requested IDs could not be hydrated."""
    return recovered < requested


if __name__ == "__main__":
    main()
