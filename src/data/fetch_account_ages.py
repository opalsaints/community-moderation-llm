#!/usr/bin/env python3
"""
Fetch account creation dates for all unique authors in a dataset.

Strategy: For each author, search Arctic Shift for their oldest known post.
Posts often contain `author_created_utc`. If not available, the oldest post's
`created_utc` serves as an upper bound on account age.

Output: JSON mapping author -> {"account_created_utc": int, "source": str}

Usage:
    python fetch_account_ages.py \
        --dataset-dir ~/data/dataset/changemyview/enriched \
        --output ~/data/enrichment/changemyview_account_ages.json
"""

import argparse
import json
import os
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

API_POSTS = "https://arctic-shift.photon-reddit.com/api/posts/search"
API_COMMENTS = "https://arctic-shift.photon-reddit.com/api/comments/search"
REQUEST_DELAY = 0.5  # be gentle with the API
SAVE_EVERY = 100


def fetch_oldest_activity(author, max_retries=3):
    """Find the oldest known activity for an author.

    Tries posts first (often has author_created_utc), then comments.
    Returns (account_created_utc, source_string) or (None, None).
    """
    # Try posts first -- they more often have author_created_utc
    for endpoint, label in [(API_POSTS, "post"), (API_COMMENTS, "comment")]:
        url = f"{endpoint}?author={author}&sort=asc&limit=1"
        req = Request(url, headers={"User-Agent": "capstone-research/1.0"})

        for attempt in range(max_retries):
            try:
                with urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode())
                    items = data.get("data", [])
                    if items:
                        item = items[0]
                        # Best case: explicit account creation date
                        acct_created = item.get("author_created_utc")
                        if acct_created and acct_created > 0:
                            return int(acct_created), f"{label}_author_created_utc"
                        # Fallback: oldest known activity timestamp
                        created = item.get("created_utc")
                        if created:
                            return int(created), f"oldest_{label}"
                    break  # no data but no error
            except (URLError, HTTPError, TimeoutError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    return None, f"error: {e}"

    return None, "not_found"


def main():
    parser = argparse.ArgumentParser(description="Fetch account creation dates")
    parser.add_argument("--dataset-dir", required=True,
                        help="Dataset directory with train.jsonl and test.jsonl")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    # Collect unique authors
    print("Collecting unique authors...", flush=True)
    authors = set()
    for split in ["train.jsonl", "test.jsonl"]:
        fp = os.path.join(args.dataset_dir, split)
        if os.path.exists(fp):
            with open(fp) as f:
                for line in f:
                    rec = json.loads(line)
                    author = rec.get("author", "")
                    if author and author != "[deleted]":
                        authors.add(author)

    print(f"  {len(authors)} unique authors")

    # Load existing results for resume support
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        print(f"  {len(existing)} already fetched (resuming)")

    needed = [a for a in authors if a not in existing]
    print(f"  {len(needed)} remaining to fetch")

    if not needed:
        print("All authors already fetched.")
        return

    # Estimate time
    est_minutes = len(needed) * REQUEST_DELAY * 2 / 60  # 2 requests per author worst case
    print(f"  Estimated time: {est_minutes:.0f} minutes")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fetched = 0
    errors = 0
    sources = {}

    for i, author in enumerate(needed):
        ts, source = fetch_oldest_activity(author)

        if ts:
            existing[author] = {
                "account_created_utc": ts,
                "source": source,
            }
            fetched += 1
            sources[source] = sources.get(source, 0) + 1
        else:
            existing[author] = {
                "account_created_utc": None,
                "source": source,
            }
            errors += 1

        # Progress + periodic save
        if (i + 1) % SAVE_EVERY == 0 or i == len(needed) - 1:
            with open(args.output, "w") as f:
                json.dump(existing, f, indent=2)
            elapsed_pct = 100 * (i + 1) / len(needed)
            print(f"  [{i+1}/{len(needed)} ({elapsed_pct:.0f}%)] "
                  f"fetched={fetched} errors={errors} "
                  f"sources={dict(sources)}", flush=True)

        time.sleep(REQUEST_DELAY)

    # Final save
    with open(args.output, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nDone. {fetched} fetched, {errors} errors.")
    print(f"Sources: {dict(sources)}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
