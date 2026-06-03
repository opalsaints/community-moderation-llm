#!/usr/bin/env python3
"""
Fetch post titles from Arctic Shift API for all unique link_ids in a dataset.

Creates a sidecar JSON mapping link_id -> post title, used by the enriched
training pipeline without modifying the original dataset files.

Usage:
    python fetch_post_titles.py --dataset-dir ./dataset/changemyview/random_split \
                                --output ./enrichment/changemyview_post_titles.json

    python fetch_post_titles.py --dataset-dir ./dataset/changemyview/random_split \
                                --output ./enrichment/changemyview_post_titles.json \
                                --train-file ../extracted/changemyview.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Arctic Shift posts search -- accepts comma-separated IDs
API_IDS = "https://arctic-shift.photon-reddit.com/api/posts/ids"
BATCH_SIZE = 100  # max IDs per request
REQUEST_DELAY = 1.0


def fetch_posts_by_ids(post_ids, max_retries=3):
    """Fetch posts by their IDs from Arctic Shift."""
    ids_str = ",".join(f"t3_{pid}" for pid in post_ids)
    url = f"{API_IDS}?ids={ids_str}"
    req = Request(url, headers={"User-Agent": "capstone-research/1.0"})

    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data.get("data", []), None
        except (URLError, HTTPError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retry {attempt + 1} ({e}), waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                return [], str(e)
    return [], "Max retries exceeded"


def collect_link_ids(dataset_dir, train_file=None):
    """Collect all unique link_ids from dataset files and optionally raw extracted data."""
    link_ids = set()

    # From dataset splits
    for fname in ["train.jsonl", "test.jsonl"]:
        fp = os.path.join(dataset_dir, fname)
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                lid = d.get("link_id", "")
                if lid:
                    link_ids.add(lid.replace("t3_", ""))

    # From raw extracted data (has all comments, not just sampled)
    if train_file and os.path.exists(train_file):
        print(f"  Also scanning {train_file} for link_ids...")
        with open(train_file) as f:
            data = json.load(f)
        for category in ["removed", "approved"]:
            for c in data.get(category, []):
                lid = c.get("link_id", "")
                if lid:
                    link_ids.add(lid.replace("t3_", ""))

    return sorted(link_ids)


def main():
    parser = argparse.ArgumentParser(description="Fetch post titles from Arctic Shift")
    parser.add_argument("--dataset-dir", required=True, help="Directory with train/test.jsonl")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--train-file", default=None,
                        help="Optional: raw extracted JSON to get ALL link_ids (not just sampled)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file (skip already-fetched IDs)")
    args = parser.parse_args()

    # Collect link_ids
    print("Collecting link_ids...", flush=True)
    link_ids = collect_link_ids(args.dataset_dir, args.train_file)
    print(f"  Found {len(link_ids)} unique link_ids")

    # Resume support
    existing = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing)} already fetched")
        link_ids = [lid for lid in link_ids if lid not in existing and f"t3_{lid}" not in existing]
        print(f"  Remaining: {len(link_ids)} to fetch")

    titles = dict(existing)

    # Fetch in batches
    total = len(link_ids)
    fetched = 0
    failed = 0

    for i in range(0, total, BATCH_SIZE):
        batch = link_ids[i:i + BATCH_SIZE]
        posts, error = fetch_posts_by_ids(batch)

        if error:
            print(f"\n  Error at batch {i}: {error}", flush=True)
            failed += len(batch)
            continue

        for post in posts:
            pid = post.get("id", "")
            title = post.get("title", "")
            if pid and title:
                titles[pid] = title

        fetched += len(batch)

        if fetched % 500 == 0 or fetched == total:
            print(f"\r  Fetched {fetched}/{total} ({len(titles)} titles found, {failed} failed)",
                  end="", flush=True)
            # Periodic save
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(titles, f, ensure_ascii=False)

        time.sleep(REQUEST_DELAY)

    # Final save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(titles, f, indent=2, ensure_ascii=False)

    print(f"\n\nDone: {len(titles)} titles saved to {args.output}")
    print(f"  Fetched: {fetched}, Failed: {failed}, From resume: {len(existing)}")


if __name__ == "__main__":
    main()
