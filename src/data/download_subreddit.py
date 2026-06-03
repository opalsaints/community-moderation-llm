#!/usr/bin/env python3
"""
Download all comments for a subreddit from Arctic Shift API.

Paginates through the search API (100 comments per request) and saves
all comments with full metadata including _meta fields for removal detection.

Usage:
    python download_subreddit.py AskHistorians --after 2024-01-01 --before 2025-01-01
    python download_subreddit.py --all --after 2024-01-01 --before 2025-01-01
    python download_subreddit.py --all --after 2024-01-01 --before 2025-01-01 --output-dir data/raw/

Output: NDJSON file per subreddit (one JSON object per line), same format as .zst dumps.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

API_BASE = "https://arctic-shift.photon-reddit.com/api/comments/search"
BATCH_SIZE = 100
REQUEST_DELAY = 0.5  # seconds between requests (be polite)


def api_get(params, max_retries=5):
    """Make a GET request to the Arctic Shift comments search API."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API_BASE}?{query}"
    req = Request(url, headers={"User-Agent": "capstone-research/1.0"})

    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                if data.get("error"):
                    if "Timeout" in str(data["error"]) or "slow down" in str(data["error"]):
                        wait = 15 * (attempt + 1)
                        print(f"\n    Rate limited, waiting {wait}s...", end="", flush=True)
                        time.sleep(wait)
                        continue
                    return data, data["error"]
                return data, None
        except (URLError, HTTPError, TimeoutError, OSError, ConnectionError) as e:
            wait = 10 * (2 ** attempt)  # exponential backoff: 10, 20, 40, 80, 160s
            if attempt < max_retries - 1:
                print(f"\n    Network error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            return None, str(e)
    return None, "Max retries exceeded"


def download_subreddit(subreddit, after, before, output_dir, max_comments=None):
    """Download all comments for a subreddit in a time range."""
    output_file = os.path.join(output_dir, f"{subreddit}.ndjson")

    # Resume support: check if file exists and find last timestamp
    resume_after = after
    existing_count = 0
    if os.path.exists(output_file):
        # Count existing lines and find the last timestamp
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                existing_count += 1
                try:
                    comment = json.loads(line)
                    ts = comment.get("created_utc", 0)
                    if ts:
                        resume_after = str(ts)
                except json.JSONDecodeError:
                    pass

        if existing_count > 0:
            print(f"  Resuming from {existing_count} existing comments (after={resume_after})")

    # Open in append mode for resume support
    total = existing_count
    empty_batches = 0

    with open(output_file, "a") as f:
        current_after = resume_after

        while True:
            params = {
                "subreddit": subreddit,
                "after": current_after,
                "before": before,
                "limit": BATCH_SIZE,
                "sort": "asc",
            }

            data, error = api_get(params)

            if error:
                print(f"\n  Error at after={current_after}: {error}")
                # Flush what we have and stop this subreddit
                f.flush()
                break

            comments = data.get("data", [])
            if not comments:
                empty_batches += 1
                if empty_batches >= 2:
                    break
                time.sleep(2)
                continue

            empty_batches = 0

            for comment in comments:
                f.write(json.dumps(comment, ensure_ascii=False) + "\n")

            total += len(comments)
            last_ts = comments[-1].get("created_utc", 0)
            current_after = str(last_ts)

            # Check cap
            if max_comments and total >= max_comments:
                f.flush()
                print(f"\n  Reached cap of {max_comments:,} comments")
                break

            # Progress update + periodic flush
            if total % 1000 == 0 or len(comments) < BATCH_SIZE:
                f.flush()
                ts_str = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d") if last_ts else "?"
                print(f"\r  {subreddit}: {total:,} comments (through {ts_str})", end="", flush=True)

            time.sleep(REQUEST_DELAY)

    print(f"\r  {subreddit}: {total:,} comments total -> {output_file}")
    return total


def classify_comment(comment):
    """Quick classification for progress reporting."""
    body = comment.get("body", "")
    meta = comment.get("_meta", {}) or {}
    if body == "[removed]":
        return "removed_no_text"
    if meta.get("removal_type") == "removed":
        return "captured_removed"
    if body == "[deleted]":
        return "deleted"
    return "approved"


def report_stats(filepath):
    """Quick stats on downloaded file."""
    cats = {"captured_removed": 0, "approved": 0, "removed_no_text": 0, "deleted": 0}
    total = 0
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                cat = classify_comment(c)
                cats[cat] = cats.get(cat, 0) + 1
                total += 1
            except:
                pass

    cr = cats["captured_removed"]
    ap = cats["approved"]
    rate = cr / total if total > 0 else 0
    print(f"    Stats: {total:,} total, {cr:,} captured_removed ({rate:.1%}), {ap:,} approved")
    return cats


def main():
    parser = argparse.ArgumentParser(description="Download subreddit comments from Arctic Shift")
    parser.add_argument("subreddit", nargs="?", help="Subreddit name")
    parser.add_argument("--all", action="store_true", help="Download all subreddits from config")
    parser.add_argument("--config", default="configs/subreddits.json", help="Config file")
    parser.add_argument("--after", default="2024-01-01", help="Start date (default: 2024-01-01)")
    parser.add_argument("--before", default="2025-01-01", help="End date (default: 2025-01-01)")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--max-per-sub", type=int, default=750000,
                        help="Max comments per subreddit (default: 750K, enough for 5K removed at 1%% rate)")
    parser.add_argument("--stats-only", action="store_true", help="Just report stats on existing files")
    args = parser.parse_args()

    if args.all:
        with open(args.config) as f:
            config = json.load(f)
        subreddits = config.get("subreddits") or (config.get("seen", []) + config.get("unseen", []))
    elif args.subreddit:
        subreddits = [args.subreddit]
    else:
        parser.error("Provide a subreddit name or --all")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.stats_only:
        for sub in subreddits:
            fp = os.path.join(args.output_dir, f"{sub}.ndjson")
            if os.path.exists(fp):
                print(f"r/{sub}:")
                report_stats(fp)
            else:
                print(f"r/{sub}: no data")
        return

    print(f"Downloading {len(subreddits)} subreddits")
    print(f"Period: {args.after} to {args.before}")
    print(f"Output: {args.output_dir}/")
    print()

    print(f"Max per sub: {args.max_per_sub:,}")
    print()

    failed = []
    for sub in subreddits:
        try:
            total = download_subreddit(sub, args.after, args.before, args.output_dir,
                                       max_comments=args.max_per_sub)
            if total > 0:
                report_stats(os.path.join(args.output_dir, f"{sub}.ndjson"))
        except Exception as e:
            print(f"\n  FAILED r/{sub}: {e}")
            failed.append(sub)
        print()

    if failed:
        print(f"\nFailed subreddits ({len(failed)}): {', '.join(failed)}")
        print("Re-run with resume support to retry these.")


if __name__ == "__main__":
    main()
