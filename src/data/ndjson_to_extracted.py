#!/usr/bin/env python3
"""
Bridge: download_subreddit.py NDJSON output -> build_dataset.py input format.

download_subreddit.py writes one NDJSON file per subreddit (one JSON object
per line) from the Arctic Shift API. build_dataset.py expects the same
{sub}.json format that process_zst_dump.py produces from .zst dumps:

    {
      "removed":  [comment_obj, ...],   # gold: captured text + mod-removed
      "approved": [comment_obj, ...]    # normal comments
    }

plus a parent index file {sub}_parent_index.json mapping comment_id -> body
for parent-context resolution in build_dataset.py.

Classification logic is a direct port of process_zst_dump.py:57-72.

Usage:
    # Single sub:
    python ndjson_to_extracted.py --input-dir ~/data/raw_2026 \\
        --output-dir ~/data/extracted_2026 --subs changemyview

    # All NDJSON files in the input dir:
    python ndjson_to_extracted.py --input-dir ~/data/raw_2026 \\
        --output-dir ~/data/extracted_2026 --all
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path


def stream_ndjson(filepath):
    """Yield one JSON object per line from an NDJSON file."""
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def classify_comment(comment):
    """Port of process_zst_dump.py:57-72. Classify by removal status.

    Returns one of:
      - "captured_removed": has readable body AND mod-removed (gold)
      - "approved": normal live comment
      - "removed_no_text": body=="[removed]" (original text lost, skip)
      - "deleted_no_text": body=="[deleted]" (user self-deleted, skip)
      - "captured_deleted": has readable body but user-deleted (skip)
    """
    body = comment.get("body", "")
    meta = comment.get("_meta", {}) or {}
    removal_type = meta.get("removal_type")

    if body == "[removed]":
        return "removed_no_text"
    if body == "[deleted]":
        return "deleted_no_text"
    if removal_type == "removed":
        return "captured_removed"
    if removal_type == "deleted" or meta.get("was_deleted_later"):
        return "captured_deleted"
    return "approved"


def process_sub(ndjson_path, output_dir, approved_reservoir_size, seed):
    """Process one subreddit's NDJSON file.

    Writes {sub}.json and {sub}_parent_index.json to output_dir.
    Returns a dict of per-category counts.

    Memory: keeps ALL captured_removed (rare, every one is precious) and a
    fixed-size reservoir sample of the approved class using Algorithm R
    (Vitter 1985). The reservoir is seeded so reruns are deterministic.
    This bounds peak memory regardless of raw NDJSON volume and produces a
    time-uniform sample of the approved class across the download window.
    """
    sub = ndjson_path.stem  # e.g. "changemyview"
    removed = []
    approved_reservoir = []
    approved_seen = 0
    parent_index = {}
    counts = defaultdict(int)
    rng = random.Random(seed)

    start = time.time()
    for comment in stream_ndjson(ndjson_path):
        counts["total"] += 1
        category = classify_comment(comment)
        counts[category] += 1

        # Index any comment with readable text for parent-context resolution
        body = comment.get("body", "")
        comment_id = comment.get("id", "")
        if comment_id and body and body not in ("[removed]", "[deleted]", ""):
            parent_index[comment_id] = body

        if category == "captured_removed":
            removed.append(comment)
        elif category == "approved":
            # Reservoir sampling (Algorithm R) for the approved class
            if len(approved_reservoir) < approved_reservoir_size:
                approved_reservoir.append(comment)
            else:
                j = rng.randint(0, approved_seen)
                if j < approved_reservoir_size:
                    approved_reservoir[j] = comment
            approved_seen += 1
        # removed_no_text / deleted_no_text / captured_deleted: drop

        if counts["total"] % 100_000 == 0:
            elapsed = time.time() - start
            print(f"    ... processed {counts['total']:,} comments "
                  f"({elapsed:.0f}s)", file=sys.stderr)

    elapsed = time.time() - start

    # Write {sub}.json
    sub_file = Path(output_dir) / f"{sub}.json"
    with open(sub_file, "w") as f:
        json.dump({"removed": removed, "approved": approved_reservoir}, f)

    # Write {sub}_parent_index.json
    index_file = Path(output_dir) / f"{sub}_parent_index.json"
    with open(index_file, "w") as f:
        json.dump(parent_index, f)

    # Record reservoir metadata for downstream verification
    counts["approved_seen"] = approved_seen
    counts["approved_written"] = len(approved_reservoir)
    counts["approved_reservoir_size"] = approved_reservoir_size

    print(f"  {sub}: {counts['total']:,} total in {elapsed:.1f}s")
    print(f"    -> captured_removed: {counts['captured_removed']:,}")
    print(f"    -> approved (seen):    {approved_seen:,}")
    print(f"    -> approved (written): {len(approved_reservoir):,} "
          f"(reservoir cap {approved_reservoir_size:,})")
    print(f"    -> dropped: {counts['removed_no_text']:,} no_text + "
          f"{counts['deleted_no_text']:,} deleted + "
          f"{counts['captured_deleted']:,} user_deleted")
    print(f"    -> parent index:     {len(parent_index):,} entries")
    print(f"    -> {sub_file}")

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Convert download_subreddit.py NDJSON output to build_dataset.py input format"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing {sub}.ndjson files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for {sub}.json and {sub}_parent_index.json output")
    parser.add_argument("--subs", default=None,
                        help="Comma-separated list of subs to process (default: all .ndjson in input-dir)")
    parser.add_argument("--all", action="store_true",
                        help="Process all .ndjson files in input-dir")
    parser.add_argument("--approved-reservoir", type=int, default=200_000,
                        help="Max approved comments kept per sub via reservoir "
                             "sampling (default: 200000). build_dataset.py caps "
                             "at 5K per class, so 200K leaves ~40x headroom.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the reservoir sampler (default: 42)")
    args = parser.parse_args()

    input_dir = Path(os.path.expanduser(args.input_dir))
    output_dir = Path(os.path.expanduser(args.output_dir))

    if not input_dir.is_dir():
        parser.error(f"--input-dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect subs to process
    if args.subs:
        sub_names = [s.strip() for s in args.subs.split(",")]
        ndjson_files = [input_dir / f"{s}.ndjson" for s in sub_names]
        missing = [p for p in ndjson_files if not p.exists()]
        if missing:
            parser.error(f"Missing NDJSON files: {[str(p) for p in missing]}")
    elif args.all:
        ndjson_files = sorted(input_dir.glob("*.ndjson"))
        if not ndjson_files:
            parser.error(f"No .ndjson files found in {input_dir}")
    else:
        parser.error("Provide --subs LIST or --all")

    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Subs:       {len(ndjson_files)}")
    print()

    all_stats = {}
    for ndjson_path in ndjson_files:
        sub = ndjson_path.stem
        print(f"Processing r/{sub}...")
        counts = process_sub(ndjson_path, output_dir,
                             args.approved_reservoir, args.seed)
        all_stats[sub] = counts
        print()

    # Summary table
    print("=" * 70)
    print(f"{'Subreddit':<25} {'Total':>10} {'Removed':>10} {'Approved':>10} {'Usable%':>9}")
    print("-" * 70)
    for sub, c in sorted(all_stats.items(), key=lambda kv: kv[1].get("total", 0), reverse=True):
        total = c.get("total", 0)
        cr = c.get("captured_removed", 0)
        ap = c.get("approved", 0)
        pct = f"{cr / total * 100:.1f}%" if total > 0 else "N/A"
        print(f"{sub:<25} {total:>10,} {cr:>10,} {ap:>10,} {pct:>9}")

    # Save combined stats
    stats_file = output_dir / "extract_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nCombined stats: {stats_file}")


if __name__ == "__main__":
    main()
