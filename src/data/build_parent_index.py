#!/usr/bin/env python3
"""
Build parent context index from downloaded NDJSON files.

Reads the raw NDJSON files (from download_subreddit.py or process_zst_dump.py)
and creates a parent index mapping comment_id -> body for all comments with
readable text. This index is then used by build_dataset.py to resolve
parent_id references and provide thread context to models.

Usage:
    python build_parent_index.py --input-dir data/raw/ --output-dir data/extracted/
    python build_parent_index.py --input-dir data/raw/ --output-dir data/extracted/ --subs AskHistorians,science
"""

import argparse
import json
import os
import sys
from pathlib import Path


def build_index_from_ndjson(filepath):
    """Build parent index and extract removed/approved from an NDJSON file.

    Returns:
        removed: list of captured_removed comments
        approved: list of approved comments
        parent_index: dict of comment_id -> body
        stats: dict of category counts
    """
    parent_index = {}
    removed = []
    approved = []
    stats = {"total": 0, "captured_removed": 0, "approved": 0,
             "removed_no_text": 0, "deleted": 0, "other": 0}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                comment = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["total"] += 1
            body = comment.get("body", "")
            comment_id = comment.get("id", "")
            meta = comment.get("_meta", {}) or {}
            removal_type = meta.get("removal_type")

            # Index all comments with readable text
            if comment_id and body and body not in ("[removed]", "[deleted]", ""):
                parent_index[comment_id] = body

            # Classify
            if body == "[removed]":
                stats["removed_no_text"] += 1
            elif body == "[deleted]":
                stats["deleted"] += 1
            elif removal_type == "removed":
                stats["captured_removed"] += 1
                removed.append(comment)
            elif removal_type == "deleted" or meta.get("was_deleted_later"):
                stats["other"] += 1
            else:
                stats["approved"] += 1
                approved.append(comment)

    return removed, approved, parent_index, stats


def main():
    parser = argparse.ArgumentParser(description="Build parent index from NDJSON files")
    parser.add_argument("--input-dir", default="data/raw", help="Directory with .ndjson files")
    parser.add_argument("--output-dir", default="data/extracted", help="Output directory")
    parser.add_argument("--subs", type=str, default=None, help="Comma-separated subreddits (default: all)")
    parser.add_argument("--config", default="configs/subreddits.json", help="Config file for --all")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subs:
        subreddits = [s.strip() for s in args.subs.split(",")]
    else:
        # Find all .ndjson files in input dir
        subreddits = [f.stem for f in input_dir.glob("*.ndjson")]

    if not subreddits:
        print("No subreddits found. Download data first with download_subreddit.py")
        sys.exit(1)

    print(f"Processing {len(subreddits)} subreddits")
    print(f"Input: {input_dir}/")
    print(f"Output: {output_dir}/")
    print()

    for sub in sorted(subreddits):
        ndjson_file = input_dir / f"{sub}.ndjson"
        if not ndjson_file.exists():
            print(f"  {sub}: no NDJSON file found, skipping")
            continue

        print(f"  Processing r/{sub}...", end=" ", flush=True)
        removed, approved, parent_index, stats = build_index_from_ndjson(ndjson_file)

        # Save in the same format that build_dataset.py expects
        # 1. Extracted comments (removed + approved)
        extracted_file = output_dir / f"{sub}.json"
        with open(extracted_file, "w") as f:
            json.dump({"removed": removed, "approved": approved}, f)

        # 2. Parent index
        index_file = output_dir / f"{sub}_parent_index.json"
        with open(index_file, "w") as f:
            json.dump(parent_index, f)

        cr = stats["captured_removed"]
        ap = stats["approved"]
        rate = cr / stats["total"] if stats["total"] > 0 else 0
        print(f"{stats['total']:,} total, {cr:,} removed ({rate:.1%}), "
              f"{ap:,} approved, {len(parent_index):,} indexed")

    print("\nDone. Run build_dataset.py --input-dir", output_dir, "to create train/test splits.")


if __name__ == "__main__":
    main()
