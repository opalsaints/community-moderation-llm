#!/usr/bin/env python3
"""
Process Arctic Shift .zst monthly comment dumps.
Streams NDJSON without fully decompressing to disk.

Usage:
    python process_zst_dump.py scan RC_2024-06.zst --subreddits science,Conservative,de
    python process_zst_dump.py extract RC_2024-06.zst --subreddits science --output-dir ./extracted/
    python process_zst_dump.py stats RC_2024-06.zst

Requires: pip install zstandard
"""

import argparse
import json
import sys
import os
import time
from collections import defaultdict

try:
    import zstandard as zstd
except ImportError:
    print("Install zstandard: pip install zstandard")
    sys.exit(1)


def stream_comments(filepath):
    """Stream comments from a .zst NDJSON file without decompressing to disk."""
    dctx = zstd.ZstdDecompressor()
    with open(filepath, "rb") as fh:
        reader = dctx.stream_reader(fh)
        buffer = b""
        while True:
            chunk = reader.read(2 ** 22)  # 4MB chunks
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split(b"\n")
            buffer = lines[-1]  # keep incomplete last line
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        # process remaining buffer
        if buffer.strip():
            try:
                yield json.loads(buffer)
            except json.JSONDecodeError:
                pass


def classify_comment(comment):
    """Classify a comment into categories based on removal status."""
    body = comment.get("body", "")
    meta = comment.get("_meta", {})
    removal_type = meta.get("removal_type")

    if body == "[removed]":
        return "removed_no_text"
    elif body == "[deleted]":
        return "deleted_no_text"
    elif removal_type == "removed":
        return "captured_removed"  # gold: has text + mod-removed
    elif removal_type == "deleted" or meta.get("was_deleted_later"):
        return "captured_deleted"  # user-deleted, has text
    else:
        return "approved"


def cmd_scan(args):
    """Scan a dump file and report per-subreddit statistics."""
    target_subs = None
    if args.subreddits:
        target_subs = set(s.strip().lower() for s in args.subreddits.split(","))

    stats = defaultdict(lambda: defaultdict(int))
    total = 0
    start = time.time()

    for comment in stream_comments(args.file):
        total += 1
        sub = comment.get("subreddit", "unknown")

        if target_subs and sub.lower() not in target_subs:
            continue

        category = classify_comment(comment)
        stats[sub][category] += 1
        stats[sub]["total"] += 1

        if total % 500_000 == 0:
            elapsed = time.time() - start
            print(f"  ... processed {total:,} comments ({elapsed:.0f}s)", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\nProcessed {total:,} comments in {elapsed:.1f}s\n")

    # Print results
    header = f"{'Subreddit':<25} {'Total':>8} {'Approved':>9} {'CaptRem':>8} {'CaptDel':>8} {'NoText':>8} {'Usable%':>8}"
    print(header)
    print("-" * len(header))

    for sub in sorted(stats.keys(), key=lambda s: stats[s]["total"], reverse=True):
        s = stats[sub]
        t = s["total"]
        cr = s["captured_removed"]
        usable_pct = f"{cr/t*100:.1f}%" if t > 0 else "N/A"
        print(f"{sub:<25} {t:>8} {s['approved']:>9} {cr:>8} {s['captured_deleted']:>8} {s['removed_no_text']:>8} {usable_pct:>8}")

    # Save to JSON
    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(stats), f, indent=2)
        print(f"\nSaved to {args.output}")


def cmd_extract(args):
    """Extract usable comments (captured_removed + approved) for target subreddits.

    Also builds a parent context index: maps comment IDs to body text for all
    comments with readable text. This allows resolving parent_id references
    later in build_dataset.py to provide thread context to models.
    """
    target_subs = set(s.strip().lower() for s in args.subreddits.split(","))
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect comments per subreddit
    collected = defaultdict(lambda: {"removed": [], "approved": []})
    # Parent context index: comment_id -> body (for resolving parent_id later)
    parent_index = defaultdict(dict)
    total = 0
    start = time.time()

    for comment in stream_comments(args.file):
        total += 1
        sub = comment.get("subreddit", "unknown")

        if sub.lower() not in target_subs:
            continue

        body = comment.get("body", "")
        comment_id = comment.get("id", "")

        # Index ALL comments with readable text for parent resolution
        if comment_id and body and body not in ("[removed]", "[deleted]", ""):
            parent_index[sub][comment_id] = body

        category = classify_comment(comment)

        if category == "captured_removed":
            collected[sub]["removed"].append(comment)
        elif category == "approved":
            collected[sub]["approved"].append(comment)

        if total % 500_000 == 0:
            elapsed = time.time() - start
            print(f"  ... processed {total:,} comments ({elapsed:.0f}s)", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\nProcessed {total:,} comments in {elapsed:.1f}s\n")

    for sub, data in collected.items():
        outfile = os.path.join(args.output_dir, f"{sub}.json")
        # Append to existing file if it exists (for multi-month processing)
        if os.path.exists(outfile):
            with open(outfile) as f:
                existing = json.load(f)
            existing["removed"].extend(data["removed"])
            existing["approved"].extend(data["approved"])
            data = existing
        with open(outfile, "w") as f:
            json.dump(data, f)
        print(f"{sub}: {len(data['removed'])} removed, {len(data['approved'])} approved -> {outfile}")

        # Save parent context index
        index_file = os.path.join(args.output_dir, f"{sub}_parent_index.json")
        existing_index = {}
        if os.path.exists(index_file):
            with open(index_file) as f:
                existing_index = json.load(f)
        existing_index.update(parent_index.get(sub, {}))
        with open(index_file, "w") as f:
            json.dump(existing_index, f)
        print(f"  Parent index: {len(existing_index):,} entries -> {index_file}")


def cmd_stats(args):
    """Quick overall stats for a dump file (no subreddit filter)."""
    categories = defaultdict(int)
    total = 0
    start = time.time()

    for comment in stream_comments(args.file):
        total += 1
        categories[classify_comment(comment)] += 1

        if total % 1_000_000 == 0:
            elapsed = time.time() - start
            print(f"  ... {total:,} comments ({elapsed:.0f}s)", file=sys.stderr)

    elapsed = time.time() - start
    print(f"\nTotal: {total:,} comments in {elapsed:.1f}s")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:,} ({count/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Process Arctic Shift .zst dumps")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    scan_p = subparsers.add_parser("scan", help="Per-subreddit statistics")
    scan_p.add_argument("file", help="Path to .zst file")
    scan_p.add_argument("--subreddits", help="Comma-separated list of subreddits to track")
    scan_p.add_argument("--output", help="Save stats to JSON file")

    # extract
    ext_p = subparsers.add_parser("extract", help="Extract usable comments")
    ext_p.add_argument("file", help="Path to .zst file")
    ext_p.add_argument("--subreddits", required=True, help="Comma-separated subreddits")
    ext_p.add_argument("--output-dir", default="./extracted", help="Output directory")

    # stats
    stats_p = subparsers.add_parser("stats", help="Quick overall stats")
    stats_p.add_argument("file", help="Path to .zst file")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
