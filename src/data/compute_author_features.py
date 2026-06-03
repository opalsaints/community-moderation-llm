#!/usr/bin/env python3
"""
Compute temporal author features for each comment in a subreddit dataset.

For each comment, computes features based ONLY on the author's activity
BEFORE that comment's timestamp (no label leakage, no temporal leakage).

Features computed:
  1. n_prior:        author's total prior comments in this subreddit
  2. vel_24h:        author's comments in the 24 hours before this comment
  3. avg_score:      average score of author's prior comments
  4. unique_threads: number of distinct threads the author posted in before
  5. days_active:    days since the author's first comment in this sub
  6. is_first:       boolean, True if this is the author's first comment
  7. max_in_thread:  most comments by this author in any single prior thread

Input:  raw extracted JSON (from process_zst_dump.py) containing ALL comments
Output: JSON mapping comment_id -> {7 features}

Usage:
    python compute_author_features.py \
        --input ./extracted/changemyview.json \
        --output ./enrichment/changemyview_author_features.json

    python compute_author_features.py \
        --input ./extracted/changemyview.json \
        --output ./enrichment/changemyview_author_features.json \
        --dataset-ids ./dataset/changemyview/random_split/test.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def compute_features(all_comments, target_ids=None):
    """Compute temporal author features for each comment.

    Args:
        all_comments: list of comment dicts, will be sorted by created_utc
        target_ids: if set, only compute features for these comment IDs
                    (still uses ALL comments to build author histories)

    Returns:
        dict mapping comment_id -> feature dict
    """
    # Sort chronologically
    all_comments.sort(key=lambda c: c.get("created_utc", 0))

    # Running author state
    # For each author, track: list of (timestamp, score, link_id)
    author_history = defaultdict(list)

    features = {}
    total = len(all_comments)

    for i, comment in enumerate(all_comments):
        cid = comment.get("id", "")
        author = comment.get("author", "")
        timestamp = comment.get("created_utc", 0)
        score = comment.get("score", 1)
        link_id = comment.get("link_id", "")

        # Skip if we only want specific IDs and this isn't one
        if target_ids is not None and cid not in target_ids:
            # Still record in history for future lookups
            author_history[author].append((timestamp, score, link_id))
            continue

        # Compute features from author's history BEFORE this comment
        history = author_history[author]

        if not history:
            # First comment by this author
            feat = {
                "n_prior": 0,
                "vel_24h": 0,
                "avg_score": 0.0,
                "unique_threads": 0,
                "days_active": 0,
                "is_first": True,
                "max_in_thread": 0,
            }
        else:
            n_prior = len(history)

            # Velocity: comments in prior 24h
            cutoff_24h = timestamp - 86400
            vel_24h = sum(1 for ts, _, _ in history if ts > cutoff_24h)

            # Average score
            scores = [s for _, s, _ in history]
            avg_score = round(sum(scores) / len(scores), 1)

            # Unique threads
            threads = set(lid for _, _, lid in history)
            unique_threads = len(threads)

            # Days active
            first_ts = history[0][0]
            days_active = round((timestamp - first_ts) / 86400, 1)

            # Max comments in any single thread
            thread_counts = defaultdict(int)
            for _, _, lid in history:
                thread_counts[lid] += 1
            max_in_thread = max(thread_counts.values()) if thread_counts else 0

            feat = {
                "n_prior": n_prior,
                "vel_24h": vel_24h,
                "avg_score": avg_score,
                "unique_threads": unique_threads,
                "days_active": days_active,
                "is_first": False,
                "max_in_thread": max_in_thread,
            }

        features[cid] = feat

        # Add this comment to history AFTER computing features
        author_history[author].append((timestamp, score, link_id))

        # Progress
        if (i + 1) % 100000 == 0:
            print(f"\r  Processed {i+1}/{total} comments, {len(features)} features computed",
                  end="", flush=True)

    print(f"\r  Processed {total}/{total} comments, {len(features)} features computed")
    return features


def load_target_ids(dataset_paths):
    """Load comment IDs from dataset files to limit feature computation."""
    ids = set()
    for fp in dataset_paths:
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                cid = d.get("id", "")
                if cid:
                    ids.add(cid)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Compute temporal author features")
    parser.add_argument("--input", required=True,
                        help="Raw extracted JSON file (from process_zst_dump.py)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--dataset-ids", nargs="*", default=None,
                        help="Optional: only compute features for comment IDs in these JSONL files "
                             "(still uses ALL comments for history building)")
    args = parser.parse_args()

    # Load raw data
    print(f"Loading {args.input}...", flush=True)
    with open(args.input) as f:
        data = json.load(f)

    all_comments = data.get("removed", []) + data.get("approved", [])
    print(f"  {len(all_comments)} total comments")

    # Optionally limit which IDs get features
    target_ids = None
    if args.dataset_ids:
        target_ids = load_target_ids(args.dataset_ids)
        print(f"  Computing features for {len(target_ids)} target comment IDs")

    # Compute features
    print("Computing temporal author features...", flush=True)
    features = compute_features(all_comments, target_ids)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(features, f, indent=2)

    print(f"\nSaved {len(features)} feature records to {args.output}")

    # Summary stats
    if features:
        n_first = sum(1 for v in features.values() if v["is_first"])
        avg_prior = sum(v["n_prior"] for v in features.values()) / len(features)
        max_vel = max(v["vel_24h"] for v in features.values())
        print(f"  First-time commenters: {n_first} ({100*n_first/len(features):.0f}%)")
        print(f"  Avg prior comments: {avg_prior:.1f}")
        print(f"  Max 24h velocity: {max_vel}")


if __name__ == "__main__":
    main()
