#!/usr/bin/env python3
"""
Enrich v2: Add all missing features to the existing enriched dataset.

Adds the following feature groups:
  1. Post metadata (from Arctic Shift): time_since_post, thread_size, post_score, post_upvote_ratio
  2. Interaction features: score_density, velocity_ratio, reply_length_ratio
  3. Text-derived features: caps_ratio, exclamation_count, question_count,
     avg_word_length, unique_word_ratio, sentence_count, has_profanity_signal
  4. Account age proxy: account_age_proxy (from author_fullname base36 ID)
  5. Controversiality: already in data, just ensure it's in feature set
  6. Thread approximations: thread_comment_count (from our own data)

Usage:
    python enrich_v2.py \
        --dataset-dir ~/data/dataset/changemyview/enriched \
        --account-ages ~/data/enrichment/changemyview_account_ages.json \
        --output-dir ~/data/dataset/changemyview/enriched_v2

    # Skip API fetch if post metadata already saved:
    python enrich_v2.py \
        --dataset-dir ~/data/dataset/changemyview/enriched \
        --account-ages ~/data/enrichment/changemyview_account_ages.json \
        --post-metadata ~/data/enrichment/changemyview_post_metadata.json \
        --output-dir ~/data/dataset/changemyview/enriched_v2
"""

import argparse
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# Arctic Shift API
API_IDS = "https://arctic-shift.photon-reddit.com/api/posts/ids"
BATCH_SIZE = 100
REQUEST_DELAY = 1.0


# =============================================================================
# 1. Post metadata fetching
# =============================================================================

def fetch_posts_by_ids(post_ids, max_retries=3):
    """Fetch posts by IDs from Arctic Shift."""
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


def fetch_all_post_metadata(link_ids, output_path=None):
    """Fetch metadata for all unique posts. Returns dict: post_id -> metadata."""
    # Check for existing data
    existing = {}
    if output_path and os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing post metadata entries")

    # Find missing
    needed = [pid for pid in link_ids if pid not in existing]
    if not needed:
        print(f"  All {len(link_ids)} posts already fetched")
        return existing

    print(f"  Need to fetch {len(needed)} posts ({len(existing)} cached)")

    batches = [needed[i:i+BATCH_SIZE] for i in range(0, len(needed), BATCH_SIZE)]
    fetched = 0
    errors = 0

    for batch_idx, batch in enumerate(batches):
        posts, err = fetch_posts_by_ids(batch)
        if err:
            errors += 1
            print(f"  Batch {batch_idx+1}/{len(batches)}: ERROR {err}")
        else:
            for p in posts:
                pid = p.get("id", "")
                existing[pid] = {
                    "created_utc": p.get("created_utc"),
                    "num_comments": p.get("num_comments"),
                    "score": p.get("score"),
                    "upvote_ratio": p.get("upvote_ratio"),
                    "author": p.get("author"),
                    "over_18": p.get("over_18", False),
                }
                fetched += 1

        if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
            print(f"  Batch {batch_idx+1}/{len(batches)}: "
                  f"{fetched} fetched, {errors} errors", flush=True)

        if batch_idx < len(batches) - 1:
            time.sleep(REQUEST_DELAY)

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Saved {len(existing)} post metadata entries to {output_path}")

    return existing


# =============================================================================
# 2. Text-derived features
# =============================================================================

# Simple profanity/toxicity signal words (not exhaustive, just directional)
PROFANITY_PATTERNS = [
    r'\bfuck\w*\b', r'\bshit\w*\b', r'\bass\b', r'\basshole\w*\b',
    r'\bbitch\w*\b', r'\bdamn\w*\b', r'\bcrap\b', r'\bhell\b',
    r'\bstfu\b', r'\bstfu\b', r'\bwtf\b', r'\blmao\b', r'\blmfao\b',
    r'\bidiot\w*\b', r'\bstupid\b', r'\bmoron\w*\b', r'\bdumb\w*\b',
    r'\btroll\w*\b', r'\bkill\s+your', r'\bdie\b', r'\bkys\b',
    r'\bretard\w*\b', r'\bcringe\b',
]
PROFANITY_RE = re.compile('|'.join(PROFANITY_PATTERNS), re.IGNORECASE)


def compute_text_features(text):
    """Compute text-derived features from comment body."""
    if not text or not text.strip():
        return {
            "caps_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "avg_word_length": 0.0,
            "unique_word_ratio": 0.0,
            "sentence_count": 0,
            "profanity_count": 0,
            "has_profanity_signal": False,
            "paragraph_count": 1,
            "starts_with_quote": False,
            "link_count": 0,
        }

    # Character-level
    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    caps_ratio = len(upper_chars) / len(alpha_chars) if alpha_chars else 0.0

    exclamation_count = text.count("!")
    question_count = text.count("?")

    # Word-level
    words = text.split()
    word_lengths = [len(w.strip(string.punctuation)) for w in words]
    word_lengths = [wl for wl in word_lengths if wl > 0]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0.0

    lower_words = [w.lower().strip(string.punctuation) for w in words]
    lower_words = [w for w in lower_words if w]
    unique_word_ratio = len(set(lower_words)) / len(lower_words) if lower_words else 0.0

    # Sentence-level (approximate)
    sentence_count = len(re.split(r'[.!?]+', text.strip()))

    # Profanity
    profanity_matches = PROFANITY_RE.findall(text)
    profanity_count = len(profanity_matches)

    # Structure
    paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
    starts_with_quote = text.lstrip().startswith(">")
    link_count = len(re.findall(r'https?://', text))

    return {
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "avg_word_length": round(avg_word_length, 2),
        "unique_word_ratio": round(unique_word_ratio, 4),
        "sentence_count": sentence_count,
        "profanity_count": profanity_count,
        "has_profanity_signal": profanity_count > 0,
        "paragraph_count": paragraph_count,
        "starts_with_quote": starts_with_quote,
        "link_count": link_count,
    }


# =============================================================================
# 4. Main enrichment
# =============================================================================

def enrich_record(rec, post_metadata, account_ages, thread_counts):
    """Add all v2 features to a single record.

    CRITICAL: All temporal features are RELATIVE to the comment's own timestamp.
    No absolute timestamps are added as features. This prevents temporal leakage
    and makes features interpretable to the model.
    """
    enriched = dict(rec)
    comment_created = rec.get("created_utc", 0)

    # --- Post metadata features (RELATIVE to comment time) ---
    link_id = rec.get("link_id", "").replace("t3_", "")
    pmeta = post_metadata.get(link_id, {})

    post_created = pmeta.get("created_utc")
    if post_created and comment_created:
        enriched["time_since_post_hours"] = round(
            (comment_created - post_created) / 3600, 2
        )
    else:
        enriched["time_since_post_hours"] = -1  # missing

    # NOTE: thread_size_api and post_score reflect FINAL state, not state at
    # comment time. These are flagged as potentially leaky. The local thread
    # count is a safer proxy (only counts comments in our dataset).
    enriched["thread_size_api"] = pmeta.get("num_comments", 0) or 0
    enriched["post_is_nsfw"] = pmeta.get("over_18", False)

    # --- Thread count from our own data (safer, no future leakage) ---
    enriched["thread_comment_count_local"] = thread_counts.get(link_id, 0)

    # --- Account age (RELATIVE to comment time) ---
    author = rec.get("author", "")
    acct_info = account_ages.get(author, {})
    acct_created = acct_info.get("account_created_utc")
    if acct_created and comment_created and acct_created < comment_created:
        enriched["account_age_days"] = round(
            (comment_created - acct_created) / 86400, 1
        )
    else:
        enriched["account_age_days"] = -1  # unknown

    # --- Interaction features ---
    word_count = rec.get("word_count", 1)
    score = rec.get("score", 0)
    enriched["score_per_word"] = round(score / max(word_count, 1), 4)

    n_prior = rec.get("author_n_prior", 0)
    vel_24h = rec.get("author_vel_24h", 0)
    enriched["velocity_ratio"] = round(
        vel_24h / max(n_prior, 1), 4
    )

    parent_wc = rec.get("parent_word_count", 0)
    enriched["reply_length_ratio"] = round(
        word_count / max(parent_wc, 1), 4
    ) if parent_wc > 0 else 0.0

    # --- Text features ---
    body = rec.get("body", "")
    text_feats = compute_text_features(body)
    for k, v in text_feats.items():
        enriched[k] = v

    # --- Controversiality (already in data, ensure clean) ---
    enriched["controversiality"] = rec.get("controversiality", 0)

    # --- Time-of-day features (cyclical, no absolute date exposed) ---
    if comment_created:
        import datetime
        dt = datetime.datetime.fromtimestamp(comment_created, tz=datetime.timezone.utc)
        enriched["hour_of_day"] = dt.hour
        enriched["day_of_week"] = dt.weekday()  # 0=Mon, 6=Sun
        enriched["is_weekend"] = dt.weekday() >= 5
    else:
        enriched["hour_of_day"] = -1
        enriched["day_of_week"] = -1
        enriched["is_weekend"] = False

    return enriched


def process_split(input_path, output_path, post_metadata, account_ages, thread_counts):
    """Process one split file."""
    if not os.path.exists(input_path):
        print(f"  Skipping {input_path} (not found)")
        return 0

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    enriched = [
        enrich_record(rec, post_metadata, account_ages, thread_counts)
        for rec in records
    ]

    with open(output_path, "w") as f:
        for e in enriched:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return len(enriched)


def main():
    parser = argparse.ArgumentParser(description="Enrich v2: add all missing features")
    parser.add_argument("--dataset-dir", required=True,
                        help="Enriched v1 dataset directory")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for enriched v2 dataset")
    parser.add_argument("--post-metadata", default=None,
                        help="Pre-fetched post metadata JSON (skip API calls)")
    parser.add_argument("--post-metadata-output", default=None,
                        help="Where to save fetched post metadata")
    parser.add_argument("--account-ages", default=None,
                        help="Account ages JSON from fetch_account_ages.py")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip API fetching (use only local data)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all records to compute global stats
    print("Loading enriched v1 dataset...", flush=True)
    all_records = []
    for split in ["train.jsonl", "test.jsonl"]:
        fp = os.path.join(args.dataset_dir, split)
        if os.path.exists(fp):
            with open(fp) as f:
                for line in f:
                    all_records.append(json.loads(line))
    print(f"  {len(all_records)} total records")

    # Collect unique link_ids
    link_ids = set()
    for r in all_records:
        lid = r.get("link_id", "").replace("t3_", "")
        if lid:
            link_ids.add(lid)
    print(f"  {len(link_ids)} unique posts")

    # Compute thread counts from our data
    thread_counts = Counter()
    for r in all_records:
        lid = r.get("link_id", "").replace("t3_", "")
        if lid:
            thread_counts[lid] += 1

    # Load account ages
    account_ages = {}
    if args.account_ages and os.path.exists(args.account_ages):
        print(f"\nLoading account ages from {args.account_ages}...")
        with open(args.account_ages) as f:
            account_ages = json.load(f)
        n_with_age = sum(1 for v in account_ages.values()
                         if v.get("account_created_utc"))
        print(f"  {len(account_ages)} entries, {n_with_age} with creation date")
    else:
        print("\nWARNING: No account ages file provided. account_age_days will be -1.")

    # Fetch post metadata
    post_metadata = {}
    if args.post_metadata and os.path.exists(args.post_metadata):
        print(f"\nLoading pre-fetched post metadata from {args.post_metadata}...")
        with open(args.post_metadata) as f:
            post_metadata = json.load(f)
        print(f"  {len(post_metadata)} entries loaded")
    elif not args.skip_api:
        print(f"\nFetching post metadata from Arctic Shift API...")
        meta_output = args.post_metadata_output or os.path.join(
            os.path.dirname(args.dataset_dir), "post_metadata.json"
        )
        post_metadata = fetch_all_post_metadata(list(link_ids), meta_output)
    else:
        print("\nSkipping API fetch (--skip-api)")

    # Process each split
    for split in ["train.jsonl", "test.jsonl"]:
        input_path = os.path.join(args.dataset_dir, split)
        output_path = os.path.join(args.output_dir, split)
        print(f"\nProcessing {split}...", flush=True)
        n = process_split(input_path, output_path, post_metadata, account_ages,
                          thread_counts)
        print(f"  Written {n} enriched v2 records to {output_path}")

    # Show new field count and sample
    sample_path = os.path.join(args.output_dir, "train.jsonl")
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            sample = json.loads(f.readline())
        print(f"\nTotal fields per record: {len(sample)}")

        new_fields = [
            # Post-relative temporal
            "time_since_post_hours",
            # Thread
            "thread_size_api", "post_is_nsfw", "thread_comment_count_local",
            # Account age (relative to comment time)
            "account_age_days",
            # Interaction
            "score_per_word", "velocity_ratio", "reply_length_ratio",
            # Text-derived
            "caps_ratio", "exclamation_count", "question_count",
            "avg_word_length", "unique_word_ratio", "sentence_count",
            "profanity_count", "has_profanity_signal", "paragraph_count",
            "starts_with_quote", "link_count",
            # Existing but now explicitly in feature set
            "controversiality",
            # Time-of-day (cyclical, no absolute date)
            "hour_of_day", "day_of_week", "is_weekend",
        ]
        print(f"\nNew v2 fields ({len(new_fields)}):")
        for field in new_fields:
            val = sample.get(field, "MISSING")
            print(f"  {field:<30s} = {val}")

    # Save metadata
    meta = {
        "source": args.dataset_dir,
        "n_records": len(all_records),
        "n_posts": len(link_ids),
        "n_unique_authors": len(set(r.get("author", "") for r in all_records)),
        "post_metadata_fetched": len(post_metadata),
        "account_ages_loaded": len(account_ages),
        "temporal_design": "All temporal features are RELATIVE to the comment's "
                           "own created_utc. No absolute timestamps are exposed "
                           "as features. This prevents temporal leakage.",
        "new_fields_added": [
            "time_since_post_hours", "thread_size_api", "post_is_nsfw",
            "thread_comment_count_local", "account_age_days",
            "score_per_word", "velocity_ratio", "reply_length_ratio",
            "caps_ratio", "exclamation_count", "question_count",
            "avg_word_length", "unique_word_ratio", "sentence_count",
            "profanity_count", "has_profanity_signal", "paragraph_count",
            "starts_with_quote", "link_count",
            "controversiality", "hour_of_day", "day_of_week", "is_weekend",
        ],
        "leakage_notes": {
            "thread_size_api": "Reflects final thread size, not size at comment time. "
                               "Use thread_comment_count_local as safer alternative.",
            "account_age_days": "Computed as (comment_created_utc - account_created_utc). "
                                "Safe: account creation is fixed in time.",
            "time_since_post_hours": "Computed as (comment_created_utc - post_created_utc). "
                                     "Safe: both timestamps are from time of creation.",
        },
    }
    meta_path = os.path.join(args.output_dir, "enrichment_v2_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
