#!/usr/bin/env python3
"""
Build training/test datasets from extracted per-subreddit JSON files.

Applies preprocessing filters, samples balanced classes, splits train/test,
and outputs final datasets ready for model training.

Usage:
    python build_dataset.py --input-dir ./extracted/ --output-dir ./dataset/ --subs AskHistorians,iran,changemyview,PublicFreakout,darknet
    python build_dataset.py --input-dir ./extracted/ --output-dir ./dataset/ --subs AskHistorians --max-per-class 5000 --test-ratio 0.2

Input: per-subreddit JSON files from process_zst_dump.py extract
Output: per-subreddit train/test JSONL files + dataset statistics
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

# Known bot accounts whose names don't contain "bot" (not caught by regex).
# The regex already catches names with "bot/Bot" at word boundaries or endings.
KNOWN_BOTS = {
    "VisualMod",
    "QualityVote",
    "SaveVideo",
    "TotesMessenger",
    "Flair_Helper",
    "ContentPolicyComplianceBot",  # contains "Bot" but listed for clarity
}

# Threshold for exact-duplicate spam detection: if the same body text appears
# this many times or more across different authors, all instances are removed.
SPAM_DUP_THRESHOLD = 5


def load_extracted(filepath):
    """Load extracted JSON from process_zst_dump.py."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("removed", []), data.get("approved", [])


def filter_comment(comment):
    """
    Return True if comment should be KEPT, False if filtered out.

    Filters (aligned with Chandrasekharan et al. 2018 and standard practice):
    1. AutoModerator
    2. Distinguished mod/admin comments
    3. Mod team removal notice accounts (e.g., darknet-ModTeam)
    4. Empty/trivial body ([removed], [deleted], empty)
    5. Very short (< 3 words, following Chandrasekharan)
    6. [deleted] author (user self-deleted)
    7. Known bot accounts (explicit blocklist)
    8. Bot-like username patterns (regex)
    """
    author = comment.get("author", "")
    body = comment.get("body", "")
    distinguished = comment.get("distinguished")

    # Filter AutoModerator
    if author == "AutoModerator":
        return False

    # Filter distinguished mod comments (mod action posts, not real comments)
    if distinguished in ("moderator", "admin"):
        return False

    # Filter mod team removal notice accounts (e.g., "darknet-ModTeam")
    if author.endswith("-ModTeam"):
        return False

    # Filter empty or trivial
    if not body or body.strip() in ("[removed]", "[deleted]", ""):
        return False

    # Filter very short (< 3 words, following Chandrasekharan et al.)
    if len(body.strip().split()) < 3:
        return False

    # Filter known bot accounts and deleted users
    if author == "[deleted]":
        return False
    if author in KNOWN_BOTS:
        return False
    # Match "Bot" or "bot" as a word boundary (e.g., "HelperBot", "sneakpeekbot")
    # but not inside normal words (e.g., "robotics", "about")
    if re.search(r'(?i)\bbot\b|Bot$|_bot_|^bot_|_bot$', author):
        return False

    return True


def remove_spam_duplicates(comments):
    """Remove exact-duplicate spam: if the same body text appears >= SPAM_DUP_THRESHOLD
    times across different authors, all instances are removed.

    Returns (filtered_comments, num_removed).
    """
    body_counts = Counter()
    body_authors = {}
    for c in comments:
        body = c.get("body", "").strip()
        body_counts[body] += 1
        if body not in body_authors:
            body_authors[body] = set()
        body_authors[body].add(c.get("author", ""))

    # Only flag as spam if the text appears many times AND from multiple authors
    # (a single prolific user repeating themselves is not necessarily spam)
    spam_bodies = {
        body for body, count in body_counts.items()
        if count >= SPAM_DUP_THRESHOLD and len(body_authors[body]) >= 2
    }

    if not spam_bodies:
        return comments, 0

    filtered = [c for c in comments if c.get("body", "").strip() not in spam_bodies]
    return filtered, len(comments) - len(filtered)


def random_split(all_comments, test_ratio, rng):
    """Standard random train/test split."""
    rng.shuffle(all_comments)
    split_idx = int(len(all_comments) * (1 - test_ratio))
    return all_comments[:split_idx], all_comments[split_idx:]


def author_stratified_split(all_comments, test_ratio, rng):
    """
    Author-stratified split: all comments from each author go to either
    train or test, never both. Prevents author leakage.
    """
    from collections import defaultdict

    # Group comments by author
    by_author = defaultdict(list)
    for c in all_comments:
        by_author[c.get("author", "unknown")].append(c)

    # Shuffle author order
    authors = list(by_author.keys())
    rng.shuffle(authors)

    # Greedily assign authors to test until we reach target size
    target_test = int(len(all_comments) * test_ratio)
    test_authors = set()
    test_count = 0
    for author in authors:
        if test_count >= target_test:
            break
        test_authors.add(author)
        test_count += len(by_author[author])

    train, test = [], []
    for author, comments in by_author.items():
        if author in test_authors:
            test.extend(comments)
        else:
            train.extend(comments)

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def compute_filter_stats(comments, label):
    """Count how many comments each filter catches (for reporting)."""
    stats = {}
    for name, fn in [
        ("automod", lambda c: c.get("author") == "AutoModerator"),
        ("distinguished", lambda c: c.get("distinguished") in ("moderator", "admin")),
        ("mod_team", lambda c: c.get("author", "").endswith("-ModTeam")),
        ("empty_trivial", lambda c: not c.get("body") or c.get("body", "").strip() in ("[removed]", "[deleted]", "")),
        ("short_lt3words", lambda c: len(c.get("body", "").strip().split()) < 3),
        ("deleted_user", lambda c: c.get("author") == "[deleted]"),
        ("known_bot", lambda c: c.get("author", "") in KNOWN_BOTS),
        ("bot_regex", lambda c: bool(re.search(r'(?i)\bbot\b|Bot$|_bot_|^bot_|_bot$', c.get("author", "")))),
    ]:
        stats[name] = sum(1 for c in comments if fn(c))
    return stats


def build_dataset_for_sub(removed, approved, max_per_class, test_ratio, seed):
    """Build balanced train/test splits for one subreddit.

    Returns both random and author-stratified splits.
    """
    rng = random.Random(seed)

    # Compute per-filter stats before filtering (for reporting)
    filter_stats_removed = compute_filter_stats(removed, "removed")
    filter_stats_approved = compute_filter_stats(approved, "approved")

    # Apply per-comment filters
    removed_filtered = [c for c in removed if filter_comment(c)]
    approved_filtered = [c for c in approved if filter_comment(c)]

    # Remove exact-duplicate spam
    removed_filtered, spam_removed_r = remove_spam_duplicates(removed_filtered)
    approved_filtered, spam_removed_a = remove_spam_duplicates(approved_filtered)

    # Shuffle
    rng.shuffle(removed_filtered)
    rng.shuffle(approved_filtered)

    # Cap at max_per_class
    removed_sample = removed_filtered[:max_per_class]
    approved_sample = approved_filtered[:max_per_class]

    # Balance: use the smaller class size
    n = min(len(removed_sample), len(approved_sample))
    removed_sample = removed_sample[:n]
    approved_sample = approved_sample[:n]

    # Assign labels
    for c in removed_sample:
        c["label"] = "removed"
    for c in approved_sample:
        c["label"] = "approved"

    # Combine
    all_comments = removed_sample + approved_sample

    # Random split
    train_rand, test_rand = random_split(list(all_comments), test_ratio, random.Random(seed))

    # Author-stratified split
    train_author, test_author = author_stratified_split(list(all_comments), test_ratio, random.Random(seed))

    # Count author overlap in random split (for reporting)
    train_authors_rand = {c.get("author") for c in train_rand}
    test_authors_rand = {c.get("author") for c in test_rand}
    author_overlap = len(train_authors_rand & test_authors_rand)

    stats = {
        "raw_removed": len(removed),
        "raw_approved": len(approved),
        "filtered_removed": len(removed_filtered),
        "filtered_approved": len(approved_filtered),
        "spam_duplicates_removed": {"removed": spam_removed_r, "approved": spam_removed_a},
        "filter_breakdown_removed": filter_stats_removed,
        "filter_breakdown_approved": filter_stats_approved,
        "sampled_per_class": n,
        "random_split": {
            "train_size": len(train_rand),
            "test_size": len(test_rand),
            "author_overlap": author_overlap,
            "unique_authors_train": len(train_authors_rand),
            "unique_authors_test": len(test_authors_rand),
        },
        "author_stratified_split": {
            "train_size": len(train_author),
            "test_size": len(test_author),
            "author_overlap": 0,
            "unique_authors_train": len({c.get("author") for c in train_author}),
            "unique_authors_test": len({c.get("author") for c in test_author}),
        },
    }

    return train_rand, test_rand, train_author, test_author, stats


def resolve_parent_context(comments, parent_index):
    """Resolve parent_id to parent body text using the parent index.

    For comments replying to other comments (t1_ prefix), looks up the
    parent comment's body in the index. For top-level comments replying
    to submissions (t3_ prefix), parent_body is left empty (submission
    context requires separate processing).

    Returns count of resolved vs unresolved parents.
    """
    resolved = 0
    unresolved_comment = 0
    top_level = 0

    for comment in comments:
        parent_id = comment.get("parent_id", "")

        if parent_id.startswith("t1_"):
            # Replying to another comment -- strip prefix and look up
            bare_id = parent_id[3:]
            parent_body = parent_index.get(bare_id, "")
            comment["parent_body"] = parent_body
            if parent_body:
                resolved += 1
            else:
                unresolved_comment += 1
        elif parent_id.startswith("t3_"):
            # Top-level comment replying to submission
            comment["parent_body"] = ""
            top_level += 1
        else:
            comment["parent_body"] = ""
            unresolved_comment += 1

    return resolved, unresolved_comment, top_level


def extract_features(comment):
    """Extract the fields we want in the final dataset."""
    meta = comment.get("_meta", {}) or {}
    return {
        "id": comment.get("id"),
        "subreddit": comment.get("subreddit"),
        "body": comment.get("body"),
        "parent_body": comment.get("parent_body", ""),
        "score": comment.get("score"),
        "author": comment.get("author"),
        "created_utc": comment.get("created_utc"),
        "parent_id": comment.get("parent_id"),
        "link_id": comment.get("link_id"),
        "is_submitter": comment.get("is_submitter"),
        "controversiality": comment.get("controversiality"),
        "author_flair_text": comment.get("author_flair_text"),
        "permalink": comment.get("permalink"),
        "label": comment.get("label"),
        "is_edited": meta.get("is_edited") or comment.get("edited", False),
    }


def write_jsonl(data, filepath):
    """Write list of dicts to JSONL file."""
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build training/test datasets")
    parser.add_argument("--input-dir", required=True, help="Directory with extracted per-sub JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory for datasets")
    parser.add_argument("--subs", required=True, help="Comma-separated list of subreddits")
    parser.add_argument("--max-per-class", type=int, default=5000, help="Max comments per class per sub (default: 5000)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    subs = [s.strip() for s in args.subs.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for sub in subs:
        input_file = os.path.join(args.input_dir, f"{sub}.json")
        if not os.path.exists(input_file):
            print(f"WARNING: {input_file} not found, skipping {sub}")
            continue

        print(f"\nProcessing r/{sub}...")
        removed, approved = load_extracted(input_file)

        # Load parent context index if available
        index_file = os.path.join(args.input_dir, f"{sub}_parent_index.json")
        parent_index = {}
        if os.path.exists(index_file):
            with open(index_file) as f:
                parent_index = json.load(f)
            print(f"  Loaded parent index: {len(parent_index):,} entries")
        else:
            print(f"  No parent index found at {index_file} (parent_body will be empty)")

        train_rand, test_rand, train_author, test_author, stats = build_dataset_for_sub(
            removed, approved, args.max_per_class, args.test_ratio, args.seed
        )

        # Resolve parent context for all splits
        all_splits = [train_rand, test_rand, train_author, test_author]
        all_comments_flat = [c for split in all_splits for c in split]
        resolved, unresolved, top_level = resolve_parent_context(all_comments_flat, parent_index)
        total_ctx = resolved + unresolved + top_level
        stats["parent_context"] = {
            "resolved": resolved,
            "unresolved_comment": unresolved,
            "top_level": top_level,
            "resolution_rate": round(resolved / max(resolved + unresolved, 1), 3),
        }
        print(f"  Parent context: {resolved} resolved, {unresolved} unresolved, {top_level} top-level "
              f"({stats['parent_context']['resolution_rate']:.1%} of comment replies resolved)")

        # Extract features for both splits
        train_rand = [extract_features(c) for c in train_rand]
        test_rand = [extract_features(c) for c in test_rand]
        train_author = [extract_features(c) for c in train_author]
        test_author = [extract_features(c) for c in test_author]

        # Write output: random split
        rand_dir = output_dir / sub / "random_split"
        rand_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(train_rand, rand_dir / "train.jsonl")
        write_jsonl(test_rand, rand_dir / "test.jsonl")

        # Write output: author-stratified split
        author_dir = output_dir / sub / "author_split"
        author_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(train_author, author_dir / "train.jsonl")
        write_jsonl(test_author, author_dir / "test.jsonl")

        all_stats[sub] = stats

        rs = stats["random_split"]
        as_ = stats["author_stratified_split"]
        print(f"  Raw: {stats['raw_removed']} removed, {stats['raw_approved']} approved")
        # Show per-filter breakdown
        fr = stats["filter_breakdown_removed"]
        fa = stats["filter_breakdown_approved"]
        for fname in fr:
            r_count, a_count = fr[fname], fa[fname]
            if r_count > 0 or a_count > 0:
                print(f"    {fname}: {r_count}r + {a_count}a = {r_count + a_count}")
        sd = stats["spam_duplicates_removed"]
        if sd["removed"] > 0 or sd["approved"] > 0:
            print(f"    spam_duplicates: {sd['removed']}r + {sd['approved']}a = {sd['removed'] + sd['approved']}")
        print(f"  After filtering: {stats['filtered_removed']} removed, {stats['filtered_approved']} approved")
        print(f"  Sampled: {stats['sampled_per_class']} per class")
        print(f"  Random split:  train={rs['train_size']}, test={rs['test_size']}, author overlap={rs['author_overlap']}")
        print(f"  Author split:  train={as_['train_size']}, test={as_['test_size']}, author overlap=0")
        print(f"  Output: {output_dir / sub}/")

    # Write combined stats
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nDataset stats saved to {stats_file}")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Subreddit':<20} {'Raw Rem':>8} {'Filtered':>9} {'Sampled':>8} {'Rand Trn':>9} {'Auth Trn':>9} {'Overlap':>8}")
    print(f"{'-'*70}")
    for sub, s in all_stats.items():
        rs = s["random_split"]
        as_ = s["author_stratified_split"]
        print(f"{sub:<20} {s['raw_removed']:>8} {s['filtered_removed']:>9} {s['sampled_per_class']:>8} {rs['train_size']:>9} {as_['train_size']:>9} {rs['author_overlap']:>8}")


if __name__ == "__main__":
    main()
