#!/usr/bin/env python3
"""
Build an enriched dataset with all features pre-joined into each record.

Takes the original random_split train/test.jsonl files and enriches each
comment with:
  - post_title        (from post titles sidecar)
  - is_top_level      (derived from parent_id)
  - is_edited         (normalised to boolean)
  - author_flair      (from author_flair_text field, or "none")
  - 7 temporal author features (from author features sidecar)

Output: new train.jsonl and test.jsonl in a parallel directory (enriched/)
with all features as top-level fields. The original dataset is never modified.

Usage:
    python build_enriched_dataset.py \
        --dataset-dir ~/data/dataset/changemyview/random_split \
        --post-titles ~/data/enrichment/changemyview_post_titles.json \
        --author-features ~/data/enrichment/changemyview_author_features.json \
        --rules-file ~/data/rules/changemyview/rules.txt \
        --output-dir ~/data/dataset/changemyview/enriched

    # For all seen subreddits at once:
    python build_enriched_dataset.py \
        --dataset-dir ~/data/dataset/changemyview/random_split \
        --post-titles ~/data/enrichment/changemyview_post_titles.json \
        --author-features ~/data/enrichment/changemyview_author_features.json \
        --rules-file ~/data/rules/changemyview/rules.txt \
        --output-dir ~/data/dataset/changemyview/enriched \
        --validate
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(filepath):
    """Load a JSON file, return empty dict if missing."""
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found, features will be empty")
        return {}
    with open(filepath) as f:
        return json.load(f)


def load_rules(filepath):
    """Load rules text file."""
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found, rules will be empty")
        return ""
    with open(filepath) as f:
        return f.read().strip()


def enrich_comment(comment, post_titles, author_features, rules_text):
    """Add all enrichment fields to a single comment dict.

    Preserves ALL original fields and adds new computed/joined fields.
    Returns a new dict (does not modify the original).
    """
    enriched = dict(comment)  # shallow copy -- keeps EVERY original field from the input

    # --- Joined fields ---

    # Post title (from Arctic Shift posts API)
    link_id = comment.get("link_id", "")
    link_id_bare = link_id.replace("t3_", "")
    enriched["post_title"] = post_titles.get(link_id_bare,
                             post_titles.get(link_id, ""))

    # --- Derived fields ---

    # is_top_level (parent_id starts with t3_ = top-level, t1_ = reply)
    parent_id = comment.get("parent_id", "")
    enriched["is_top_level"] = parent_id.startswith("t3_")

    # Normalised booleans (original raw fields preserved, these are clean versions)
    edited_raw = comment.get("is_edited") or comment.get("edited")
    enriched["is_edited_bool"] = bool(edited_raw)

    flair = comment.get("author_flair_text") or ""
    enriched["author_flair_clean"] = flair.strip() if flair else "none"
    enriched["has_author_flair"] = bool(flair and flair.strip())

    enriched["is_submitter_bool"] = bool(comment.get("is_submitter"))

    # Comment metadata
    body = comment.get("body", "")
    enriched["word_count"] = len(body.split())
    enriched["char_count"] = len(body)
    enriched["has_url"] = "http://" in body or "https://" in body
    enriched["has_markdown"] = any(m in body for m in ["**", "~~", "^", "> ", "# "])

    # Parent metadata
    parent_body = comment.get("parent_body", "")
    enriched["parent_word_count"] = len(parent_body.split()) if parent_body else 0
    enriched["has_parent_body"] = bool(parent_body)

    # --- Temporal author features (from precomputed sidecar) ---

    cid = comment.get("id", "")
    af = author_features.get(cid, {})
    enriched["author_n_prior"] = af.get("n_prior", 0)
    enriched["author_vel_24h"] = af.get("vel_24h", 0)
    enriched["author_avg_score"] = af.get("avg_score", 0.0)
    enriched["author_unique_threads"] = af.get("unique_threads", 0)
    enriched["author_days_active"] = af.get("days_active", 0)
    enriched["author_is_first"] = af.get("is_first", True)
    enriched["author_max_in_thread"] = af.get("max_in_thread", 0)
    enriched["author_features_found"] = bool(af)  # True if we had precomputed features

    # --- Subreddit rules (same for every comment, stored for self-containment) ---
    enriched["rules_text"] = rules_text

    return enriched


def process_split(input_path, output_path, post_titles, author_features, rules_text,
                   raw_by_id=None, validate=False):
    """Process one split (train or test)."""
    if not os.path.exists(input_path):
        print(f"  Skipping {input_path} (not found)")
        return 0, 0

    comments = []
    with open(input_path) as f:
        for line in f:
            comments.append(json.loads(line))

    enriched = []
    missing_titles = 0
    missing_author = 0

    for comment in comments:
        # Merge in extra raw fields if available
        if raw_by_id:
            cid = comment.get("id", "")
            extra = raw_by_id.get(cid, {})
            if extra:
                for k, v in extra.items():
                    if k not in comment:  # never overwrite existing fields
                        comment[k] = v

        e = enrich_comment(comment, post_titles, author_features, rules_text)

        if not e["post_title"]:
            missing_titles += 1
        if e["author_n_prior"] == 0 and not e["author_is_first"]:
            missing_author += 1

        enriched.append(e)

    # Validate: check for issues
    if validate:
        n_total = len(enriched)
        checks = {
            "post_title": sum(1 for e in enriched if e.get("post_title")),
            "author_feats": sum(1 for e in enriched if e.get("author_features_found")),
            "is_top_level": sum(1 for e in enriched if e.get("is_top_level")),
            "has_flair": sum(1 for e in enriched if e.get("has_author_flair")),
            "is_edited": sum(1 for e in enriched if e.get("is_edited_bool")),
            "is_submitter": sum(1 for e in enriched if e.get("is_submitter_bool")),
            "has_parent": sum(1 for e in enriched if e.get("has_parent_body")),
            "has_url": sum(1 for e in enriched if e.get("has_url")),
            "has_markdown": sum(1 for e in enriched if e.get("has_markdown")),
        }
        n_fields = len(enriched[0].keys()) if enriched else 0

        print(f"    Validation ({os.path.basename(input_path)}, n={n_total}, fields={n_fields}):")
        for name, count in checks.items():
            print(f"      {name:18s}: {count:>5}/{n_total} ({100*count/n_total:.0f}%)")

    # Write
    with open(output_path, "w") as f:
        for e in enriched:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return len(enriched), missing_titles


def main():
    parser = argparse.ArgumentParser(description="Build enriched dataset")
    parser.add_argument("--dataset-dir", required=True,
                        help="Original dataset directory (with train.jsonl, test.jsonl)")
    parser.add_argument("--post-titles", required=True,
                        help="Post titles JSON sidecar file")
    parser.add_argument("--author-features", required=True,
                        help="Author features JSON sidecar file")
    parser.add_argument("--rules-file", required=True,
                        help="Subreddit rules text file")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for enriched dataset")
    parser.add_argument("--raw-extracted", default=None,
                        help="Optional: raw extracted JSON to merge in additional fields "
                             "(author_premium, ups, downs, _meta, total_awards, etc.)")
    parser.add_argument("--validate", action="store_true",
                        help="Print detailed validation statistics")
    args = parser.parse_args()

    # Load raw extracted data (optional -- for extra fields)
    raw_by_id = {}
    if args.raw_extracted:
        print(f"Loading raw extracted data for extra fields...", flush=True)
        raw_data = load_json(args.raw_extracted)
        # Extra fields worth carrying forward (not in the 15-field dataset)
        EXTRA_FIELDS = [
            "author_fullname", "author_premium", "ups", "downs",
            "gilded", "total_awards_received", "no_follow", "stickied",
            "locked", "collapsed", "collapsed_reason_code",
            "score_hidden", "send_replies", "retrieved_on",
            "_meta",  # contains removal_type, retrieved_2nd_on, was_deleted_later
        ]
        for category in ["removed", "approved"]:
            for c in raw_data.get(category, []):
                cid = c.get("id", "")
                if cid:
                    raw_by_id[cid] = {k: c.get(k) for k in EXTRA_FIELDS if k in c}
        print(f"  Loaded extra fields for {len(raw_by_id)} comments")
        del raw_data  # free memory

    # Load sidecars
    print("Loading enrichment data...", flush=True)
    post_titles = load_json(args.post_titles)
    print(f"  Post titles: {len(post_titles)} entries")

    author_features = load_json(args.author_features)
    print(f"  Author features: {len(author_features)} entries")

    rules_text = load_rules(args.rules_file)
    print(f"  Rules: {len(rules_text)} chars")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    for split in ["train.jsonl", "test.jsonl"]:
        input_path = os.path.join(args.dataset_dir, split)
        output_path = os.path.join(args.output_dir, split)

        print(f"\nProcessing {split}...", flush=True)
        n, missing = process_split(
            input_path, output_path,
            post_titles, author_features, rules_text,
            raw_by_id=raw_by_id,
            validate=args.validate
        )
        print(f"  Written {n} enriched records to {output_path}")
        if missing > 0:
            print(f"  Missing post titles: {missing}")

    # Write a metadata file documenting what was joined
    meta = {
        "source_dataset": args.dataset_dir,
        "post_titles_file": args.post_titles,
        "author_features_file": args.author_features,
        "rules_file": args.rules_file,
        "n_post_titles": len(post_titles),
        "n_author_features": len(author_features),
        "rules_chars": len(rules_text),
        "enriched_fields_added": [
            # Joined
            "post_title",
            # Derived
            "is_top_level", "is_edited_bool", "author_flair_clean",
            "has_author_flair", "is_submitter_bool",
            "word_count", "char_count", "has_url", "has_markdown",
            "parent_word_count", "has_parent_body",
            # Temporal author features
            "author_n_prior", "author_vel_24h", "author_avg_score",
            "author_unique_threads", "author_days_active",
            "author_is_first", "author_max_in_thread", "author_features_found",
            # Rules
            "rules_text",
        ],
        "note": "All original fields from random_split are preserved. "
                "Enriched fields are ADDED, never overwriting originals.",
    }
    meta_path = os.path.join(args.output_dir, "enrichment_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
