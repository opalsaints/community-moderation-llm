#!/usr/bin/env python3
"""
Collect subreddit rules from Reddit's public JSON API.

For each subreddit, fetches the structured rules from the /about/rules.json
endpoint (no authentication required). Saves rules as text files for the
with-rules prompting condition.

Usage:
    python collect_rules.py
    python collect_rules.py --subs AskHistorians,science,politics
    python collect_rules.py --config configs/subreddits.json --output-dir data/rules/

Output per subreddit:
    {output_dir}/{subreddit}/rules.txt    -- formatted rules text for prompts
    {output_dir}/{subreddit}/rules.json   -- raw API response
"""

import argparse
import json
import os
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def fetch_rules(subreddit):
    """Fetch subreddit rules from Reddit's public API."""
    url = f"https://www.reddit.com/r/{subreddit}/about/rules.json"
    req = Request(url, headers={
        "User-Agent": "capstone-research:v1.0 (by /u/research_bot)"
    })

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data, None
    except HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        return None, f"URL error: {e.reason}"
    except Exception as e:
        return None, str(e)


def format_rules_text(rules_data):
    """Format rules into a clean text block for use in prompts.

    Matches the format used by Kumar et al. (2024) and SLM-Mod (Ma et al. 2024).
    """
    rules = rules_data.get("rules", [])
    if not rules:
        return "", 0

    lines = []
    for i, rule in enumerate(rules, 1):
        short = rule.get("short_name", "").strip()
        desc = rule.get("description", "").strip()

        if short and desc:
            lines.append(f"{i}. {short}: {desc}")
        elif short:
            lines.append(f"{i}. {short}")
        elif desc:
            lines.append(f"{i}. {desc}")

    return "\n".join(lines), len(rules)


def main():
    parser = argparse.ArgumentParser(description="Collect subreddit rules")
    parser.add_argument("--config", type=str, default="configs/subreddits.json",
                        help="Path to subreddits config (default: configs/subreddits.json)")
    parser.add_argument("--subs", type=str, default=None,
                        help="Comma-separated subreddits (overrides config)")
    parser.add_argument("--output-dir", type=str, default="data/rules",
                        help="Output directory (default: data/rules)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between API requests (default: 2.0)")
    args = parser.parse_args()

    # Get subreddit list
    if args.subs:
        subreddits = [s.strip() for s in args.subs.split(",")]
    else:
        with open(args.config) as f:
            config = json.load(f)
        subreddits = config.get("subreddits") or (config.get("seen", []) + config.get("unseen", []))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Collecting rules for {len(subreddits)} subreddits")
    print(f"Output: {args.output_dir}/")
    print()

    results = {}
    for sub in subreddits:
        print(f"  r/{sub}... ", end="", flush=True)

        data, error = fetch_rules(sub)
        if error:
            print(f"ERROR: {error}")
            results[sub] = {"status": "error", "error": error}
            time.sleep(args.delay)
            continue

        rules_text, rule_count = format_rules_text(data)

        # Save raw JSON
        sub_dir = os.path.join(args.output_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)

        with open(os.path.join(sub_dir, "rules.json"), "w") as f:
            json.dump(data, f, indent=2)

        # Save formatted text
        with open(os.path.join(sub_dir, "rules.txt"), "w") as f:
            f.write(rules_text)

        print(f"{rule_count} rules")
        results[sub] = {"status": "ok", "rule_count": rule_count}

        time.sleep(args.delay)

    # Summary
    with open(os.path.join(args.output_dir, "collection_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    err = sum(1 for r in results.values() if r["status"] == "error")
    print(f"\nDone: {ok} collected, {err} errors")

    if err > 0:
        print("Errors:")
        for sub, r in results.items():
            if r["status"] == "error":
                print(f"  r/{sub}: {r['error']}")


if __name__ == "__main__":
    main()
