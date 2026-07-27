#!/usr/bin/env python3
"""
Search YouTube for videos by person name — batch or single.

Merges youtube_video_search.py (search logic) + batch_expand_speakers.py
(orchestration) into one script.

Input:
  - One or more person names as positional args
  - --from-file: JSONL (extracts "speaker" field), JSON list (extracts "name"),
                 or plain text (one name per line)

Output:
  search_results/<safe_name>/video_list.json   (or --output-dir)

Usage:
    # Single person
    python3 search_speakers.py "Ray Dalio"

    # Multiple people
    python3 search_speakers.py "Ray Dalio" "Peter Thiel" "Sam Harris"

    # From video_list.jsonl (extracts unique speakers)
    python3 search_speakers.py --from-file video_list.jsonl

    # From a list file
    python3 search_speakers.py --from-file speakers.txt

    # Custom output
    python3 search_speakers.py "Ray Dalio" --output-dir ./my_results --max 50

    # Extra search queries to broaden coverage
    python3 search_speakers.py "Noam Chomsky" --extra-queries interview talk podcast

    # Dry run (just show what would be searched)
    python3 search_speakers.py --from-file video_list.jsonl --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

DURATION_PAT = re.compile(r"^(\d+:)?\d{1,2}:\d{2}$")


def seconds_to_timestamp(total_seconds: int) -> str:
    if total_seconds is None or total_seconds < 0:
        return "0:00"
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def sanitize_date(raw: str) -> str:
    if not raw:
        return ""
    match = re.match(r"(\d{4})(\d{2})(\d{2})", raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.match(r"(\d{4})-(\d{2})", raw)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return raw[:7]


def clean_title(title: str) -> str:
    return title.strip() if title else ""


def name_to_safe(name: str) -> str:
    """Convert 'Peter Thiel' -> 'peter_thiel'."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe


# ── Search logic (from youtube_video_search.py) ──────────────────────────────

def crawl_search_results(
    person_name: str,
    max_results: int = 30,
    min_duration: int | None = None,
    max_duration: int | None = None,
    delay: float = 1.0,
    extra_queries: list[str] | None = None
) -> list[dict]:
    """Search YouTube via yt-dlp and collect video metadata.

    Returns list of dicts sorted by upload_date descending, deduplicated by id.
    Each dict: {id, title, upload_date, duration, channel}
    """
    queries = [person_name]
    if extra_queries:
        for suffix in extra_queries:
            queries.append(f"{person_name} {suffix}")

    seen_ids: set[str] = set()
    results: list[dict] = []

    duration_filter_parts = []
    if min_duration is not None:
        duration_filter_parts.append(f"duration > {min_duration}")
    if max_duration is not None:
        duration_filter_parts.append(f"duration < {max_duration}")

    qmax = min(100, max_results)
    ytdlp_timeout = 10 * qmax

    for q_idx, query in enumerate(queries):
        if q_idx > 0:
            time.sleep(delay)

        search_str = f"ytsearch{qmax}:{query}"
        cmd = [
            "yt-dlp",
            "--proxy", "http://127.0.0.1:57890",
            "--no-cookies",
            "--js-runtime", "node",
            "--ignore-no-formats-error",
            "--skip-download",
            "--dump-json",
            "--no-warnings",
            search_str,
        ]
        if duration_filter_parts:
            cmd.extend(["--match-filter", " & ".join(duration_filter_parts)])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=ytdlp_timeout)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] yt-dlp timed out for query: {query}", file=sys.stderr)
            continue
        except FileNotFoundError:
            print("  [ERROR] yt-dlp not found. Install: pip install yt-dlp", file=sys.stderr)
            sys.exit(1)

        if proc.returncode != 0:
            if not proc.stdout.strip():
                stderr_snippet = proc.stderr.strip()[:300]
                print(f"  [WARN] Query '{query}' returned no output: {stderr_snippet}", file=sys.stderr)
                continue

        new_count = 0
        for line in proc.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            vid = entry.get("id", "")
            if not vid or vid in seen_ids:
                continue

            title = clean_title(entry.get("fulltitle") or entry.get("title", ""))
            upload_date = sanitize_date(entry.get("upload_date", ""))
            duration_sec = entry.get("duration")
            timestamp = seconds_to_timestamp(duration_sec)
            channel = (entry.get("channel") or entry.get("uploader") or "")

            results.append({
                "id": vid,
                "title": title,
                "upload_date": upload_date,
                "duration": timestamp,
                "channel": channel,
                "description": (entry.get("description") or "")[:1000],
                "language": entry.get("language") or "",
            })
            seen_ids.add(vid)
            new_count += 1

        print(f"  Query [{q_idx + 1}/{len(queries)}] '{query}' "
              f"→ {new_count} new (total unique: {len(results)})")

    results.sort(key=lambda v: v["upload_date"], reverse=True)

    return results


# ── Input loading ────────────────────────────────────────────────────────────

def load_names_from_file(path: str) -> list[str]:
    """Load person names from various file formats.

    Supports:
      - .jsonl: extracts "speaker" field from each line
      - .json:  extracts "name" field from each dict in a list
      - .txt:   one name per line (skips comments starting with #)
    """
    ext = os.path.splitext(path)[1].lower()
    names = []

    if ext == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    sp = d.get("speaker", "")
                    if sp:
                        names.append(sp)
                except json.JSONDecodeError:
                    pass
        # Deduplicate speaker names, preserving order
        seen = set()
        unique = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        names = unique

    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    n = item.get("name", "")
                    if n:
                        names.append(n)
                elif isinstance(item, str):
                    names.append(item)

    elif ext == ".txt":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    names.append(line)

    else:
        # Try JSON first, then text
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        n = item.get("name", "")
                        if n:
                            names.append(n)
                    elif isinstance(item, str):
                        names.append(item)
        except (json.JSONDecodeError, ValueError):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        names.append(line)

    return names


# ── Orchestration ────────────────────────────────────────────────────────────

def search_speakers(
    names: list[str],
    output_dir: str,
    max_results: int,
    min_duration: int | None,
    max_duration: int | None,
    delay: float,
    extra_queries: list[str] | None,
    dry_run: bool = False,
) -> dict:
    """Batch search for all speakers and save results.

    Returns summary dict.
    """
    total_people = len(names)
    print(f"\n{'=' * 60}")
    print(f"Searching YouTube for {total_people} people")
    print(f"Output: {output_dir}/")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("DRY RUN — no searches will be performed")
        for name in names:
            safe = name_to_safe(name)
            out_path = os.path.join(output_dir, safe, "video_list.json")
            print(f"  {name:30s} → {out_path}")
        print(f"\nDry run: {total_people} people, 0 searches executed")
        return {"searched": total_people, "total_videos": 0, "errors": []}

    results = []
    errors = []

    for idx, name in enumerate(names, 1):
        safe = name_to_safe(name)
        out_path = os.path.join(output_dir, safe, "video_list.json")

        # Check if already exists
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing = json.load(f)
            print(f"[{idx:>3}/{total_people}] SKIP {name} ({len(existing)} existing)")
            results.append({"name": name, "safe": safe, "count": len(existing)})
            continue

        print(f"[{idx:>3}/{total_people}] SEARCH {name} ...")

        videos = crawl_search_results(
            person_name=name,
            max_results=max_results,
            min_duration=min_duration,
            max_duration=max_duration,
            delay=delay,
            extra_queries=extra_queries
        )

        if videos:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(videos, f, indent=2, ensure_ascii=False)
            print(f"  → Saved {len(videos)} videos to {out_path}")
            results.append({"name": name, "safe": safe, "count": len(videos)})
        else:
            print(f"  → No videos found")
            results.append({"name": name, "safe": safe, "count": 0})

        if idx < total_people:
            time.sleep(delay)

    # Summary
    total_videos = sum(r["count"] for r in results)
    print(f"\n{'=' * 60}")
    print(f"SEARCH COMPLETE")
    print(f"  People searched: {len(results)}")
    print(f"  Total videos:    {total_videos}")
    if errors:
        print(f"  Errors:          {len(errors)}")
        for e in errors:
            print(f"    {e['name']}: {e['error'][:100]}")
    print(f"  Output:          {output_dir}/")
    print(f"{'=' * 60}")

    return {"searched": len(results), "total_videos": total_videos, "errors": errors}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Search YouTube for videos by person name — batch or single.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s \"Ray Dalio\"\n"
            "  %(prog)s \"Ray Dalio\" \"Peter Thiel\"\n"
            "  %(prog)s --from-file video_list.jsonl\n"
            "  %(prog)s --from-file speakers.txt --output-dir ./my_results\n"
            "  %(prog)s \"Sam Harris\" --max 100 --min-duration 300 --max-duration 7200\n"
            "  %(prog)s \"Noam Chomsky\" --extra-queries interview talk --dry-run\n"
        ),
    )
    parser.add_argument(
        "names", nargs="*",
        help="Person name(s) to search for (e.g., 'Ray Dalio')",
    )
    parser.add_argument(
        "--from-file", "-f",
        help=(
            "Load person names from file. Supports:\n"
            "  .jsonl — extracts 'speaker' field\n"
            "  .json  — extracts 'name' field from list\n"
            "  .txt   — one name per line"
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./search_results",
        help="Output directory (default: ./search_results)",
    )
    parser.add_argument(
        "--max", "-n", type=int, default=100, dest="max_results",
        help="Max videos per query (default: 100)",
    )
    parser.add_argument(
        "--min-duration", type=int, default=None,
        help="Minimum video duration in seconds (e.g., 300 = 5 min)",
    )
    parser.add_argument(
        "--max-duration", type=int, default=None,
        help="Maximum video duration in seconds (e.g., 7200 = 2 hr)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between search queries (default: 0.5)",
    )
    parser.add_argument(
        "--extra-queries", nargs="+", default=None,
        help="Extra search terms (e.g., --extra-queries interview podcast)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be searched without running yt-dlp",
    )

    args = parser.parse_args()

    # Collect names from all sources
    names = list(args.names)
    if args.from_file:
        file_names = load_names_from_file(args.from_file)
        if not file_names:
            print(f"ERROR: no names found in {args.from_file}", file=sys.stderr)
            sys.exit(1)
        # Merge, preserving order, deduplicating
        seen = set()
        merged = []
        for n in names + file_names:
            if n not in seen:
                seen.add(n)
                merged.append(n)
        names = merged

    if not names:
        parser.print_help()
        print("\nERROR: provide at least one name or use --from-file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(names)} person name(s)")
    print(f"  Output dir: {args.output_dir}")
    if args.min_duration is not None or args.max_duration is not None:
        parts = []
        if args.min_duration is not None:
            parts.append(f"≥{args.min_duration}s")
        if args.max_duration is not None:
            parts.append(f"≤{args.max_duration}s")
        print(f"  Duration:   {' & '.join(parts)}")
    if args.extra_queries:
        print(f"  Extra q:    {', '.join(args.extra_queries)}")
    print()

    summary = search_speakers(
        names=names,
        output_dir=args.output_dir,
        max_results=args.max_results,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        delay=args.delay,
        extra_queries=args.extra_queries,
        dry_run=args.dry_run,
    )

    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
