#!/usr/bin/env python3
"""
Re-split sentences >60 words via LLM, then update grouped_sentences files.

Reads:  data/grouped_sentences/<speaker>/<YYYY-MM>.txt
Writes: back to same files (only modifies lines with long sentences)
Uses:   CLEAN_API_KEY (DeepSeek) from .env

Usage:
  python3 src/03_sentence_splitting/re_split_long_sentences.py
  python3 src/03_sentence_splitting/re_split_long_sentences.py --dry-run
  python3 src/03_sentence_splitting/re_split_long_sentences.py --limit 100
"""

import json, os, re, sys, time
from collections import defaultdict

from src.utils.llm import llm_call

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GROUPED_DIR = os.path.join(PROJECT_DIR, "data", "grouped_sentences")
BACKUP_DIR = os.path.join(PROJECT_DIR, "data", "grouped_sentences_bak")
MIN_WORDS = 61  # sentences at or above this threshold get re-split
BATCH_SIZE = 15  # sentences per API call
DELAY = 1.5      # seconds between API calls

# .env is auto-loaded by src.utils.llm at import
API_KEY = os.environ.get("CLEAN_API_KEY", "")
API_BASE = os.environ.get("CLEAN_API_BASE", "").rstrip("/")
MODEL = os.environ.get("CLEAN_MODEL", "deepseek-chat")

if not API_KEY:
    print("ERROR: CLEAN_API_KEY not set")
    sys.exit(1)


# ── Collect long sentences ──
def collect_long_sentences():
    """Return list of (speaker, month_file, line_idx, text, word_count)."""
    long_sents = []
    for spk in sorted(os.listdir(GROUPED_DIR)):
        spk_dir = os.path.join(GROUPED_DIR, spk)
        if not os.path.isdir(spk_dir) or spk.startswith("."):
            continue
        for fn in sorted(os.listdir(spk_dir)):
            if not fn.endswith(".txt"):
                continue
            fpath = os.path.join(spk_dir, fn)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines):
                s = line.strip()
                wc = len(s.split())
                if wc >= MIN_WORDS:
                    long_sents.append((spk, fn, idx, s, wc))
    return long_sents


# ── LLM re-split call ──
SYSTEM_PROMPT = """You are a sentence-boundary corrector for computational linguistics.
You receive text segments numbered like "--- Segment N ---" that are too long (60+ words) because the original sentence splitter missed boundaries.

For each segment, split it into properly bounded sentences. Each output sentence must:
- End with terminal punctuation (. ! ?)
- Start with a capital letter
- Be a natural, grammatically well-formed sentence on its own

CRITICAL RULE: You MUST split long segments. A 60+ word segment almost always contains 2+ sentence boundaries. Do not output a segment as a single sentence unless it is truly a single sentence (rare).

Do NOT add, remove, or rephrase content — only split at existing natural boundaries.

Output format (strict):
1|First sentence of segment 1.
1|Second sentence of segment 1.
2|First sentence of segment 2.
2|Its second sentence.
"""


def re_split_batch(segments, start_id):
    """Send a batch of long sentences to LLM for re-splitting."""
    user_msg = "\n\n".join(
        f"--- Segment {start_id + i} ---\n{text}"
        for i, (_, _, _, text, _) in enumerate(segments)
    )

    llm_cfg = {"api_key": API_KEY, "api_base": API_BASE, "model": MODEL}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(3):
        try:
            content = llm_call(llm_cfg, messages, attempts=1,
                               temperature=0.0, max_tokens=16384, timeout=180)
            return parse_response(content)
        except Exception as e:
            err = str(e)[:80]
            if attempt < 2:
                wait = (attempt + 1) * 10
                print(f"  ⚠️  Retry {attempt+1}/3 in {wait}s: {err}")
                time.sleep(wait)
            else:
                print(f"  ❌ Failed: {err}")
                return None


def parse_response(content):
    """Parse LLM output into {seg_id: [sentences]}."""
    result = {}
    for line in content.strip().split("\n"):
        line = line.strip()
        m = re.match(r"^(\d+)\|(.+)$", line)
        if m:
            seg_id = int(m.group(1))
            sent = m.group(2).strip()
            result.setdefault(seg_id, []).append(sent)
    return result


# ── Update files ──
def apply_fixes(all_segments, split_map):
    """Group fixes by file and rewrite lines."""
    # { (speaker, month): [(line_idx, old_text, new_lines)] }
    file_fixes = defaultdict(list)
    for i, (spk, fn, idx, text, wc) in enumerate(all_segments):
        replacements = split_map.get(i)
        if replacements:
            file_fixes[(spk, fn)].append((idx, text, replacements))

    total_replaced = 0
    total_new_sents = 0
    for (spk, fn), fixes in sorted(file_fixes.items()):
        fpath = os.path.join(GROUPED_DIR, spk, fn)
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Apply fixes in reverse order (so indices stay valid)
        for idx, old_text, new_lines in sorted(fixes, key=lambda x: -x[0]):
            old_line = lines[idx].strip()
            if old_line != old_text:
                print(f"  [WARN] {spk}/{fn}:{idx+1} mismatch, skipping")
                continue
            # Replace the line with new sentences
            new_lines_with_newline = [nl + "\n" for nl in new_lines]
            lines[idx:idx+1] = new_lines_with_newline
            total_new_sents += len(new_lines)
            total_replaced += 1

        with open(fpath, "w", encoding="utf-8") as f:
            f.writelines(lines)

    return total_replaced, total_new_sents


# ── Main ──
def main():
    dry_run = "--dry-run" in sys.argv
    limit = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])

    print(f"Collecting sentences >= {MIN_WORDS} words...")
    long_sents = collect_long_sentences()
    print(f"Found {len(long_sents)} long sentences")

    if limit:
        long_sents = long_sents[:limit]
        print(f"Limited to {limit}")

    if dry_run:
        print(f"\nDistribution by word count:")
        buckets = [(61,80), (81,100), (101,150), (151,200), (201,500), (501,99999)]
        for lo, hi in buckets:
            n = sum(1 for _,_,_,_,wc in long_sents if lo <= wc <= hi)
            if n > 0:
                print(f"  {lo}-{hi}: {n}")
        print(f"\nSample long sentences (first 5):")
        for _, _, _, text, wc in long_sents[:5]:
            print(f"  [{wc}词] {text[:100]}...")
        print(f"\nDry-run complete. {len(long_sents)} sentences would be processed.")
        return

    # Backup
    import shutil
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
    shutil.copytree(GROUPED_DIR, BACKUP_DIR)
    print(f"Backup saved to {BACKUP_DIR}")

    # Process in batches
    all_results = {}
    total_batches = (len(long_sents) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_i in range(0, len(long_sents), BATCH_SIZE):
        batch = long_sents[batch_i:batch_i + BATCH_SIZE]
        batch_num = batch_i // BATCH_SIZE + 1
        print(f"\nBatch [{batch_num}/{total_batches}] ({len(batch)} sentences)...")

        result = re_split_batch(batch, batch_i)
        if result:
            all_results.update(result)
            # Print stats
            for seg_id, new_sents in result.items():
                idx = seg_id - batch_i
                if 0 <= idx < len(batch):
                    _, _, _, old_text, wc = batch[idx]
                    n_new = len(new_sents)
                    avg_new = sum(len(s.split()) for s in new_sents) / max(n_new, 1)
                    print(f"  ✓ Seg {seg_id}: {wc}→{n_new}句 (avg {avg_new:.0f}词)")
        else:
            print(f"  ✗ Batch failed, skipping")

        if batch_num < total_batches:
            time.sleep(DELAY)

    # Apply fixes
    replaced, new_sents = apply_fixes(long_sents, all_results)
    print(f"\n{'='*50}")
    print(f"Done: {replaced} sentences replaced → {new_sents} new sentences")
    print(f"Files modified: under {GROUPED_DIR}/")
    print(f"Backup: {BACKUP_DIR}/")

    # Summary
    total_after = 0
    long_after = 0
    for spk in os.listdir(GROUPED_DIR):
        spk_dir = os.path.join(GROUPED_DIR, spk)
        if not os.path.isdir(spk_dir) or spk.startswith("."):
            continue
        for fn in os.listdir(spk_dir):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(spk_dir, fn)) as f:
                for line in f:
                    s = line.strip()
                    if s:
                        total_after += 1
                        if len(s.split()) >= MIN_WORDS:
                            long_after += 1
    print(f"Total sentences: {total_after:,}")
    print(f"Still >= {MIN_WORDS} words: {long_after:,}")


if __name__ == "__main__":
    main()
