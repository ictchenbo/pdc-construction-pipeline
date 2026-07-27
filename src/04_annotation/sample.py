#!/usr/bin/env python3
"""Stratified sampling from corpus.jsonl (or anno.jsonl) by domain.

Writes a sample manifest (JSONL) that annotate.py reads.

Usage:
  python3 src/04_annotation/sample.py
  python3 src/04_annotation/sample.py --sample-fraction 0.1
  python3 src/04_annotation/sample.py --sample-fraction 0.05 --manifest /tmp/my_manifest.jsonl
  python3 src/04_annotation/sample.py --source anno --sample-fraction 0.1
  python3 src/04_annotation/sample.py --dry-run
"""

import argparse, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    PROJECT_DIR, VALIDATION_DIR,
    get_domain, VALIDATION_SAMPLE_FRACTION,
    SENTENCE_MIN_WORDS, SENTENCE_MAX_WORDS,
)
from src.utils.common import save_jsonl, load_jsonl

CORPUS_PATH = os.path.join(PROJECT_DIR, "data", "merge", "corpus.jsonl")
ANNO_PATH = os.path.join(PROJECT_DIR, "data", "merge", "anno.jsonl")


def _word_count(text: str) -> int:
    """Return the number of whitespace-delimited tokens in *text*."""
    return len(text.split())


def _resolve_source_path(source: str) -> str:
    """Return the file path for the given source name."""
    if source == "corpus":
        path = CORPUS_PATH
    elif source == "anno":
        path = ANNO_PATH
    else:
        print(f"  [ERROR] Unknown source: {source}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(path):
        print(f"  [ERROR] Source not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def load_corpus(min_words: int = SENTENCE_MIN_WORDS,
                max_words: int = SENTENCE_MAX_WORDS,
                source: str = "corpus"):
    """Load all sentences from corpus or anno source, group by domain.

    Parameters
    ----------
    min_words : int
        Minimum word count (inclusive). Sentences below this are skipped.
    max_words : int
        Maximum word count (inclusive). Sentences above this are skipped.
    source : str
        Source to load from: "corpus" (corpus.jsonl) or "anno" (anno.jsonl).
        When "anno", the sampled output includes valence / modality labels.
    """
    path = _resolve_source_path(source)
    domain_sentences = defaultdict(list)
    skipped = 0

    for rec in load_jsonl(path):
        text = rec["sentence"]
        wc = _word_count(text)
        if wc < min_words or wc > max_words:
            skipped += 1
            continue
        domain = get_domain(rec["speaker"])
        entry = {
            "video_id": rec["video_id"],
            "sentence_index": rec["sentence_index"],
            "text": text,
        }
        if source == "anno":
            entry["valence"] = rec.get("valence")
            entry["modality"] = rec.get("modality")
        domain_sentences[domain].append(entry)

    total = sum(len(v) for v in domain_sentences.values())
    print(f"  Loaded {total:,} sentences from {path}")
    if skipped:
        print(f"  Skipped {skipped:,} sentences outside [{min_words}, {max_words}] word range")
    print(f"  {len(domain_sentences)} domains")

    return domain_sentences


def sample_sentences(domain_sentences, fraction, seed=42):
    """Stratified sampling by domain. Returns list of sample dicts."""
    samples = []
    rng = random.Random(seed)

    for domain in sorted(domain_sentences.keys()):
        pool = domain_sentences[domain]
        n_sample = max(1, int(len(pool) * fraction))
        chosen = rng.sample(pool, min(n_sample, len(pool)))
        for entry in chosen:
            sample = {
                "video_id": entry["video_id"],
                "sentence_index": entry["sentence_index"],
                "text": entry["text"],
                "domain": domain,
            }
            if "valence" in entry and "modality" in entry:
                sample["valence"] = entry["valence"]
                sample["modality"] = entry["modality"]
            samples.append(sample)
        print(f"    {domain}: {len(pool):,} sentences -> sampled {len(chosen)}")

    return samples


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description="Stratified sample from corpus.jsonl or anno.jsonl")
    parser.add_argument("--sample-fraction", type=float, default=VALIDATION_SAMPLE_FRACTION)
    parser.add_argument("--manifest",
                       help="Output path for sample manifest (default: data/validation/sample_manifest.jsonl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=None,
                       help=f"Minimum word count (inclusive; default: {SENTENCE_MIN_WORDS})")
    parser.add_argument("--max-words", type=int, default=None,
                       help=f"Maximum word count (inclusive; default: {SENTENCE_MAX_WORDS})")
    parser.add_argument("--source", choices=["corpus", "anno"], default="corpus",
                       help="Source to sample from: corpus.jsonl (default) or anno.jsonl (includes valence/modality)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(VALIDATION_DIR, exist_ok=True)
    manifest_path = args.manifest or os.path.join(VALIDATION_DIR, "sample_manifest.jsonl")

    min_words = args.min_words if args.min_words is not None else SENTENCE_MIN_WORDS
    max_words = args.max_words if args.max_words is not None else SENTENCE_MAX_WORDS

    source_label = {"corpus": "corpus.jsonl", "anno": "anno.jsonl"}[args.source]
    print(f"Loading sentences from {source_label}...")
    print(f"  Sentence length filter: [{min_words}, {max_words}] words")
    domain_sentences = load_corpus(min_words=min_words, max_words=max_words, source=args.source)
    total_sents = sum(len(v) for v in domain_sentences.values())
    print(f"  {total_sents:,} eligible sentences across {len(domain_sentences)} domains")

    print(f"\nSampling {args.sample_fraction*100:.0f}% per domain (seed={args.seed})...")
    samples = sample_sentences(domain_sentences, args.sample_fraction, seed=args.seed)
    print(f"  Selected {len(samples)} sentences")

    if args.dry_run:
        print(f"\n[dry-run] Would save {len(samples)} sentences to {manifest_path}")
        return

    save_jsonl(manifest_path, samples)
    print(f"  Manifest saved: {manifest_path}")
    print(f"\nDone. Run annotate.py with this manifest to annotate (corpus source) or review (anno source).")


if __name__ == "__main__":
    main()
