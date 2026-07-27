#!/usr/bin/env python3
"""
Merge the flat video-based corpus into a single JSONL file.

Reads sentences_by_video_keep/<video_id>.txt and tsp_preannotation.jsonl,
outputs one JSONL row per sentence with video_id, speaker, sentence_index, sentence.

Usage:
  python3 src/07_publish/merge_dataset.py
  python3 src/07_publish/merge_dataset.py --output data/merge/corpus.jsonl
"""

import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROJECT_DIR
from src.utils.common import save_jsonl

SENTENCES_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data", "sentences_by_video_keep"))
TSP_MAPPING_PATH = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data", "tsp_preannotation.jsonl"))
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "data", "merge", "corpus.jsonl")


def load_speaker_mapping(path):
    """Load video_id -> speaker mapping from JSONL."""
    mapping = {}
    if not os.path.exists(path):
        print(f"[ERROR] Speaker mapping not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mapping[rec["video_id"]] = rec.get("speaker", "")
    return mapping


def merge(sentences_dir, speaker_map):
    """Yield rows from flat corpus directory."""
    video_ids = sorted([
        f.replace(".txt", "") for f in os.listdir(sentences_dir) if f.endswith(".txt")
    ])

    n_skipped_nomap = 0
    n_total = 0

    for vid in video_ids:
        speaker = speaker_map.get(vid)
        if not speaker:
            n_skipped_nomap += 1
            continue

        fpath = os.path.join(sentences_dir, f"{vid}.txt")
        with open(fpath) as f:
            for i, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                yield {
                    "video_id": vid,
                    "speaker": speaker,
                    "sentence_index": i,
                    "sentence": text,
                }
                n_total += 1

    print(f"  Videos: {len(video_ids)} total, {n_skipped_nomap} skipped (no speaker mapping)")
    print(f"  Sentences: {n_total:,}")


def main():
    parser = argparse.ArgumentParser(description="Merge corpus into single JSONL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                       help=f"Output JSONL path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    print("Loading speaker mapping...")
    speaker_map = load_speaker_mapping(TSP_MAPPING_PATH)
    print(f"  {len(speaker_map)} video->speaker entries")

    print(f"Reading sentences from: {SENTENCES_DIR}")
    rows = list(merge(SENTENCES_DIR, speaker_map))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_jsonl(args.output, rows)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
