#!/usr/bin/env python3
"""
Split diarized transcripts into sentences (one per line), saved per video file.

Reads:  ../data/audio_diarized/<video_id>.txt
        ../data/video_list.jsonl                (id → speaker, upload_date)
Writes: ../data/sentences_by_video/<video_id>.txt

Usage:
  python3 src/03_sentence_splitting/split_by_video.py              # all videos
  python3 src/03_sentence_splitting/split_by_video.py adam_tooze    # one speaker (filter)
"""

import json
import os
import re
import sys
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIARIZED_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data", "audio_diarized"))
VIDEO_LIST = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data", "video_list.jsonl"))
OUTPUT_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data", "sentences_by_video"))

# Sentence-ending pattern
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|(?<=[.!?])$")
MIN_WORDS = 6
MAX_WORDS = 80


def load_video_map():
    """Load video_list.jsonl → {video_id: {speaker, upload_date, title}}."""
    if not os.path.exists(VIDEO_LIST):
        print(f"ERROR: video list not found: {VIDEO_LIST}", file=sys.stderr)
        sys.exit(1)
    mapping = {}
    with open(VIDEO_LIST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            mapping[entry["id"]] = {
                "speaker": entry.get("speaker", ""),
                "upload_date": entry.get("upload_date", ""),
                "title": entry.get("title", ""),
            }
    return mapping


def split_into_sentences(text):
    """Split text into sentences, return list of non-empty sentences."""
    parts = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def process_all(video_map):
    """Process all diarized .txt files, group by speaker."""
    speakers = defaultdict(list)  # speaker → [(video_id, text)]
    no_speaker = []

    for fname in sorted(os.listdir(DIARIZED_DIR)):
        if not fname.endswith(".txt"):
            continue
        vid = fname[:-4]
        meta = video_map.get(vid)
        if not meta or not meta.get("speaker"):
            no_speaker.append(vid)
            continue

        fpath = os.path.join(DIARIZED_DIR, fname)
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        # No length-based pre-filter — sentence-level filtering happens downstream

        speakers[meta["speaker"]].append((vid, text))

    if no_speaker:
        print(f"  [INFO] {len(no_speaker)} video(s) without speaker mapping, skipped")

    return speakers


def process_one_speaker(video_list):
    """Split sentences for one speaker's videos, save flat to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_sents = 0
    total_filtered = 0
    total_videos = 0

    for vid, text in video_list:
        sentences = split_into_sentences(text)
        filtered = [s for s in sentences if MIN_WORDS <= len(s.split()) <= MAX_WORDS]

        if not filtered:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{vid}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for s in filtered:
                f.write(s + "\n")

        total_sents += len(sentences)
        total_filtered += len(filtered)
        total_videos += 1

    return total_videos, total_sents, total_filtered


def main():
    filter_speaker = sys.argv[1].lower().replace(" ", "_") if len(sys.argv) > 1 else None

    if not os.path.isdir(DIARIZED_DIR):
        print(f"ERROR: diarized dir not found: {DIARIZED_DIR}", file=sys.stderr)
        sys.exit(1)

    video_map = load_video_map()
    print(f"Loaded {len(video_map)} video mappings from {VIDEO_LIST}")

    speakers = process_all(video_map)
    print(f"Found {len(speakers)} speaker(s) with diarized data\n")

    if filter_speaker:
        if filter_speaker in speakers:
            speakers = {filter_speaker: speakers[filter_speaker]}
        else:
            print(f"Speaker '{filter_speaker}' not found in data")
            sys.exit(1)

    grand_videos = 0
    grand_sents = 0
    grand_filtered = 0

    for speaker_name in sorted(speakers.keys()):
        print(f"═══ {speaker_name} ═══")
        vids, sents, filtered = process_one_speaker(speakers[speaker_name])
        grand_videos += vids
        grand_sents += sents
        grand_filtered += filtered
        print(f"  → {vids} videos, {sents:,} total sents, {filtered:,} filtered (6-80w)")

    print()
    print("═" * 50)
    print(f"Done — {len(speakers)} speakers, {grand_videos} videos")
    print(f"Total sentences: {grand_sents:,}")
    print(f"Filtered (6-80w): {grand_filtered:,}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
