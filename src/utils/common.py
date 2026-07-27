#!/usr/bin/env python3
"""Shared utilities across all phases."""

import json, os, sys, time, ssl, certifi
from pathlib import Path

ssl_context = ssl.create_default_context(cafile=certifi.where())

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_load_speakers(raw_dir):
    """Discover speakers with raw data."""
    if not os.path.isdir(raw_dir):
        print(f"ERROR: raw dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)
    return sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
        and not d.startswith(".")
        and os.path.isdir(os.path.join(raw_dir, d, "transcripts"))
    ])
