#!/usr/bin/env python3
"""
Compute cross-dimensional phi correlations by speaker from annotated data.

r(pos, emp)  — positive & emphatic
r(pos, hed)  — positive & hedged
r(neg, emp)  — negative & emphatic
r(neg, hed)  — negative & hedged

Reads:  data/annotated_deepseek/<video_id>.jsonl
        ../data/video_list.jsonl  (speaker mapping)
Writes: results/cross_dim_correlations.json

Usage:
  python3 src/09_evaluation/cross_dim_correlation.py
"""

import os, json, math
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOTATED_DIR = os.path.join(PROJECT_DIR, "data", "annotated_deepseek")
VIDEO_LIST = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data", "video_list.jsonl"))
OUTPUT_PATH = os.path.join(PROJECT_DIR, "results", "cross_dim_correlations.json")


def phi_corr(n11, n10, n01, n00):
    """Compute phi coefficient from 2x2 contingency table."""
    n1_ = n11 + n10   # row 1 total
    n0_ = n01 + n00   # row 0 total
    n_1 = n11 + n01   # col 1 total
    n_0 = n10 + n00   # col 0 total
    n = n11 + n10 + n01 + n00
    denom = math.sqrt(n1_ * n0_ * n_1 * n_0)
    if denom == 0:
        return None
    return round((n11 * n00 - n10 * n01) / denom, 4)


def compute_speaker_correlations(labels):
    """Compute all 4 phi correlations for one speaker's labels."""
    # n11 = X=1 & Y=1, n10 = X=1 & Y=0, n01 = X=0 & Y=1, n00 = X=0 & Y=0
    tables = {
        "pos_emp": {"n11": 0, "n10": 0, "n01": 0, "n00": 0},
        "pos_hed": {"n11": 0, "n10": 0, "n01": 0, "n00": 0},
        "neg_emp": {"n11": 0, "n10": 0, "n01": 0, "n00": 0},
        "neg_hed": {"n11": 0, "n10": 0, "n01": 0, "n00": 0},
    }

    for l in labels:
        v = l["valence"]
        m = l["modality"]

        # r(pos, emp)
        x = 1 if v == "positive" else 0
        y = 1 if m == "emphatic" else 0
        if x and y: tables["pos_emp"]["n11"] += 1
        elif x and not y: tables["pos_emp"]["n10"] += 1
        elif not x and y: tables["pos_emp"]["n01"] += 1
        else: tables["pos_emp"]["n00"] += 1

        # r(pos, hed)
        y = 1 if m == "hedged" else 0
        if x and y: tables["pos_hed"]["n11"] += 1
        elif x and not y: tables["pos_hed"]["n10"] += 1
        elif not x and y: tables["pos_hed"]["n01"] += 1
        else: tables["pos_hed"]["n00"] += 1

        # r(neg, emp)
        x = 1 if v == "negative" else 0
        y = 1 if m == "emphatic" else 0
        if x and y: tables["neg_emp"]["n11"] += 1
        elif x and not y: tables["neg_emp"]["n10"] += 1
        elif not x and y: tables["neg_emp"]["n01"] += 1
        else: tables["neg_emp"]["n00"] += 1

        # r(neg, hed)
        y = 1 if m == "hedged" else 0
        if x and y: tables["neg_hed"]["n11"] += 1
        elif x and not y: tables["neg_hed"]["n10"] += 1
        elif not x and y: tables["neg_hed"]["n01"] += 1
        else: tables["neg_hed"]["n00"] += 1

    results = {}
    for key, t in tables.items():
        r = phi_corr(t["n11"], t["n10"], t["n01"], t["n00"])
        if r is not None:
            results[key] = {"phi": r, "n11": t["n11"], "n": t["n11"] + t["n10"] + t["n01"] + t["n00"]}
    return results


# ── Load speaker mapping ──
speaker_map = {}
if os.path.exists(VIDEO_LIST):
    with open(VIDEO_LIST) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            speaker_map[e["id"]] = e.get("speaker", "")

# ── Group labels by speaker ──
speaker_labels = defaultdict(list)
total_labels = 0
files_loaded = 0

for fname in sorted(os.listdir(ANNOTATED_DIR)):
    if not fname.endswith(".jsonl"): continue
    vid = fname.replace(".jsonl", "")
    spk = speaker_map.get(vid)
    if not spk: continue
    path = os.path.join(ANNOTATED_DIR, fname)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                speaker_labels[spk].append(obj)
                total_labels += 1
            except json.JSONDecodeError:
                continue
    files_loaded += 1

print(f"Loaded {files_loaded} files, {total_labels:,} labels across {len(speaker_labels)} speakers")

# ── Compute per-speaker correlations ──
all_results = {}
for spk in sorted(speaker_labels):
    labels = speaker_labels[spk]
    if len(labels) < 10:
        continue  # skip speakers with too few sentences
    corr = compute_speaker_correlations(labels)
    all_results[spk] = {
        "n": len(labels),
        **corr,
    }

# ── Compute overall (all speakers pooled) ──
all_labels = [l for labels in speaker_labels.values() for l in labels]
overall = compute_speaker_correlations(all_labels)
overall["n"] = len(all_labels)

# ── Summary table ──
print(f"\n{'='*80}")
print(f"Cross-Dimension Correlations by Speaker")
print(f"{'='*80}")
print(f"{'Speaker':<25} {'n':>6} {'r(p,em)':>8} {'r(p,he)':>8} {'r(n,em)':>8} {'r(n,he)':>8}")
print(f"{'─'*80}")

# Sort by n descending
sorted_spks = sorted(all_results.items(), key=lambda x: -x[1]["n"])
for spk, data in sorted_spks:
    pe = data.get("pos_emp", {}).get("phi", "")
    ph = data.get("pos_hed", {}).get("phi", "")
    ne = data.get("neg_emp", {}).get("phi", "")
    nh = data.get("neg_hed", {}).get("phi", "")
    pe_s = f"{pe:.3f}" if isinstance(pe, float) else ""
    ph_s = f"{ph:.3f}" if isinstance(ph, float) else ""
    ne_s = f"{ne:.3f}" if isinstance(ne, float) else ""
    nh_s = f"{nh:.3f}" if isinstance(nh, float) else ""
    print(f"{spk:<25} {data['n']:>6} {pe_s:>8} {ph_s:>8} {ne_s:>8} {nh_s:>8}")

print(f"{'─'*80}")
print(f"{'OVERALL (pooled)':<25} {overall['n']:>6}", end="")
for key in ["pos_emp", "pos_hed", "neg_emp", "neg_hed"]:
    v = overall.get(key, {}).get("phi", "")
    print(f" {v:>7.3f}" if isinstance(v, float) else f" {'':>7}", end="")
print()

# ── Save ──
output = {
    "overall": overall,
    "by_speaker": all_results,
    "metadata": {
        "total_labels": total_labels,
        "total_speakers": len(all_results),
        "method": "phi coefficient (binary correlation)",
        "dimensions": ["pos_emp", "pos_hed", "neg_emp", "neg_hed"],
    }
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved: {OUTPUT_PATH}")
print(f"Speakers with >= 10 sentences: {len(all_results)}")
