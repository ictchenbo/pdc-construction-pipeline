#!/usr/bin/env python3
"""
Monthly Negative Affect vs Emphatic Certainty — Keyword Lexicon Scoring.
Reads per-video cleaned transcripts, groups by month, applies keyword lexicon
with negation handling, computes cross-dimensional correlations.

Output: results/keyword_lexicon_eval.json (cross-speaker summary)
       + results/keyword_monthly_dalio.json (Dalio monthly breakdown)
"""
import json, os, re, sys
from collections import defaultdict
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from keyword_lexicon import (
    count_matches, NEGATIVE, POSITIVE, MODALITY,
    NEGATABLE_POSITIVE, NEGATION_PATTERNS
)

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
CLEANED_DIR = os.path.join(BASE, "data", "cleaned")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATES_PATH = os.path.expanduser("~/ray_dalio_analysis/video_dates_cache.json")
NORM_BASE = 30000

SPEAKERS = [
    ("ray_dalio", "Ray Dalio"),
    ("cathie_wood", "Cathie Wood"),
    ("kenneth_rogoff", "Kenneth Rogoff"),
    ("peter_zeihan", "Peter Zeihan"),
]

def load_video_dates():
    with open(DATES_PATH) as f:
        return json.load(f)

def process_speaker(speaker, dates):
    spk_dir = os.path.join(CLEANED_DIR, speaker)
    if not os.path.isdir(spk_dir):
        return None

    monthly = defaultdict(lambda: defaultdict(float))
    char_counts = defaultdict(int)
    video_counts = defaultdict(int)

    for fname in sorted(os.listdir(spk_dir)):
        if not fname.endswith(".txt") or fname == "cleaned_summary.json":
            continue
        vid = fname.replace(".txt", "")
        ym = dates.get(vid)
        if not ym:
            continue
        with open(os.path.join(spk_dir, fname)) as f:
            text = f.read().lower()
        text = re.sub(r'[^a-z\s/-]', ' ', text)

        char_counts[ym] += len(text)
        video_counts[ym] += 1
        matches = count_matches(text)
        for cat, info in matches.items():
            monthly[ym][cat] = monthly[ym].get(cat, 0) + info["count"]

    months = sorted(monthly.keys())
    if len(months) < 3:
        return None

    neg_cats = list(NEGATIVE.keys())
    emph_cat = "Emphatic/Certain"
    hedge_cat = "Hedging/Doubt"

    monthly_data = []
    neg_vals, emph_vals, hedge_vals = [], [], []

    for ym in months:
        n_videos = max(video_counts[ym], 1)
        avg_char = char_counts[ym] / n_videos  # L̄m = average chars per video
        norm = avg_char / NORM_BASE            # scale to 30K baseline

        neg_raw = sum(monthly[ym].get(c, 0) for c in neg_cats)
        emph_raw = monthly[ym].get(emph_cat, 0)
        hedge_raw = monthly[ym].get(hedge_cat, 0)
        pos_raw = sum(monthly[ym].get(c, 0) for c in POSITIVE.keys())

        # f_raw × 30000 / L̄m = f_raw × 30000 × Nm / Lm
        nv = neg_raw / max(norm, 0.001)
        ev = emph_raw / max(norm, 0.001)
        hv = hedge_raw / max(norm, 0.001)
        pv = pos_raw / max(norm, 0.001)

        neg_vals.append(nv)
        emph_vals.append(ev)
        hedge_vals.append(hv)

        monthly_data.append({
            "ym": ym,
            "videos": video_counts[ym],
            "chars": char_counts[ym],
            "neg_norm": round(nv, 1),
            "emph_norm": round(ev, 1),
            "hedge_norm": round(hv, 1),
            "pos_norm": round(pv, 1),
            "net_sentiment": round(pv - nv, 1),
        })

    r_ne, p_ne = pearsonr(neg_vals, emph_vals)
    r_nh, p_nh = pearsonr(neg_vals, hedge_vals)

    return {
        "speaker": speaker,
        "display_name": speaker.replace("_", " ").title().replace("Ray Dalio","Ray Dalio"),
        "n_videos": sum(video_counts.values()),
        "n_months": len(months),
        "r_neg_emphatic": round(r_ne, 4),
        "p_neg_emphatic": round(p_ne, 6),
        "r_neg_hedged": round(r_nh, 4),
        "p_neg_hedged": round(p_nh, 6),
        "months": months,
        "monthly_data": monthly_data,
    }

# ── Main ──
print("=" * 70)
print("KEYWORD LEXICON SCORING — All Speakers")
print("=" * 70)

dates = load_video_dates()
all_results = {}

for spk, display in SPEAKERS:
    result = process_speaker(spk, dates)
    if result is None:
        print(f"\n{display}: no data or insufficient months")
        continue

    all_results[spk] = result
    r, p = result["r_neg_emphatic"], result["p_neg_emphatic"]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"\n{display} ({result['n_videos']} videos, {result['n_months']} months)")
    print(f"  r(neg, emph) = {r:.4f}{sig}  (p={p:.6f})")
    print(f"  r(neg, hedge) = {result['r_neg_hedged']:.4f}  (p={result['p_neg_hedged']:.6f})")

# ── Summary table ──
print("\n" + "=" * 70)
print("CROSS-SPEAKER SUMMARY")
print("=" * 70)
print(f"{'Speaker':<20} {'Videos':>7} {'Months':>7} {'r(neg,emph)':>13} {'p-value':>12}")
print("-" * 62)
for spk, display in SPEAKERS:
    r = all_results.get(spk)
    if not r:
        continue
    p = r["p_neg_emphatic"]
    p_str = f"{p:.4f}" if p and p > 0.0001 else "<0.0001" if p else "n/a"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{display:<20} {r['n_videos']:>7} {r['n_months']:>7} {r['r_neg_emphatic']:>13.4f}{sig} {p_str:>12}")

# ── Save ──
out = {
    "description": "Keyword lexicon scoring — per-video avg normalization",
    "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "lexicon_patterns": {
        "positive": sum(len(v) for v in POSITIVE.values()),
        "negative": sum(len(v) for v in NEGATIVE.values()),
        "modality": sum(len(v) for v in MODALITY.values()),
    },
    "results": {k: {
        "speaker": v["speaker"],
        "n_videos": v["n_videos"],
        "n_months": v["n_months"],
        "r_neg_emphatic": v["r_neg_emphatic"],
        "p_neg_emphatic": v["p_neg_emphatic"],
        "r_neg_hedged": v["r_neg_hedged"],
        "p_neg_hedged": v["p_neg_hedged"],
    } for k, v in all_results.items()},
}

out_path = os.path.join(RESULTS_DIR, "keyword_lexicon_eval.json")
with open(out_path, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")

# Also save Dalio monthly breakdown for chart
if "ray_dalio" in all_results:
    dalio = all_results["ray_dalio"]
    dalio_out = {
        "title": "Dalio Negative Affect vs Emphatic Certainty — Keyword Lexicon",
        "n_videos": dalio["n_videos"],
        "n_months": dalio["n_months"],
        "r": dalio["r_neg_emphatic"],
        "p": dalio["p_neg_emphatic"],
        "months": dalio["months"],
        "data": dalio["monthly_data"],
    }
    d_path = os.path.join(RESULTS_DIR, "keyword_monthly_dalio.json")
    with open(d_path, "w") as f:
        json.dump(dalio_out, f, ensure_ascii=False, indent=2)
    print(f"Saved Dalio monthly: {d_path}")
