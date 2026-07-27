#!/usr/bin/env python3
"""
compare_diarization_keyword — Keyword-based VTT vs AssemblyAI comparison.

Applies keyword lexicon scoring per-video (same unit as LLM annotation),
computes per-speaker r(neg,emph) and r(neg,hedged), cross-source deltas,
aggregated by domain.

Input:
  - data/sentences/<source>/<speaker>/<video_id>.txt

Output:
  - results/keyword_comparison.json

Usage:
  python3 src/03_analysis/compare_diarization_keyword.py
"""

import json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keyword_lexicon import count_matches, NEGATIVE, MODALITY

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    PROJECT_DIR, ANALYSIS_RESULTS_DIR,
    get_domain, DOMAIN_ORDER, DOMAIN_LABELS
)
from src.utils.common import save_json, VIDEO_TXT_RE

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SENTENCES_VTT = os.path.join(PROJECT_DIR, "data", "sentences", "vtt")
SENTENCES_ASR = os.path.join(PROJECT_DIR, "data", "sentences", "assemblyai")
NORM_BASE = 30000


def process_source(sentences_dir, speaker):
    """Keyword-score one source for one speaker at video level."""
    spk_dir = os.path.join(sentences_dir, speaker)
    if not os.path.isdir(spk_dir):
        return None

    neg_cats = list(NEGATIVE.keys())
    emph_cat = "Emphatic/Certain"
    hedge_cat = "Hedging/Doubt"

    neg_vals, emph_vals, hedge_vals = [], [], []
    n_videos = 0

    for fname in sorted(os.listdir(spk_dir)):
        if not VIDEO_TXT_RE.match(fname):
            continue
        with open(os.path.join(spk_dir, fname)) as f:
            text = f.read().lower()
        text = re.sub(r'[^a-z\s/-]', ' ', text)

        # Normalize by video character length
        norm = len(text) / NORM_BASE
        if norm < 0.001:
            continue

        matches = count_matches(text)
        neg_raw = sum(matches.get(c, {}).get("count", 0) for c in neg_cats)
        emph_raw = matches.get(emph_cat, {}).get("count", 0)
        hedge_raw = matches.get(hedge_cat, {}).get("count", 0)

        neg_vals.append(neg_raw / norm)
        emph_vals.append(emph_raw / norm)
        hedge_vals.append(hedge_raw / norm)
        n_videos += 1

    if n_videos < 4:
        return None

    stats = {"n_videos": n_videos}

    if HAS_SCIPY and len(neg_vals) >= 4:
        r_ne, p_ne = pearsonr(neg_vals, emph_vals)
        r_nh, p_nh = pearsonr(neg_vals, hedge_vals)
        stats.update({
            "r_neg_emph": round(r_ne, 4), "p_neg_emph": round(p_ne, 6),
            "r_neg_hedged": round(r_nh, 4), "p_neg_hedged": round(p_nh, 6),
        })
    else:
        stats.update({"r_neg_emph": None, "r_neg_hedged": None,
                       "p_neg_emph": None, "p_neg_hedged": None})
    return stats


def discover_common_speakers():
    vtt = set(d for d in os.listdir(SENTENCES_VTT)
              if os.path.isdir(os.path.join(SENTENCES_VTT, d)))
    asr = set(d for d in os.listdir(SENTENCES_ASR)
              if os.path.isdir(os.path.join(SENTENCES_ASR, d)))
    return sorted(vtt & asr)


def aggregate_by_domain(results):
    domain_data = defaultdict(lambda: {
        "Δr_neg_emph": [], "Δr_neg_hedged": [], "reversal_count": 0,
        "vtt_r_ne": [], "vtt_r_nh": [], "asr_r_ne": [], "asr_r_nh": [],
        "speakers": [],
    })

    for r in results:
        d = r["domain"]
        dd = domain_data[d]
        dd["speakers"].append(r["speaker"])

        vtt_ne = r["vtt"].get("r_neg_emph")
        asr_ne = r["assemblyai"].get("r_neg_emph")
        vtt_nh = r["vtt"].get("r_neg_hedged")
        asr_nh = r["assemblyai"].get("r_neg_hedged")

        if vtt_ne is not None and asr_ne is not None:
            dd["Δr_neg_emph"].append(asr_ne - vtt_ne)
            dd["vtt_r_ne"].append(vtt_ne)
            dd["asr_r_ne"].append(asr_ne)
            if vtt_ne * asr_ne < 0:
                dd["reversal_count"] += 1
        if vtt_nh is not None and asr_nh is not None:
            dd["Δr_neg_hedged"].append(asr_nh - vtt_nh)
            dd["vtt_r_nh"].append(vtt_nh)
            dd["asr_r_nh"].append(asr_nh)

    summary = {}
    for domain in DOMAIN_ORDER:
        if domain not in domain_data:
            continue
        d = domain_data[domain]

        def mean_std(vals):
            if not vals: return None, None
            return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

        def mean_abs(vals):
            if not vals: return None
            return round(float(np.mean([abs(v) for v in vals])), 4)

        summary[domain] = {
            "label": DOMAIN_LABELS.get(domain, domain),
            "n_speakers": len(d["speakers"]),
            "Δr_neg_emph_mean": mean_std(d["Δr_neg_emph"])[0],
            "Δr_neg_hedged_mean": mean_std(d["Δr_neg_hedged"])[0],
            "mean_abs_delta_r_ne": mean_abs(d["Δr_neg_emph"]),
            "mean_abs_delta_r_nh": mean_abs(d["Δr_neg_hedged"]),
            "sign_reversal_pct": round(d["reversal_count"] / len(d["speakers"]), 3) if d["speakers"] else 0,
            "vtt_r_ne_mean": mean_std(d["vtt_r_ne"])[0],
            "asr_r_ne_mean": mean_std(d["asr_r_ne"])[0],
            "vtt_r_nh_mean": mean_std(d["vtt_r_nh"])[0],
            "asr_r_nh_mean": mean_std(d["asr_r_nh"])[0],
        }
    return summary


def main():
    speakers = discover_common_speakers()
    # Remove politics speakers
    speakers = [s for s in speakers if get_domain(s) != "politics"]
    if not speakers:
        print("No common speakers found.")
        sys.exit(1)

    print(f"Keyword comparison (video-level): {len(speakers)} speakers\n")

    results = []
    for spk in speakers:
        s_vtt = process_source(SENTENCES_VTT, spk)
        s_asr = process_source(SENTENCES_ASR, spk)
        if s_vtt is None or s_asr is None:
            print(f"  {spk}: skipped (insufficient data)")
            continue

        delta = {}
        for k in ("r_neg_emph", "r_neg_hedged"):
            delta[k] = round(s_asr[k] - s_vtt[k], 4) if s_vtt[k] is not None and s_asr[k] is not None else None

        domain = get_domain(spk)
        vtt_ne = s_vtt.get("r_neg_emph", "?")
        asr_ne = s_asr.get("r_neg_emph", "?")
        print(f"  {spk} ({domain}): VTT r(N,E)={vtt_ne}  ASR r(N,E)={asr_ne}  VTT n_vid={s_vtt['n_videos']}")

        results.append({
            "speaker": spk, "domain": domain,
            "vtt": s_vtt, "assemblyai": s_asr, "delta": delta,
        })

    domain_summary = aggregate_by_domain(results)

    print(f"\n=== Domain Summary (video-level) ===")
    print(f"{'Domain':<25} {'n':>3} {'Δr(N,E)':>9} {'Δr(N,H)':>9} {'Rev%':>6}")
    print("-" * 55)
    for domain in DOMAIN_ORDER:
        if domain in domain_summary:
            s = domain_summary[domain]
            print(f"{s['label']:<25} {s['n_speakers']:>3} "
                  f"{s['Δr_neg_emph_mean'] or '?':>9} {s['Δr_neg_hedged_mean'] or '?':>9} "
                  f"{s['sign_reversal_pct']:>6.0%}")

    out = {"method": "keyword (video-level)", "n_speakers": len(results),
           "per_speaker": results, "domain_summary": domain_summary}
    out_path = os.path.join(ANALYSIS_RESULTS_DIR, "keyword_comparison.json")
    save_json(out_path, out)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
