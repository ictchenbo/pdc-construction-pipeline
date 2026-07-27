#!/usr/bin/env python3
"""
compare_diarization_llm — LLM-based VTT vs AssemblyAI comparison.

Per-speaker analysis: proportions, r(neg,emph), r(neg,hedged) for each source,
cross-source deltas, aggregated by domain.

Input:
  - data/llm_classification/deepseek/<source>/<speaker>/<video_id>.jsonl

Output:
  - results/llm_comparison.json

Usage:
  python3 src/03_analysis/compare_diarization_llm.py
"""

import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    PROJECT_DIR, ANALYSIS_RESULTS_DIR,
    get_domain, DOMAIN_ORDER, DOMAIN_LABELS
)
from src.utils.common import load_jsonl, save_json, VIDEO_JSONL_RE

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

CLASSIFY_DIR = os.path.join(PROJECT_DIR, "data", "llm_classification", "deepseek")


def load_speaker(speaker, source):
    """Per-video annotation counts for one speaker+source."""
    spk_dir = os.path.join(CLASSIFY_DIR, source, speaker)
    if not os.path.isdir(spk_dir):
        return {}
    videos = {}
    for f in sorted(os.listdir(spk_dir)):
        if not VIDEO_JSONL_RE.match(f):
            continue
        vid = f.replace(".jsonl", "")
        counts = {"pos": 0, "neg": 0, "emphatic": 0, "hedged": 0, "total": 0}
        for r in load_jsonl(os.path.join(spk_dir, f)):
            counts["total"] += 1
            v, m = r.get("valence"), r.get("modality")
            if v == "positive":
                counts["pos"] += 1
            elif v == "negative":
                counts["neg"] += 1
            if m == "emphatic":
                counts["emphatic"] += 1
            elif m == "hedged":
                counts["hedged"] += 1
        if counts["total"] >= 5:
            videos[vid] = counts
    return videos


def speaker_stats(videos):
    """Aggregate proportions and cross-dimensional correlations from per-video counts."""
    if not videos:
        return None
    total = sum(v["total"] for v in videos.values())
    pos = sum(v["pos"] for v in videos.values())
    neg = sum(v["neg"] for v in videos.values())
    emph = sum(v["emphatic"] for v in videos.values())
    hedged = sum(v["hedged"] for v in videos.values())

    stats = {
        "n_videos": len(videos), "n_sentences": total,
        "p_positive": round(pos / total, 4) if total else 0,
        "p_negative": round(neg / total, 4) if total else 0,
        "p_emphatic": round(emph / total, 4) if total else 0,
        "p_hedged": round(hedged / total, 4) if total else 0,
    }

    # Per-video cross-dimensional correlations
    neg_pcts = [v["neg"] / v["total"] for v in videos.values()]
    emph_pcts = [v["emphatic"] / v["total"] for v in videos.values()]
    hedged_pcts = [v["hedged"] / v["total"] for v in videos.values()]

    if HAS_SCIPY and len(neg_pcts) >= 4:
        r_ne, p_ne = pearsonr(neg_pcts, emph_pcts)
        r_nh, p_nh = pearsonr(neg_pcts, hedged_pcts)
        stats["r_neg_emph"] = round(r_ne, 4)
        stats["p_neg_emph"] = round(p_ne, 6)
        stats["r_neg_hedged"] = round(r_nh, 4)
        stats["p_neg_hedged"] = round(p_nh, 6)
    else:
        for k in ("r_neg_emph", "p_neg_emph", "r_neg_hedged", "p_neg_hedged"):
            stats[k] = None

    return stats


def discover_common_speakers():
    vtt = set(d for d in os.listdir(os.path.join(CLASSIFY_DIR, "vtt"))
              if os.path.isdir(os.path.join(CLASSIFY_DIR, "vtt", d)))
    asr = set(d for d in os.listdir(os.path.join(CLASSIFY_DIR, "assemblyai"))
              if os.path.isdir(os.path.join(CLASSIFY_DIR, "assemblyai", d)))
    return sorted(vtt & asr)


# ── Domain aggregation ──
def aggregate_by_domain(results):
    """Per-domain mean, std for deltas and r differences."""
    domain_data = defaultdict(lambda: {
        "Δneg": [], "Δemph": [], "Δhedged": [], "Δr_neg_emph": [], "Δr_neg_hedged": [],
        "vtt_r_ne": [], "vtt_r_nh": [], "asr_r_ne": [], "asr_r_nh": [],
        "reversal_count": 0, "speakers": [],
    })

    for r in results:
        domain = r["domain"]
        d = domain_data[domain]
        d["speakers"].append(r["speaker"])
        d["Δneg"].append(r["delta"]["p_negative"])
        d["Δemph"].append(r["delta"]["p_emphatic"])
        d["Δhedged"].append(r["delta"]["p_hedged"])

        vtt_ne = r["vtt"].get("r_neg_emph")
        asr_ne = r["assemblyai"].get("r_neg_emph")
        vtt_nh = r["vtt"].get("r_neg_hedged")
        asr_nh = r["assemblyai"].get("r_neg_hedged")

        if vtt_ne is not None and asr_ne is not None:
            d["Δr_neg_emph"].append(asr_ne - vtt_ne)
            d["vtt_r_ne"].append(vtt_ne)
            d["asr_r_ne"].append(asr_ne)
            if vtt_ne * asr_ne < 0:  # sign reversal
                d["reversal_count"] += 1
        if vtt_nh is not None and asr_nh is not None:
            d["Δr_neg_hedged"].append(asr_nh - vtt_nh)
            d["vtt_r_nh"].append(vtt_nh)
            d["asr_r_nh"].append(asr_nh)

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
            "Δneg_mean": mean_std(d["Δneg"])[0],
            "Δemph_mean": mean_std(d["Δemph"])[0],
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
    if not speakers:
        print("No common speakers found.")
        sys.exit(1)

    print(f"LLM comparison: {len(speakers)} speakers\n")

    results = []
    for spk in speakers:
        vtt = load_speaker(spk, "vtt")
        asr = load_speaker(spk, "assemblyai")
        s_vtt = speaker_stats(vtt)
        s_asr = speaker_stats(asr)
        if s_vtt is None or s_asr is None:
            continue

        delta = {
            k: round(s_asr[k] - s_vtt[k], 4) if s_vtt[k] is not None and s_asr[k] is not None else None
            for k in ("p_positive", "p_negative", "p_emphatic", "p_hedged")
        }

        domain = get_domain(spk)
        vtt_ne = s_vtt.get("r_neg_emph", "?")
        asr_ne = s_asr.get("r_neg_emph", "?")
        print(f"  {spk} ({domain}): VTT r(N,E)={vtt_ne}  ASR r(N,E)={asr_ne}")

        results.append({
            "speaker": spk, "domain": domain,
            "vtt": s_vtt, "assemblyai": s_asr, "delta": delta,
        })

    domain_summary = aggregate_by_domain(results)

    # Print domain summary
    print(f"\n=== Domain Summary ===")
    print(f"{'Domain':<25} {'n':>3} {'Δneg':>7} {'Δemph':>7} {'Δr(N,E)':>9} {'Δr(N,H)':>9} {'Rev%':>6}")
    print("-" * 72)
    for domain in DOMAIN_ORDER:
        if domain in domain_summary:
            s = domain_summary[domain]
            print(f"{s['label']:<25} {s['n_speakers']:>3} "
                  f"{s['Δneg_mean'] or '?':>7} {s['Δemph_mean'] or '?':>7} "
                  f"{s['Δr_neg_emph_mean'] or '?':>9} {s['Δr_neg_hedged_mean'] or '?':>9} "
                  f"{s['sign_reversal_pct']:>6.0%}")

    out = {"method": "LLM", "n_speakers": len(results),
           "per_speaker": results, "domain_summary": domain_summary}
    out_path = os.path.join(ANALYSIS_RESULTS_DIR, "llm_comparison.json")
    save_json(out_path, out)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
