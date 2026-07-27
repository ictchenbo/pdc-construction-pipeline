#!/usr/bin/env python3
"""
compare_synthesis — Cross-Method Pipeline Sensitivity Synthesis.

Loads LLM and keyword comparison results, merges them into a single view
per speaker/domain, and reports which method is more sensitive to the
preprocessing pipeline choice (VTT vs AssemblyAI).

Input:
  - results/llm_comparison.json
  - results/keyword_comparison.json

Output:
  - results/pipeline_sensitivity.json

Usage:
  python3 src/03_analysis/compare_synthesis.py
"""

import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import ANALYSIS_RESULTS_DIR, DOMAIN_ORDER, DOMAIN_LABELS
from src.utils.common import load_json


def load(path, label):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None
    return load_json(path)


def main():
    llm = load(os.path.join(ANALYSIS_RESULTS_DIR, "llm_comparison.json"), "LLM")
    kw = load(os.path.join(ANALYSIS_RESULTS_DIR, "keyword_comparison.json"), "keyword")

    if llm is None or kw is None:
        print("Run compare_diarization_llm.py and compare_diarization_keyword.py first.")
        sys.exit(1)

    # Index by speaker
    llm_by_spk = {r["speaker"]: r for r in llm.get("per_speaker", [])}
    kw_by_spk = {r["speaker"]: r for r in kw.get("per_speaker", [])}
    common = sorted(set(llm_by_spk) & set(kw_by_spk))

    print(f"Pipeline Sensitivity Synthesis: {len(common)} speakers\n")

    # ── Per-speaker comparison ──
    rows = []
    for spk in common:
        l, k = llm_by_spk[spk], kw_by_spk[spk]
        domain = l["domain"]

        # LLM deltas
        l_delta_ne = l["vtt"].get("r_neg_emph") and l["assemblyai"].get("r_neg_emph") and \
            round(l["assemblyai"]["r_neg_emph"] - l["vtt"]["r_neg_emph"], 4)
        l_delta_nh = l["vtt"].get("r_neg_hedged") and l["assemblyai"].get("r_neg_hedged") and \
            round(l["assemblyai"]["r_neg_hedged"] - l["vtt"]["r_neg_hedged"], 4)
        # Keyword deltas
        k_delta_ne = k["vtt"].get("r_neg_emph") and k["assemblyai"].get("r_neg_emph") and \
            round(k["assemblyai"]["r_neg_emph"] - k["vtt"]["r_neg_emph"], 4)
        k_delta_nh = k["vtt"].get("r_neg_hedged") and k["assemblyai"].get("r_neg_hedged") and \
            round(k["assemblyai"]["r_neg_hedged"] - k["vtt"]["r_neg_hedged"], 4)

        # Which method is more stable? (smaller |Δ|)
        def more_stable(d1, d2):
            if d1 is None or d2 is None:
                return "?"
            return "LLM" if abs(d1) < abs(d2) else "KW" if abs(d2) < abs(d1) else "tie"

        print(f"  {spk} ({domain}): "
              f"LLM Δr(N,E)={l_delta_ne}  KW Δr(N,E)={k_delta_ne}  "
              f"→ {more_stable(l_delta_ne, k_delta_ne)} more stable")

        rows.append({
            "speaker": spk, "domain": domain,
            "llm": {"Δr_neg_emph": l_delta_ne, "Δr_neg_hedged": l_delta_nh,
                    "Δneg": l["delta"]["p_negative"]},
            "keyword": {"Δr_neg_emph": k_delta_ne, "Δr_neg_hedged": k_delta_nh},
        })

    # ── Domain-level summary ──
    domain_agg = defaultdict(lambda: {
        "llm_Δr_ne": [], "kw_Δr_ne": [], "llm_stable": 0, "kw_stable": 0, "n": 0,
    })

    for r in rows:
        d = domain_agg[r["domain"]]
        d["n"] += 1
        if r["llm"]["Δr_neg_emph"] is not None:
            d["llm_Δr_ne"].append(abs(r["llm"]["Δr_neg_emph"]))
        if r["keyword"]["Δr_neg_emph"] is not None:
            d["kw_Δr_ne"].append(abs(r["keyword"]["Δr_neg_emph"]))

        l_ne, k_ne = r["llm"]["Δr_neg_emph"], r["keyword"]["Δr_neg_emph"]
        if l_ne is not None and k_ne is not None:
            if abs(l_ne) < abs(k_ne):
                d["llm_stable"] += 1
            elif abs(k_ne) < abs(l_ne):
                d["kw_stable"] += 1

    print(f"\n=== Pipeline Sensitivity by Domain ===")
    print(f"{'Domain':<25} {'n':>3} {'LLM|Δr|':>8} {'KW|Δr|':>8} {'LLM更稳':>8} {'KW更稳':>8}")
    print("-" * 65)
    for domain in DOMAIN_ORDER:
        if domain not in domain_agg:
            continue
        d = domain_agg[domain]

        def m(vals):
            return round(float(np.mean(vals)), 4) if vals else None

        print(f"{DOMAIN_LABELS.get(domain, domain):<25} {d['n']:>3} "
              f"{str(m(d['llm_Δr_ne'])):>8} {str(m(d['kw_Δr_ne'])):>8} "
              f"{d['llm_stable']:>8} {d['kw_stable']:>8}")

    out = {
        "description": "Cross-method pipeline sensitivity synthesis",
        "n_speakers": len(rows),
        "per_speaker": rows,
        "domain_summary": {
            domain: {
                "label": DOMAIN_LABELS.get(domain, domain),
                "n": d["n"],
                "mean_abs_llm_delta_r_ne": round(float(np.mean(d["llm_Δr_ne"])), 4) if d["llm_Δr_ne"] else None,
                "mean_abs_kw_delta_r_ne": round(float(np.mean(d["kw_Δr_ne"])), 4) if d["kw_Δr_ne"] else None,
                "llm_stable_count": d["llm_stable"],
                "kw_stable_count": d["kw_stable"],
            }
            for domain, d in domain_agg.items()
        },
    }

    out_path = os.path.join(ANALYSIS_RESULTS_DIR, "pipeline_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
