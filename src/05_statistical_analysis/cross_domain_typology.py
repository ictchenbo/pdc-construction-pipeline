#!/usr/bin/env python3
"""
Phase 5b: Cross-Domain Typology of Valence-Modality Coupling.

RQ3: Compare cross-dimensional correlation patterns across speaker domains.

Hypotheses:
  - Academia & Policy: Stronger negative-hedging coupling
  - Media & Investing: Weaker or reversed coupling
  - Politics: Intermediate

Input:
  - data/llm_classification/<provider>/<speaker>/<YYYY-MM>.jsonl

Output:
  - results/domain_typology.json
  - figures/domain_coupling_bars.png (optional)

Usage:
  python3 src/08_statistical_analysis/cross_domain_typology.py
  python3 src/08_statistical_analysis/cross_domain_typology.py --provider deepseek
"""

import argparse, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    PROJECT_DIR, ANALYSIS_RESULTS_DIR, SPEAKER_DOMAINS,
    get_domain, get_display_name, discover_speakers, RAW_DIR,
    MIN_MONTHS_FOR_CORRELATION, DOMAIN_ORDER, DOMAIN_LABELS
)
from src.utils.common import load_jsonl, save_json

try:
    from scipy.stats import pearsonr, f_oneway, kruskal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_speaker_correlations(speaker, provider="deepseek"):
    """Compute r(neg,emph) and r(neg,hedged) for one speaker from LLM data."""
    llm_dir = os.path.join(PROJECT_DIR, "data", "llm_classification", provider, speaker)
    if not os.path.isdir(llm_dir):
        return None
    
    monthly = defaultdict(lambda: {"neg": 0, "emph": 0, "hedged": 0, "total": 0})
    
    for fname in sorted(os.listdir(llm_dir)):
        if not fname.endswith(".jsonl") or fname.startswith("_"):
            continue
        ym = fname.replace(".jsonl", "")
        for r in load_jsonl(os.path.join(llm_dir, fname)):
            monthly[ym]["total"] += 1
            if r.get("valence") == "negative":
                monthly[ym]["neg"] += 1
            if r.get("modality") == "emphatic":
                monthly[ym]["emph"] += 1
            if r.get("modality") == "hedged":
                monthly[ym]["hedged"] += 1
    
    months = sorted(monthly.keys())
    neg_pcts = [monthly[m]["neg"] / monthly[m]["total"] for m in months if monthly[m]["total"] >= 5]
    emph_pcts = [monthly[m]["emph"] / monthly[m]["total"] for m in months if monthly[m]["total"] >= 5]
    hedged_pcts = [monthly[m]["hedged"] / monthly[m]["total"] for m in months if monthly[m]["total"] >= 5]
    
    result = {"speaker": speaker, "domain": get_domain(speaker), "n_months": len(months)}
    
    if len(neg_pcts) >= MIN_MONTHS_FOR_CORRELATION and HAS_SCIPY:
        r_ne, p_ne = pearsonr(neg_pcts, emph_pcts)
        r_nh, p_nh = pearsonr(neg_pcts, hedged_pcts)
        result["r_neg_emph"] = round(r_ne, 4)
        result["p_neg_emph"] = round(p_ne, 4)
        result["r_neg_hedged"] = round(r_nh, 4)
        result["p_neg_hedged"] = round(p_nh, 4)
    else:
        result["r_neg_emph"] = None
        result["r_neg_hedged"] = None
    
    return result


def domain_anova(all_results):
    """Test whether r(neg,hedged) differs significantly across domains."""
    domain_values = defaultdict(list)
    for r in all_results:
        if r and r.get("r_neg_hedged") is not None:
            domain_values[r["domain"]].append(r["r_neg_hedged"])
    
    if not HAS_SCIPY or len(domain_values) < 2:
        return {"note": "Insufficient data or scipy missing"}
    
    groups = [vals for vals in domain_values.values() if len(vals) >= 2]
    if len(groups) >= 2:
        h_stat, p_val = kruskal(*groups)
        return {"kruskal_h": round(h_stat, 4), "p": round(p_val, 4)}
    return {"note": "Insufficient groups for test"}


def build_domain_summary(all_results):
    """Build per-domain summary of coupling patterns."""
    domain_summary = defaultdict(lambda: {
        "speakers": [], "r_neg_emph": [], "r_neg_hedged": [],
        "emph_sig_count": 0, "hedged_sig_count": 0
    })
    
    for r in all_results:
        if not r:
            continue
        domain = r.get("domain", "unknown")
        domain_summary[domain]["speakers"].append(r["speaker"])
        
        if r.get("r_neg_emph") is not None:
            domain_summary[domain]["r_neg_emph"].append(r["r_neg_emph"])
            if r.get("p_neg_emph", 1.0) < 0.05:
                domain_summary[domain]["emph_sig_count"] += 1
        
        if r.get("r_neg_hedged") is not None:
            domain_summary[domain]["r_neg_hedged"].append(r["r_neg_hedged"])
            if r.get("p_neg_hedged", 1.0) < 0.05:
                domain_summary[domain]["hedged_sig_count"] += 1
    
    # Aggregate
    summary = {}
    for domain in DOMAIN_ORDER:
        if domain in domain_summary:
            d = domain_summary[domain]
            ne = d["r_neg_emph"]
            nh = d["r_neg_hedged"]
            summary[domain] = {
                "label": DOMAIN_LABELS.get(domain, domain),
                "n_speakers": len(d["speakers"]),
                "r_neg_emph_mean": round(np.mean(ne), 4) if ne else None,
                "r_neg_emph_std": round(np.std(ne), 4) if ne else None,
                "r_neg_hedged_mean": round(np.mean(nh), 4) if nh else None,
                "r_neg_hedged_std": round(np.std(nh), 4) if nh else None,
                "emph_sig_fraction": round(d["emph_sig_count"] / len(d["speakers"]), 3) if d["speakers"] else 0,
                "hedged_sig_fraction": round(d["hedged_sig_count"] / len(d["speakers"]), 3) if d["speakers"] else 0,
            }
    
    return summary


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description="Phase 5b: Cross-Domain Typology")
    parser.add_argument("--provider", default="deepseek")
    args = parser.parse_args()
    
    if not HAS_SCIPY:
        print("ERROR: scipy not installed")
        sys.exit(1)
    
    speakers = discover_speakers(RAW_DIR)
    print(f"Computing domain typology for {len(speakers)} speakers\n")
    
    all_results = []
    for spk in speakers:
        r = compute_speaker_correlations(spk, args.provider)
        if r:
            all_results.append(r)
            print(f"  {spk} ({r['domain']}): "
                  f"r(neg,emph)={r.get('r_neg_emph','N/A')}, "
                  f"r(neg,hedged)={r.get('r_neg_hedged','N/A')}, "
                  f"n={r['n_months']}")
    
    summary = build_domain_summary(all_results)
    anova_result = domain_anova(all_results)
    
    print("\n=== Domain Typology Summary ===")
    print(f"{'Domain':<25} {'n':>4} {'r(NE)':>8} {'r(NH)':>8} {'E-sig%':>7} {'H-sig%':>7}")
    print("-" * 65)
    for domain in DOMAIN_ORDER:
        if domain in summary:
            s = summary[domain]
            print(f"{s['label']:<25} {s['n_speakers']:>4} "
                  f"{s['r_neg_emph_mean'] or 'N/A':>8} "
                  f"{s['r_neg_hedged_mean'] or 'N/A':>8} "
                  f"{s['emph_sig_fraction']:>7.2f} "
                  f"{s['hedged_sig_fraction']:>7.2f}")
    
    print(f"\nKruskal-Wallis test: H={anova_result.get('kruskal_h','N/A')}, p={anova_result.get('p','N/A')}")
    
    # Save
    out = {"per_speaker": all_results, "domain_summary": summary, "anova": anova_result}
    out_path = os.path.join(ANALYSIS_RESULTS_DIR, "domain_typology.json")
    save_json(out_path, out)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
