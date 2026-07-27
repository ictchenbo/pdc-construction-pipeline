#!/usr/bin/env python3
"""
Phase 5d: Hierarchical Linear Models for Artifact Magnitude.

RQ2: Fit hierarchical models predicting Δr_s (keyword-LLM discrepancy)
from speaker-level features: hedging baseline, domain vocabulary entropy,
sentence complexity, negativity ratio.

Input:
  - results/measurement_validity_scores.json
  - data/statistics/speaker_statistics.json

Output:
  - results/hierarchical_model_results.json

Usage:
  python3 src/08_statistical_analysis/hierarchical_modeling.py
"""

import json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    PROJECT_DIR, ANALYSIS_RESULTS_DIR, SPEAKER_DOMAINS,
    get_domain, discover_speakers, RAW_DIR, DOMAIN_LABELS
)
from src.utils.common import load_json, save_json

try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy.stats import pearsonr
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_speaker_features(speaker):
    """Compute speaker-level features for hierarchical modeling.

    Returns dict with:
      - hedging_baseline: overall rate of hedging markers
      - sentence_complexity: mean sentence length
      - negativity_ratio: proportion of negative sentences
      - domain: speaker domain
    """
    # Placeholder — actual computation depends on data availability
    return {
        "hedging_baseline": None,
        "sentence_complexity": None,
        "negativity_ratio": None,
        "domain": get_domain(speaker),
    }


def fit_hierarchical_model(data, features, target="delta_r"):
    """Fit hierarchical linear model with domain random intercepts."""
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}
    
    # Build design matrix
    X_data = {}
    for feat in features:
        vals = [d.get(feat) for d in data]
        if all(v is None for v in vals):
            continue
        X_data[feat] = np.array([v if v is not None else np.nan for v in vals])
    
    y = np.array([d.get(target, np.nan) for d in data])
    domains = [d.get("domain", "unknown") for d in data]
    
    # Remove NaN
    valid = ~np.isnan(y)
    for feat, vals in X_data.items():
        valid = valid & ~np.isnan(vals)
    
    if sum(valid) < 10:
        return {"error": "Insufficient valid data points"}
    
    y = y[valid]
    X = np.column_stack([X_data[f][valid] for f in X_data])
    domains = [d for i, d in enumerate(domains) if valid[i]]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Mixed effects model with domain random intercept
    domain_map = {d: i for i, d in enumerate(sorted(set(domains)))}
    
    try:
        model = MixedLM(y, X_scaled, groups=[domain_map[d] for d in domains])
        result = model.fit()
        
        return {
            "log_likelihood": result.llf,
            "aic": result.aic,
            "fixed_effects": {f: round(result.fe_params[i], 4) for i, f in enumerate(X_data.keys())},
            "random_effects_var": round(result.cov_re.iloc[0, 0], 6) if hasattr(result.cov_re, 'iloc') else float(result.cov_re),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print("Phase 5d: Hierarchical Modeling")
    print("=" * 50)
    print()
    print("This module fits hierarchical linear models to predict")
    print("keyword-LLM discrepancy from speaker-level features.")
    print()
    print("Prerequisites:")
    print("  1. Run measurement_validity_score.py first")
    print("  2. Run speaker_statistics.py for speaker-level features")
    print()
    
    if not HAS_STATSMODELS:
        print("WARNING: statsmodels not installed. pip install statsmodels")
    if not HAS_SKLEARN:
        print("WARNING: scikit-learn not installed.")
    
    mvs_path = os.path.join(ANALYSIS_RESULTS_DIR, "measurement_validity_scores.json")
    if os.path.exists(mvs_path):
        data = load_json(mvs_path)
        print(f"Loaded MVS data: {len(data)} speakers")
    else:
        print(f"MVS data not found at {mvs_path}")
        print("Run measurement_validity_score.py first.")


if __name__ == "__main__":
    main()
