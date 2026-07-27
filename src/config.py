#!/usr/bin/env python3
"""
Centralized configuration for the PDC pipeline.

All paths, model settings, speaker domains, and analysis parameters
are defined here. Individual phase scripts import from this module.

Usage:
    from src.config import PROJECT_DIR, RAW_DIR, SPEAKER_DOMAINS
"""

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
CLEANED_DIR = os.path.join(PROJECT_DIR, "data", "cleaned")
GROUPED_BY_MONTH_DIR = os.path.join(PROJECT_DIR, "data", "grouped_by_month")
GROUPED_SENTENCES_DIR = os.path.join(PROJECT_DIR, "data", "grouped_sentences")
LLM_CLASSIFICATION_DIR = os.path.join(PROJECT_DIR, "data", "llm_classification")
VALIDATION_DIR = os.path.join(PROJECT_DIR, "data", "validation")
BASELINE_RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "baseline_results")
ANALYSIS_RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# ── Ensure output dirs ────────────────────────────────────────────────────
for _d in [CLEANED_DIR, GROUPED_BY_MONTH_DIR, GROUPED_SENTENCES_DIR,
           LLM_CLASSIFICATION_DIR, VALIDATION_DIR, BASELINE_RESULTS_DIR,
           ANALYSIS_RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── .env loading ──────────────────────────────────────────────────────────
def _load_dotenv():
    dotenv_path = os.path.join(PROJECT_DIR, ".env")
    if os.path.exists(dotenv_path):
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                os.environ.setdefault(key, val)

_load_dotenv()

# ── LLM Configuration ─────────────────────────────────────────────────────
LLM_CONFIGS = {
    "deepseek": {
        "api_key_env": "CLEAN_API_KEY",
        "api_base_env": "CLEAN_API_BASE",
        "model_env": "CLEAN_MODEL",
        "default_model": "deepseek-v4-flash",
    },
    "deepseek_pro": {
        "api_key_env": "CLEAN_API_KEY",
        "api_base_env": "CLEAN_API_BASE",
        "model_env": "CLEAN_MODEL1",
        "default_model": "deepseek-v4-pro",
    },
    "gpt5": {
        "api_key_env": "AUDIT_API_KEY",
        "api_base_env": "AUDIT_API_BASE",
        "model_env": "AUDIT_MODEL",
        "default_model": "gpt-5.5",
    }
}

def get_llm_config(provider: str) -> dict:
    """Resolve LLM provider config with env var overrides."""
    cfg = LLM_CONFIGS.get(provider)
    if not cfg:
        raise ValueError(f"Unknown LLM provider: {provider}. Known: {list(LLM_CONFIGS)}")
    api_key = os.environ.get(cfg["api_key_env"], "")
    api_base = os.environ.get(cfg["api_base_env"], "").rstrip("/")
    model = os.environ.get(cfg["model_env"], "") or cfg["default_model"]
    if not api_key:
        raise ValueError(f"Missing API key for {provider}: set {cfg['api_key_env']} in .env")
    return {
        "provider": provider,
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
    }

# ── Speaker Domain Mapping ────────────────────────────────────────────────
# Maps speaker slug → (display_name, domain)
# Based on research proposal Section 3.2

SPEAKER_DOMAINS = {
    # Finance/Investing (28)
    "ray_dalio": ("Ray Dalio", "finance"),
    "cathie_wood": ("Cathie Wood", "finance"),
    "bill_ackman": ("Bill Ackman", "finance"),
    "bill_gross": ("Bill Gross", "finance"),
    "warren_buffett": ("Warren Buffett", "finance"),
    "jamie_dimon": ("Jamie Dimon", "finance"),
    "stanley_druckenmiller": ("Stanley Druckenmiller", "finance"),
    "howard_marks": ("Howard Marks", "finance"),
    "larry_fink": ("Larry Fink", "finance"),
    "david_tepper": ("David Tepper", "finance"),
    "paul_tudor_jones": ("Paul Tudor Jones", "finance"),
    "mohamed_el_erian": ("Mohamed El-Erian", "finance"),
    "nouriel_roubini": ("Nouriel Roubini", "finance"),
    "david_einhorn": ("David Einhorn", "finance"),
    "jeremy_grantham": ("Jeremy Grantham", "finance"),
    "jeff_gundlach": ("Jeff Gundlach", "finance"),
    "david_rosenberg": ("David Rosenberg", "finance"),
    "michael_burry": ("Michael Burry", "finance"),
    "george_soros": ("George Soros", "finance"),
    "charlie_munger": ("Charlie Munger", "finance"),

    # Finance (hyphenated variant)
    "mohamed_el-erian": ("Mohamed El-Erian", "finance"),

    # Politics/Government (42)
    "donald_trump": ("Donald Trump", "politics"),
    "joe_biden": ("Joe Biden", "politics"),
    "hillary_clinton": ("Hillary Clinton", "politics"),
    "bernie_sanders": ("Bernie Sanders", "politics"),
    "nancy_pelosi": ("Nancy Pelosi", "politics"),
    "boris_johnson": ("Boris Johnson", "politics"),
    "emmanuel_macron": ("Emmanuel Macron", "politics"),
    "narendra_modi": ("Narendra Modi", "politics"),
    "recep_erdogan": ("Recep Erdogan", "politics"),
    "viktor_orban": ("Viktor Orban", "politics"),
    "volodymyr_zelenskyy": ("Volodymyr Zelenskyy", "politics"),
    "jd_vance": ("JD Vance", "politics"),
    "nikki_haley": ("Nikki Haley", "politics"),
    "ron_desantis": ("Ron DeSantis", "politics"),
    "gavin_newsom": ("Gavin Newsom", "politics"),
    "pete_buttigieg": ("Pete Buttigieg", "politics"),
    "ted_cruz": ("Ted Cruz", "politics"),
    "josh_hawley": ("Josh Hawley", "politics"),
    "marco_rubio": ("Marco Rubio", "politics"),
    "mitt_romney": ("Mitt Romney", "politics"),
    "elizabeth_warren": ("Elizabeth Warren", "politics"),
    "tulsi_gabbard": ("Tulsi Gabbard", "politics"),
    "rand_paul": ("Rand Paul", "politics"),
    "newt_gingrich": ("Newt Gingrich", "politics"),
    "dan_crenshaw": ("Dan Crenshaw", "politics"),
    "tom_cotton": ("Tom Cotton", "politics"),
    "david_cameron": ("David Cameron", "politics"),
    "antony_blinken": ("Antony Blinken", "politics"),
    "ursula_von_der_leyen": ("Ursula von der Leyen", "politics"),
    "mark_esper": ("Mark Esper", "politics"),
    "james_mattis": ("James Mattis", "politics"),
    "mike_pompeo": ("Mike Pompeo", "politics"),

    # Academia/Economics (22)
    "kenneth_rogoff": ("Kenneth Rogoff", "academia"),
    "daron_acemoglu": ("Daron Acemoglu", "academia"),
    "paul_krugman": ("Paul Krugman", "academia"),
    "joseph_stiglitz": ("Joseph Stiglitz", "academia"),
    "carmen_reinhart": ("Carmen Reinhart", "academia"),
    "barry_eichengreen": ("Barry Eichengreen", "academia"),
    "robert_shiller": ("Robert Shiller", "academia"),
    "niall_ferguson": ("Niall Ferguson", "academia"),
    "jeffrey_sachs": ("Jeffrey Sachs", "academia"),
    "branko_milanovic": ("Branko Milanovic", "academia"),
    "raghuram_rajan": ("Raghuram Rajan", "academia"),
    "paul_romer": ("Paul Romer", "academia"),
    "tyler_cowen": ("Tyler Cowen", "academia"),
    "noam_chomsky": ("Noam Chomsky", "academia"),
    "richard_wolff": ("Richard Wolff", "academia"),
    "yanis_varoufakis": ("Yanis Varoufakis", "academia"),
    "lawrence_summers": ("Lawrence Summers", "academia"),
    "mariana_mazzucato": ("Mariana Mazzucato", "academia"),
    "michael_pettis": ("Michael Pettis", "academia"),
    "stephanie_kelton": ("Stephanie Kelton", "academia"),
    "stephen_roach": ("Stephen Roach", "academia"),

    # Central Banking/Policy (9)
    "jerome_powell": ("Jerome Powell", "central_banking"),
    "janet_yellen": ("Janet Yellen", "central_banking"),
    "ben_bernanke": ("Ben Bernanke", "central_banking"),
    "alan_greenspan": ("Alan Greenspan", "central_banking"),
    "christine_lagarde": ("Christine Lagarde", "central_banking"),
    "mario_draghi": ("Mario Draghi", "central_banking"),
    "neel_kashkari": ("Neel Kashkari", "central_banking"),
    "jim_bullard": ("Jim Bullard", "central_banking"),
    "john_williams": ("John Williams", "central_banking"),

    # Geopolitics/Strategy (10)
    "peter_zeihan": ("Peter Zeihan", "geopolitics"),
    "henry_kissinger": ("Henry Kissinger", "geopolitics"),
    "john_mearsheimer": ("John Mearsheimer", "geopolitics"),
    "ian_bremmer": ("Ian Bremmer", "geopolitics"),
    "fiona_hill": ("Fiona Hill", "geopolitics"),
    "condoleezza_rice": ("Condoleezza Rice", "geopolitics"),
    "robert_gates": ("Robert Gates", "geopolitics"),
    "richard_haass": ("Richard Haass", "geopolitics"),
    "stephen_walt": ("Stephen Walt", "geopolitics"),
    "john_bolton": ("John Bolton", "geopolitics"),
    "graham_allison": ("Graham Allison", "geopolitics"),
    "eliot_cohen": ("Eliot Cohen", "geopolitics"),
    "emma_ashford": ("Emma Ashford", "geopolitics"),
    "michael_mcfaul": ("Michael McFaul", "geopolitics"),
    "robert_kagan": ("Robert Kagan", "geopolitics"),
    "samantha_power": ("Samantha Power", "geopolitics"),

    # Technology/Business (18)
    "elon_musk": ("Elon Musk", "technology"),
    "sam_altman": ("Sam Altman", "technology"),
    "peter_thiel": ("Peter Thiel", "technology"),
    "marc_andreessen": ("Marc Andreessen", "technology"),
    "eric_weinstein": ("Eric Weinstein", "technology"),
    "scott_bessent": ("Scott Bessent", "technology"),
    "david_solomon": ("David Solomon", "finance"),
    "lloyd_blankfein": ("Lloyd Blankfein", "finance"),
    "vivek_ramaswamy": ("Vivek Ramaswamy", "technology"),

    # Media/Commentary (14)
    "ben_shapiro": ("Ben Shapiro", "media"),
    "tucker_carlson": ("Tucker Carlson", "media"),
    "jordan_peterson": ("Jordan Peterson", "media"),
    "ezra_klein": ("Ezra Klein", "media"),
    "sam_harris": ("Sam Harris", "media"),
    "bari_weiss": ("Bari Weiss", "media"),
    "fareed_zakaria": ("Fareed Zakaria", "media"),
    "david_brooks": ("David Brooks", "media"),
    "thomas_friedman": ("Thomas Friedman", "media"),
    "bret_stephens": ("Bret Stephens", "media"),
    "douglas_murray": ("Douglas Murray", "media"),
    "charles_murray": ("Charles Murray", "media"),
    "andrew_sullivan": ("Andrew Sullivan", "media"),
    "matt_taibbi": ("Matt Taibbi", "media"),
    "glenn_greenwald": ("Glenn Greenwald", "media"),
    "anne_applebaum": ("Anne Applebaum", "media"),
    "adam_tooze": ("Adam Tooze", "media"),
}

DOMAIN_ORDER = [
    "finance", "politics", "academia", "central_banking",
    "geopolitics", "technology", "media"
]

DOMAIN_LABELS = {
    "finance": "Finance/Investing",
    "politics": "Politics/Government",
    "academia": "Academia/Economics",
    "central_banking": "Central Banking/Policy",
    "geopolitics": "Geopolitics/Strategy",
    "technology": "Technology/Business",
    "media": "Media/Commentary",
}

def discover_speakers(raw_dir=None):
    """Return list of speaker slugs that have raw VTT data."""
    if raw_dir is None:
        raw_dir = RAW_DIR
    if not os.path.isdir(raw_dir):
        return []
    return sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
        and not d.startswith(".")
        and os.path.isdir(os.path.join(raw_dir, d, "transcripts"))
    ])

def get_domain(speaker_slug):
    """Return domain string for a speaker slug, or 'unknown'."""
    return SPEAKER_DOMAINS.get(speaker_slug, (speaker_slug, "unknown"))[1]

def get_display_name(speaker_slug):
    """Return display name for a speaker slug."""
    return SPEAKER_DOMAINS.get(speaker_slug, (speaker_slug.replace("_", " ").title(), "unknown"))[0]

# ── Analysis Parameters ───────────────────────────────────────────────────
SENTENCE_MIN_WORDS = 8
SENTENCE_MAX_WORDS = 60
LLM_BATCH_SIZE = 50
VALIDATION_SAMPLE_FRACTION = 0.10  # 10% per domain for multi-model validation
KEYWORD_NORM_BASE = 30000  # characters per normalization unit
MIN_MONTHLY_SENTENCES = 5  # minimum sentences per month for correlation
MIN_MONTHS_FOR_CORRELATION = 6  # minimum months for computing r

# ── Multi-Model Validation Providers ──────────────────────────────────────
VALIDATION_PROVIDERS = ["deepseek", "gpt5", "claude", "doubao"]

# ── Major Events for Temporal Analysis ────────────────────────────────────
MAJOR_EVENTS = {
    "covid_start": "2020-03",
    "covid_vaccine": "2020-12",
    "ukraine_invasion": "2022-02",
    "fed_hikes_start": "2022-03",
    "svb_collapse": "2023-03",
    "us_election_2024": "2024-11",
}
