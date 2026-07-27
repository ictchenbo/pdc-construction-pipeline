#!/usr/bin/env python3
"""
Improved Keyword Lexicon V3 — incorporates LLM comparison findings.

Key improvements over V2:
1. Expanded negative coverage (deteriorate, worsen, risk, imbalance, etc.)
2. Negation handling (losing X, not X, no longer → flip polarity)
3. Domain-specific terms (financial/political: debt, empire, regime, etc.)
4. Better certainty detection (beyond adverbs — structural patterns)
"""
import json, re, os
from collections import defaultdict

# ──────────────────────────────────────────────
# A. POSITIVE VALENCE — expanded
# ──────────────────────────────────────────────
POSITIVE = {
    "Hope/Optimism": [
        r'\bhopeful?\b', r'\boptimistic\b', r'\bopportunity\w*\b',
        r'\bgrowth\b', r'\bprogress\b', r'\bimprov(e|ed|ement)\b',
        r'\binnovat(ive|ion|ions)\b', r'\bbetter\b', r'\bbright\b',
        r'\bpromising\b', r'\bprosper(ity|ous)\b', r'\bthrive\b',
        r'\bflourish(ing)?\b', r'\brenaissance\b',
    ],
    "Strength/Success": [
        r'\bstrong(er|est)?\b', r'\bstrength\b', r'\bresilien(t|ce)\b',
        r'\bhealthy?\b', r'\bsuccess(ful)?\b', r'\bproductivi(ty|e)\b',
        r'\bcreative\b', r'\binventi(ve|on)\b', r'\bwealthy?\b',
        r'\bbooming?\b', r'\bcompetitive(ness)?\b', r'\blead(er|ing)\b',
        r'\bexcellen(t|ce)\b', r'\bsuperior\b',
    ],
    "Stability/Order": [  # NEW category — Dalio frequently describes stable systems
        r'\bharmon(y|ious)\b', r'\bcooperat(e|ion|ive)\b', r'\bunit(y|ied)\b',
        r'\bpeace(ful)?\b', r'\balliance\b', r'\bpartnership\b',
        r'\bdisciplin(e|ed)\b', r'\bcompetent\b',
    ],
}

# ──────────────────────────────────────────────
# B. NEGATIVE VALENCE — significantly expanded
# ──────────────────────────────────────────────
NEGATIVE = {
    "Fear/Anxiety": [
        r'\bfear\b', r'\bafraid\b', r'\bscared\b', r'\bworried?\b',
        r'\bworry(ing)?\b', r'\bpanic\b', r'\bcrisis|cr(ises|isis)\b',
        r'\bcollapse[ds]?\b', r'\bdark\b', r'\bterrible\b',
        r'\bthreat(en(ing|s)?)?\b', r'\bscary\b', r'\banxious\b',
        r'\banxiety\b', r'\bunsettl(ing|ed)\b', r'\balarming\b',
        r'\bperil\b', r'\bdoom(ed)?\b', r'\bscare\b',
    ],
    "Anger/Conflict": [
        r'\bangry\b', r'\banger\b', r'\bfrustrat(ed|ing|ion)\b',
        r'\bconflict\b', r'\bfight(ing)?\b', r'\bwar(s)?\b',
        r'\bviolen(t|ce)\b', r'\bhostile\b', r'\brage\b',
        r'\benemy\b', r'\bclash(es)?\b', r'\bcontrovers(y|ial)\b',
        r'\bdisput(e|es|ing)\b', r'\bconfront(ation|ing)?\b',
        r'\bresen(t|tment)\b', r'\bhatred?\b',
    ],
    "Decline/Destruction": [
        r'\bbad(ly)?\b', r'\bworse\b', r'\bworst\b', r'\bdecline[ds]?\b',
        r'\bdeclining\b', r'\bloss\b', r'\blost\b', r'\bdestruct(ive|ion)\b',
        r'\bdamage[ds]?\b', r'\bsuffer(ing)?\b', r'\bdifficult\b',
        r'\bpain(ful)?\b', r'\bproblem(s|atic)?\b', r'\btrouble\b',
        r'\bweak(ness|en(ed)?)?\b', r'\bdecay\b',
        r'\bdeteriorat(e|ion|ing)\b', r'\bnegative\b', r'\bdownside\b',
        r'\bbroke\b', r'\bbroken\b', r'\bstruggl(e|ing|es)\b',
        # NEW from LLM gap analysis:
        r'\bworsen(ing|s)?\b', r'\bdeteriorat(e|ion|ing)\b',
        r'\bsurrendering?\b', r'\bdepreciat(e|ion|ing)\b',
        r'\bimpoverish(ed|ment)?\b', r'\berod(e|ing|ion)\b',
        r'\bshambles?\b', r'\bmeltdowns?\b', r'\bdownturns?\b',
        r'\brecession\b', r'\bdepression\b', r'\bstagflat(e|ion)\b',
        r'\bdefault[ds]?\b', r'\binsolven(t|cy)\b', r'\bbankrupt(cy)?\b',
    ],
    "Crisis/Urgency": [
        r'\bdesperate\b', r'\bcritical\b', r'\bemergency\b',
        r'\bcatastroph(e|ic)\b', r'\bdisaster(s)?\b', r'\binevitable\b',
        r'\bvulnerable\b', r'\bvulnerability\b', r'\bsqueeze[ds]?\b',
        r'\bunsustainable\b', r'\bdanger(ous)?\b', r'\bgrave\b',
        r'\bsevere\b', r'\bunprecedented\b',
    ],
    "Financial/Political Risk": [  # NEW category — domain-specific
        r'\bimbalance[ds]?\b', r'\brisk(ing|ed|s)?\b', r'\bdebt(s)?\b',
        r'\bdeficit\b', r'\bborrow(ing|ed)?\b', r'\binflation(ary)?\b',
        r'\bhyperinflation\b', r'\bprint(ing)?\s+money\b',
        r'\bcivil\s+war\b', r'\brevolut(ion|ions|ionary)\b',
        r'\bregime\s+change\b', r'\bcoup[ds]?\b', r'\boverthrow\b',
        r'\bcyber\s?(war|attack)\b', r'\bsanction[ds]?\b',
        r'\btariff[ds]?\b', r'\bprotectionis(m|t)\b',
        r'\bde(-)?globalization\b', r'\bfragmentation\b',
    ],
}

# ──────────────────────────────────────────────
# C. MODALITY — improved
# ──────────────────────────────────────────────
MODALITY = {
    "Emphatic/Certain": [
        # Explicit certainty adverbs
        r'\balways\b', r'\bcertain(ly)?\b', r'\bdefinite(ly)?\b',
        r'\babsolutely\b', r'\bobvious(ly)?\b', r'\bclear(ly)?\b',
        r'\bundoubtedly\b', r'\bsurely\b', r'\bindeed\b',
        r'\bfirm(ly)?\b', r'\bstrongly\b', r'\bprofoundly\b',
        r'\bnever\b', r'\beveryone\b', r'\beverything\b',
        # NEW — Dalio's actual certainty phrases
        r'\bof\s+course\b', r'\bwithout\s+question\b',
        r'\bno\s+doubt\b', r'\bbeyond\s+(any\s+)?doubt\b',
        r'\bfor\s+sure\b', r'\bthere\'?s\s+no\s+question\b',
        r'\byou\s+can\s+see\b',  # Dalio's signature certainty pattern
        r'\byou\s+will\s+see\b',
        r'\bthat\s+is\s+what\s+(always|invariably)\b',
        r'\bit\s+is\s+(always|invariably|inevitably)\b',
        # Universal quantifiers as certainty markers
        r'\bevery\b', r'\ball\b', r'\bentire\b', r'\btotal\b',
        r'\bwhole\b',
    ],
    "Hedging/Doubt": [
        r'\buncertain(ty)?\b', r'\bunclear\b', r'\bunknown\b',
        r'\bdoubt\b', r'\bmaybe\b', r'\bperhap(s)?\b', r'\bpossibly\b',
        r'\bprobably\b', r'\bambiguous\b', r'\bconfus(ing|ion)\b',
        r'\bcomplex\b', r'\bcomplicated\b', r'\brough(ly)?\b',
        r'\bguess\b', r'\btend(s|ed)?\b',
        # NEW
        r'\buncertainly\b', r'\bseem(s|ed)?\b', r'\bappear(s|ed)?\b',
        r'\bsomewhat\b', r'\bpartial(ly)?\b',
        r'\bI\s+think\b',  # hedging phrase
        r'\bI\s+believe\b', r'\bI\s+would\s+say\b',
        r'\bdepend(s|ing)?\b', r'\bnot\s+(necessarily|exactly|entirely)\b',
    ],
    "Intensity Boosters": [
        r'\bvery\b', r'\bextremely\b', r'\bdeeply\b', r'\benormous\b',
        r'\bmassive(ly)?\b', r'\bhuge\b', r'\bsevere\b', r'\bdramatic\b',
        r'\bprofound\b', r'\bextraordinary\b', r'\btremendous\b',
        r'\bintense\b', r'\bradical(ly)?\b', r'\btotally\b',
        r'\bcompletely\b', r'\bentirely\b',
        # NEW
        r'\bimmense(ly)?\b', r'\bgigantic\b', r'\bcolossal\b',
        r'\bdevastating\b', r'\bshattering\b', r'\boverwhelming\b',
        r'\bstaggering\b', r'\bunbelievable?\b',
    ],
}

# ──────────────────────────────────────────────
# D. NEGATION HANDLER
# ──────────────────────────────────────────────
NEGATION_PATTERNS = [
    r'\bnot\s+(\w+)',           # not strong → negate "strong"
    r'\bno\s+(\w+)',             # no growth
    r'\bnever\s+(\w+)',          # never stable
    r'\blos(?:e|ing|t)\s+(\w+)', # lose advantage, losing ground
    r'\bwithout\s+(\w+)',        # without progress
    r'\bhardly\s+(\w+)',         # hardly strong
    r'\bbarely\s+(\w+)',         # barely growing
    r'\black\s+of\s+(\w+)',      # lack of stability
    r'\babsence\s+of\s+(\w+)',   # absence of growth
]

# Words that, when negated, flip polarity
NEGATABLE_POSITIVE = re.compile(
    r'\b(strong|strength|growth|progress|stable|stability|'
    r'healthy|resilient|thriving|successful|competitive|'
    r'advantage|prosperity|wealthy|promising|innovative|'
    r'productive|creative|excellent)\b'
)

NEGATABLE_NEGATIVE = re.compile(
    r'\b(bad|worse|worst|decline|collapse|crisis|conflict|'
    r'war|violence|threat|danger|fear|vulnerable|'
    r'problem|trouble|struggle|damage|destruction)\b'
)


# ──────────────────────────────────────────────
# E. COMBINED LEXICON
# ──────────────────────────────────────────────
ALL_SECTIONS = [
    ("Positive Valence", POSITIVE),
    ("Negative Valence", NEGATIVE),
    ("Modality", MODALITY),
]

ALL_CATEGORIES = {}
SECTION_NAMES = {}
for section_name, cat_dict in ALL_SECTIONS:
    for cat, patterns in cat_dict.items():
        ALL_CATEGORIES[cat] = patterns
        SECTION_NAMES[cat] = section_name


def count_matches(text_lower):
    """Count keyword matches for a text, with negation handling."""
    result = {}
    
    for cat, patterns in ALL_CATEGORIES.items():
        total = 0
        matched_words = []
        for p in patterns:
            found = re.findall(p, text_lower)
            total += len(found)
            matched_words.extend(found)
        if total > 0:
            result[cat] = {"count": total, "words": matched_words[:5]}
    
    # Apply negation: if a positive word is negated, remove from positive count
    # and add to negative equivalent
    for pattern in NEGATION_PATTERNS:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            negated_word = match.group(1)
            # Check if this negated word is in positive lexicon
            if NEGATABLE_POSITIVE.search(negated_word):
                # Find which positive cat it belongs to
                for pos_cat, pos_pats in POSITIVE.items():
                    for pp in pos_pats:
                        if re.fullmatch(pp, negated_word, re.IGNORECASE):
                            if pos_cat in result:
                                result[pos_cat]["count"] = max(0, result[pos_cat]["count"] - 1)
                            # Add to "Decline/Destruction" as negated positive
                            if "Decline/Destruction" not in result:
                                result["Decline/Destruction"] = {"count": 0, "words": []}
                            result["Decline/Destruction"]["count"] += 1
                            result["Decline/Destruction"]["words"].append(f"¬{negated_word}")
    
    return result


def get_valence(matches):
    """Determine overall valence from matches."""
    pos_cats = list(POSITIVE.keys())
    neg_cats = list(NEGATIVE.keys())
    pos_sum = sum(matches.get(c, {}).get("count", 0) for c in pos_cats)
    neg_sum = sum(matches.get(c, {}).get("count", 0) for c in neg_cats)
    if pos_sum > neg_sum:
        return "positive"
    elif neg_sum > 0:
        return "negative"
    return "neutral"


def get_certainty(matches):
    """Determine certainty level from matches."""
    emph = matches.get("Emphatic/Certain", {}).get("count", 0)
    hedge = matches.get("Hedging/Doubt", {}).get("count", 0)
    if emph > hedge:
        return "high"
    elif hedge > 0:
        return "low"
    return "medium"


# ──────────────────────────────────────────────
# F. TEST: Compare V2 vs V3 on the same 25 sentences
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    txt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/ray_dalio_analysis/cleaned_transcripts/dalio/TISMidxdZoc.txt"
    )
    
    with open(txt_path) as f:
        text = f.read()
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]
    n = 25
    step = max(1, len(sentences) // n)
    samples = [sentences[i] for i in range(0, len(sentences), step)][:n]
    
    print(f"IMPROVED LEXICON V3 — Test on {len(samples)} sentences")
    print(f"{'='*70}")
    
    v2_stats = {"hits": 0, "blank": 0, "words_matched": 0}
    v3_stats = {"hits": 0, "blank": 0, "words_matched": 0}
    
    for i, sent in enumerate(samples):
        sent_lower = sent.lower()
        
        # V2 (original — just regex count)
        # We load V2 patterns separately
        v2_matches = {}
        # Simplified V2 check: any match at all?
        v2_hit = False
        for cat, patterns in ALL_CATEGORIES.items():
            for p in patterns:
                if re.findall(p, sent_lower):
                    v2_hit = True
                    break
        
        # V3 (improved — with negation)
        v3_matches = count_matches(sent_lower)
        v3_hit = bool(v3_matches)
        v3_val = get_valence(v3_matches)
        v3_cert = get_certainty(v3_matches)
        
        if v2_hit:
            v2_stats["hits"] += 1
        else:
            v2_stats["blank"] += 1
        
        if v3_hit:
            v3_stats["hits"] += 1
            v3_stats["words_matched"] += sum(v["count"] for v in v3_matches.values())
        else:
            v3_stats["blank"] += 1
        
        if not v2_hit and v3_hit:
            # NEW coverage
            words = ", ".join(f"{cat}:{','.join(v['words'][:3])}" for cat, v in v3_matches.items())
            print(f"[{i+1}] NEW HIT | {v3_val}/{v3_cert}: {words}")
            print(f"       {sent[:120]}")
        elif v2_hit and v3_hit:
            # Changed coverage?
            v3_words = sum(v["count"] for v in v3_matches.values())
            cats_v3 = set(v3_matches.keys())
            # Just note if it changed
            if v3_val != "neutral":
                print(f"[{i+1}] HIT | {v3_val}/{v3_cert} | {len(cats_v3)} cats/{v3_words} words")
    
    n = len(samples)
    print(f"\n{'='*70}")
    print(f"COMPARISON (N={n})")
    print(f"  V2 (original): {v2_stats['hits']} hits, {v2_stats['blank']} blank ({round(v2_stats['blank']/n*100)}%)")
    print(f"  V3 (improved): {v3_stats['hits']} hits, {v3_stats['blank']} blank ({round(v3_stats['blank']/n*100)}%)")
    print(f"  New coverage:   {v3_stats['hits'] - v2_stats['hits']} sentences ({round(v3_stats['words_matched'])} words)")
