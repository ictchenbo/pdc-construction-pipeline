#!/usr/bin/env python3
"""
TSP pre-annotation using only YouTube-provided data (title, description, VTT).

No ASR/diarization results are used — this is a pre-ASR filter.

Output: JSONL with TSP prediction + confidence + evidence for each video.
"""

import json, os, re, sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data"))

VIDEO_LIST   = os.path.join(DATA_DIR, "video_list.jsonl")
VTT_DIR      = os.path.join(DATA_DIR, "vtt_raw")
META_FILE    = os.path.join(DATA_DIR, "audio_diarized_metadata.jsonl")
OUTPUT       = os.path.join(DATA_DIR, "tsp_preannotation.jsonl")

# ── Heuristics ──
INTERVIEW_KEYWORDS = [
    "on ", " with ", " interview", " talks ", " discusses ", " joins ",
    " conversation", " live", " says ", " explains ", " moderated by",
    " in conversation", " speaks with", " sits down with",
]

DOC_KEYWORDS = [
    "biography", "the man who", "the story of", "rise and fall",
    "documentary", "profile", "history of", "legend of", "life and times",
    "a biography", "the true story", "full documentary",
]

PANEL_KEYWORDS = [
    "panel", "roundtable", "moderated by", "featuring", "and ",
]

LOW_CONFIDENCE_TITLE_PATTERNS = [
    r"^(The\s+)?(Story|Truth|History|Rise|Fall|Life|Legend)\s+(of|behind|about)",
    r"(documentary|biography|profile|exposed)\s*:?\s*$",
    r"^(Who|What|Why|How)\s+(is|was|are|were)\s+",
    r"(evil|puppet|master|exposed|destroy)",
]


def load_vtt_text(vid):
    """Load VTT subtitle text for a video."""
    for ext in [".en.vtt", ".vtt"]:
        p = os.path.join(VTT_DIR, f"{vid}{ext}")
        if os.path.exists(p):
            text = open(p, encoding="utf-8", errors="replace").read()
            # Strip VTT headers, timestamps, and markup
            lines = []
            for line in text.split("\n"):
                line = line.strip()
                if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
                    continue
                if "-->" in line or re.match(r"^\d{2}:\d{2}", line):
                    continue
                # Strip <c> tags and other VTT markup
                line = re.sub(r"<[^>]+>", "", line)
                line = line.strip()
                if line:
                    lines.append(line)
            return " ".join(lines)
    return ""


def speaker_in_title(title, speaker):
    """Check if speaker's name appears in the title."""
    if not speaker:
        return False
    # Full name
    name = speaker.replace("_", " ").lower()
    title_lower = title.lower()
    if name in title_lower:
        return True
    # Last name only (more distinctive)
    parts = name.split()
    if len(parts) > 1:
        if parts[-1] in title_lower:
            return True
    return False


def tsp_from_title(title, speaker):
    """
    TSP evidence from title alone.
    Returns (predicted_tsp, confidence, reasons).
    """
    title_lower = title.lower()
    has_speaker = speaker_in_title(title, speaker)
    has_interview_kw = any(kw in title_lower for kw in INTERVIEW_KEYWORDS)
    has_doc_kw = any(kw in title_lower for kw in DOC_KEYWORDS)
    has_panel_kw = any(kw in title_lower for kw in PANEL_KEYWORDS)
    is_low_conf = any(re.search(p, title_lower) for p in LOW_CONFIDENCE_TITLE_PATTERNS)

    reasons = []
    if has_speaker:
        reasons.append("speaker_in_title")
    if has_interview_kw:
        reasons.append("interview_keyword")
    if has_doc_kw:
        reasons.append("doc_keyword")
    if has_panel_kw:
        reasons.append("panel_keyword")
    if is_low_conf:
        reasons.append("low_conf_pattern")

    if not has_speaker:
        return 1, "low", reasons + ["no_speaker_in_title"]
    if has_doc_kw or is_low_conf:
        return 2, "medium", reasons
    if has_interview_kw and not has_panel_kw:
        return 4, "high", reasons
    if has_panel_kw:
        return 3, "medium", reasons
    # Speaker in title but no format signal
    return 3, "medium", reasons


def tsp_from_description(desc, speaker):
    """TSP evidence from description."""
    if not desc:
        return None, "none", []
    desc_lower = desc.lower()
    has_speaker = speaker.replace("_", " ").lower() in desc_lower if speaker else False
    has_interview = any(kw in desc_lower for kw in [
        "sits down with", "conversation with", "joins", "interview",
        "discusses with", "speaks with", "in conversation",
    ])
    has_doc = any(kw in desc_lower for kw in [
        "documentary", "biography of", "the story of", "profile of",
        "this film", "this documentary",
    ])
    has_panel = any(kw in desc_lower for kw in ["panel", "roundtable", "moderated by"])

    reasons = []
    if has_speaker:
        reasons.append("speaker_in_desc")
    if has_interview:
        reasons.append("interview_in_desc")
    if has_doc:
        reasons.append("doc_in_desc")
    if has_panel:
        reasons.append("panel_in_desc")

    if has_doc and not has_interview:
        return 2, "high", reasons
    if has_interview and has_speaker:
        return 4, "high", reasons + ["desc_confirms_interview"]
    if has_panel:
        return 3, "medium", reasons
    if has_speaker:
        return 3, "low", reasons
    return None, "none", reasons


def tsp_from_vtt(vtt_text, speaker):
    """TSP evidence from VTT subtitle content."""
    if not vtt_text:
        return None, "none", []

    words = vtt_text.split()
    word_count = len(words)
    question_count = vtt_text.count("?")
    q_ratio = question_count / max(word_count, 1)

    has_speaker_name = False
    if speaker:
        for part in speaker.replace("_", " ").lower().split():
            if part in vtt_text.lower() and len(part) > 2:
                has_speaker_name = True
                break

    reasons = [f"vtt_{word_count}w", f"q_ratio={q_ratio:.3f}"]
    if has_speaker_name:
        reasons.append("speaker_in_vtt")

    # Very short VTT → minimal content
    if word_count < 200:
        return 1, "high", reasons + ["vtt_too_short"]
    if word_count < 500:
        return 2, "medium", reasons + ["vtt_short"]

    # High question ratio → dialogue structure (interview)
    if q_ratio > 0.003:
        if has_speaker_name:
            return 4, "high", reasons + ["dialogue_with_speaker"]
        return 3, "medium", reasons + ["dialogue_detected"]

    # Low question ratio → monologue/narration
    if word_count > 2000 and has_speaker_name:
        return 3, "low", reasons + ["long_monologue"]
    if word_count > 5000:
        return 3, "low", reasons + ["long_vtt_no_dialogue"]

    return 2, "low", reasons + ["unclear_format"]


def predict_tsp(meta_record, speaker):
    """Combine all signals into a single TSP prediction from a metadata record."""
    vid = meta_record["id"]
    title = meta_record.get("title", "")
    desc = meta_record.get("description", "")

    # 1. Title
    tsp_t, conf_t, reasons_t = tsp_from_title(title, speaker)

    # 2. Description
    tsp_d, conf_d, reasons_d = tsp_from_description(desc, speaker)

    # 3. VTT
    vtt_text = load_vtt_text(vid)
    tsp_v, conf_v, reasons_v = tsp_from_vtt(vtt_text, speaker)

    # ── Combine ──
    # Collect all TSP votes with confidence weights
    votes = []
    if tsp_t is not None:
        weight = {"high": 3, "medium": 2, "low": 1}.get(conf_t, 1)
        votes.append((tsp_t, weight))
    if tsp_d is not None:
        weight = {"high": 3, "medium": 2, "low": 1}.get(conf_d, 1)
        votes.append((tsp_d, weight))
    if tsp_v is not None:
        weight = {"high": 3, "medium": 2, "low": 1}.get(conf_v, 1)
        votes.append((tsp_v, weight))

    # Weighted average
    total_weight = sum(w for _, w in votes)
    if total_weight > 0:
        weighted_tsp = round(sum(t * w for t, w in votes) / total_weight, 1)
    else:
        weighted_tsp = 1  # default: no information

    # Confidence: all three agree at high → high, else lower
    unique_tsp = set(v[0] for v in votes)
    high_sources = sum(1 for _, w in votes if w >= 2)
    if len(unique_tsp) == 1 and conf_t == "high" and conf_v == "high":
        overall_conf = "high"
    elif len(unique_tsp) <= 2 and high_sources >= 2:
        overall_conf = "high"
    elif len(unique_tsp) <= 2:
        overall_conf = "medium"
    else:
        overall_conf = "low"

    # Decide if human review needed
    needs_review = overall_conf == "low" or (
        weighted_tsp < 3 and overall_conf != "high"
    )

    return {
        "video_id": vid,
        "speaker": speaker,
        "title": title[:100],
        "channel": meta_record.get("channel", "")[:40],
        "tsp_prediction": weighted_tsp,
        "confidence": overall_conf,
        "needs_review": needs_review,
        "tsp_title": tsp_t,
        "conf_title": conf_t,
        "reasons_title": reasons_t,
        "tsp_desc": tsp_d,
        "conf_desc": conf_d,
        "reasons_desc": reasons_d,
        "tsp_vtt": tsp_v,
        "conf_vtt": conf_v,
        "reasons_vtt": reasons_v,
        "vtt_words": len(vtt_text.split()) if vtt_text else 0,
    }


def main():
    # Load metadata records from audio_diarized_metadata.jsonl
    meta_records = {}
    with open(META_FILE) as f:
        for line in f:
            r = json.loads(line.strip())
            meta_records[r["id"]] = r
    print(f"Loaded {len(meta_records)} metadata records from {META_FILE}")

    # Get all diarized video IDs
    diarized = sorted(
        p.stem for p in Path(os.path.join(DATA_DIR, "audio_diarized")).glob("*.txt")
        if p.stem != "_run_log"
    )
    print(f"Videos to annotate: {len(diarized)}")

    results = []
    counts = Counter()
    review_count = 0

    for i, vid in enumerate(diarized):
        meta = meta_records.get(vid)
        if not meta:
            print(f"[{i+1}/{len(diarized)}] {vid} ⏭️  no metadata")
            continue
        speaker = meta.get("speaker", "")
        result = predict_tsp(meta, speaker)
        results.append(result)
        counts[f"TSP_{int(result['tsp_prediction'])}"] += 1
        if result["needs_review"]:
            review_count += 1

        tsp = result["tsp_prediction"]
        conf = result["confidence"]
        rev = "⚠" if result["needs_review"] else "✓"
        print(f"[{i+1}/{len(diarized)}] {vid} TSP={tsp} ({conf}) {rev} "
              f"title={result['reasons_title']} vtt={result['reasons_vtt'][:2]}")

    # Sort by TSP score ascending (lowest first = most suspicious first)
    results.sort(key=lambda r: (r["tsp_prediction"], 0 if r["needs_review"] else 1, r.get("vtt_words", 0)))

    # Save main output
    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save TSP annotation file for the grading tool (grade 0-4 = TSP 1-5)
    TSP_GRADE_FILE = os.path.join(DATA_DIR, "tsp_grade_items.jsonl")
    with open(TSP_GRADE_FILE, "w") as f:
        for r in results:
            tsp = int(round(r["tsp_prediction"]))
            grade = max(0, min(4, tsp - 1))  # TSP 1→0, 2→1, 3→2, 4→3, 5→4
            entry = {
                "speaker": r["speaker"],
                "video_id": r["video_id"],
                "title": r.get("title", ""),
                "channel": r.get("channel", ""),
                "grade": tsp,
                "grade_hint": grade,
                "confidence": r["confidence"],
                "needs_review": r["needs_review"],
                "reasons": r.get("reasons_title", []) + r.get("reasons_vtt", []),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n  TSP grade items: {TSP_GRADE_FILE} ({len(results)} files, sorted by TSP asc)")

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    for level in sorted(counts.keys()):
        print(f"  {level}: {counts[level]:>4d}")
    print(f"\n  Need human review: {review_count} ({review_count/len(results)*100:.1f}%)")
    print(f"\n  Output: {OUTPUT}")


if __name__ == "__main__":
    main()
