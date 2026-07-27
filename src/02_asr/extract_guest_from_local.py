#!/usr/bin/env python3
"""
Extract target (guest) speaker from audio_diarized_local/ multi-speaker JSON files.

Rule+LLM decision tree matching the paper (Section 3.3):
  1 speaker            → that speaker (duration rule)
  2 speakers:
    ratio < 0.3        → longer speaker (duration rule)
    ratio ≥ 0.3        → LLM identifies guest with speaker name hint
  3+ speakers          → LLM identifies guest with speaker name hint

Writes guest-only text to audio_diarized_local_guest/<video_id>.txt

Usage:
  python3 src/02_asr/extract_guest_from_local.py
  python3 src/02_asr/extract_guest_from_local.py --dry-run
"""

import json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.llm import llm_call

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

LOCAL_JSON_DIR  = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "audio_diarized_local"))
OUT_DIR         = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "audio_diarized_local_guest"))
VIDEO_LIST      = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "video_list.jsonl"))
os.makedirs(OUT_DIR, exist_ok=True)

# .env is auto-loaded by src.utils.llm at import
def make_llm_cfg():
    return {
        "api_key": os.environ.get("CLEAN_API_KEY", ""),
        "api_base": os.environ.get("CLEAN_API_BASE", "").rstrip("/"),
        "model": os.environ.get("CLEAN_MODEL", "deepseek-chat"),
    }

# ── Load video → speaker mapping ──
def load_speaker_map():
    mapping = {}
    if not os.path.exists(VIDEO_LIST):
        return mapping
    with open(VIDEO_LIST) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            mapping[e["id"]] = e.get("speaker", "")
    return mapping

# ── LLM-based speaker identification ──
def identify_target_by_llm(speaker_texts, llm_cfg, target_name_hint=None):
    """Call LLM to identify the guest speaker. Returns (target_spk, labels or None)."""
    target_display = target_name_hint.replace("_", " ").title() if target_name_hint else "the featured person"

    # Build sample text for each speaker (skip UNKNOWN)
    valid_speakers = sorted([spk for spk in speaker_texts if spk != "UNKNOWN"])
    SAMPLES_PER_SPK = 20
    CHARS_PER_SPK = 1200
    samples = []
    for spk in valid_speakers:
        texts = speaker_texts[spk]
        n = len(texts)
        if n > SAMPLES_PER_SPK:
            indices = [int(i * n / SAMPLES_PER_SPK) for i in range(SAMPLES_PER_SPK)]
            sampled = [texts[i] for i in indices]
        else:
            sampled = texts
        sample = "\n".join(sampled)
        samples.append(f"--- {spk} ---\n{sample[:CHARS_PER_SPK]}")

    messages = [
        {"role": "system",
         "content": f"You are identifying speakers in an interview. The target guest is {target_display}. "
                    "For each speaker, determine if they are the MAIN GUEST (the person being interviewed, giving opinions) "
                    "or the INTERVIEWER/HOST (asking questions, introducing topics). "
                    "Output valid JSON only: {\"SPEAKER_00\": \"guest\"|\"interviewer\", ...}"},
        {"role": "user",
         "content": "Identify each speaker:\n\n" + "\n\n".join(samples)}
    ]

    for attempt in range(3):
        try:
            content = llm_call(llm_cfg, messages, attempts=1,
                               temperature=0.0, max_tokens=1024, timeout=60)

            # Extract JSON from response (handles code fences and loose text)
            m = re.search(r"\{[^{}]+\}", content, re.DOTALL)
            if not m:
                if attempt < 2:
                    time.sleep((attempt + 1) * 5)
                continue

            labels = json.loads(m.group())

            # Normalize keys: try exact match first, then case-insensitive
            for spk in valid_speakers:
                val = labels.get(spk)
                if val is None:
                    for k, v in labels.items():
                        if k.lower().replace(" ", "_") == spk.lower():
                            val = v
                            break
                if val and str(val).strip().lower() == "guest":
                    return spk, labels

            # Valid JSON but no guest label → fall back immediately
            break

        except Exception:
            if attempt < 2:
                time.sleep((attempt + 1) * 5)
            else:
                pass

    # Fallback: most words (excluding UNKNOWN)
    word_counts = {s: sum(len(t.split()) for t in texts)
                   for s, texts in speaker_texts.items() if s != "UNKNOWN"}
    return max(word_counts, key=word_counts.get), None


# ── Extract guest from one JSON ──
def extract_guest(json_path, llm_cfg, target_name_hint=None, dry_run=False):
    with open(json_path) as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        return None, None

    # Group by speaker, excluding UNKNOWN segments
    speaker_texts = defaultdict(list)
    word_counts = Counter()
    for seg in segments:
        spk = seg["speaker"]
        if spk == "UNKNOWN":
            continue
        t = seg.get("text", "").strip()
        speaker_texts[spk].append(t)
        word_counts[spk] += len(t.split())

    n_speakers = len(word_counts)
    if n_speakers == 0:
        return None, None

    method = ""
    target = None
    labels = None

    if n_speakers == 1:
        target = list(word_counts.keys())[0]
        method = "duration (1 spk)"
    elif n_speakers == 2:
        primary = word_counts.most_common(1)[0][0]
        secondary = word_counts.most_common(2)[1][0]
        ratio = word_counts[secondary] / max(word_counts[primary], 1)
        if ratio < 0.3:
            target = primary
            method = f"duration (2 spk, ratio={ratio:.3f})"
        else:
            method = f"llm (2 spk, ratio={ratio:.3f})"
    else:
        method = f"llm ({n_speakers} spk)"

    if target is None:
        if dry_run:
            # dry_run: use rule-based approximation (most words)
            target = max(word_counts, key=word_counts.get)
            method = method.replace("llm", "llm(dry)")
        else:
            target, labels = identify_target_by_llm(speaker_texts, llm_cfg, target_name_hint=target_name_hint)
            if target is None:
                target = max(word_counts, key=word_counts.get)
                method = method.replace("llm", "llm(fallback)")

    guest_text = "\n".join(speaker_texts[target])
    info = {
        "target_speaker": target,
        "method": method,
        "n_speakers": n_speakers,
        "total_words": sum(word_counts.values()),
        "guest_words": len(guest_text.split()),
        "ratio": round(word_counts[target] / max(sum(word_counts.values()), 1), 3),
        "target_name_hint": target_name_hint,
    }
    return guest_text, info


# ── Main ──
def main():
    dry_run = "--dry-run" in sys.argv

    speaker_map = load_speaker_map()
    print(f"Loaded {len(speaker_map)} speaker mappings from video_list.jsonl")

    llm_cfg = None if dry_run else make_llm_cfg()
    if llm_cfg:
        print(f"LLM: {llm_cfg['model']}")
    else:
        print("LLM: N/A (dry run)")

    json_files = sorted(Path(LOCAL_JSON_DIR).glob("*.json"))
    json_files = [f for f in json_files if f.stem not in ("quality_report",)]
    print(f"Local JSON files: {len(json_files)}")
    print()

    stats = {"processed": 0, "duration_rule": 0, "llm_ok": 0, "llm_fallback": 0, "no_segments": 0, "errors": 0}
    log = []

    # Pre-build set of already processed files for skip mode
    processed_ids = {f.replace('.txt', '') for f in os.listdir(OUT_DIR) if f.endswith('.txt')}
    skipped_existing = 0

    for i, jp in enumerate(json_files):
        vid = jp.stem
        if vid in processed_ids:
            skipped_existing += 1
            continue
        hint = speaker_map.get(vid)
        prefix = f"[{i+1}/{len(json_files)}]"

        try:
            guest_text, info = extract_guest(jp, llm_cfg, target_name_hint=hint, dry_run=dry_run)
            if guest_text is None:
                print(f"{prefix} {vid} ⏭️  no segments")
                stats["no_segments"] += 1
                continue

            if not dry_run:
                out_path = os.path.join(OUT_DIR, f"{vid}.txt")
                with open(out_path, "w") as f:
                    f.write(guest_text)

            # Classify method
            m = info["method"]
            is_llm = m.startswith("llm")
            if m == "llm(fallback)":
                stats["llm_fallback"] += 1
            elif is_llm:
                stats["llm_ok"] += 1
            else:
                stats["duration_rule"] += 1
            stats["processed"] += 1

            marker = "✓" if not is_llm else ("⚠" if "fallback" in m else "🤖")
            if dry_run:
                marker = "[DRY]"
            print(f"{prefix} {vid} {marker} {info['guest_words']:>5}w/{info['total_words']:>5}w ({info['ratio']:.0%}) "
                  f"spk={info['target_speaker']} [{info['method']:>25s}] "
                  f"hint={hint or 'none'}")

            log.append({"video_id": vid, **info})

            if is_llm and not dry_run and "fallback" not in m:
                time.sleep(0.5)

        except Exception as e:
            stats["errors"] += 1
            print(f"{prefix} {vid} ❌ {e}")

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Processed:           {stats['processed']}")
    print(f"    - Duration rule:   {stats['duration_rule']}")
    print(f"    - LLM success:     {stats['llm_ok']}")
    print(f"    - LLM fallback:    {stats['llm_fallback']}")
    print(f"  No segments:         {stats['no_segments']}")
    print(f"  Errors:              {stats['errors']}")

    if not dry_run:
        out_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".txt") and not f.startswith("_")]
        print(f"\n  Output: {OUT_DIR}/ ({len(out_files)} files)")

        summary = {
            "source": str(LOCAL_JSON_DIR), "output": str(OUT_DIR),
            "stats": stats, "files": log,
        }
        with open(os.path.join(OUT_DIR, "_extraction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"{'='*60}")

if __name__ == "__main__":
    main()
