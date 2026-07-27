#!/usr/bin/env python3
"""Run extract_guest_from_local.py only for missing files."""
import os, sys, json, time
from pathlib import Path

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_DIR)

from src.utils.llm import llm_call

# Replicate paths from original script
LOCAL_JSON_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "audio_diarized_local"))
OUT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "audio_diarized_local_guest"))
os.makedirs(OUT_DIR, exist_ok=True)

# Import the functions from the original script
sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "02_asr"))

# We'll just copy the needed functions inline from the original to avoid import issues
# ── Direct copy of needed functions ──

def make_llm_cfg():
    return {
        "api_key": os.environ.get("CLEAN_API_KEY", ""),
        "api_base": os.environ.get("CLEAN_API_BASE", "").rstrip("/"),
        "model": os.environ.get("CLEAN_MODEL", "deepseek-chat"),
    }

def identify_target_by_llm(speaker_texts, llm_cfg, target_name_hint=None):
    import re
    from collections import defaultdict
    target_display = target_name_hint.replace("_", " ").title() if target_name_hint else "the featured person"
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
            m = re.search(r"\{[^{}]+\}", content, re.DOTALL)
            if not m:
                if attempt < 2:
                    time.sleep((attempt + 1) * 5)
                continue
            labels = json.loads(m.group())
            for spk in valid_speakers:
                val = labels.get(spk)
                if val is None:
                    for k, v in labels.items():
                        if k.lower().replace(" ", "_") == spk.lower():
                            val = v
                            break
                if val and str(val).strip().lower() == "guest":
                    return spk, labels
            break
        except Exception:
            if attempt < 2:
                time.sleep((attempt + 1) * 5)
            else:
                pass
    word_counts = {s: sum(len(t.split()) for t in texts)
                   for s, texts in speaker_texts.items() if s != "UNKNOWN"}
    return max(word_counts, key=word_counts.get), None


def extract_guest(json_path, llm_cfg, target_name_hint=None):
    from collections import Counter, defaultdict
    with open(json_path) as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        return None, None

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


# ── Find missing files ──
existing = set(f.replace('.txt', '') for f in os.listdir(OUT_DIR) if f.endswith('.txt'))
all_json = sorted(Path(LOCAL_JSON_DIR).glob("*.json"))
all_json = [f for f in all_json if f.stem not in ("quality_report",)]
missing = [f for f in all_json if f.stem not in existing]

# Load speaker map
VIDEO_LIST = os.path.abspath(os.path.join(PROJECT_DIR, "..", "data", "video_list.jsonl"))
speaker_map = {}
if os.path.exists(VIDEO_LIST):
    with open(VIDEO_LIST) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            speaker_map[e["id"]] = e.get("speaker", "")

print(f"Total JSON: {len(all_json)}, Already done: {len(existing)}, Missing: {len(missing)}")

llm_cfg = make_llm_cfg()
print(f"LLM: {llm_cfg['model']}")
print()

stats = {"processed": 0, "duration_rule": 0, "llm_ok": 0, "llm_fallback": 0, "no_segments": 0, "errors": 0}

for i, jp in enumerate(missing):
    vid = jp.stem
    hint = speaker_map.get(vid)
    prefix = f"[{i+1}/{len(missing)}]"

    try:
        guest_text, info = extract_guest(jp, llm_cfg, target_name_hint=hint)
        if guest_text is None:
            print(f"{prefix} {vid} ⏭️  no segments")
            stats["no_segments"] += 1
            continue

        out_path = os.path.join(OUT_DIR, f"{vid}.txt")
        with open(out_path, "w") as f:
            f.write(guest_text)

        m = info["method"]
        is_llm = m.startswith("llm")
        if m == "llm(fallback)":
            stats["llm_fallback"] += 1
        elif is_llm:
            stats["llm_ok"] += 1
        else:
            stats["duration_rule"] += 1
        stats["processed"] += 1

        marker = "⚠" if "fallback" in m else ("🤖" if is_llm else "✓")
        print(f"{prefix} {vid} {marker} {info['guest_words']:>5}w/{info['total_words']:>5}w ({info['ratio']:.0%}) "
              f"spk={info['target_speaker']} [{info['method']:>25s}] "
              f"hint={hint or 'none'}")

        if is_llm and "fallback" not in m:
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

out_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".txt") and not f.startswith("_")]
print(f"\n  Output: {OUT_DIR}/ ({len(out_files)} files)")
print(f"{'='*60}")
