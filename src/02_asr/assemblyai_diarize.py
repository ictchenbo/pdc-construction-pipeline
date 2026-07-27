#!/usr/bin/env python3
"""
AssemblyAI-based speaker diarization.
Uploads audio, gets diarized transcript, extracts target speaker.

Usage:
  export ASSEMBLYAI_API_KEY=xxx
  python3 assemblyai_diarize.py --speaker ray_dalio
  python3 assemblyai_diarize.py --speaker ray_dalio --video-id rog0QA0s3WI
"""

import json, os, sys, time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_DIR = os.path.join(PROJECT_DIR, "data", "audio")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "diarized_assemblyai")
os.makedirs(OUT_DIR, exist_ok=True)

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
if not API_KEY:
    # try .env
    dotenv = os.path.join(PROJECT_DIR, ".env")
    if os.path.exists(dotenv):
        with open(dotenv) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                    v = v[1:-1]
                os.environ.setdefault(k, v)
    API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")

import assemblyai as aai

aai.settings.api_key = API_KEY


def transcribe_with_diarization(audio_path):
    """Upload audio, get diarized transcript. Returns AssemblyAI transcript object."""
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        language_code="en",
    )
    for attempt in range(5):
        try:
            transcript = aai.Transcriber().transcribe(audio_path, config=config)
            break
        except Exception as e:
            if attempt < 4:
                wait = (attempt + 1) * 30
                print(f"  ⚠️ Retry {attempt+1}/5 in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI error: {transcript.error}")
    return transcript


def extract_target_speaker(transcript, speaker_name, llm_cfg=None):
    """Extract target speaker from diarized transcript.

    Strategy:
    1. Count words per speaker → assume the one with most words is the guest
    2. If ambiguous (ratio > 0.3 for second speaker), use LLM to ID
    """
    speaker_words = {}
    speaker_texts = {}
    for utterance in transcript.utterances:
        spk = utterance.speaker
        if spk not in speaker_words:
            speaker_words[spk] = 0
            speaker_texts[spk] = []
        text = utterance.text.strip()
        speaker_words[spk] += len(text.split())
        speaker_texts[spk].append(text)

    if len(speaker_words) == 1:
        return list(speaker_words.keys())[0], "single_speaker"

    # Duration heuristic
    primary = max(speaker_words, key=speaker_words.get)
    secondary = min(speaker_words, key=speaker_words.get)
    ratio = speaker_words[secondary] / max(speaker_words[primary], 1)

    if ratio < 0.25:
        return primary, "duration_clear"
    elif llm_cfg:
        return _llm_identify(speaker_texts, speaker_name, llm_cfg), "llm"
    else:
        return primary, "duration_ambiguous"


def _llm_identify(speaker_texts, target_name, llm_cfg):
    """Ask LLM to identify which speaker is the target."""
    import urllib.request, ssl, certifi, re

    samples = []
    for spk, texts in sorted(speaker_texts.items()):
        sample = "\n".join(texts[:5])
        samples.append(f"--- Speaker {spk} ---\n{sample[:600]}")

    system = f"You are identifying speakers in an interview. The target guest is {target_name}. Determine which speaker is the MAIN GUEST (giving opinions and analysis). Output JSON: {{\"Speaker_A\": \"guest\"|\"interviewer\", ...}}"

    payload = json.dumps({
        "model": llm_cfg["model"],
        "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": "\n\n".join(samples)}],
        "temperature": 0.0, "max_tokens": 256,
    }).encode()

    url = f"{llm_cfg['api_base']}/chat/completions"
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {llm_cfg['api_key']}")
    req.add_header("User-Agent", "Mozilla/5.0")
    ctx = ssl.create_default_context(cafile=certifi.where())

    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=30)
        content = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            labels = json.loads(m.group())
            for spk in speaker_texts:
                if labels.get(spk) == "guest":
                    return spk
    except Exception:
        pass
    # Fallback to duration
    wc = {s: sum(len(t.split()) for t in texts) for s, texts in speaker_texts.items()}
    return max(wc, key=wc.get)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def load_llm_cfg():
    from pathlib import Path
    dotenv = os.path.join(PROJECT_DIR, ".env")
    if os.path.exists(dotenv):
        with open(dotenv) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                    v = v[1:-1]
                os.environ.setdefault(k, v)
    return {
        "api_key": os.environ.get("CLEAN_API_KEY", ""),
        "api_base": os.environ.get("CLEAN_API_BASE", "").rstrip("/"),
        "model": os.environ.get("CLEAN_MODEL", "deepseek-chat"),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default="ray_dalio")
    parser.add_argument("--video-id", default=None)
    args = parser.parse_args()

    audio_spk_dir = os.path.join(AUDIO_DIR, args.speaker)
    out_spk_dir = os.path.join(OUT_DIR, args.speaker)
    os.makedirs(out_spk_dir, exist_ok=True)

    audio_files = sorted(f for f in os.listdir(audio_spk_dir) if f.endswith(('.m4a', '.mp3', '.wav')))
    if args.video_id:
        audio_files = [f for f in audio_files if f.startswith(args.video_id)]

    llm_cfg = load_llm_cfg()
    print(f"Speaker: {args.speaker} | Files: {len(audio_files)}")

    for i, fn in enumerate(audio_files):
        vid = fn.rsplit(".", 1)[0]
        out_path = os.path.join(out_spk_dir, f"{vid}.txt")
        audio_path = os.path.join(audio_spk_dir, fn)

        print(f"\n[{i+1}/{len(audio_files)}] {vid}")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            print(f"  ⏭️  already done")
            continue

        try:
            # Upload & transcribe
            print(f"  📤 Uploading ({os.path.getsize(audio_path)//1024//1024}MB)...", end="", flush=True)
            t0 = time.time()
            transcript = transcribe_with_diarization(audio_path)
            t1 = time.time()
            print(f" done in {t1-t0:.0f}s")
            n_speakers = len(set(u.speaker for u in transcript.utterances))
            print(f"  Utterances: {len(transcript.utterances)}, Speakers: {n_speakers}")

            # Identify target speaker
            target, method = extract_target_speaker(transcript, args.speaker.replace("_", " ").title(), llm_cfg)
            print(f"  Target: {target} ({method})")

            # Extract guest text
            guest_text = []
            total_words = 0
            for utt in transcript.utterances:
                nw = len(utt.text.split())
                total_words += nw
                if utt.speaker == target:
                    guest_text.append(utt.text)

            full_text = "\n".join(guest_text)
            guest_words = len(full_text.split())

            with open(out_path, "w") as f:
                f.write(full_text)

            gr = guest_words / max(total_words, 1)
            print(f"  ✅ Guest: {guest_words}w / {total_words}w (ratio={gr:.3f})")

        except Exception as e:
            print(f"  ❌ {e}")

    # Summary
    total = sum(1 for f in os.listdir(out_spk_dir) if f.endswith('.txt'))
    print(f"\nDone: {total} files in {out_spk_dir}/")
