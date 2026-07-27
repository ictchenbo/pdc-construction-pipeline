#!/usr/bin/env python3
"""
Pipeline Quality Audit (Phase 1: Clean + Diarize + Split).
Samples 3 speakers per domain × 1 video each = 21 videos.
Uses a secondary LLM to evaluate 3 processing stages per video.

Stages:
  1. VTT → Cleaned:    punctuation, capitalization, disfluency removal
  2. Cleaned → Diarized: interviewer removal accuracy
  3. Diarized → Sentences: sentence boundary quality

Scoring per stage: 0 (broken), 1 (minor issues), 2 (good)

Output: data/inspection/pipeline_audit.json + .txt report

Usage:
  python3 src/08_statistical_analysis/pipeline_quality_audit.py
  python3 src/08_statistical_analysis/pipeline_quality_audit.py --endpoint https://... --key sk-... --model gpt-4

  If --endpoint/--key not provided, reads from .env:
    AUDIT_API_BASE, AUDIT_API_KEY, AUDIT_MODEL
"""

import argparse, json, os, random, re, sys, time, urllib.request, ssl
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
CLEANED_DIR = os.path.join(PROJECT_DIR, "data", "cleaned")
DIARIZED_DIR = os.path.join(PROJECT_DIR, "data", "diarized")
SENTENCES_DIR = os.path.join(PROJECT_DIR, "data", "sentences_by_video")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "inspection")

from src.config import SPEAKER_DOMAINS, DOMAIN_ORDER

# ── Prompt templates per stage ──
PROMPTS = {
    "clean": {
        "system": "You are a transcript quality evaluator. You will see a raw YouTube subtitle (VTT) and its LLM-cleaned version. Rate the cleaning quality on a scale of 0, 1, or 2.",
        "user_template": """RAW VTT (before cleaning):
{Raw}

CLEANED TEXT (after LLM processing):
{Cleaned}

Evaluate the cleaning quality. Check:
- Are punctuation and capitalization correctly added?
- Are obvious disfluencies (um, uh, repeated words) removed?
- Is the text coherent and readable?
- Are there any entire paragraphs of garbled text?

Score: 2 = excellent (clean, well-punctuated, very minor issues)
Score: 1 = acceptable (mostly good but some missing punctuation or minor errors)
Score: 0 = poor (significant errors, garbled text, large sections without punctuation)

Output ONLY: {{"score": <int>, "reason": "<1-2 sentences>"}}""",
        "max_score": 2,
    },
    "diarize": {
        "system": "You are a speaker diarization evaluator. You will see a cleaned interview transcript and its diarized (guest-only) version. Rate the diarization quality.",
        "user_template": """CLEANED TEXT (before diarization, may contain interviewer speech):
{Cleaned}

DIARIZED TEXT (after speaker filtering, guest-only):
{Diarized}

Evaluate the diarization quality. Check:
- Was interviewer speech (introductory remarks, questions, "Welcome...", "Next topic...") correctly removed?
- Was guest speech incorrectly removed?
- Does the diarized text contain only the guest's own words?

Score: 2 = excellent (clean separation, guest content preserved)
Score: 1 = minor issues (a few interviewer lines remain, or a sentence of guest lost)
Score: 0 = poor (major contamination or large sections lost)

Output ONLY: {{"score": <int>, "reason": "<1-2 sentences>"}}""",
        "max_score": 2,
    },
    "split": {
        "system": "You are a sentence boundary evaluator. You will see a diarized transcript and its sentence-split version. Rate the sentence splitting quality.",
        "user_template": """DIARIZED TEXT (before sentence splitting):
{Diarized}

SENTENCE-SPLIT TEXT (sample of 10 sentences):
{Sentences}

Evaluate the sentence splitting. Check:
- Are sentences correctly separated at natural boundaries?
- Are there sentences that are clearly too long (>60 words, run-on)?
- Are there sentence fragments that should have been part of a larger sentence?

Score: 2 = excellent (clean sentence boundaries)
Score: 1 = minor issues (a few run-on sentences or fragments)
Score: 0 = poor (many bad splits)

Output ONLY: {{"score": <int>, "reason": "<1-2 sentences>"}}""",
        "max_score": 2,
    },
}


# ── Function definitions ──

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", help="LLM API base URL")
    parser.add_argument("--key", help="LLM API key")
    parser.add_argument("--model", help="LLM model name (default: gpt-4o)")
    parser.add_argument("--dry-run", action="store_true", help="Only sample, no API calls")
    parser.add_argument("--stages", default="clean,diarize,split",
                        help="Comma-separated stages to run: clean,diarize,split (default: all)")
    args = parser.parse_args()
    dry_run = args.dry_run

    # Parse stages
    valid_stages = {"clean", "diarize", "split"}
    stages = set(s.strip() for s in args.stages.split(","))
    unknown = stages - valid_stages
    if unknown:
        print(f"ERROR: Unknown stages: {unknown}. Valid: {sorted(valid_stages)}")
        sys.exit(1)
    if not stages:
        stages = valid_stages

    config = {
        "endpoint": args.endpoint, "key": args.key, "model": args.model,
        "temp": 0.0, "max_tokens": 2048,
    }

    # From .env
    env_path = os.path.join(PROJECT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                    v = v[1:-1]
                if k == "AUDIT_API_BASE" and not config["endpoint"]:
                    config["endpoint"] = v
                if k == "AUDIT_API_KEY" and not config["key"]:
                    config["key"] = v
                if k == "AUDIT_MODEL" and not config["model"]:
                    config["model"] = v

    if not config["endpoint"] or not config["key"]:
        if not dry_run:
            print("ERROR: Provide --endpoint and --key, or set AUDIT_API_BASE and AUDIT_API_KEY in .env")
            sys.exit(1)
        else:
            config["endpoint"] = "dry-run"
            config["key"] = "dry-run"

    if not config["endpoint"].endswith("/chat/completions") and not dry_run:
        config["endpoint"] = config["endpoint"].rstrip("/") + "/chat/completions"

    return config, dry_run, stages


def validate_endpoint(endpoint, key, model):
    """Quick connectivity test before starting the audit."""
    try:
        payload = json.dumps({
            "model": model, "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0, "max_tokens": 1,
        }).encode()
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(endpoint, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "Mozilla/5.0",
        })
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return True, f"HTTP {resp.getcode()} (endpoint reachable)"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        if e.code in (401, 403):
            return False, f"认证失败 (HTTP {e.code}): 请检查 API Key"
        if e.code == 404:
            return False, f"端点不存在 (HTTP 404): 请检查 URL 是否包含 /chat/completions"
        try:
            d = json.loads(body)
            if "error" in d and "model" in str(d["error"]).lower():
                return True, f"端点正常 (HTTP {e.code}: model mismatch, 不影响)"
        except json.JSONDecodeError:
            pass
        return False, f"HTTP {e.code}: {body[:200]}"
    except urllib.error.URLError as e:
        return False, f"无法连接到端点: {str(e.reason)[:120]}"
    except ssl.SSLError as e:
        return False, f"SSL证书错误: {str(e)[:120]}"
    except Exception as e:
        return False, f"连接失败: {type(e).__name__}: {str(e)[:120]}"


def call_llm(endpoint, key, model, system, user):
    """Call the audit LLM, return (success, response_or_error)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "reasoning_effort": "none",
        "temperature": 0.0,
        "max_tokens": 2048,
    }).encode()
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(endpoint, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": "Mozilla/5.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.getcode()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        if body.startswith("<"):
            body = re.sub(r'<[^>]+>', ' ', body).strip()
            body = re.sub(r'\s+', ' ', body)[:200]
        return False, f"HTTP {e.code}: {body}"
    except urllib.error.URLError as e:
        return False, f"网络错误: {str(e.reason)[:120]}"
    except ssl.SSLError as e:
        return False, f"SSL错误: {str(e)[:120]}"
    except Exception as e:
        return False, f"连接错误: {type(e).__name__}: {str(e)[:120]}"

    if status != 200:
        return False, f"HTTP {status}: {raw[:200]}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"JSON解析失败: {raw[:200]}"

    if "error" in data:
        return False, f"API错误: {json.dumps(data['error'])[:200]}"

    if "choices" not in data or not data["choices"]:
        return False, f"无choices: {str(data)[:200]}"

    return True, data["choices"][0]["message"]["content"].strip()


def parse_response(content, stage):
    """Parse JSON from LLM response. Returns dict with score + reason."""
    if content is None:
        return {"stage": stage, "score": -1, "reason": "空响应"}

    # Try direct JSON parse
    try:
        d = json.loads(content)
        if "score" in d:
            score = int(d["score"])
            if score not in (0, 1, 2):
                return {"stage": stage, "score": -1, "reason": f"无效分数 {score}: {content[:100]}"}
            return {"stage": stage, "score": score, "reason": d.get("reason", "").strip()}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to extract JSON from markdown code block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if not m:
        m = re.search(r'\{[^{}]*"score"[^{}]*\}', content, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1) if '```' in content else m.group())
            if "score" in d:
                score = int(d["score"])
                if score in (0, 1, 2):
                    return {"stage": stage, "score": score, "reason": d.get("reason", "").strip()}
        except (json.JSONDecodeError, ValueError):
            pass

    return {"stage": stage, "score": -1, "reason": f"解析失败: {content[:120]}"}


def extract_vtt_plaintext(vtt_text):
    """Strip VTT formatting to get only spoken text for fair comparison.

    YouTube VTT files contain: WEBVTT header, timestamps, <c> word-level
    timing tags, speaker labels (>>), alignment metadata, and duplicate
    lines (one with tags, one plain).  This extracts only the clean
    plain-text lines, skipping metadata and tagged duplicates.
    """
    lines = []
    for line in vtt_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip metadata lines
        if line in ("WEBVTT", "Kind: captions", "Language: en"):
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        # Skip timestamp lines
        if "-->" in line and line[0].isdigit():
            continue
        # Skip alignment metadata
        if line.startswith("align:"):
            continue
        # Skip lines with word-level timing tags (they have duplicates without tags)
        if "<c>" in line or "</c>" in line:
            continue
        # Strip speaker labels and HTML entities
        line = line.replace("&gt;&gt;", "").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        # Skip now-empty lines after stripping
        if not line or line.isspace():
            continue
        lines.append(line)
    # Deduplicate consecutive duplicates
    result = []
    for line in lines:
        if not result or line != result[-1]:
            result.append(line)
    return "\n".join(result)


def align_vtt_to_cleaned(vtt_plaintext, cleaned_text, window_chars=2500):
    """Find the VTT segment that corresponds to the cleaned text's beginning.

    The cleaning pipeline already strips interviewer speech, so the raw
    VTT starts with interviewer intro while cleaned text starts with guest
    speech.  We anchor on distinctive content phrases, progressively
    shortening until we find a match (the VTT may contain filler words
    like "um" that the cleaned text has removed).
    """
    cleaned_first = cleaned_text.strip()
    vtt_lower = vtt_plaintext.lower()

    # Try progressively shorter anchors
    for anchor_len in (120, 80, 50, 30):
        anchor = cleaned_first[:anchor_len].rsplit(" ", 1)[0]
        anchor_lower = anchor.lower()
        idx = vtt_lower.find(anchor_lower)
        if idx >= 0:
            break

    if idx < 0:
        # Try key content words from the first sentence
        words = cleaned_first[:200].split()
        content_words = [w for w in words if len(w) > 5 and w.lower()
                         not in ("there", "their", "these", "those", "about",
                                 "which", "would", "could", "should")]
        for i in range(len(content_words) - 2):
            phrase = " ".join(content_words[i:i+3])
            idx = vtt_lower.find(phrase.lower())
            if idx >= 0:
                break

    if idx < 0:
        # Fallback: skip first 30% of VTT (typical interviewer intro zone)
        idx = len(vtt_plaintext) // 3

    # Include some context before the match
    start = max(0, idx - 300)
    end = min(len(vtt_plaintext), idx + window_chars)
    return vtt_plaintext[start:end]


def sample_videos():
    """Pick 3 speakers per domain, 1 video each."""
    selected = []
    for domain in DOMAIN_ORDER:
        spks = sorted([s for s, (_, d) in SPEAKER_DOMAINS.items() if d == domain])
        if not spks:
            continue
        chosen = random.sample(spks, min(3, len(spks)))
        for spk in chosen:
            vtt_dir = os.path.join(RAW_DIR, spk, "transcripts")
            if not os.path.isdir(vtt_dir):
                continue
            if not os.path.isdir(os.path.join(CLEANED_DIR, spk)):
                continue
            if not os.path.isdir(os.path.join(DIARIZED_DIR, spk)):
                continue
            if not os.path.isdir(os.path.join(SENTENCES_DIR, spk)):
                continue
            vtt_ids = {fn.replace(".en.vtt", "").replace(".vtt", "")
                       for fn in os.listdir(vtt_dir) if fn.endswith(".vtt")}
            cleaned_ids = {fn[:-4] for fn in os.listdir(os.path.join(CLEANED_DIR, spk))
                           if fn.endswith(".txt") and not fn.startswith("_")}
            diarized_ids = {fn[:-4] for fn in os.listdir(os.path.join(DIARIZED_DIR, spk))
                            if fn.endswith(".txt") and not fn.startswith("_")}
            sentences_ids = {fn[:-4] for fn in os.listdir(os.path.join(SENTENCES_DIR, spk))
                             if fn.endswith(".txt")}

            common = sorted(vtt_ids & cleaned_ids & diarized_ids & sentences_ids)
            if common:
                vid = random.choice(common)
                selected.append({"speaker": spk, "domain": domain, "video_id": vid})
    return selected


def interpolate_template(template, **kwargs):
    """Safely replace {Placeholder} tokens in template with actual text.
    Uses str.replace() instead of str.format() to avoid errors when the
    text contains literal { or } characters (common in VTT transcripts)."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", value)
    return result


def run_audit(config, dry_run, stages):
    """Run the pipeline audit. Returns list of result dicts."""
    videos = sample_videos()
    num_stages = len(stages)
    print(f"Sampled {len(videos)} videos from {len(set(v['domain'] for v in videos))} domains")
    if not dry_run:
        print(f"Estimated API calls: {len(videos) * num_stages} ({num_stages} stage(s))")
        print(f"Estimated cost: ~${0.01 * len(videos) * num_stages:.2f} (GPT-4o pricing)")
    print()

    results = []
    for vi, v in enumerate(videos):
        spk = v["speaker"]
        vid = v["video_id"]
        domain = v["domain"]
        print(f"[{vi+1}/{len(videos)}] {spk}/{vid} ({domain})", end="", flush=True)

        # Read the raw files
        vtt_file = os.path.join(RAW_DIR, spk, "transcripts", f"{vid}.en.vtt")
        if not os.path.exists(vtt_file):
            vtt_file = os.path.join(RAW_DIR, spk, "transcripts", f"{vid}.vtt")
        if not os.path.exists(vtt_file):
            print("  SKIP (no VTT)")
            continue

        cleaned_file = os.path.join(CLEANED_DIR, spk, f"{vid}.txt")
        diarized_file = os.path.join(DIARIZED_DIR, spk, f"{vid}.txt")
        sentences_file = os.path.join(SENTENCES_DIR, spk, f"{vid}.txt")

        try:
            with open(vtt_file, encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            with open(cleaned_file, encoding="utf-8", errors="replace") as f:
                cleaned_text = f.read()
            with open(diarized_file, encoding="utf-8", errors="replace") as f:
                diarized_text = f.read()
            with open(sentences_file, encoding="utf-8", errors="replace") as f:
                sentences_text = f.read()
        except Exception as e:
            stages = [{"stage": s, "score": -1, "reason": f"读取失败: {type(e).__name__}"}
                      for s in ("clean", "diarize", "split")]
            results.append({"speaker": spk, "video_id": vid, "domain": domain, "stages": stages})
            print(f"  FILE ERROR: {e}", flush=True)
            continue

        # Extract plaintext from VTT and align to the cleaned text's content.
        # The cleaning pipeline strips interviewer speech, so raw VTT begins
        # with interviewer intro while cleaned text starts with guest speech.
        # We use the cleaned text as an anchor to find the matching VTT segment.
        vtt_plain = extract_vtt_plaintext(raw_text)
        raw_sample = align_vtt_to_cleaned(vtt_plain, cleaned_text)[:2000]
        cleaned_sample = cleaned_text[:1500]
        diarized_sample = diarized_text[:1500]
        sent_lines = [l.strip() for l in sentences_text.split("\n") if l.strip()][:10]
        sent_sample = "\n".join(sent_lines)

        video_result = {"speaker": spk, "video_id": vid, "domain": domain}
        stage_results = []
        errors_shown = []

        if dry_run:
            video_result["stages"] = [
                {"stage": s, "score": 2, "reason": "(dry run)"}
                for s in ("clean", "diarize", "split") if s in stages
            ]
            results.append(video_result)
            print(f"  (dry run)", flush=True)
            continue

        ep, key, model = config["endpoint"], config["key"], config["model"]

        # Stage 1: Clean
        if "clean" in stages:
            try:
                user_msg = interpolate_template(
                    PROMPTS["clean"]["user_template"],
                    Raw=raw_sample, Cleaned=cleaned_sample,
                )
                ok, resp = call_llm(ep, key, model, PROMPTS["clean"]["system"], user_msg)
                if ok:
                    stage_results.append(parse_response(resp, "clean"))
                    print("  C", end="", flush=True)
                else:
                    stage_results.append({"stage": "clean", "score": -1, "reason": resp})
                    print("  C✗", end="", flush=True)
                    errors_shown.append(f"clean: {resp[:150]}")
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:120]}"
                stage_results.append({"stage": "clean", "score": -1, "reason": msg})
                print("  C✗", end="", flush=True)
                errors_shown.append(f"clean: {msg}")
        else:
            print("  C·", end="", flush=True)

        # Stage 2: Diarize
        if "diarize" in stages:
            try:
                user_msg = interpolate_template(
                    PROMPTS["diarize"]["user_template"],
                    Cleaned=cleaned_sample, Diarized=diarized_sample,
                )
                ok, resp = call_llm(ep, key, model, PROMPTS["diarize"]["system"], user_msg)
                if ok:
                    stage_results.append(parse_response(resp, "diarize"))
                    print(" D", end="", flush=True)
                else:
                    stage_results.append({"stage": "diarize", "score": -1, "reason": resp})
                    print(" D✗", end="", flush=True)
                    errors_shown.append(f"diarize: {resp[:150]}")
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:120]}"
                stage_results.append({"stage": "diarize", "score": -1, "reason": msg})
                print(" D✗", end="", flush=True)
                errors_shown.append(f"diarize: {msg}")
        else:
            print(" D·", end="", flush=True)

        # Stage 3: Split
        if "split" in stages:
            try:
                user_msg = interpolate_template(
                    PROMPTS["split"]["user_template"],
                    Diarized=diarized_sample, Sentences=sent_sample,
                )
                ok, resp = call_llm(ep, key, model, PROMPTS["split"]["system"], user_msg)
                if ok:
                    stage_results.append(parse_response(resp, "split"))
                    print(" S", flush=True)
                else:
                    stage_results.append({"stage": "split", "score": -1, "reason": resp})
                    print(" S✗", flush=True)
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:120]}"
                stage_results.append({"stage": "split", "score": -1, "reason": msg})
                print(" S✗", flush=True)
        else:
            print(" S·", flush=True)

        video_result["stages"] = stage_results
        results.append(video_result)

        if errors_shown:
            for e in errors_shown:
                print(f"    ↳ {e}", flush=True)

        # Save intermediate
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "pipeline_audit.json"), "w") as f:
            json.dump({"config": {"model": config["model"]}, "results": results}, f, indent=2)

    return results


def generate_report(results):
    """Print and save the audit report."""
    if not results:
        print("No results to report.")
        return

    stage_scores = defaultdict(list)
    stage_issues = defaultdict(list)
    domain_scores = defaultdict(lambda: defaultdict(list))

    for r in results:
        for s in r["stages"]:
            stage_scores[s["stage"]].append(s["score"])
            domain_scores[r["domain"]][s["stage"]].append(s["score"])
            if 0 <= s["score"] <= 1:
                stage_issues[s["stage"]].append({
                    "speaker": r["speaker"], "video_id": r["video_id"],
                    "score": s["score"], "reason": s.get("reason", ""),
                })

    print(f"\n{'='*70}")
    print("  Pipeline Quality Audit Report")
    print(f"{'='*70}")

    print("\n  Overall (per stage):")
    print(f"  {'Stage':<12} {'N':>4} {'Mean':>6} {'Score 0':>8} {'Score 1':>8} {'Score 2':>8}")
    print(f"  {'─'*50}")
    for stage in ["clean", "diarize", "split"]:
        scores = stage_scores[stage]
        valid = [s for s in scores if s >= 0]
        if valid:
            n0 = sum(1 for s in valid if s == 0)
            n1 = sum(1 for s in valid if s == 1)
            n2 = sum(1 for s in valid if s == 2)
            print(f"  {stage:<12} {len(valid):>4} {sum(valid)/len(valid):>5.2f} {n0:>8} {n1:>8} {n2:>8}")

    print("\n  By domain:")
    print(f"  {'Domain':<22} {'C-mean':>7} {'D-mean':>7} {'S-mean':>7} {'Overall':>8}")
    print(f"  {'─'*55}")
    for domain in DOMAIN_ORDER:
        ds = domain_scores.get(domain)
        if not ds:
            continue
        c_m = sum(s for s in ds["clean"] if s >= 0) / max(1, len([s for s in ds["clean"] if s >= 0]))
        d_m = sum(s for s in ds["diarize"] if s >= 0) / max(1, len([s for s in ds["diarize"] if s >= 0]))
        s_m = sum(s for s in ds["split"] if s >= 0) / max(1, len([s for s in ds["split"] if s >= 0]))
        all_s = [s for s in ds["clean"] + ds["diarize"] + ds["split"] if s >= 0]
        overall = sum(all_s) / max(1, len(all_s))
        print(f"  {domain:<22} {c_m:>7.2f} {d_m:>7.2f} {s_m:>7.2f} {overall:>8.2f}")

    # Issues section
    error_count = sum(1 for r in results for s in r["stages"] if s["score"] < 0)
    total_issues = sum(len(v) for v in stage_issues.values())
    if total_issues + error_count > 0:
        print(f"\n  问题 ({total_issues} 质量问题 + {error_count} 错误):")
        print(f"  {'─'*70}")
        for r in results:
            for s in r["stages"]:
                if s["score"] < 0:
                    print(f"  ❌ [{s['stage']}/{r['speaker']}/{r['video_id'][:10]}] {s['reason'][:100]}")
        for stage in ["clean", "diarize", "split"]:
            for issue in stage_issues[stage][:5]:
                print(f"  ⚠️ [{stage}/{issue['speaker']}/{issue['video_id'][:8]}] score={issue['score']}: {issue['reason'][:80]}")

    # Save human-readable report
    report_path = os.path.join(OUT_DIR, "pipeline_audit.txt")
    with open(report_path, "w") as f:
        f.write("Pipeline Quality Audit Report\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"[{r['domain']}] {r['speaker']}/{r['video_id']}\n")
            for s in r["stages"]:
                score_str = "?" if s["score"] < 0 else str(s["score"])
                f.write(f"  {s['stage']}: score={score_str}  {s.get('reason', '')}\n")
            f.write("\n")

    # Final overall grade
    all_valid = [s["score"] for r in results for s in r["stages"] if s["score"] >= 0]
    if all_valid:
        overall_mean = sum(all_valid) / len(all_valid)
        pct_2 = sum(1 for s in all_valid if s == 2) / len(all_valid)
        pct_atleast1 = sum(1 for s in all_valid if s >= 1) / len(all_valid)
        grade = "A" if pct_2 >= 0.8 else "B" if pct_atleast1 >= 0.8 else "C" if pct_atleast1 >= 0.6 else "D"

        print(f"\n  {'─'*55}")
        print(f"  总评: {grade} (均分={overall_mean:.2f}/2, {pct_2:.0%}满分, {pct_atleast1:.0%}≥1分)")
        print(f"  有效评分: {len(all_valid)}/{len(results)*3}, 错误: {sum(1 for r in results for s in r['stages'] if s['score']<0)}")
        print(f"  按领域均分:")
        for domain in DOMAIN_ORDER:
            ds = domain_scores.get(domain)
            if not ds or not ds.get("clean"):
                continue
            all_ds = [s for s in ds["clean"] + ds["diarize"] + ds["split"] if s >= 0]
            if all_ds:
                print(f"    {domain:22s} {sum(all_ds)/len(all_ds):.2f} ({len(all_ds)} samples)")

    print(f"\nFull report: {report_path}")
    print(f"JSON data: {os.path.join(OUT_DIR, 'pipeline_audit.json')}")


# ── Entry point ──

def main():
    random.seed(42)

    config, dry_run, stages = load_config()
    print(f"Pipeline Quality Audit")
    print(f"  Model: {config['model']}")
    print(f"  Stages: {', '.join(sorted(stages))}")
    print(f"  Endpoint: {config['endpoint'][:60]}...")
    print()

    # Pre-flight connectivity check
    if not dry_run:
        ok, msg = validate_endpoint(config["endpoint"], config["key"], config["model"])
        if not ok:
            print(f"❌ 端点连接失败: {msg}")
            print(f"   请检查 .env 中的 AUDIT_API_BASE 和 AUDIT_API_KEY")
            sys.exit(1)
        print(f"✅ 端点连接: {msg}")
        print()

    results = run_audit(config, dry_run, stages)

    if dry_run:
        print("\nDry run complete. Use without --dry-run to actually evaluate.")
        sys.exit(0)

    generate_report(results)


if __name__ == "__main__":
    main()
