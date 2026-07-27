#!/usr/bin/env python3
"""Annotate a sample manifest with LLM providers, with optional thinking mode.

Unifies gpt.py / gpt_thinking.py / deepseek.py / deepseek_thinking.py.

Usage:
  python3 src/04_annotation/annotate.py --provider gpt5
  python3 src/04_annotation/annotate.py --provider deepseek --thinking
  python3 src/04_annotation/annotate.py --provider gpt5 --thinking
  python3 src/04_annotation/annotate.py --provider deepseek_pro --thinking
  python3 src/04_annotation/annotate.py --provider gpt5 --manifest /tmp/my_manifest.jsonl
  python3 src/04_annotation/annotate.py --provider gpt5 --dry-run
"""

import argparse
import json
import os
import random
import ssl
import sys
import time
import traceback

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    PROJECT_DIR, VALIDATION_DIR,
    get_llm_config,
)
from src.utils.common import save_jsonl, load_jsonl

CTX = ssl._create_unverified_context()

SYSTEM_PROMPT = """You are a linguistic annotation assistant. For each sentence, output:
sentence_index|valence|modality

Valence: positive / negative / neutral
Modality: emphatic / hedged / neutral
Be conservative: prefer neutral when unsure. One line per sentence. No extra text."""


# ── Thinking-param helpers ──────────────────────────────────────────────

# Providers that use `thinking: {"type": "enabled"|"disabled"}`  (DeepSeek-style)
_DEEPSEEK_STYLE = {"deepseek", "deepseek_pro"}

# Providers that use `reasoning: {"effort": "high"}` (OpenAI-style reasoning)
_REASONING_STYLE = {"gpt5"}


def _thinking_params(provider: str, thinking: bool) -> dict:
    """Return extra payload keys for thinking / non-thinking mode.

    Different providers expose thinking via different API parameters.
    """
    if provider in _DEEPSEEK_STYLE:
        if thinking:
            return {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
        else:
            return {"thinking": {"type": "disabled"}}
    elif provider in _REASONING_STYLE:
        if thinking:
            return {"reasoning": {"effort": "xhigh"}}
        else:
            return {"reasoning": {"effort": "none"}}
    else:
        # Unknown provider — don't touch thinking params at all
        return {}


# ── Annotation logic ────────────────────────────────────────────────────

def load_existing_keys(provider_path):
    """Load already-annotated keys from an existing per-provider output file."""
    keys = set()
    if not os.path.exists(provider_path):
        return keys
    for rec in load_jsonl(provider_path):
        keys.add(f"{rec.get('video_id', '')}:{rec.get('sentence_index', -1)}")
    return keys


def annotate_with_provider(samples, llm_cfg, provider_name, provider_path,
                           thinking=False, dry_run=False):
    """Annotate sample sentences with one LLM provider.

    Appends results batch-by-batch so partial runs can resume.
    """
    existing_keys = load_existing_keys(provider_path)
    pending = [s for s in samples
               if f"{s['video_id']}:{s['sentence_index']}" not in existing_keys]

    if not pending:
        print(f"    All {len(samples)} entries already annotated -- skipping")
        return load_jsonl(provider_path)

    print(f"    {len(existing_keys)} done, {len(pending)} pending")

    if dry_run:
        print(f"    [DRY RUN] would annotate {len(pending)} sentences")
        results = []
        for s in pending:
            results.append({
                "video_id": s["video_id"],
                "sentence_index": s["sentence_index"],
                "text": s["text"],
                "valence": random.choice(["positive", "negative", "neutral"]),
                "modality": random.choice(["emphatic", "hedged", "neutral"]),
            })
        return results

    results = list(load_jsonl(provider_path)) if existing_keys else []
    batch_size = 50

    url = f"{llm_cfg['api_base']}/chat/completions"
    print("requesting LLM: ", url)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm_cfg['api_key']}"
    }

    for idx in range(0, len(pending), batch_size):
        print(f'calling LLM[{provider_name}] batch {idx}')
        batch = pending[idx:idx + batch_size]
        numbered = "\n".join(f"{i}. {s['text']}" for i, s in enumerate(batch))

        payload = {
            "model": llm_cfg["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": numbered}
            ],
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        payload.update(_thinking_params(provider_name, thinking))

        batch_results = []
        success = False
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=180)
                resp.raise_for_status()
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                # Parse pipe-delimited output: sentence_index|valence|modality
                for line in content.split('\n'):
                    parts = line.strip().split('|')
                    if len(parts) < 3:
                        continue
                    try:
                        rel_idx = int(parts[0].strip())
                        v = parts[1].strip().lower()
                        m = parts[2].strip().lower()
                        if (rel_idx < len(batch)
                                and v in ('positive', 'negative', 'neutral')
                                and m in ('emphatic', 'hedged', 'neutral')):
                            s = batch[rel_idx]
                            batch_results.append({
                                "video_id": s["video_id"],
                                "sentence_index": s["sentence_index"],
                                "text": s["text"],
                                "valence": v,
                                "modality": m,
                            })
                    except (ValueError, IndexError):
                        continue
                success = True
                break
            except Exception as e:
                traceback.print_exc()
                if attempt == 2:
                    print(f"    [ERROR] batch {idx // batch_size + 1} failed: {e}")
                time.sleep(2 ** attempt)

        if success:
            results.extend(batch_results)
            with open(provider_path, "a") as f:
                for r in batch_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        time.sleep(1)

    return results


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Annotate sample manifest with one LLM provider (optionally with thinking mode). "
                    "Provider name must match a key in src.config.LLM_CONFIGS."
    )
    parser.add_argument("--manifest",
                        help="Path to sample manifest (default: data/validation/sample_manifest.jsonl)")
    parser.add_argument("--provider", default="gpt5",
                        help="LLM provider key in src.config.LLM_CONFIGS (default: gpt5)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking / reasoning mode")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(VALIDATION_DIR, "sample_manifest.jsonl")

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        print("Run src/04_annotation/sample.py first, or specify --manifest", file=sys.stderr)
        sys.exit(1)

    samples = load_jsonl(manifest_path)
    print(f"Loaded {len(samples)} sentences from manifest: {manifest_path}")

    os.makedirs(VALIDATION_DIR, exist_ok=True)
    provider = args.provider

    # Build output path that includes thinking mode and manifest identifier.
    #   manifest=sample_manifest_0.01_v3, --provider gpt5              → gpt5_annotations_default_0.01_v3.jsonl
    #   manifest=sample_manifest_0.01_v3, --provider gpt5 --thinking   → gpt5_annotations_thinking_0.01_v3.jsonl
    #   manifest=sample_manifest_w6-80.0.01_v1, --provider deepseek    → deepseek_annotations_default_w6-80.0.01_v1.jsonl
    manifest_base = os.path.basename(manifest_path)           # sample_manifest_0.01_v3.jsonl
    if manifest_base.startswith("sample_manifest_"):
        manifest_tag = manifest_base[len("sample_manifest_"):-len(".jsonl")]
    else:
        manifest_tag = manifest_base.rsplit(".", 1)[0]
    tag = "thinking" if args.thinking else "default"
    provider_path = os.path.join(VALIDATION_DIR, f"{provider}_annotations_{tag}_{manifest_tag}.jsonl")

    print(f"\nAnnotating with {provider} (thinking={args.thinking}), output to {provider_path}")

    if not args.dry_run:
        if os.path.exists(provider_path) and os.path.getsize(provider_path) > 0:
            existing = load_jsonl(provider_path)
            n_expected = len(samples)
            if len(existing) >= n_expected:
                print(f"  [SKIP] {provider}: output already complete ({len(existing)}/{n_expected})")
                return
            print(f"  [RESUME] {provider}: {len(existing)}/{n_expected} existing")

        try:
            cfg = get_llm_config(provider)
        except ValueError as e:
            print(f"  [SKIP] {provider}: {e}")
            sys.exit(1)
    else:
        cfg = None

    results = annotate_with_provider(
        samples, cfg, provider,
        provider_path, thinking=args.thinking, dry_run=args.dry_run
    )
    print(f"  {provider}: {len(results)} annotations ({provider_path})")


if __name__ == "__main__":
    main()
