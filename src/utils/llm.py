#!/usr/bin/env python3
"""Project-wide LLM API + .env utilities.

All scripts that call OpenAI-compatible APIs share this module.
Import via:  from src.utils.llm import llm_retry
"""

import json, os, time

import requests
from dotenv import load_dotenv

# ── Project root and .env ────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(_here))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))


# ══════════════════════════════════════════════════════════════════════════
#  API call
# ══════════════════════════════════════════════════════════════════════════

def llm_retry(llm_cfg, messages, attempts=3, temperature=0.0,
              max_tokens=4096, timeout=120):
    """OpenAI-compatible chat completion with retry. Returns parsed JSON body."""
    url = f"{llm_cfg['api_base']}/chat/completions"
    payload = {
        "model": llm_cfg["model"], 
        "messages": messages,
        "temperature": temperature, 
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {llm_cfg['api_key']}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    }

    for n in range(attempts):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if n == attempts - 1:
                raise
            time.sleep(2 ** n)


def llm_call(llm_cfg, messages, attempts=3, temperature=0.0,
             max_tokens=2048, timeout=120):
    """Convenience wrapper around llm_retry that returns the content string."""
    body = llm_retry(llm_cfg, messages, attempts=attempts,
                     temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return body["choices"][0]["message"]["content"]
