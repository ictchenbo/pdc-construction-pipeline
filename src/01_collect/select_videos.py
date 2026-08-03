#!/usr/bin/env python3
"""
Pipeline: consolidate search results → download VTT subtitles → filter non-speaker content → select → download audio.

Reads video_list.jsonl, downloads missing VTT subtitles via yt-dlp,
marks unavailable/no_captions videos, enriches metadata to flag non-speaker
content, selects IDs for audio download (balanced by speaker, VTT optional),
then downloads audio in parallel.

Modes:
    python select_videos.py consolidate     # Step 0: Build JSONL from search results
    python select_videos.py download-vtt    # Phase 1: Download missing VTTs
    python select_videos.py filter          # Phase 2: Flag non-speaker content
    python select_videos.py select          # Phase 3: Select N IDs for audio
    python select_videos.py download-audio  # Phase 4: Download audio (parallel)
    python select_videos.py full            # download-vtt + filter + select + download-audio

Step 0 — consolidate:
    Reads search_results/<speaker>/video_list.json per speaker (produced by
    search_speakers.py) and merges them into a flat video_list.jsonl with
    a "speaker" field and (when available) "description" and "language".
    Preserves existing entries' status fields. Run once after search or
    whenever search results are refreshed.

Phase 1 — download-vtt:
    For each video in video_list.jsonl without a .en.vtt file in vtt_raw/,
    try to download automatic English subtitles via yt-dlp (android client).
    If the video is unavailable → mark "status": "unavailable".
    If no captions exist → mark "status": "no_captions".

Phase 2 — filter:
    For each video with a VTT file, scores speaker-name evidence in
    title/channel/transcript (identity scoring) and checks description
    for reposted documentaries, news reports, or non-English content.
    Uses description/language already on the entry if present; otherwise
    falls back to yt-dlp --dump-json fetch. Flagged entries get
    "status": "filtered" + "filter_reason".

Phase 3 — select:
    From entries with no audio and no exclusion status, select up to N IDs
    sampling evenly across speakers. VTT subtitles are not required.
    Output: JSONL with full entry data.

Phase 4 — download-audio:
    Reads the list from Phase 3, downloads m4a audio via yt-dlp in
    parallel (default 3 workers), skips already-downloaded files.
"""

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path


# ── Paths (adjust as needed) ──────────────────────────────────────────────────
DATA_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

VIDEO_LIST = f"{DATA_BASE}/video_list.jsonl"
VTT_DIR = f"{DATA_BASE}/vtt_raw"
AUDIO_DIR = f"{DATA_BASE}/audio_raw"
META_CACHE = f"{DATA_BASE}/video_meta.jsonl"
YT_DLP = shutil.which("yt-dlp") or "yt-dlp"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_entries(path: str) -> list[dict]:
    """Load all JSONL entries into a list."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_entries(path: str, entries: list[dict]):
    """Atomically write entries back as JSONL."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def get_existing_vtt_ids(vtt_dir: str) -> set[str]:
    """Return set of video IDs that already have .en.vtt files."""
    ids = set()
    if not os.path.isdir(vtt_dir):
        return ids
    for f in os.listdir(vtt_dir):
        if f.endswith(".en.vtt"):
            ids.add(f.replace(".en.vtt", ""))
    return ids


def get_existing_audio_ids(audio_dir: str) -> set[str]:
    """Return set of video IDs whose audio has already been downloaded."""
    ids = set()
    if not os.path.isdir(audio_dir):
        return ids
    for f in os.listdir(audio_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".m4a", ".wav", ".mp3", ".flac", ".ogg", ".aac", ".opus"):
            ids.add(stem)
    return ids


# ── Phase 0: Consolidate per-speaker search results into JSONL ────────────

def consolidate(args):
    """Build/update video_list.jsonl from per-speaker search results.

    Reads search_results/<speaker>/video_list.json files produced by
    search_speakers.py, flattens them into the shared video_list.jsonl.
    Preserves existing entries' status/identity fields if the output
    already exists. New (speaker, video) pairs are appended; existing
    entries' metadata fields (title, description, language) are refreshed
    from search output.

    After consolidate, run 'enrich' or 'filter' to fill in descriptions
    for entries that came from older search results without description.
    """
    search_dir = Path(args.search_dir)
    output = args.video_list

    if not search_dir.exists():
        print(f"Search results directory not found: {search_dir}")
        print("Run search_speakers.py first to generate per-speaker search results.")
        return

    # Load existing entries keyed by (speaker, id)
    existing: dict[tuple[str, str], dict] = {}
    if os.path.exists(output):
        for e in load_entries(output):
            existing[(e.get("speaker", ""), e.get("id", ""))] = e

    # Walk search_results/<speaker>/video_list.json
    speaker_dirs = sorted(d for d in search_dir.iterdir() if d.is_dir())
    if not speaker_dirs:
        print(f"No speaker directories found in {search_dir}")
        return

    updated = 0
    new_count = 0
    entries: list[dict] = []

    for sp_dir in speaker_dirs:
        json_path = sp_dir / "video_list.json"
        if not json_path.exists():
            continue
        speaker_name = sp_dir.name
        try:
            videos = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"  Skipping unreadable: {json_path}")
            continue
        for v in videos:
            vid = v.get("id", "")
            if not vid:
                continue
            key = (speaker_name, vid)
            if key in existing:
                entry = existing[key]
                # Refresh metadata fields from new search output
                for k in ("title", "upload_date", "duration", "channel"):
                    if v.get(k) and v[k] != entry.get(k):
                        entry[k] = v[k]
                # Populate description/language if missing from entry
                changed = False
                if not entry.get("description") and v.get("description"):
                    entry["description"] = v["description"]
                    changed = True
                if not entry.get("language") and v.get("language"):
                    entry["language"] = v["language"]
                    changed = True
                if changed:
                    updated += 1
                entries.append(entry)
            else:
                entry: dict = {
                    "id": vid,
                    "speaker": speaker_name,
                    "title": v.get("title", ""),
                    "upload_date": v.get("upload_date", ""),
                    "duration": v.get("duration", ""),
                    "channel": v.get("channel", ""),
                }
                if v.get("description"):
                    entry["description"] = v["description"]
                if v.get("language"):
                    entry["language"] = v["language"]
                new_count += 1
                entries.append(entry)

    if not entries:
        print("No entries found in search results.")
        return

    write_entries(output, entries)
    print(f"Consolidated {len(entries)} entries from {len(speaker_dirs)} speakers")
    if new_count:
        print(f"  New entries:  {new_count}")
    if updated:
        print(f"  Updated:      {updated}")
    missing_desc = sum(1 for e in entries if not e.get("description"))
    missing_lang = sum(1 for e in entries if not e.get("language"))
    if missing_desc:
        print(f"  Missing description: {missing_desc}")
        print("  → Run 'filter' phase to fetch descriptions via yt-dlp.")
    if missing_lang:
        print(f"  Missing language:    {missing_lang}")
    print(f"  Output:       {output}")


# ── Phase 1: Download missing VTTs ────────────────────────────────────────────

def download_vtt(args):
    """Download missing VTT subtitles and mark unavailable videos."""
    print("=" * 60)
    print("Phase 1: Downloading missing VTT subtitles")
    print("=" * 60)

    entries = load_entries(args.video_list)
    existing_vtt = get_existing_vtt_ids(args.vtt_dir)
    print(f"Total entries: {len(entries)}")
    print(f"Existing VTT files: {len(existing_vtt)}")

    # Find entries that need VTT download (no VTT, not excluded by status)
    needs_vtt = []
    already_excluded = 0
    for e in entries:
        vid = e.get("id", "")
        if not vid:
            continue
        s = e.get("status")
        if s in ("unavailable", "no_captions", "audio_failed", "filtered"):
            already_excluded += 1
            continue
        if vid not in existing_vtt:
            needs_vtt.append(e)

    print(f"Already excluded (status): {already_excluded}")
    print(f"Videos needing VTT download: {len(needs_vtt)}")

    if not needs_vtt:
        print("No VTT downloads needed. Skipping download phase.")
        return

    # Check yt-dlp availability
    if not shutil.which(YT_DLP):
        print(f"ERROR: yt-dlp not found at '{YT_DLP}'. Install it first.")
        sys.exit(1)

    # Download each missing VTT
    downloaded = 0
    marked_unavailable = 0
    marked_no_captions = 0
    failed_other = 0
    skipped_existing = 0
    total = len(needs_vtt)

    for idx, entry in enumerate(needs_vtt, 1):
        vid = entry["id"]

        # Double-check: VTT might have been created by a prior run in this session
        if os.path.exists(os.path.join(args.vtt_dir, f"{vid}.en.vtt")):
            skipped_existing += 1
            continue

        url = f"https://youtu.be/{vid}"
        print(f"\n[{idx}/{total}] {entry.get('speaker','?'):>20s}  {vid}  {entry.get('title','')[:50]}")

        try:
            result = subprocess.run(
                [
                    YT_DLP,
                    "--write-auto-sub",
                    "--sub-lang", "en",
                    "--sub-format", "vtt",
                    "--skip-download",
                    "--sleep-requests", str(args.sleep_requests),
                    "--extractor-args", "youtube:player_client=android",
                    "--no-warnings",
                    "-o", os.path.join(args.vtt_dir, "%(id)s"),
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=args.ytdlp_timeout,
            )
            stderr = result.stderr.lower()
            stdout = result.stdout.lower()

            # Check for "unavailable" signal in yt-dlp output
            if result.returncode != 0:
                if re.search(r"video unavailable|private video|age.restrict|removed|not available|playback restricted|community guidelines", stderr):
                    entry["status"] = "unavailable"
                    print(f"    ↳ MARKED UNAVAILABLE: {stderr.split('ERROR:')[-1].strip()[:80]}")
                    marked_unavailable += 1
                else:
                    print(f"    ↳ FAILED (other error): {stderr.split('ERROR:')[-1].strip()[:80]}")
                    failed_other += 1
            else:
                # Success check: did the VTT file actually get created?
                if os.path.exists(os.path.join(args.vtt_dir, f"{vid}.en.vtt")):
                    downloaded += 1
                    print(f"    ↳ VTT downloaded ✓")
                else:
                    # yt-dlp returned 0 but no VTT — subtitles are genuinely missing
                    entry["status"] = "no_captions"
                    print(f"    ↳ No captions available — marked as no_captions")
                    marked_no_captions += 1

        except subprocess.TimeoutExpired:
            print(f"    ↳ TIMEOUT (>{args.ytdlp_timeout}s)")
            failed_other += 1
        except FileNotFoundError:
            print(f"    ↳ yt-dlp not found!")
            sys.exit(1)

        # Be polite to YouTube
        if idx < total:
            time.sleep(args.sleep_interval)

    # Write back updated JSONL with any newly-marked unavailable entries
    write_entries(args.video_list, entries)

    # Summary
    print("\n" + "=" * 60)
    print("Phase 1 Summary:")
    print(f"  Total needing VTT:     {total}")
    print(f"  Downloaded:            {downloaded}")
    print(f"  Already existed:       {skipped_existing}")
    print(f"  Marked unavailable:    {marked_unavailable}")
    print(f"  No captions avail:     {marked_no_captions}")
    print(f"  Failed (other):        {failed_other}")
    print(f"  Updated:               {args.video_list}")
    print("=" * 60)


# ── Phase 2: Quality filter — flag non-speaker content ─────────────────────

# Exclusion rules (adapted from filter_videos.py)
_DESC_REPOST_DOC = re.compile(
    r'(?:mini.documentary|this\s+(?:new\s+)?documentary\s+(?:on|about|reveals))',
    re.IGNORECASE)
_DESC_ABOUT_PERSON = re.compile(
    r'\b(?:biography\s+of|book\s+(?:of|about)\s+the\b.*?(?:secretary|treasury|fed|chair))',
    re.IGNORECASE)
_DESC_NEWS_REPORT = re.compile(
    r'\b(?:lightning\s+rod\s+for\s+conspiracy\b)',
    re.IGNORECASE)
_DESC_DOC_TROPE = re.compile(
    r'\b(?:rags\s+to\s+riches|untold\s+story)\b', re.IGNORECASE)
_NON_ENGLISH_LANGS = {'it', 'fr', 'de', 'es', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi'}


def _check_video_filter(meta: dict) -> tuple[bool, str]:
    """Check a video's metadata for non-speaker content signals.

    Returns (excluded, reason). If excluded is False, reason is empty.
    """
    desc = (meta.get('description') or '').strip()

    if _DESC_REPOST_DOC.search(desc):
        return True, "repost_documentary"
    if _DESC_ABOUT_PERSON.search(desc):
        return True, "about_person_not_by"
    if _DESC_NEWS_REPORT.search(desc):
        return True, "news_report_about"
    if _DESC_DOC_TROPE.search(desc):
        return True, "doc_trope"

    lang = (meta.get('language') or '').lower()
    if lang and lang in _NON_ENGLISH_LANGS:
        return True, f"lang:{lang}"

    return False, ""


# ── Identity scoring (merged from check_video_identity.py) ─────────────────
# Scores each video by checking whether its title, channel, and transcript
# contain the speaker's name. Used to filter out reposted/non-speaker content
# that passes the description-based check above.

_INTERVIEW_TERMS = {
    "interview", "conversation", "talk", "speech", "lecture", "keynote",
    "podcast", "debate", "remarks", "address", "testimony", "hearing",
    "forum", "summit", "panel", "q&a", "qa", "fireside", "live", "full",
    "discusses", "explains", "on ",
}

_RISK_TERMS = {
    "documentary", "biography", "history of", "rise and fall", "story of",
    "explained", "profile", "tribute", "almost crashed", "who is",
    "why ", "how ", "news", "breaking", "analysis of", "reacts to",
    "exposed", "secret", "scandal", "compilation",
}

_GENERIC_NEWS_CHANNEL_TERMS = {
    "news", "pbs", "frontline", "cnn", "msnbc", "fox", "cnbc", "bbc",
    "bloomberg", "reuters", "ap archive", "associated press", "guardian",
}

_SURNAME_STOPWORDS = {
    "news", "power", "rice", "wood", "gross", "marks", "gates", "hill",
    "warren", "cotton", "johnson", "williams", "brooks", "rubio",
}

_INTRO_PATTERNS = [
    "our guest", "my guest", "joining us", "joined by", "welcome",
    "pleased to welcome", "today we have", "with us",
    "in conversation with", "keynote", "speaker is", "this is",
]

_SELF_INTRO_PATTERNS = [
    "i am", "i'm", "my name is", "this is",
]

# Slugs whose display name can't be derived by simple title-case.
_DISPLAY_NAME_OVERRIDES = {
    "mohamed_el_erian": "Mohamed El-Erian",
    "mohamed_el-erian": "Mohamed El-Erian",
    "volodymyr_zelenskyy": "Volodymyr Zelenskyy",
    "recep_erdogan": "Recep Tayyip Erdogan",
    "lawrence_summers": "Lawrence Summers",
    "robert_shiller": "Robert Shiller",
    "marc_andreessen": "Marc Andreessen",
    "neel_kashkari": "Neel Kashkari",
    "james_mattis": "James Mattis",
    "jd_vance": "JD Vance",
    "mohamed_a_el_erian": "Mohamed A. El-Erian",
}

# Name variants not derivable from slug or display name.
_ALIAS_OVERRIDES = {
    "jd_vance": {"j d vance", "j.d. vance", "j.d vance", "jd vance"},
    "mohamed_el_erian": {"mohamed el erian", "mohamed el-erian"},
    "mohamed_el-erian": {"mohamed el erian", "mohamed el-erian"},
    "volodymyr_zelenskyy": {"volodymyr zelensky", "zelensky", "zelenskyy"},
    "recep_erdogan": {"recep tayyip erdogan", "erdogan"},
    "robert_shiller": {"bob shiller", "robert j shiller"},
    "lawrence_summers": {"larry summers", "lawrence summers"},
    "james_mattis": {"jim mattis", "james mattis"},
    "neel_kashkari": {"neel kashkari", "neal kashkari"},
    "marc_andreessen": {"mark andreessen", "marc andreessen"},
}


def _identity_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "'")
    text = re.sub(r"[^a-z0-9&'+ -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _identity_compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _identity_display_name(slug: str) -> str:
    """Convert a speaker slug (e.g. 'ray_dalio') to a display name."""
    if slug in _DISPLAY_NAME_OVERRIDES:
        return _DISPLAY_NAME_OVERRIDES[slug]
    return slug.replace("_", " ").title()


def _identity_name_aliases(display_name: str, slug: str) -> set[str]:
    """Build a set of search aliases for a speaker's name."""
    aliases = {
        _identity_normalize(display_name),
        _identity_normalize(slug.replace("_", " ")),
    }
    parts = [p for p in re.split(r"\s+", _identity_normalize(display_name)) if p]
    if len(parts) >= 2:
        aliases.add(f"{parts[0]} {parts[-1]}")
        last = parts[-1]
        if len(last) >= 5 and last not in _SURNAME_STOPWORDS:
            aliases.add(last)
        initials = "".join(p[0] for p in parts if p)
        if len(initials) >= 2:
            aliases.add(initials)

    aliases.update(_identity_normalize(a) for a in _ALIAS_OVERRIDES.get(slug, set()))
    return {a for a in aliases if a}


def _identity_contains_alias(text: str, aliases: set[str]) -> tuple[bool, str]:
    """Check if any alias appears as a whole word in text."""
    padded = f" {text} "
    for alias in sorted(aliases, key=len, reverse=True):
        if len(alias) <= 2:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return True, alias
            continue
        if f" {alias} " in padded:
            return True, alias
    return False, ""


def _identity_has_all_name_tokens(text: str, display_name: str) -> bool:
    """Check if all significant name tokens appear in text (not necessarily as a phrase)."""
    tokens = [
        t for t in _identity_normalize(display_name).split()
        if len(t) > 1 and t not in {"von", "der", "de", "of", "the"}
    ]
    if len(tokens) < 2:
        return False
    return all(re.search(rf"\b{re.escape(t)}\b", text) for t in tokens)


def _identity_read_vtt_text(vtt_path: str, max_chars: int = 200_000) -> str:
    """Read and clean VTT subtitle text."""
    raw = Path(vtt_path).read_text(encoding="utf-8", errors="ignore")[:max_chars]
    lines: list[str] = []
    previous = ""
    for line in raw.splitlines():
        line = html.unescape(line.strip())
        if not line:
            continue
        if line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line or re.match(r"^\d{2}:\d{2}:\d{2}\.", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = line.replace(">>", " ")
        line = _identity_compact_text(line)
        if not line or line == previous:
            continue
        previous = line
        lines.append(line)
    return _identity_compact_text(" ".join(lines))


def _identity_excerpt_around(text: str, needle: str, radius: int = 130) -> str:
    if not text or not needle:
        return ""
    idx = text.find(needle)
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    return _identity_compact_text(text[start:end])


def _identity_find_intro_evidence(norm_text: str, aliases: set[str]) -> tuple[bool, str]:
    """Check transcript beginning for host intro patterns indicating the speaker appeared."""
    search_window = norm_text[:8000]
    for alias in sorted(aliases, key=len, reverse=True):
        if len(alias) < 4:
            continue
        for pattern in _INTRO_PATTERNS:
            if re.search(rf"\b{re.escape(pattern)}\b.{{0,120}}\b{re.escape(alias)}\b", search_window):
                return True, f"{pattern} ... {alias}"
            if re.search(rf"\b{re.escape(alias)}\b.{{0,80}}\b{re.escape(pattern)}\b", search_window):
                return True, f"{alias} ... {pattern}"
    return False, ""


def _identity_find_self_intro_evidence(norm_text: str, aliases: set[str]) -> tuple[bool, str]:
    """Check transcript beginning for self-introduction patterns."""
    search_window = norm_text[:8000]
    for alias in sorted(aliases, key=len, reverse=True):
        if len(alias) < 4:
            continue
        for pattern in _SELF_INTRO_PATTERNS:
            if re.search(rf"\b{re.escape(pattern)}\b.{{0,40}}\b{re.escape(alias)}\b", search_window):
                return True, f"{pattern} ... {alias}"
    return False, ""


def _identity_transcript_evidence(vtt_dir: str, video_id: str, display_name: str, aliases: set[str]) -> dict:
    """Score whether the video's transcript confirms the speaker is the subject."""
    vtt_path = os.path.join(vtt_dir, f"{video_id}.en.vtt")
    if not os.path.exists(vtt_path):
        return {
            "transcript_path": "",
            "transcript_score": 0,
            "transcript_alias_count": 0,
            "transcript_intro": "",
            "transcript_self_intro": "",
            "transcript_excerpt": "",
        }

    text = _identity_read_vtt_text(vtt_path)
    norm_text = _identity_normalize(text)
    first_window = norm_text[:8000]

    alias_count = 0
    best_alias = ""
    for alias in sorted(aliases, key=len, reverse=True):
        if len(alias) < 4:
            continue
        hits = len(re.findall(rf"\b{re.escape(alias)}\b", norm_text))
        if hits > alias_count:
            alias_count = hits
            best_alias = alias

    intro_hit, intro_reason = _identity_find_intro_evidence(norm_text, aliases)
    self_intro_hit, self_intro_reason = _identity_find_self_intro_evidence(norm_text, aliases)
    early_full_name = _identity_has_all_name_tokens(first_window, display_name)

    score = 0
    if intro_hit:
        score += 4
    if self_intro_hit:
        score += 4
    if early_full_name:
        score += 3
    if alias_count >= 3:
        score += 3
    elif alias_count >= 1:
        score += 1

    excerpt = ""
    if best_alias:
        excerpt = _identity_excerpt_around(norm_text, best_alias)

    # Derive a display-friendly relative path
    rel_path = vtt_path
    if vtt_path.startswith(DATA_BASE):
        rel_path = vtt_path[len(DATA_BASE.rstrip("/")) + 1:]

    return {
        "transcript_path": rel_path,
        "transcript_score": score,
        "transcript_alias_count": alias_count,
        "transcript_intro": intro_reason,
        "transcript_self_intro": self_intro_reason,
        "transcript_excerpt": excerpt,
    }


def _identity_score_video(slug: str, video: dict, vtt_dir: str | None = None) -> dict:
    """Score a video's metadata + transcript for speaker-identity evidence.

    Returns a dict with identity_score, identity_status, identity_reasons,
    and identity_transcript_* fields for storage on the JSONL entry.
    """
    display_name = _identity_display_name(slug)
    title = video.get("title", "") or ""
    channel = video.get("channel", "") or ""
    combined = _identity_normalize(f"{title} {channel}")
    title_norm = _identity_normalize(title)
    channel_norm = _identity_normalize(channel)
    aliases = _identity_name_aliases(display_name, slug)

    title_hit, title_alias = _identity_contains_alias(title_norm, aliases)
    channel_hit, channel_alias = _identity_contains_alias(channel_norm, aliases)
    all_tokens_hit = _identity_has_all_name_tokens(combined, display_name)
    interview_hit = any(term in title_norm for term in _INTERVIEW_TERMS)
    risk_hits = sorted(term for term in _RISK_TERMS if term in title_norm)
    generic_channel = any(term in channel_norm for term in _GENERIC_NEWS_CHANNEL_TERMS)

    metadata_score = 0
    reasons: list[str] = []

    if title_hit:
        metadata_score += 4
        reasons.append(f"title_name:{title_alias}")
    if channel_hit:
        metadata_score += 3
        reasons.append(f"channel_name:{channel_alias}")
    if all_tokens_hit and not title_hit:
        metadata_score += 3
        reasons.append("all_name_tokens")
    if interview_hit:
        metadata_score += 1
        reasons.append("interview_or_speech_term")
    if generic_channel and not title_hit:
        metadata_score -= 1
        reasons.append("generic_news_channel_without_title_name")
    if risk_hits:
        metadata_score -= min(3, len(risk_hits))
        reasons.append("risk_terms:" + "|".join(risk_hits[:4]))
    if not (title_hit or channel_hit or all_tokens_hit):
        metadata_score -= 4
        reasons.append("no_name_match")

    evidence = {}
    transcript_score = 0
    if vtt_dir is not None:
        evidence = _identity_transcript_evidence(vtt_dir, video.get("id", ""), display_name, aliases)
        transcript_score = evidence["transcript_score"]
        if transcript_score:
            reasons.append(f"transcript_evidence:{transcript_score}")
        elif evidence.get("transcript_path") and metadata_score <= 0:
            reasons.append("transcript_no_name_evidence")

    score = metadata_score + transcript_score

    if score >= 4:
        status = "likely_match"
    elif score >= 1:
        status = "review"
    else:
        status = "high_risk"

    result = {
        "identity_score": score,
        "identity_status": status,
        "identity_reasons": "; ".join(reasons),
        "identity_transcript_path": evidence.get("transcript_path", ""),
        "identity_transcript_score": evidence.get("transcript_score", 0),
        "identity_transcript_alias_count": evidence.get("transcript_alias_count", 0),
        "identity_transcript_intro": evidence.get("transcript_intro", ""),
        "identity_transcript_self_intro": evidence.get("transcript_self_intro", ""),
        "identity_transcript_excerpt": evidence.get("transcript_excerpt", ""),
    }
    return result


def filter_videos(args):
    """Enrich video metadata and flag non-target-speaker content.

    For each video in video_list.jsonl that has a VTT file, fetches
    description/language via yt-dlp --dump-json and applies exclusion
    rules. Also scores each video's title, channel, and transcript for
    speaker-name evidence (identity scoring). Flagged entries get
    "status": "filtered" + "filter_reason". Identity scores are stored
    on the entry for downstream analysis."""

    print("=" * 60)
    print("Phase 2: Filtering non-speaker video content")
    print("=" * 60)

    entries = load_entries(args.video_list)
    existing_vtt = get_existing_vtt_ids(args.vtt_dir)
    print(f"Total entries: {len(entries)}")
    print(f"Existing VTT files: {len(existing_vtt)}")

    # Candidates: not already excluded (VTT optional — no-VTT entries
    # are still filtered via description + title/channel identity scoring)
    candidates = []
    candidates_no_vtt = []
    for e in entries:
        vid = e.get("id", "")
        s = e.get("status", "")
        if not vid:
            continue
        if s in ("unavailable", "no_captions", "filtered"):
            continue
        if vid in existing_vtt:
            candidates.append(e)
        else:
            candidates_no_vtt.append(e)

    # no-VTT candidates are also eligible for filtering
    candidates.extend(candidates_no_vtt)

    if not candidates:
        print("No candidates to filter.")
        return

    print(f"Candidates to check: {len(candidates)}")

    # Load metadata cache
    cached = {}
    if os.path.exists(args.meta_cache):
        with open(args.meta_cache) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    cached[d.get('id', '')] = d
    print(f"Cached metadata: {len(cached)}")

    # Candidates that lack description/language on their entry (need yt-dlp fetch)
    need_meta = [e for e in candidates
                 if not e.get("description") or not e.get("language")]

    # Determine which need actual yt-dlp fetch
    if args.meta_fetch_all:
        to_fetch = candidates
    else:
        to_fetch = [e for e in need_meta if e["id"] not in cached]

    if to_fetch:
        print(f"Fetching metadata for {len(to_fetch)} videos via yt-dlp...")
        cache_fh = open(args.meta_cache, "a")
        done = 0
        for entry in to_fetch:
            vid = entry["id"]
            done += 1
            if done > 1:
                time.sleep(args.sleep_interval)

            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                result = subprocess.run(
                    [YT_DLP, "--dump-json", "--no-download", "--skip-download",
                     "--extractor-args", "youtube:player_client=android",
                     "--sleep-requests", str(args.sleep_requests),
                     url],
                    capture_output=True, text=True, timeout=args.ytdlp_timeout,
                )
                if result.returncode == 0:
                    d = json.loads(result.stdout)
                    d["_speaker"] = entry.get("speaker", "?")
                    cached[vid] = d
                    cache_fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                    # Populate description/language on the entry for future runs
                    if not entry.get("description") and d.get("description"):
                        entry["description"] = (d.get("description") or "")[:1000]
                    if not entry.get("language") and d.get("language"):
                        entry["language"] = d.get("language", "")
                else:
                    stderr = result.stderr.lower()
                    if re.search(r"video unavailable|private video|age\.restrict|removed|not available", stderr):
                        entry["status"] = "unavailable"
                        print(f"  [{done}/{len(to_fetch)}] {vid} → unavailable")
                    else:
                        print(f"  [{done}/{len(to_fetch)}] {vid} → fetch error")
            except subprocess.TimeoutExpired:
                print(f"  [{done}/{len(to_fetch)}] {vid} → timeout")
            except json.JSONDecodeError:
                print(f"  [{done}/{len(to_fetch)}] {vid} → bad JSON")
            except Exception as e:
                print(f"  [{done}/{len(to_fetch)}] {vid} → {e}")

            if done % 50 == 0 and done < len(to_fetch):
                cache_fh.flush()

        cache_fh.close()

    # Run filter rules + identity scoring on candidates
    flagged = 0
    passed = 0
    identity_flagged = 0
    identity_scores: dict[str, list[int]] = defaultdict(list)
    for e in candidates:
        # Identity scoring (from title/channel in entry + VTT transcript).
        # This doesn't need yt-dlp metadata, so it runs for all candidates.
        identity_result = _identity_score_video(
            e.get("speaker", "unknown"), e, args.vtt_dir
        )
        e["identity_score"] = identity_result["identity_score"]
        e["identity_status"] = identity_result["identity_status"]
        e["identity_reasons"] = identity_result["identity_reasons"]
        # Store transcript evidence fields
        for fld in ("identity_transcript_path", "identity_transcript_score",
                     "identity_transcript_alias_count", "identity_transcript_intro",
                     "identity_transcript_self_intro", "identity_transcript_excerpt"):
            if fld in identity_result:
                e[fld] = identity_result[fld]
        identity_scores[e.get("speaker", "unknown")].append(identity_result["identity_score"])

        # Description-based filter: prefer entry-level description, fallback to cache
        if e.get("description") and e.get("language"):
            # Can filter directly from entry fields — no yt-dlp needed
            excluded, reason = _check_video_filter(e)
        else:
            meta = cached.get(e["id"])
            if not meta:
                passed += 1
                continue
            if e.get("status") in ("unavailable",):
                continue
            excluded, reason = _check_video_filter(meta)
            # Populate entry from cache for future runs
            if not e.get("description") and meta.get("description"):
                e["description"] = (meta.get("description") or "")[:1000]
            if not e.get("language") and meta.get("language"):
                e["language"] = meta.get("language", "")

        # Auto-filter videos with no speaker-name evidence (score ≤ 0)
        if not excluded and identity_result["identity_status"] == "high_risk":
            excluded = True
            reason = "low_identity_score"

        if excluded:
            e["status"] = "filtered"
            e["filter_reason"] = reason
            flagged += 1
            if reason == "low_identity_score":
                identity_flagged += 1
        else:
            passed += 1

    # Write back updated JSONL
    write_entries(args.video_list, entries)

    # Summary
    print(f"\n{'=' * 60}")
    print("Phase 2 Summary:")
    print(f"  Checked:             {len(candidates)}  (with VTT: {len(candidates) - len(candidates_no_vtt)}, without VTT: {len(candidates_no_vtt)})")
    print(f"  Passed:              {passed}")
    print(f"  Flagged:             {flagged}")
    if identity_flagged:
        print(f"    via identity:      {identity_flagged}")
    print(f"  Identity scoring:")
    print(f"    likely_match:      {sum(1 for e in candidates if e.get('identity_status') == 'likely_match')}")
    print(f"    review:            {sum(1 for e in candidates if e.get('identity_status') == 'review')}")
    print(f"    high_risk:         {sum(1 for e in candidates if e.get('identity_status') == 'high_risk')}")
    print(f"  Updated:             {args.video_list}")

    if flagged:
        print(f"\nFlagged videos:")
        for e in entries:
            if e.get("status") == "filtered":
                id_label = f" id:{e.get('identity_score','?')}/{e.get('identity_status','?')}" if "identity_score" in e else ""
                print(f"  {e['id']:15s} {e.get('speaker','?'):>20s}  [{e.get('filter_reason','')}]{id_label}  {e.get('title','')[:50]}")

    # Speakers with lowest average identity score
    low_confidence = sorted(
        identity_scores.items(),
        key=lambda x: sum(x[1]) / len(x[1]),
    )[:5]
    if low_confidence:
        print(f"\nLowest avg identity scores by speaker:")
        for spk, scores in low_confidence:
            avg = sum(scores) / len(scores)
            print(f"  {spk:>25s}  avg={avg:.1f}  n={len(scores)}")

    print("=" * 60)


# ── Phase 3: Select video IDs for audio download ─────────────────────────────

def select_for_audio(args):
    """Select video IDs for audio download, even without VTT subtitles.

    From entries with no audio file and no exclusion status (unavailable /
    filtered / etc.), select up to N IDs sampling evenly across speakers.
    VTT is not required — audio can be downloaded independently.
    Output: JSONL with full entry data."""
    print("=" * 60)
    print("Phase 2: Selecting video IDs for audio download")
    print("=" * 60)

    entries = load_entries(args.video_list)
    existing_audio = get_existing_audio_ids(args.audio_dir)

    print(f"Total entries:               {len(entries)}")
    print(f"Existing audio files:        {len(existing_audio)}")

    # Group entries by speaker, filtering to eligible candidates
    # Eligible: no audio file AND not marked unavailable/filtered
    # (VTT not required — audio can be downloaded without subtitles)
    per_speaker: dict[str, list[str]] = defaultdict(list)
    excluded_has_audio = 0
    excluded_unavailable = 0
    excluded_no_captions = 0

    for e in entries:
        vid = e.get("id", "")
        speaker = e.get("speaker", "unknown")
        if not vid:
            continue

        if e.get("status") == "unavailable":
            excluded_unavailable += 1
            continue

        if e.get("status") == "no_captions":
            excluded_no_captions += 1
            continue

        if e.get("status") == "audio_failed":
            continue

        if e.get("status") == "filtered":
            continue

        if vid in existing_audio:
            excluded_has_audio += 1
            continue

        per_speaker[speaker].append(vid)

    print(f"Available for selection:     {sum(len(v) for v in per_speaker.values())}")
    print(f"  Excluded (has audio):      {excluded_has_audio}")
    print(f"  Excluded (unavailable):    {excluded_unavailable}")
    print(f"  Excluded (no captions):    {excluded_no_captions}")

    if not per_speaker:
        print("No candidates available for audio download.")
        return []

    # Build a lookup from video_id to entry
    entries_by_id: dict[str, dict] = {}
    for e in entries:
        vid = e.get("id", "")
        if vid:
            entries_by_id[vid] = e

    # Sample evenly across speakers (or select all with --all)
    rng = __import__("random").Random(args.seed)
    speaker_list = sorted(per_speaker.keys())
    selected_ids: list[str] = []

    if args.all:
        for sp in speaker_list:
            ids = per_speaker[sp]
            selected_ids.extend(ids)
        print(f"  --all mode: selecting all {len(selected_ids)} eligible videos")
    elif args.per_speaker > 0:
        for sp in speaker_list:
            ids = per_speaker[sp]
            rng.shuffle(ids)
            selected_ids.extend(ids[:args.per_speaker])
    else:
        for sp in speaker_list:
            ids = per_speaker[sp]
            rng.shuffle(ids)
            selected_ids.extend(ids)

    # Shuffle interleaved results so order isn't grouped by speaker
    rng.shuffle(selected_ids)

    selected_entries = [entries_by_id[vid] for vid in selected_ids if vid in entries_by_id]
    speaker_count = len(set(e.get("speaker", "unknown") for e in selected_entries))

    # Write output as JSONL (like video_list.jsonl)
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for e in selected_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nSelected {len(selected_entries)} videos from {speaker_count} speakers:")
    for e in selected_entries[:15]:
        print(f"  {e.get('speaker', 'unknown'):>25s}  {e.get('id', '')}")
    if len(selected_entries) > 15:
        print(f"  ... and {len(selected_entries) - 15} more")
    print(f"\nSaved to {output_path}")

    return selected_entries


# ── Phase 3: Download audio for selected videos ──────────────────────────────

def download_audio(args):
    """Download audio for video IDs from the selection list using yt-dlp (parallel)."""
    print("=" * 60)
    print("Phase 3: Downloading audio files")
    print("=" * 60)

    # Load entries for status lookup
    entries_map: dict[str, dict] = {}
    if os.path.exists(args.video_list):
        for e in load_entries(args.video_list):
            entries_map[e.get("id", "")] = e

    # Read video IDs to download — JSONL format (like video_list.jsonl)
    tasks: list[str] = []
    input_file = args.audio_input or args.output
    if not os.path.exists(input_file):
        print(f"ERROR: input file not found: {input_file}")
        print("Run 'select' mode first, or pass --audio-input <video_list.jsonl>")
        return

    for e in load_entries(input_file):
        vid = e.get("id", "")
        if vid:
            tasks.append(vid)
    print(f"Loaded {len(tasks)} video IDs from {input_file}")

    # Dedup
    tasks = list(set(tasks))
    print(f"Videos to download: {len(tasks)} (workers={args.audio_workers})")

    if not tasks:
        print("Nothing to download.")
        return

    os.makedirs(args.audio_dir, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ydl_base = [
        YT_DLP,
        "--retries", str(args.audio_retries),
        "--fragment-retries", str(args.audio_retries),
        "--sleep-requests", str(args.sleep_requests),
        "--extractor-args", "youtube:player_client=android",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "m4a",
        "-o", "%(id)s.%(ext)s",
    ]

    def download_one(video_id: str) -> tuple[str, bool, str]:
        """Download audio for one video; returns (video_id, ok, message)."""
        out_path = os.path.join(args.audio_dir, f"{video_id}.m4a")
        if os.path.exists(out_path):
            return (video_id, True, "already exists")

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            result = subprocess.run(
                ydl_base + [url],
                capture_output=True, text=True, timeout=args.audio_timeout,
                cwd=args.audio_dir,
            )
            if result.returncode == 0 and os.path.exists(out_path):
                return (video_id, True, "ok")

            stderr = result.stderr.lower()
            # Check for unavailable / age-restricted
            if re.search(r"video unavailable|private video|age.restrict|removed|not available|playback restricted|community guidelines", stderr):
                # Mark in JSONL
                entry = entries_map.get(video_id)
                if entry and not entry.get("status"):
                    entry["status"] = "unavailable"
                    write_entries(args.video_list, list(entries_map.values()))
                return (video_id, False, "unavailable")

            err = result.stderr.strip()[-200:] if result.stderr else "unknown error"
            return (video_id, False, err)
        except subprocess.TimeoutExpired:
            return (video_id, False, "timeout")
        except Exception as e:
            return (video_id, False, str(e)[-200:])

    start = time.time()
    ok = fail = 0
    unavailable_count = 0
    with ThreadPoolExecutor(max_workers=args.audio_workers) as ex:
        fut_to_vid = {ex.submit(download_one, vid): vid for vid in tasks}
        done_total = len(fut_to_vid)
        done = 0
        for fut in as_completed(fut_to_vid):
            vid, is_ok, msg = fut.result()
            done += 1
            if is_ok:
                ok += 1
            else:
                fail += 1
                if msg == "unavailable":
                    unavailable_count += 1
            elapsed = time.time() - start
            mark = "✓" if is_ok else "✗"
            print(f"  [{done}/{done_total}] {elapsed:.0f}s {mark} {vid} | {msg[:80]}")
            time.sleep(10)

    elapsed = time.time() - start
    print(f"\nDone: {ok} OK, {fail} FAIL ({unavailable_count} unavailable) in {elapsed:.0f}s")
    print(f"Output: {args.audio_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: consolidate → download VTT → filter → select → download audio."
    )
    parser.add_argument(
        "mode",
        choices=["consolidate", "download-vtt", "filter", "select", "download-audio", "full"],
        help=(
            "consolidate:  Build video_list.jsonl from search_results/<speaker>/video_list.json. "
            "download-vtt: Download missing VTT subtitles, mark unavailable/no_captions. "
            "filter:       Check identity & description; flag non-speaker content. "
            "select:       Select video IDs for audio download (balanced by speaker). "
            "download-audio: Download audio for the selected IDs (parallel). "
            "Use --audio-input to read from a JSONL video list directly."
            "full:         download-vtt + filter + select + download-audio."
        ),
    )

    # Shared options
    parser.add_argument("--video-list", default=VIDEO_LIST, help="Path to video_list.jsonl")
    parser.add_argument("--vtt-dir", default=VTT_DIR, help="Directory for VTT files")
    parser.add_argument("--audio-dir", default=AUDIO_DIR, help="Directory for audio files")
    parser.add_argument("--meta-cache", default=META_CACHE, help="yt-dlp metadata cache file (default: video_meta.jsonl)")

    # Step 0 options (consolidate)
    parser.add_argument("--search-dir", default="./search_results", help="Search results directory (default: ./search_results)")

    # Phase 1 / 2 options
    parser.add_argument("--sleep-requests", type=float, default=1.0, help="Seconds between yt-dlp requests (default: 1.0)")
    parser.add_argument("--sleep-interval", type=float, default=1.0, help="Seconds between videos (default: 1.0)")
    parser.add_argument("--ytdlp-timeout", type=int, default=120, help="yt-dlp timeout per video in seconds (default: 120)")

    # Phase 2 options
    parser.add_argument("--meta-fetch-all", action="store_true", help="Re-fetch all metadata (ignore cache)")

    # Phase 3 options
    parser.add_argument("--per-speaker", type=int, default=4, help="Max per speaker (default: 4; 0 = no limit)")
    parser.add_argument("--all", action="store_true", help="Select ALL eligible video IDs (ignores --per-speaker)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", default=os.path.join(DATA_BASE, "download_audio.jsonl"), help="Output file path (default: $(DATA_BASE)/download_audio.jsonl)")

    # Phase 4 options
    parser.add_argument("--audio-input", default="", help="Input JSONL video list (defaults to --output from select)")
    parser.add_argument("--audio-workers", type=int, default=3, help="Parallel download workers (default: 3)")
    parser.add_argument("--audio-retries", type=int, default=10, help="yt-dlp retry count (default: 10)")
    parser.add_argument("--audio-timeout", type=int, default=7200, help="Timeout per audio download in seconds (default: 7200 = 2h)")

    args = parser.parse_args()

    if args.mode == "consolidate":
        consolidate(args)
    elif args.mode == "download-vtt":
        download_vtt(args)
    elif args.mode == "filter":
        filter_videos(args)
    elif args.mode == "select":
        select_for_audio(args)
    elif args.mode == "download-audio":
        download_audio(args)
    elif args.mode == "full":
        download_vtt(args)
        filter_videos(args)
        select_for_audio(args)
        download_audio(args)


if __name__ == "__main__":
    main()
