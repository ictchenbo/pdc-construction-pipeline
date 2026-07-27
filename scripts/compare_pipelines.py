#!/usr/bin/env python3
"""
Compare AssemblyAI vs local (Whisper+pyannote) diarization pipelines.

Usage:
    python compare_pipelines.py                                # summary report
    python compare_pipelines.py --assemblyai-dir PATH          # override AssemblyAI dir
    python compare_pipelines.py --local-dir PATH               # override local pipeline dir

AssemblyAI TXT files are plain text (no speaker labels, no timestamps) --
already filtered to guest-only speech.  The comparison therefore focuses on:
  - Cost (API vs local GPU)
  - Throughput
  - Transcript word counts / coverage
  - Text overlap when local pipeline results become available
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "data"  # ../data/

DEFAULT_ASSEMBLYAI_DIR = DATA_ROOT / "diarized_assemblyai"
DEFAULT_LOCAL_DIR = DATA_ROOT / "diarized_whisper_pyannote"


def parse_assemblyai_txt(text: str) -> list[dict]:
    """Parse AssemblyAI TXT output.  Files are plain text (already filtered to
    guest-only speech).  Paragraph breaks delimit segments."""
    segments = []
    for para in text.strip().split("\n\n"):
        para = para.strip()
        if not para:
            continue
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if lines:
            segments.append({"speaker": "GUEST", "text": " ".join(lines)})
    if not segments and text.strip():
        segments.append({"speaker": "GUEST", "text": text.strip()})
    return segments


def load_all_assemblyai(assemblyai_dir: Path) -> dict[str, list[dict]]:
    """Load all AssemblyAI TXT files across all speakers."""
    results = {}
    for speaker_dir in sorted(assemblyai_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for txt_path in sorted(speaker_dir.glob("*.txt")):
            video_id = txt_path.stem
            text = txt_path.read_text(encoding="utf-8")
            results[video_id] = parse_assemblyai_txt(text)
    return results


def file_stats(segments: list[dict]) -> dict:
    """Guest-only word count summary."""
    if not segments:
        return {"n_segments": 0, "total_words": 0, "guest_words": 0}
    total = sum(len(s["text"].split()) for s in segments)
    return {"n_segments": len(segments), "total_words": total, "guest_words": total}


def cost_estimate(total_minutes: float) -> dict:
    """Cost comparison."""
    aa_rate = 0.015           # $/min AssemblyAI Best tier
    gpu_wattage = 350          # W (RTX 4090D TDP)
    elect_rate = 0.24          # $/kWh
    local_speedup = 60.0 / 40 * 60  # 40 sec per 40 min → ~60x realtime

    aa_cost = total_minutes * aa_rate
    local_hours = (total_minutes * 60) / local_speedup / 3600
    local_kwh = local_hours * gpu_wattage / 1000
    local_cost = local_kwh * elect_rate

    return {
        "total_minutes": round(total_minutes),
        "assemblyai_cost": round(aa_cost, 2),
        "local_cost": round(local_cost, 2),
        "local_hours": round(local_hours, 2),
        "cost_ratio": round(aa_cost / local_cost, 1) if local_cost > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare diarization pipelines")
    parser.add_argument("--assemblyai-dir", default=str(DEFAULT_ASSEMBLYAI_DIR))
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR))
    args = parser.parse_args()

    aa_dir = Path(args.assemblyai_dir)

    # ── Load ──────────────────────────────────────────────
    print(f"Loading AssemblyAI files from {aa_dir} ...")
    aa_all = load_all_assemblyai(aa_dir)
    print(f"  {len(aa_all)} files loaded")

    # ── Per-file stats ────────────────────────────────────
    stats = {vid: file_stats(segs) for vid, segs in aa_all.items()}

    total_words = sum(s["total_words"] for s in stats.values())
    word_counts = [s["total_words"] for s in stats.values()]

    print(f"\n{'=' * 60}")
    print(f"  AssemblyAI Diarization Summary ({len(aa_all)} files)")
    print(f"{'=' * 60}")
    print(f"  Total words:        {total_words:,}")
    print(f"  Words/file:          mean={statistics.mean(word_counts):.0f}"
          f"  median={statistics.median(word_counts):.0f}")
    print(f"  Total speakers:      {len({s.name for s in aa_dir.iterdir() if s.is_dir()})}")

    # ── Cost ──────────────────────────────────────────────
    avg_duration_min = total_words / 150 / len(aa_all) if aa_all else 30  # ~150 wpm
    total_min = avg_duration_min * len(aa_all)
    cost = cost_estimate(total_min)

    print(f"\n{'=' * 60}")
    print(f"  Cost Estimate ({len(aa_all)} files, ~{avg_duration_min:.0f} min avg)")
    print(f"{'=' * 60}")
    print(f"  AssemblyAI (Best):  ${cost['assemblyai_cost']:.2f}  @ $0.015/min")
    print(f"  Local (GPU 4090D):  ${cost['local_cost']:.2f}  ({cost['local_hours']:.1f}h compute)")
    print(f"  Ratio:              {cost['cost_ratio']:.0f}x cheaper (local pipeline)")


if __name__ == "__main__":
    main()
