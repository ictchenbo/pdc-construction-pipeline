#!/usr/bin/env python3
"""
gpu_worker.py — GPU-side batch processor for ASR pipeline.

Deployed and triggered by pipeline_manager.py (local orchestrator) via SSH.
Processes a list of files: convert m4a→wav → transcribe+diarize → write output.

Usage (on GPU server):
    python3 gpu_worker.py --files abc123,def456,ghi789
    python3 gpu_worker.py --filelist /root/_batch.txt
    python3 gpu_worker.py --status-only   # report current progress only

Each file goes through:
    /root/input/{id}.m4a  ──ffmpeg──→  /root/autodl-tmp/{id}.wav  ──whisper+pyannote──→  /root/output_large-v3/{id}.json + .txt

Worker writes progress markers:
    /root/_worker_state.json   — JSON file tracking which files succeeded/failed
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Ensure PyTorch uses its bundled cuDNN (avoid version mismatch with system cuDNN)
_cudnn_path = "/root/miniconda3/lib/python3.8/site-packages/torch/lib"
if os.path.exists(_cudnn_path):
    os.environ.setdefault("LD_LIBRARY_PATH", "")
    if _cudnn_path not in os.environ["LD_LIBRARY_PATH"]:
        os.environ["LD_LIBRARY_PATH"] = f"{_cudnn_path}:{os.environ['LD_LIBRARY_PATH']}"

import numpy as np
import torchaudio
torchaudio.set_audio_backend("sox_io")
from faster_whisper import WhisperModel, BatchedInferencePipeline
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation

# ── Config (overridable via env) ──
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INPUT_DIR = os.environ.get("GPU_INPUT_DIR", "/root/input")
WORK_DIR = os.environ.get("GPU_WORK_DIR", "/root/autodl-tmp")
OUTPUT_DIR = os.environ.get("GPU_OUTPUT_DIR", "/root/output_large-v3")
WORKER_STATE = os.environ.get("GPU_WORKER_STATE", "/root/_worker_state.json")
MODEL_SIZE = "large-v3"
BEAM_SIZE = 1
BATCH_SIZE = 16
LANGUAGE = "en"
COMPUTE_TYPE = "float16"
VAD_FILTER = True
CHUNK_DURATION = 1800      # 30min chunks for long audio diarization
SKIP_DIAR_SHORT = 10       # short audio skips diarization

# Lazy-loaded models (shared across files in this batch)
_whisper_model = None
_diar_pipeline = None


# ═══════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("[worker] Loading Whisper model...", flush=True)
        base = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE_TYPE)
        _whisper_model = BatchedInferencePipeline(model=base)
        print("[worker] Whisper loaded", flush=True)
    return _whisper_model


def get_diar():
    global _diar_pipeline
    if _diar_pipeline is None:
        print("[worker] Loading diarization pipeline...", flush=True)
        import torch
        p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        if torch.cuda.is_available():
            for _, inf in p._inferences.items():
                if hasattr(inf, "to"):
                    inf.to(torch.device("cuda"))
        _diar_pipeline = p
        print("[worker] Diarization pipeline loaded", flush=True)
    return _diar_pipeline


# ═══════════════════════════════════════════
#  Core processing
# ═══════════════════════════════════════════

def convert_m4a_to_wav(file_id: str) -> bool:
    """Convert m4a → wav via ffmpeg. Returns success."""
    src = Path(INPUT_DIR) / f"{file_id}.m4a"
    dst = Path(WORK_DIR) / f"{file_id}.wav"
    if dst.exists():
        return True  # already converted
    if not src.exists():
        print(f"  [worker] ⚠ m4a not found: {src}", flush=True)
        return False

    cmd = ["ffmpeg", "-y",
           "-i", str(src),
           "-ac", "1", "-ar", "16000",
           "-sample_fmt", "s16",
           str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  [worker] ❌ ffmpeg failed: {r.stderr[:300]}", flush=True)
        return False
    if not dst.exists() or dst.stat().st_size < 1000:
        print(f"  [worker] ❌ wav too small or missing: {dst}", flush=True)
        return False
    return True


def merge_diarization(whisper_segments, diar_result):
    """Merge Whisper transcription with speaker diarization results."""
    if diar_result is None:
        return [{"start": s.start, "end": s.end, "speaker": None, "text": s.text}
                for s in whisper_segments]

    merged = []
    for ws in whisper_segments:
        seg = Segment(ws.start, ws.end)
        cropped = diar_result.crop(seg)
        if cropped:
            best_label = None
            best_dur = 0.0
            for s_ov, trk, label in cropped.itertracks(yield_label=True):
                dur = (s_ov & seg).duration
                if dur > best_dur:
                    best_dur = dur
                    best_label = label
            speaker = best_label or "UNKNOWN"
        else:
            speaker = "UNKNOWN"
        merged.append({
            "start": ws.start,
            "end": ws.end,
            "speaker": speaker,
            "text": ws.text
        })
    return merged


def transcribe_one(file_id: str) -> bool:
    """Transcribe + diarize one file. Returns success."""
    wav_path = Path(WORK_DIR) / f"{file_id}.wav"
    out_json = Path(OUTPUT_DIR) / f"{file_id}.json"
    out_txt = Path(OUTPUT_DIR) / f"{file_id}.txt"

    # Skip if already done
    if out_json.exists() and out_txt.exists():
        print(f"  [worker] ⏭ Already done: {file_id}", flush=True)
        return True

    if not wav_path.exists():
        print(f"  [worker] ⚠ wav not found: {wav_path}", flush=True)
        return False

    import librosa
    import torch

    try:
        t0 = time.time()
        whisper = get_whisper()

        # 1. Whisper transcription
        segments, info = whisper.transcribe(
            str(wav_path), batch_size=BATCH_SIZE,
            beam_size=BEAM_SIZE, language=LANGUAGE,
            vad_filter=VAD_FILTER
        )
        whisper_segs = list(segments)
        t1 = time.time()
        print(f"  [worker] 🎤 {file_id}: {len(whisper_segs)} segments ({t1-t0:.1f}s)", flush=True)

        # 2. Speaker diarization (skip for short audio)
        duration = librosa.get_duration(path=str(wav_path))
        if duration < SKIP_DIAR_SHORT:
            print(f"  [worker] ⏩ {file_id}: short ({duration:.0f}s), diarization skipped", flush=True)
            diar_result = None
        elif duration > CHUNK_DURATION:
            print(f"  [worker] ✂️ {file_id}: long ({duration/60:.0f}min), chunking", flush=True)
            diar = get_diar()
            diar_results = []
            chunk_start = 0.0
            while chunk_start < duration:
                chunk_end = min(chunk_start + CHUNK_DURATION, duration)
                print(f"    Chunk {chunk_start/60:.0f}-{chunk_end/60:.0f}min", flush=True)
                chunk_wav = f"/tmp/{file_id}_chunk_{int(chunk_start)}.wav"
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav_path),
                     "-ss", str(chunk_start), "-to", str(chunk_end),
                     "-ac", "1", "-ar", "16000", chunk_wav],
                    capture_output=True, timeout=300
                )
                chunk_diar = diar({"audio": chunk_wav})
                shifted = Annotation()
                for seg, trk, label in chunk_diar.itertracks(yield_label=True):
                    shifted[Segment(seg.start + chunk_start, seg.end + chunk_start)] = label
                diar_results.append(shifted)
                os.remove(chunk_wav)
                chunk_start = chunk_end

            combined = Annotation()
            for anno in diar_results:
                for seg, trk, label in anno.itertracks(yield_label=True):
                    combined[seg] = label
            diar_result = combined
        else:
            diar = get_diar()
            diar_result = diar({"audio": str(wav_path)})

        t2 = time.time()
        n_speakers = len(diar_result.labels()) if diar_result is not None else 0
        print(f"  [worker] 👤 {file_id}: {n_speakers} speakers ({t2-t1:.1f}s)", flush=True)

        # 3. Merge
        merged = merge_diarization(whisper_segs, diar_result)

        # 4. Write output
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "filename": file_id,
                "language": info.language,
                "duration": info.duration,
                "segments": merged
            }, f, ensure_ascii=False, indent=2)

        with open(out_txt, "w", encoding="utf-8") as f:
            cur = None
            for m in merged:
                sp = m["speaker"] or "UNKNOWN"
                if sp != cur:
                    f.write(f"\n[{sp}]\n")
                    cur = sp
                f.write(f"{m['text']} ")

        print(f"  [worker] ✅ {file_id}: {time.time()-t0:.1f}s total", flush=True)
        return True

    except Exception as e:
        print(f"  [worker] ❌ {file_id}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Write error marker
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(Path(OUTPUT_DIR) / f"{file_id}.error", "w") as f:
            f.write(str(e) + "\n" + traceback.format_exc())
        return False


# ═══════════════════════════════════════════
#  Worker state persistence
# ═══════════════════════════════════════════

def load_state() -> dict:
    """Load worker state from JSON file."""
    if os.path.exists(WORKER_STATE):
        try:
            with open(WORKER_STATE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"batch_id": "", "files": {}, "started_at": "", "completed_at": ""}


def save_state(state: dict):
    """Save worker state to JSON file."""
    Path(WORKER_STATE).parent.mkdir(parents=True, exist_ok=True)
    with open(WORKER_STATE, "w") as f:
        json.dump(state, f, indent=2)


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GPU worker for ASR pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--files", help="Comma-separated list of file IDs to process")
    group.add_argument("--filelist", help="Path to file containing one file ID per line")
    group.add_argument("--status-only", action="store_true", help="Print current state and exit")
    args = parser.parse_args()

    if args.status_only:
        state = load_state()
        print(json.dumps(state, indent=2))
        return

    # Resolve file list
    file_ids = []
    if args.files:
        file_ids = [f.strip() for f in args.files.split(",") if f.strip()]
    elif args.filelist:
        with open(args.filelist) as f:
            file_ids = [line.strip() for line in f if line.strip()]

    if not file_ids:
        print("[worker] No files to process", flush=True)
        return

    batch_id = f"batch_{int(time.time())}"
    state = load_state()
    state["batch_id"] = batch_id
    state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["completed_at"] = ""
    state["files"] = {fid: {"status": "pending"} for fid in file_ids}
    save_state(state)

    print(f"[worker] 🚀 Batch {batch_id}: {len(file_ids)} files", flush=True)
    print(f"   input:  {INPUT_DIR}", flush=True)
    print(f"   work:   {WORK_DIR}", flush=True)
    print(f"   output: {OUTPUT_DIR}", flush=True)
    print(flush=True)

    success = 0
    failed = []
    for idx, fid in enumerate(file_ids):
        print(f"[worker] [{idx+1}/{len(file_ids)}] {fid}", flush=True)

        # Step 1: Convert m4a → wav
        state["files"][fid]["status"] = "converting"
        save_state(state)
        if not convert_m4a_to_wav(fid):
            state["files"][fid]["status"] = "failed"
            state["files"][fid]["error"] = "conversion failed"
            save_state(state)
            failed.append(fid)
            continue

        # Step 2: Transcribe + diarize
        state["files"][fid]["status"] = "transcribing"
        save_state(state)
        if transcribe_one(fid):
            state["files"][fid]["status"] = "done"
            save_state(state)
            success += 1
        else:
            state["files"][fid]["status"] = "failed"
            state["files"][fid]["error"] = "transcription failed"
            save_state(state)
            failed.append(fid)

    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_state(state)

    print(f"\n[worker] {'='*50}", flush=True)
    print(f"[worker] ✅ Batch complete: {success}/{len(file_ids)} succeeded", flush=True)
    if failed:
        print(f"[worker] ❌ Failed: {', '.join(failed)}", flush=True)
    print(f"[worker] State saved to {WORKER_STATE}", flush=True)


if __name__ == "__main__":
    main()
