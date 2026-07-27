#!/usr/bin/env python3
"""
pipeline_manager.py — ASR Pipeline Orchestrator (local machine).

Data sources:
  - Video IDs from ../data/video_list.jsonl   (source of truth, includes speaker names)
  - Audio files from ../data/audio_raw/        (.m4a format)
  - Local results to ../data/audio_diarized_local/  (Whisper + pyannote output)
  - AssemblyAI results in ../data/audio_diarized_assemblyai/{speaker}/*.txt   (skip if present)

Flow: upload m4a → convert to wav → transcribe+diarize → download results.
Resilient to GPU server downtime — processed data is safe locally.

Usage:
    # First-time: create manifest, show status
    python pipeline_manager.py --init-manifest
    python pipeline_manager.py --status

    # Process a batch and download results
    python pipeline_manager.py

    # Run continuously
    python pipeline_manager.py --loop

    # Custom batch sizes
    python pipeline_manager.py --batch-upload-mb 5000 --batch-process 10

    # Reset failed files for retry
    python pipeline_manager.py --reset-failed
"""

import os
import sys
import time
import json
import subprocess
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════
#  Default Configuration
# ═══════════════════════════════════════════

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data"))

DEFAULTS = {
    # Local data paths
    "video_list": os.path.join(DATA_DIR, "video_list.jsonl"),
    "local_audio_dir": os.path.join(DATA_DIR, "audio_raw"),
    "local_output_dir": os.path.join(DATA_DIR, "audio_diarized_local"),
    "assemblyai_dir": os.path.join(DATA_DIR, "audio_diarized_assemblyai"),
    "manifest_path": os.path.join(DATA_DIR, "audio_diarized_local", "manifest.jsonl"),
    "worker_script": os.path.join(PROJECT_DIR, "gpu_worker.py"),

    # GPU server connection
    "gpu_host": "root@connect.cqa1.seetacloud.com",
    "gpu_port": "34848",
    "gpu_input_dir": "/root/input",
    "gpu_work_dir": "/root/autodl-tmp",
    "gpu_output_dir": "/root/output_large-v3",
    "gpu_worker_path": "/root/gpu_worker.py",
    "gpu_worker_state": "/root_worker_state.json",
    "ssh_opts": "-o StrictHostKeyChecking=no -o ConnectTimeout=10",

    # Batch sizes & limits
    "upload_batch_mb": 100,       # Max MB per upload batch
    "process_batch_size": 5,        # Files per GPU worker invocation
    "download_batch_size": 20,      # Files to check + download per cycle
    "min_free_gb": 2.0,            # Pause upload if GPU disk below this
    "max_retries": 3,              # Max auto-retry for failed files
    "poll_interval": 60,           # Seconds between main loop iterations
    "ssh_timeout": 30,
    "scp_timeout": 300,
}

# ═══════════════════════════════════════════
#  Manifest
# ═══════════════════════════════════════════

STATES = ["pending", "uploading", "uploaded", "processing", "done", "failed"]


class Manifest:
    """JSONL-based state manifest — the single source of truth.
    One line per video ID: {"id":"abc123","state":"done","speaker":"adam_tooze",...}
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._entries = {}
        self._dirty = False
        self.load()

    def load(self):
        self._entries = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            self._entries[entry["id"]] = entry
                        except (json.JSONDecodeError, KeyError):
                            continue

    def flush(self):
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for eid in sorted(self._entries.keys()):
                f.write(json.dumps(self._entries[eid], ensure_ascii=False) + "\n")
        self._dirty = False

    def get(self, file_id: str) -> dict:
        return self._entries.get(file_id, {})

    def set_state(self, file_id: str, state: str, **extra):
        entry = self._entries.setdefault(file_id, {"id": file_id, "retries": 0})
        entry["state"] = state
        entry["updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        entry.update(extra)
        self._dirty = True

    def files_in_state(self, state: str) -> list:
        return [eid for eid, e in self._entries.items() if e.get("state") == state]

    def all(self) -> dict:
        return dict(self._entries)

    def count_by_state(self) -> dict:
        counts = {s: 0 for s in STATES}
        for e in self._entries.values():
            s = e.get("state", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    def total(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════
#  SSH / SCP Helpers
# ═══════════════════════════════════════════

class GPUConnection:
    """Wrap SSH and SCP operations to the GPU server."""

    def __init__(self, config: dict):
        self.config = config
        self._host = config["gpu_host"]
        self._port = config["gpu_port"]
        self._ssh_opts = config["ssh_opts"]
        self._timeout = config["ssh_timeout"]
        self._scp_timeout = config["scp_timeout"]

    def _ssh_cmd(self, remote_cmd: str) -> list:
        return ["ssh", "-p", self._port, *self._ssh_opts.split(), self._host, remote_cmd]

    def _scp_cmd(self, src: str, dst: str) -> list:
        return ["scp", "-P", self._port, *self._ssh_opts.split(), src, dst]

    def check_alive(self) -> bool:
        try:
            r = subprocess.run(self._ssh_cmd("echo alive"), capture_output=True, text=True, timeout=self._timeout)
            return r.returncode == 0 and "alive" in r.stdout
        except (subprocess.TimeoutExpired, OSError):
            return False

    def ssh(self, cmd: str, timeout: int = None) -> subprocess.CompletedProcess:
        return subprocess.run(self._ssh_cmd(cmd), capture_output=True, text=True, timeout=timeout or self._timeout)

    def scp_to(self, local_path: str, remote_path: str) -> bool:
        try:
            r = subprocess.run(self._scp_cmd(local_path, f"{self._host}:{remote_path}"), capture_output=True, text=True, timeout=self._scp_timeout)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def scp_from(self, remote_path: str, local_path: str) -> bool:
        try:
            r = subprocess.run(self._scp_cmd(f"{self._host}:{remote_path}", local_path), capture_output=True, text=True, timeout=self._scp_timeout)
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def list_dir(self, remote_path: str, suffix: str = "") -> list:
        pattern = f"*{suffix}" if suffix else ""
        r = self.ssh(f"ls {remote_path}/{pattern} 2>/dev/null | xargs -I{{}} basename {{}} {suffix}")
        if r.returncode != 0 or not r.stdout.strip():
            return []
        return [line.strip() for line in r.stdout.strip().split("\n") if line.strip()]

    def check_free_gb(self) -> float:
        r = self.ssh("df -BG / | tail -1 | awk '{print $4}' | tr -d 'G'")
        try:
            return float(r.stdout.strip())
        except (ValueError, IndexError):
            return 0.0

    def remove_file(self, remote_path: str) -> bool:
        r = self.ssh(f"rm -f {remote_path}")
        return r.returncode == 0

    def file_exists(self, remote_path: str) -> bool:
        r = self.ssh(f"test -f {remote_path} && echo yes")
        return "yes" in r.stdout.strip()


# ═══════════════════════════════════════════
#  Pipeline Phases
# ═══════════════════════════════════════════

class Pipeline:
    """Orchestrates upload → process → download → cleanup."""

    def __init__(self, config: dict, manifest: Manifest, gpu: GPUConnection):
        self.cfg = config
        self.mft = manifest
        self.gpu = gpu
        self.log_buf = []

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)
        self.log_buf.append(msg)

    # ── Phase: Upload ────────────────────────────────────

    def phase_upload(self) -> int:
        """Upload pending m4a files to GPU in configurable batches."""
        pending = self.mft.files_in_state("pending")
        if not pending:
            return 0

        free_gb = self.gpu.check_free_gb()
        if free_gb < self.cfg["min_free_gb"]:
            self.log(f"  ⏸️  GPU disk {free_gb:.1f}GB < {self.cfg['min_free_gb']}GB threshold, upload paused")
            return 0

        local_dir = Path(self.cfg["local_audio_dir"])
        if not local_dir.exists():
            self.log(f"  ❌ Local audio dir not found: {local_dir}")
            return 0

        batch = []
        batch_bytes = 0
        max_bytes = self.cfg["upload_batch_mb"] * 1024 * 1024

        for fid in pending:
            src = local_dir / f"{fid}.m4a"
            if not src.exists():
                self.log(f"  ⚠ {fid}.m4a not found locally, removing from manifest")
                self.mft._entries.pop(fid, None)
                continue
            fsize = src.stat().st_size
            if batch_bytes + fsize > max_bytes:
                break
            batch.append((fid, fsize, src))
            batch_bytes += fsize

        if not batch:
            return 0

        self.log(f"  📤 Uploading {len(batch)} files ({batch_bytes/1024/1024:.0f}MB)...")
        uploaded = 0
        for fid, fsize, src in batch:
            self.mft.set_state(fid, "uploading")
            dst = f"{self.cfg['gpu_input_dir']}/{fid}.m4a"
            if self.gpu.scp_to(str(src), dst):
                self.mft.set_state(fid, "uploaded", m4a_size=fsize)
                uploaded += 1
                self.log(f"    ✅ {fid}.m4a ({fsize/1024/1024:.0f}MB)")
            else:
                self.log(f"    ❌ {fid}.m4a upload failed")
                self.mft.set_state(fid, "failed", error="scp failed")

        self.mft.flush()
        return uploaded

    # ── Phase: Process (GPU Worker) ──────────────────────

    def phase_process(self) -> int:
        """Deploy & run GPU worker for uploaded files. Returns count dispatched."""
        uploaded = self.mft.files_in_state("uploaded")
        if not uploaded:
            return 0

        batch = uploaded[:self.cfg["process_batch_size"]]
        batch_list = "\n".join(batch)
        r = self.gpu.ssh(f"cat > /tmp/_pipeline_batch.txt << 'BATCH_END'\n{batch_list}\nBATCH_END\n")
        if r.returncode != 0:
            self.log(f"  ❌ Failed to create batch list on GPU: {r.stderr[:200]}")
            return 0

        self.gpu.ssh(f"mkdir -p {self.cfg['gpu_output_dir']}")

        for fid in batch:
            self.mft.set_state(fid, "processing")
        self.mft.flush()

        worker_remote = self.cfg["gpu_worker_path"]
        if not self.gpu.scp_to(self.cfg["worker_script"], worker_remote):
            self.log("  ❌ Failed to SCP worker script to GPU")
            for fid in batch:
                self.mft.set_state(fid, "uploaded")
            self.mft.flush()
            return 0

        worker_cmd = (
            f"cd /root && "
            f"export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.8/site-packages/torch/lib:${{LD_LIBRARY_PATH:-}} && "
            f"nohup /root/miniconda3/bin/python {worker_remote} "
            f"--filelist /tmp/_pipeline_batch.txt "
            f"> /root/_worker_batch.log 2>&1 &"
        )
        self.gpu.ssh(worker_cmd, timeout=15)
        self.log(f"  🚀 Dispatched {len(batch)} files to GPU worker")
        return len(batch)

    # ── Phase: Download ──────────────────────────────────

    def phase_download(self) -> int:
        """Download completed results from GPU. Returns count downloaded."""
        remote_output_dir = self.cfg["gpu_output_dir"]
        remote_jsons = set(self.gpu.list_dir(remote_output_dir, ".json"))

        remote_errors_raw = self.gpu.ssh(f"ls {remote_output_dir}/*.error 2>/dev/null | xargs -I{{}} basename {{}} .error")
        remote_errors = set(remote_errors_raw.stdout.strip().split()) if remote_errors_raw.stdout.strip() else set()

        processing = set(self.mft.files_in_state("processing"))
        uploaded = set(self.mft.files_in_state("uploaded"))

        done_on_gpu = (processing | uploaded) & remote_jsons
        failed_on_gpu = (processing | uploaded) & remote_errors

        if not done_on_gpu and not failed_on_gpu:
            return 0

        local_dir = Path(self.cfg["local_output_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0

        for fid in done_on_gpu:
            remote_json = f"{remote_output_dir}/{fid}.json"
            remote_txt = f"{remote_output_dir}/{fid}.txt"
            local_json = local_dir / f"{fid}.json"
            local_txt = local_dir / f"{fid}.txt"

            json_ok = self.gpu.scp_from(remote_json, str(local_json))
            txt_ok = self.gpu.scp_from(remote_txt, str(local_txt))

            if json_ok and txt_ok:
                self.mft.set_state(fid, "done")
                downloaded += 1
            else:
                self.log(f"  ⚠ {fid}: partial download (json={json_ok}, txt={txt_ok})")

        for fid in failed_on_gpu:
            retries = self.mft.get(fid).get("retries", 0)
            if retries < self.cfg["max_retries"]:
                self.mft.set_state(fid, "uploaded", retries=retries + 1)
                self.log(f"  🔄 {fid}: will retry ({retries + 1}/{self.cfg['max_retries']})")
            else:
                self.mft.set_state(fid, "failed", retries=retries)
                self.log(f"  ❌ {fid}: failed after {retries} retries")
            self.gpu.scp_from(f"{remote_output_dir}/{fid}.error", str(local_dir / f"{fid}.error"))

        self.mft.flush()

        if downloaded:
            self.log(f"  📥 Downloaded {downloaded} files to {self.cfg['local_output_dir']}")
            for fid in list(done_on_gpu)[:min(3, len(done_on_gpu))]:
                local_json = local_dir / f"{fid}.json"
                if local_json.exists():
                    try:
                        with open(local_json) as f:
                            data = json.load(f)
                        n_seg = len(data.get("segments", []))
                        speakers = set(s.get("speaker") for s in data.get("segments", []))
                        dur = data.get("duration", 0)
                        self.log(f"    📄 {fid}: {n_seg} segments, {len(speakers)} speakers, {dur:.0f}s")
                    except Exception:
                        pass

        return downloaded

    # ── Phase: Cleanup GPU ───────────────────────────────

    def phase_cleanup(self) -> int:
        """Remove processed files from GPU to free space."""
        done_ids = set(self.mft.files_in_state("done"))
        if not done_ids:
            return 0

        cleaned = 0
        for fid in done_ids:
            wav_remote = f"{self.cfg['gpu_work_dir']}/{fid}.wav"
            if self.gpu.file_exists(wav_remote):
                self.gpu.remove_file(wav_remote)
                cleaned += 1
            m4a_remote = f"{self.cfg['gpu_input_dir']}/{fid}.m4a"
            if self.gpu.file_exists(m4a_remote):
                self.gpu.remove_file(m4a_remote)

        if cleaned:
            self.log(f"  🧹 Cleaned {cleaned} wav files from GPU")
        return cleaned

    # ── Phase: Recover Stuck States ──────────────────────

    def phase_recover(self) -> int:
        """Recover files stuck in transient states (after crash/restart)."""
        recovered = 0

        for fid in self.mft.files_in_state("uploading"):
            remote_m4a = f"{self.cfg['gpu_input_dir']}/{fid}.m4a"
            self.mft.set_state(fid, "uploaded" if self.gpu.file_exists(remote_m4a) else "pending")
            recovered += 1

        remote_output_dir = self.cfg["gpu_output_dir"]
        remote_done = set(self.gpu.list_dir(remote_output_dir, ".json"))
        for fid in list(self.mft.files_in_state("processing")):
            if fid in remote_done:
                continue
            self.mft.set_state(fid, "uploaded", retries=self.mft.get(fid).get("retries", 0) + 1)
            self.log(f"  🔄 {fid}: re-queued from processing → uploaded")
            recovered += 1

        for fid in self.mft.files_in_state("failed"):
            entry = self.mft.get(fid)
            retries = entry.get("retries", 0)
            if retries < self.cfg["max_retries"]:
                self.mft.set_state(fid, "uploaded", retries=retries)
                self.log(f"  🔄 {fid}: retrying (attempt {retries + 1}/{self.cfg['max_retries']})")
                recovered += 1

        if recovered:
            self.mft.flush()
        return recovered

    # ── Report ───────────────────────────────────────────

    def report(self):
        counts = self.mft.count_by_state()
        total = sum(counts.values())
        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        pending = counts.get("pending", 0)
        processing = counts.get("processing", 0)
        in_flight = counts.get("uploading", 0) + counts.get("uploaded", 0)

        progress = done / total * 100 if total > 0 else 0

        gpu_alive = self.gpu.check_alive()
        free_gb = self.gpu.check_free_gb() if gpu_alive else 0

        print(f"\n{'='*60}")
        print(f"  ASR Pipeline Status")
        print(f"{'='*60}")
        print(f"  GPU:      {'🟢 Online' if gpu_alive else '🔴 Offline'}"
              f"{f'  ({free_gb:.1f}GB free)' if gpu_alive else ''}")
        print(f"  Total:    {total} files")
        print(f"  Pending:  {pending}")
        print(f"  In flight:{in_flight} (uploading+uploaded)")
        print(f"  GPU work: {processing}")
        print(f"  Done:     {done}  ({progress:.1f}%)")
        print(f"  Failed:   {failed}")
        print(f"  Output:   {self.cfg['local_output_dir']}")
        print(f"{'='*60}\n")


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR Pipeline Orchestrator — upload → transcribe → download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--once", action="store_true", help="Run one full cycle (default)")
    parser.add_argument("--status", action="store_true", help="Show pipeline status and exit")
    parser.add_argument("--init-manifest", action="store_true",
                        help="Create manifest from video_list.jsonl + filter already-done files")
    parser.add_argument("--reset-failed", action="store_true", help="Reset all failed files to pending")
    parser.add_argument("--batch-upload-mb", type=int, default=None,
                        help=f"Max MB per upload batch (default {DEFAULTS['upload_batch_mb']})")
    parser.add_argument("--batch-process", type=int, default=None,
                        help=f"Files per GPU worker batch (default {DEFAULTS['process_batch_size']})")
    parser.add_argument("--batch-download", type=int, default=None,
                        help=f"Max files to download per cycle (default {DEFAULTS['download_batch_size']})")
    parser.add_argument("--poll-interval", type=int, default=None,
                        help=f"Seconds between loop iterations (default {DEFAULTS['poll_interval']})")
    parser.add_argument("--max-retries", type=int, default=None,
                        help=f"Max retries per file (default {DEFAULTS['max_retries']})")
    return parser.parse_args()


def init_manifest(config: dict) -> Manifest:
    """Create manifest from video_list.jsonl.

    Reads all video IDs from video_list.jsonl, checks which have .m4a in
    audio_raw/, and marks as done if already in audio_diarized_local/ or
    audio_diarized_assemblyai/ (any speaker subdirectory). New pending files
    are those with .m4a but without existing results.
    """
    manifest = Manifest(config["manifest_path"])
    existing = manifest.all()

    # Load video list
    video_path = Path(config["video_list"])
    if not video_path.exists():
        print(f"❌ video_list.jsonl not found: {video_path}")
        sys.exit(1)

    video_entries = {}  # id → metadata
    with open(video_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    video_entries[data["id"]] = data
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"📋 video_list.jsonl: {len(video_entries)} total entries")
    ids_with_m4a = set()

    # Find which IDs have .m4a files
    audio_dir = Path(config["local_audio_dir"])
    if audio_dir.exists():
        for p in audio_dir.glob("*.m4a"):
            ids_with_m4a.add(p.stem)
    print(f"📁 audio_raw/: {len(ids_with_m4a)} .m4a files")

    # Only consider IDs that exist in video_list AND have .m4a
    valid_ids = set(video_entries.keys()) & ids_with_m4a
    print(f"🎯 Valid (in both video_list + audio_raw): {len(valid_ids)}")

    # Find already-processed files in audio_diarized_local/
    local_done = set()
    local_dir = Path(config["local_output_dir"])
    if local_dir.exists():
        for p in local_dir.glob("*.json"):
            txt = local_dir / f"{p.stem}.txt"
            if txt.exists() and p.stem != "quality_report":
                local_done.add(p.stem)
    print(f"✅ audio_diarized_local/ done: {len(local_done)}")

    # Find already-processed files in audio_diarized_assemblyai/
    aa_done = set()
    aa_dir = Path(config["assemblyai_dir"])
    if aa_dir.exists():
        for p in aa_dir.rglob("*.txt"):
            aa_done.add(p.stem)
    print(f"✅ audio_diarized_assemblyai/ done: {len(aa_done)}")

    already_done = local_done | aa_done
    new_pending = valid_ids - already_done - set(existing.keys())
    already_pending = valid_ids - already_done - new_pending

    # Add new entries
    for fid in sorted(new_pending):
        meta = video_entries.get(fid, {})
        entry = {
            "id": fid, "state": "pending", "retries": 0,
            "speaker": meta.get("speaker", ""),
            "title": meta.get("title", ""),
            "updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest._entries[fid] = entry

    # Mark already-done files in manifest as done (update existing entries too)
    for fid in valid_ids & already_done:
        if fid in existing:
            manifest.set_state(fid, "done")
        else:
            meta = video_entries.get(fid, {})
            manifest.set_state(fid, "done",
                              speaker=meta.get("speaker", ""),
                              title=meta.get("title", ""))

    manifest.flush()

    counts = manifest.count_by_state()
    print(f"\n📊 Manifest summary:")
    print(f"   Pending:  {counts.get('pending', 0)}")
    print(f"   Done:     {counts.get('done', 0)}")
    print(f"   Failed:   {counts.get('failed', 0)}")
    print(f"   Total:    {len(manifest.all())}")

    return manifest


def reset_failed(config: dict):
    manifest = Manifest(config["manifest_path"])
    count = 0
    for fid in manifest.files_in_state("failed"):
        manifest.set_state(fid, "pending", retries=0)
        count += 1
    manifest.flush()
    print(f"🔄 Reset {count} failed files to pending")


def run_cycle(pipeline: Pipeline) -> dict:
    activity = {"uploaded": 0, "dispatched": 0, "downloaded": 0, "cleaned": 0}
    if not pipeline.gpu.check_alive():
        pipeline.log("🔴 GPU server unreachable — skipping all phases")
        return activity
    activity["uploaded"] = pipeline.phase_upload()
    activity["dispatched"] = pipeline.phase_process()
    activity["downloaded"] = pipeline.phase_download()
    activity["cleaned"] = pipeline.phase_cleanup()
    return activity


def main():
    args = parse_args()
    config = dict(DEFAULTS)

    if args.batch_upload_mb is not None:
        config["upload_batch_mb"] = args.batch_upload_mb
    if args.batch_process is not None:
        config["process_batch_size"] = args.batch_process
    if args.batch_download is not None:
        config["download_batch_size"] = args.batch_download
    if args.poll_interval is not None:
        config["poll_interval"] = args.poll_interval
    if args.max_retries is not None:
        config["max_retries"] = args.max_retries

    if args.init_manifest:
        init_manifest(config)
        return

    if args.reset_failed:
        reset_failed(config)
        return

    manifest = Manifest(config["manifest_path"])

    if not manifest.all():
        print("⚠️ Manifest is empty. Run with --init-manifest to create from video_list.jsonl.")
        if args.status:
            return
        print("   Continuing with empty manifest.\n")

    gpu = GPUConnection(config)
    pipeline = Pipeline(config, manifest, gpu)

    if args.status:
        pipeline.report()
        return

    if args.loop:
        pipeline.log("🔄 ASR Pipeline started (loop mode)")
        pipeline.log(f"   Upload batch: {config['upload_batch_mb']}MB")
        pipeline.log(f"   Process batch: {config['process_batch_size']} files")
        pipeline.log(f"   Poll interval: {config['poll_interval']}s")
        pipeline.log(f"   Max retries: {config['max_retries']}")
        pipeline.log("")

        consecutive_empty = 0
        while True:
            try:
                recovered = pipeline.phase_recover()
                activity = run_cycle(pipeline)
                total_active = sum(activity.values())

                if total_active == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0

                if total_active > 0 or consecutive_empty >= 10:
                    pipeline.report()
                    consecutive_empty = 0

                pipeline.mft.flush()

                counts = manifest.count_by_state()
                total = sum(counts.values())
                if total > 0 and counts.get("done", 0) >= total:
                    pipeline.log("🎉 All files processed!")
                    pipeline.report()
                    break

                time.sleep(config["poll_interval"])

            except KeyboardInterrupt:
                pipeline.log("\n⏹️  Interrupted by user")
                pipeline.mft.flush()
                break
            except Exception as e:
                pipeline.log(f"❌ Pipeline error: {e}")
                traceback.print_exc()
                pipeline.mft.flush()
                time.sleep(config["poll_interval"])

    else:
        pipeline.log("▶️ ASR Pipeline (single run)")
        pipeline.phase_recover()

        if not gpu.check_alive():
            pipeline.log("🔴 GPU server unreachable, nothing to do")
            pipeline.report()
            return

        pipeline.phase_upload()
        pipeline.phase_process()
        pipeline.phase_download()
        pipeline.phase_cleanup()
        pipeline.report()


if __name__ == "__main__":
    main()
