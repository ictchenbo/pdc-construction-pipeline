# Public Discourse Corpus (PDC) — Research Pipeline

Codebase for "The Public Discourse Corpus (PDC): A Large-Scale Dual-Dimension
Valence-Modality Annotated Dataset of 124 Public Figures."

## Overview

A multi-stage pipeline that collects public-facing speech (YouTube interviews,
talks, podcasts) for 124 notable figures across 7 domains, transcribes and
diarizes the audio, splits it into sentences, and annotates each sentence for
**valence** (positive / negative / neutral) and **modality** (emphatic / hedged /
neutral) using LLMs. The result is a large-scale, dual-dimension dataset for
studying how public figures express certainty and affect.

## Project Structure

```
pdc-construction-pipeline/
├── .env.example              # API key template (copy to .env, fill in keys)
├── .gitignore
├── README.md                 # This file
├── scripts/                  # Standalone helper scripts
│   ├── compare_pipelines.py       # Compare AssemblyAI vs local diarization
│   ├── deepseek_annotate_full.sh  # Batch annotation (DeepSeek, full corpus)
│   └── gpt_annotate_part.sh       # Batch annotation (GPT, sampled subset)
│
├── src/
│   ├── config.py             # Centralized paths, LLM config, speaker domains
│   │
│   ├── 01_collect/           # Data collection
│   │   ├── search_speakers.py     # YouTube video search by person
│   │   ├── select_videos.py       # VTT download → filter → audio download
│   │   └── tsp_preannotate.py     # TSP pre-annotation (pre-ASR filter)
│   │
│   ├── 02_asr/               # ASR + speaker diarization
│   │   ├── pipeline_manager.py    # Orchestrator (local macOS → GPU via SSH)
│   │   ├── gpu_worker.py          # GPU worker (Whisper large-v3 + pyannote)
│   │   ├── assemblyai_diarize.py  # AssemblyAI diarization (alternative)
│   │   ├── extract_guest_from_local.py  # Extract target speaker (rule + LLM)
│   │   ├── extract_guest_missing.py     # Backfill missing extractions
│   │   └── quality_check.py       # Transcription quality assessment
│   │
│   ├── 03_sentence_splitting/  # Sentence segmentation
│   │   ├── split_by_video.py      # Per-video sentence splitting
│   │   ├── merge_sentence_lines.py # Merge broken ASR lines
│   │   └── re_split_long_sentences.py  # LLM re-split of 60+ word sentences
│   │
│   ├── 04_annotation/        # Valence & modality annotation
│   │   ├── annotate.py            # LLM annotation (multi-provider, unified)
│   │   └── sample.py              # Stratified sampling by domain
│   │
│   ├── 05_statistical_analysis/  # Statistical analysis
│   │   ├── cross_domain_typology.py     # RQ3: cross-domain typology
│   │   ├── temporal_event_analysis.py   # RQ6: event-driven shifts
│   │   ├── hierarchical_modeling.py    # RQ2: artifact magnitude predictors
│   │   ├── pipeline_quality_audit.py   # Clean → diarize → split audit
│   │   ├── compare_diarization_keyword.py  # Keyword: VTT vs AssemblyAI
│   │   ├── compare_diarization_llm.py      # LLM: VTT vs AssemblyAI
│   │   ├── compare_synthesis.py            # Cross-method sensitivity
│   │   ├── keyword_lexicon.py             # Keyword lexicon V3
│   │   └── keyword_scoring.py             # Monthly keyword scoring
│   │
│   ├── 06_evaluation/        # Evaluation
│   │   ├── cross_dim_correlation.py  # Cross-dimensional phi correlations
│   │   └── stratified_sample.py      # Proportional stratified sampling
│   │
│   ├── 07_publish/           # Dataset assembly
│   │   └── merge_dataset.py       # Merge video corpus → single JSONL
│   │
│   ├── 09_quality_assurance/ # Quality assurance
│   │   ├── pipeline.py            # 3-stage QA (rule → LLM → export)
│   │   └── audit_speaker_domains.py  # Audit domain assignments via GPT
│   │
│   └── utils/                # Shared utilities
│       ├── common.py              # JSON/JSONL I/O, speaker discovery
│       ├── llm.py                 # OpenAI-compatible API client + .env
│       └── clean_asr_repetition.py # Collapse ASR repetition artifacts
│
└── data/                     # (git-ignored) Analysis outputs
    └── merge/                     # corpus.jsonl, anno.jsonl, sampled.jsonl
```

> **Note:** Raw pipeline data (audio, VTT, diarized transcripts, sentences)
> lives in a **sibling `../data/`** directory shared with the parent research
> project — not inside this repo. Only small derived artifacts
> (`data/merge/*.jsonl`) reside here.

## Data Flow

```
video_list.jsonl  (source of truth: video IDs + speaker names)
    │
    ▼ [01_collect/search_speakers.py]
search_results/<speaker>/video_list.json
    │
    ▼ [01_collect/select_videos.py]
../data/vtt_raw/ + ../data/audio_raw/     (subtitles + audio)
    │
    ▼ [01_collect/tsp_preannotate.py]
../data/tsp_preannotation.jsonl            (TSP pre-ASR filter)
    │
    ├───▼ [02_asr/pipeline_manager.py + gpu_worker.py]─── (Whisper+pyannote)
    │   ../data/audio_diarized_local/     (multi-speaker JSON + TXT)
    │
    ├───▼ [02_asr/assemblyai_diarize.py]────────────────── (AssemblyAI)
    │   ../data/audio_diarized_assemblyai/{speaker}/*.txt
    │
    ▼ [02_asr/extract_guest_from_local.py]
../data/audio_diarized_local_guest/       (guest-only transcripts)
    │
    ▼ [03_sentence_splitting/split_by_video.py]
../data/sentences_by_video/               (one sentence per line)
    │
    ▼ [03_sentence_splitting/merge_sentence_lines.py]
../data/sentences_by_video_keep/          (cleaned sentence lines)
    │
    ▼ [07_publish/merge_dataset.py]
data/merge/corpus.jsonl                   (one JSON row per sentence)
    │
    ▼ [04_annotation/sample.py]
data/validation/sample_manifest.jsonl     (stratified sample)
    │
    ▼ [04_annotation/annotate.py]
data/merge/anno.jsonl                     (valence + modality labels)
    │
    ├───▼ [06_evaluation/]───────────────────────────────
    │   cross_dim_correlation.py → results/cross_dim_correlations.json
    │   stratified_sample.py     → data/merge/sampled.jsonl
    │
    └───▼ [05_statistical_analysis/]──────────────────────
        cross_domain_typology.py   → results/domain_typology.json      (RQ3)
        temporal_event_analysis.py → results/temporal_event_analysis.json (RQ6)
        hierarchical_modeling.py  → results/hierarchical_model_results.json (RQ2)
        pipeline_quality_audit.py → data/inspection/pipeline_audit.json
        compare_*.py              → results/*_comparison.json
```

## Pipeline Stages

| Stage | Research Question | Key Scripts |
|-------|-------------------|-------------|
| **Collect** | -- | `search_speakers.py`, `select_videos.py`, `tsp_preannotate.py` |
| **ASR + Diarize** | -- | `pipeline_manager.py`, `gpu_worker.py`, `assemblyai_diarize.py`, `extract_guest_from_local.py` |
| **Sentence Split** | -- | `split_by_video.py`, `merge_sentence_lines.py`, `re_split_long_sentences.py` |
| **Annotate** | -- | `annotate.py` (valence + modality), `sample.py` |
| **Evaluate** | -- | `cross_dim_correlation.py`, `stratified_sample.py` |
| **Analyze** | RQ2, RQ3, RQ6 | `hierarchical_modeling.py`, `cross_domain_typology.py`, `temporal_event_analysis.py` |
| **QA** | -- | `pipeline.py`, `audit_speaker_domains.py`, `pipeline_quality_audit.py` |
| **Publish** | -- | `merge_dataset.py` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt        # or: pip install python-dotenv requests ...

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys (see below)

# 3. Run individual stages (each script is self-contained)
python3 src/01_collect/search_speakers.py "Ray Dalio"
python3 src/02_asr/pipeline_manager.py --status
python3 src/04_annotation/annotate.py --provider deepseek --dry-run

# 4. Batch annotation via shell wrapper
bash scripts/deepseek_annotate_full.sh
bash scripts/gpt_annotate_part.sh
```

Most scripts accept `--help` for full usage. Stage scripts read config from
`src/config.py` and API keys from `.env`.

## Speaker Domains (124 speakers, 7 domains)

Defined in `src/config.py` → `SPEAKER_DOMAINS`:

| Domain | Count | Example speakers |
|--------|-------|------------------|
| Finance/Investing | ~28 | Dalio, Wood, Buffett, Ackman, Dimon |
| Politics/Government | ~42 | Trump, Biden, Sanders, Zelenskyy, Macron |
| Academia/Economics | ~22 | Rogoff, Acemoglu, Krugman, Stiglitz |
| Central Banking/Policy | ~9 | Powell, Yellen, Bernanke, Lagarde |
| Geopolitics/Strategy | ~15 | Zeihan, Kissinger, Mearsheimer |
| Technology/Business | ~9 | Musk, Altman, Thiel |
| Media/Commentary | ~18 | Shapiro, Carlson, Peterson |

## API Configuration (.env)

Copy `.env.example` → `.env` and fill in your keys:

```bash
# ── Transcript Cleaning (OpenAI-compatible, e.g. DeepSeek) ──
CLEAN_API_KEY=your_api_key_here
CLEAN_API_BASE=https://api.deepseek.com
CLEAN_MODEL=deepseek-v4-flash
CLEAN_MODEL1=deepseek-v4-pro

# ── Audit / QA (OpenAI-compatible, e.g. GPT) ──
AUDIT_API_BASE=https://api.856868.xyz/v1
AUDIT_API_KEY=your_api_key_here
AUDIT_MODEL=gpt-5.5

# ── AssemblyAI (Diarization) ──
ASSEMBLYAI_API_KEY=your_api_key_here
```

Three independent API roles:
- **CLEAN** — LLM for transcript cleaning, guest extraction, sentence re-splitting
- **AUDIT** — Secondary LLM for quality audits and domain-assignment verification
- **ASSEMBLYAI** — Speaker diarization (alternative to local Whisper+pyannote)

## Dependencies

- Python 3.12+
- `python-dotenv` — `.env` loading
- `requests` — OpenAI-compatible API calls (no OpenAI SDK needed)
- `numpy`, `scipy` — statistics, correlations
- `faster-whisper` — ASR (GPU worker)
- `pyannote.audio` — speaker diarization (GPU worker)
- `librosa`, `ffmpeg` — audio conversion
- `scikit-learn` — classification metrics
- `statsmodels` — hierarchical modeling

## Research Questions

| RQ | Question | Analysis |
|----|----------|----------|
| RQ1 | Artifact universality | `pipeline_quality_audit.py` |
| RQ2 | Speaker feature predictors | `hierarchical_modeling.py` |
| RQ3 | Domain patterns | `cross_domain_typology.py` |
| RQ4 | Inter-annotator agreement | `annotate.py` (multi-provider) |
| RQ5 | Benchmark | `keyword_scoring.py`, `compare_*.py` |
| RQ6 | Event-driven shifts | `temporal_event_analysis.py` |
