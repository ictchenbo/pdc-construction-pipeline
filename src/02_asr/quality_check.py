#!/usr/bin/env python3
"""
转写质量检查器
基于转写管道生成文件中的 JSON 文件进行自动质量评估
输出：每个文件的质量评分 + 详细指标 + 综合报告

质量维度:
  A. 文本质量 (text_quality)
  B. 说话人分离稳定性 (diarization_stability)
  C. 时序完整性 (temporal_quality)
  D. 异常检测 (anomaly_detection)
"""

import os
import sys
import json
import re
import math
from pathlib import Path
from collections import Counter

import numpy as np

# ── 配置 ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "..", "data"))
OUTPUT_DIR = os.path.join(DATA_DIR, 'audio_diarized_local')
REPORT_FILE = "diarization_quality_report.json"
VERBOSE = True
# ─────────

FILLER_WORDS = {
    "um", "uh", "ah", "er", "erm", "hmm", "mm-hmm", "uh-huh",
    "like", "you know", "i mean", "sort of", "kind of",
    "actually", "basically", "literally", "right", "okay",
    "so", "well",
}
CAPTION_PATTERNS = [
    re.compile(r'^[A-Z\s]+$'),  # ALL CAPS like "LAUGHTER", "MUSIC"
    re.compile(r'^♪.+♪$'),      # music notes
    re.compile(r'^\[.*\]$'),     # [Music], [Applause]
]
NON_EN_PATTERN = re.compile(r'[^\x00-\x7F]')


# ═══════════════════════════════════════════
#  A. 文本质量
# ═══════════════════════════════════════════

def check_text_quality(segments: list, duration: float) -> dict:
    """文本质量检查"""
    texts = [s["text"].strip() for s in segments]
    speakers = [s["speaker"] for s in segments]
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]

    word_counts = [len(t.split()) if t else 0 for t in texts]
    total_words = sum(word_counts)

    # 1. 语速 (words per second)
    wps = total_words / duration if duration > 0 else 0

    # 2. 单/双词段比例 (可能表示碎片化)
    short_seg_ratio = sum(1 for w in word_counts if w <= 2) / len(segments) if segments else 0

    # 3. 空文本段
    empty_seg_ratio = sum(1 for w in word_counts if w == 0) / len(segments) if segments else 0

    # 4. 填充词比例
    all_words = []
    for t in texts:
        all_words.extend(t.lower().split())
    filler_count = sum(1 for w in all_words if w.strip(".,!?;:\"'") in FILLER_WORDS)
    filler_ratio = filler_count / len(all_words) if all_words else 0

    # 5. 大写标注段（LAUGHTER, MUSIC等）
    caption_texts = sum(1 for t in texts if any(p.match(t) for p in CAPTION_PATTERNS))
    caption_ratio = caption_texts / len(segments) if segments else 0

    # 6. 非英文字符
    non_en_segments = sum(1 for t in texts if NON_EN_PATTERN.search(t))

    # 综合评分 (0-100, 越高越好)
    score = 100.0
    # 语速: 正常 2-4 wps, 太低可能漏词, 太高可能重复
    if wps < 1.0:
        score -= 15
    elif wps > 5.0:
        score -= 10
    # 短段比例
    if short_seg_ratio > 0.5:
        score -= 20
    elif short_seg_ratio > 0.3:
        score -= 10
    # 空段
    if empty_seg_ratio > 0.05:
        score -= 15
    # 填充词 (适量正常, 太多说明犹豫多或模型问题)
    if filler_ratio > 0.15:
        score -= 10
    elif filler_ratio > 0.25:
        score -= 20
    # 非英文字符
    if non_en_segments > 0:
        score -= 10

    return {
        "total_words": total_words,
        "words_per_second": round(wps, 2),
        "avg_words_per_segment": round(np.mean(word_counts), 1) if word_counts else 0,
        "short_segment_ratio": round(short_seg_ratio, 3),
        "short_segment_count": sum(1 for w in word_counts if w <= 2),
        "empty_segment_ratio": round(empty_seg_ratio, 3),
        "filler_word_ratio": round(filler_ratio, 3),
        "filler_word_count": filler_count,
        "caption_segment_ratio": round(caption_ratio, 3),
        "caption_segment_count": caption_texts,
        "non_english_segments": non_en_segments,
        "score": max(0, round(score, 1)),
        "flags": _flags_text(word_counts, wps, filler_ratio, short_seg_ratio),
    }


def _flags_text(word_counts, wps, filler_ratio, short_seg_ratio):
    flags = []
    if wps < 1.0:
        flags.append("WARN:very_low_wps")
    elif wps > 5.0:
        flags.append("WARN:high_wps")
    if short_seg_ratio > 0.5:
        flags.append("FAIL:excessive_short_segments")
    elif short_seg_ratio > 0.3:
        flags.append("WARN:many_short_segments")
    if filler_ratio > 0.2:
        flags.append("WARN:high_filler_ratio")
    return flags


# ═══════════════════════════════════════════
#  B. 说话人分离稳定性
# ═══════════════════════════════════════════

def check_diarization_stability(segments: list) -> dict:
    """说话人分离质量检查"""
    if not segments:
        return {"score": 0, "speaker_count": 0, "flags": ["FAIL:no_segments"]}

    # 说话人统计
    speaker_to_segs = {}
    for s in segments:
        sp = s["speaker"]
        dur = s["end"] - s["start"]
        text = s["text"].strip()
        if sp not in speaker_to_segs:
            speaker_to_segs[sp] = {"count": 0, "total_dur": 0, "durations": [], "words": []}
        speaker_to_segs[sp]["count"] += 1
        speaker_to_segs[sp]["total_dur"] += dur
        speaker_to_segs[sp]["durations"].append(dur)
        speaker_to_segs[sp]["words"].append(len(text.split()))

    total_speakers = len(speaker_to_segs)
    unknown_ratio = speaker_to_segs.get("UNKNOWN", {}).get("count", 0) / len(segments) if segments else 0

    # 各说话人的段时长统计
    sp_durations = []
    short_sp_segments = 0
    for sp, data in speaker_to_segs.items():
        sp_durations.extend(data["durations"])
        for d in data["durations"]:
            if d < 1.0:
                short_sp_segments += 1

    # 极短段比例（<1s 可能表示不稳定）
    very_short_ratio = short_sp_segments / len(segments) if segments else 0

    # 段时长分布
    all_durs = [s["end"] - s["start"] for s in segments]
    dur_mean = np.mean(all_durs) if all_durs else 0
    dur_std = np.std(all_durs) if all_durs else 0
    dur_cv = dur_std / dur_mean if dur_mean > 0 else 0  # 变异系数

    # 说话人切换频率
    speaker_changes = sum(1 for i in range(1, len(segments)) if segments[i]["speaker"] != segments[i-1]["speaker"])
    change_rate = speaker_changes / len(segments) if segments else 0

    # 各说话人的时间占比
    total_dur = sum(s["end"] - s["start"] for s in segments)
    sp_shares = []
    for sp, data in speaker_to_segs.items():
        share = data["total_dur"] / total_dur if total_dur > 0 else 0
        sp_shares.append((sp, share))
    sp_shares.sort(key=lambda x: -x[1])

    # 综合评分
    score = 100.0
    if total_speakers == 0:
        score -= 50
    elif total_speakers == 1:
        score -= 10  # 单人(可能无对话)
    elif total_speakers >= 8:
        score -= 25  # 过多说话人（可能过分离）
    elif total_speakers >= 6:
        score -= 10

    if very_short_ratio > 0.2:
        score -= 20
    elif very_short_ratio > 0.1:
        score -= 10

    if unknown_ratio > 0.3:
        score -= 25
    elif unknown_ratio > 0.1:
        score -= 10

    if change_rate > 0.6:
        score -= 15  # 切换过于频繁
    if dur_cv > 2.0:
        score -= 10  # 段时长变异过大

    return {
        "speaker_count": total_speakers,
        "speakers": [{"name": sp, "share_pct": round(sh * 100, 1)}
                     for sp, sh in sp_shares],
        "unknown_ratio": round(unknown_ratio, 3),
        "very_short_segment_ratio": round(very_short_ratio, 3),
        "very_short_segment_count": short_sp_segments,
        "avg_segment_duration": round(dur_mean, 2),
        "segment_duration_cv": round(dur_cv, 2),
        "speaker_changes": speaker_changes,
        "change_rate": round(change_rate, 3),
        "score": max(0, round(score, 1)),
        "flags": _flags_diarization(total_speakers, very_short_ratio, unknown_ratio, change_rate),
    }


def _flags_diarization(n_speakers, very_short_ratio, unknown_ratio, change_rate):
    flags = []
    if n_speakers >= 8:
        flags.append("FAIL:too_many_speakers")
    elif n_speakers >= 6:
        flags.append("WARN:many_speakers")
    if n_speakers <= 1:
        flags.append("WARN:single_speaker")
    if unknown_ratio > 0.3:
        flags.append("FAIL:high_unknown_ratio")
    elif unknown_ratio > 0.1:
        flags.append("WARN:some_unknown")
    if very_short_ratio > 0.2:
        flags.append("FAIL:excessive_short_segments")
    elif very_short_ratio > 0.1:
        flags.append("WARN:many_short_segments")
    if change_rate > 0.6:
        flags.append("WARN:high_turn_rate")
    return flags


# ═══════════════════════════════════════════
#  C. 时序完整性
# ═══════════════════════════════════════════

def check_temporal_quality(segments: list, duration: float) -> dict:
    """时序完整性检查"""
    if not segments or duration <= 0:
        return {"score": 0, "flags": ["FAIL:no_data"]}

    starts = np.array([s["start"] for s in segments])
    ends = np.array([s["end"] for s in segments])

    # 1. 覆盖比例
    seg_durations = ends - starts
    total_covered = np.sum(seg_durations)
    coverage_ratio = total_covered / duration if duration > 0 else 0

    # 2. 间隙分析
    sorted_segs = sorted(segments, key=lambda x: x["start"])
    gaps = []
    for i in range(1, len(sorted_segs)):
        gap = sorted_segs[i]["start"] - sorted_segs[i-1]["end"]
        if gap > 0.5:  # >0.5s 算间隙
            gaps.append(gap)

    total_gap = sum(gaps) if gaps else 0
    gap_ratio = total_gap / duration if duration > 0 else 0
    max_gap = max(gaps) if gaps else 0

    # 3. 重叠检测
    overlaps = []
    total_overlap = 0.0
    for i in range(1, len(sorted_segs)):
        ov = sorted_segs[i-1]["end"] - sorted_segs[i]["start"]
        if ov > 0.1:  # >100ms 算重叠
            overlaps.append(ov)
            total_overlap += ov

    overlap_ratio = total_overlap / duration if duration > 0 else 0

    score = 100.0
    if coverage_ratio < 0.5:
        score -= 30
    elif coverage_ratio < 0.7:
        score -= 15

    if gap_ratio > 0.2:
        score -= 20
    elif gap_ratio > 0.1:
        score -= 10

    if max_gap > 30:
        score -= 15
    elif max_gap > 10:
        score -= 5

    if overlap_ratio > 0.05:
        score -= 10

    return {
        "duration_seconds": round(duration, 1),
        "coverage_ratio": round(coverage_ratio, 3),
        "total_gap_seconds": round(total_gap, 2),
        "gap_ratio": round(gap_ratio, 3),
        "max_gap_seconds": round(max_gap, 2),
        "gap_count": len(gaps),
        "total_overlap_seconds": round(total_overlap, 2),
        "overlap_ratio": round(overlap_ratio, 3),
        "overlap_count": len(overlaps),
        "score": max(0, round(score, 1)),
        "flags": _flags_temporal(coverage_ratio, gap_ratio, max_gap, overlap_ratio),
    }


def _flags_temporal(coverage, gap_ratio, max_gap, overlap_ratio):
    flags = []
    if coverage < 0.5:
        flags.append("FAIL:low_coverage")
    elif coverage < 0.7:
        flags.append("WARN:moderate_coverage")
    if gap_ratio > 0.2:
        flags.append("FAIL:large_gaps")
    elif gap_ratio > 0.1:
        flags.append("WARN:some_gaps")
    if max_gap > 30:
        flags.append("WARN:long_silence")
    if overlap_ratio > 0.05:
        flags.append("WARN:overlap")
    return flags


# ═══════════════════════════════════════════
#  D. 异常检测
# ═══════════════════════════════════════════

def check_anomalies(segments: list, duration: float) -> dict:
    """异常模式检测"""
    texts = [s["text"].strip() for s in segments]

    # 1. 重复短语检测 (trigram 级别)
    all_lower = " ".join(t.lower() for t in texts)
    words = all_lower.split()
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]

    repeated = {}
    if trigrams:
        tri_counter = Counter(trigrams)
        repeated = {k: v for k, v in tri_counter.items() if v >= 5 and len(k.split()) == 3}

    # 2. 长单词语段 (可能是噪音)
    long_word_segs = sum(1 for t in texts if len(t) > 50 and " " not in t)

    # 3. 特殊字符过多
    special_char_segs = sum(1 for t in texts if re.search(r'[^\w\s.,!?;\'\"-]', t))

    # 4. 过短的段 (< 0.5s 却有内容)
    ultrashort_with_text = sum(1 for s in segments
                               if s["end"] - s["start"] < 0.5 and len(s["text"].strip()) > 0)

    # 5. 过长的段 (> 30s 无说话人切换)
    long_no_change = sum(1 for s in segments if s["end"] - s["start"] > 30)

    # 6. 末尾异常 (末尾几段非常短或有特殊标记)
    tail_segs = texts[-5:] if len(texts) >= 5 else texts
    tail_caption = sum(1 for t in tail_segs if any(p.match(t) for p in CAPTION_PATTERNS))

    flags = []
    anomaly_count = 0

    if repeated:
        flags.append(f"WARN:repeated_trigrams:{len(repeated)}")
        anomaly_count += 1

    if long_word_segs > 3:
        flags.append("WARN:long_word_segments")
        anomaly_count += 1

    if special_char_segs > 5:
        flags.append("WARN:special_characters")
        anomaly_count += 1

    if ultrashort_with_text > 10:
        flags.append("WARN:ultrashort_with_text")
        anomaly_count += 1

    if long_no_change > 3:
        flags.append("WARN:long_segments_no_change")
        anomaly_count += 1

    if tail_caption >= 3:
        flags.append("INFO:trailing_captions")

    score = max(0, 100 - anomaly_count * 15)

    return {
        "repeated_trigrams": len(repeated),
        "repeated_trigram_examples": list(repeated.keys())[:5] if repeated else [],
        "long_word_segments": long_word_segs,
        "special_character_segments": special_char_segs,
        "ultrashort_with_text": ultrashort_with_text,
        "long_segments_no_change": long_no_change,
        "trailing_captions": tail_caption,
        "score": score,
        "flags": flags,
    }


# ═══════════════════════════════════════════
#  综合评分
# ═══════════════════════════════════════════

def compute_overall(text_q: dict, diar_q: dict, temp_q: dict, anom_q: dict) -> dict:
    """计算综合评分"""
    weights = {
        "text_quality": 0.30,
        "diarization_stability": 0.35,
        "temporal_quality": 0.20,
        "anomaly_detection": 0.15,
    }

    overall = (text_q["score"] * weights["text_quality"] +
               diar_q["score"] * weights["diarization_stability"] +
               temp_q["score"] * weights["temporal_quality"] +
               anom_q["score"] * weights["anomaly_detection"])

    all_flags = (text_q.get("flags", []) + diar_q.get("flags", []) +
                 temp_q.get("flags", []) + anom_q.get("flags", []))
    fails = [f for f in all_flags if f.startswith("FAIL")]
    warnings = [f for f in all_flags if f.startswith("WARN")]

    if overall >= 80:
        grade = "GOOD"
    elif overall >= 60:
        grade = "FAIR"
    elif overall >= 40:
        grade = "POOR"
    else:
        grade = "BAD"

    return {
        "overall_score": round(overall, 1),
        "grade": grade,
        "fail_count": len(fails),
        "warning_count": len(warnings),
        "fails": fails,
        "warnings": warnings,
        "weights": weights,
    }


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════

def analyze_one_file(json_path: str) -> dict:
    """分析单个转写结果文件"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return {"file": str(json_path), "error": str(e), "_skip": True}

    # 跳过非转写结果的 JSON（如 quality_report.json）
    if "segments" not in data:
        return {"file": str(json_path), "error": "not a transcription JSON (no 'segments' key)", "_skip": True}

    segments = data.get("segments", [])
    duration = data.get("duration", 0)
    filename = data.get("filename", Path(json_path).stem)

    text_q = check_text_quality(segments, duration)
    diar_q = check_diarization_stability(segments)
    temp_q = check_temporal_quality(segments, duration)
    anom_q = check_anomalies(segments, duration)
    overall = compute_overall(text_q, diar_q, temp_q, anom_q)

    return {
        "file": filename,
        "language": data.get("language", "?"),
        "num_segments": len(segments),
        "overall": overall,
        "text_quality": text_q,
        "diarization": diar_q,
        "temporal": temp_q,
        "anomaly": anom_q,
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        print(f"❌ Output directory not found: {OUTPUT_DIR}")
        sys.exit(1)

    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        print("❌ No JSON files found")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  转写质量检查报告")
    print(f"  共 {len(json_files)} 个文件")
    print(f"{'='*70}\n")

    results = []
    grades_counter = Counter()
    fail_files = []

    for jf in json_files:
        result = analyze_one_file(str(jf))
        if result.get("_skip"):
            filename = Path(str(jf)).stem
            print(f"⏭️ Skipping {filename}: {result.get('error', 'unknown')}")
            continue
        results.append(result)

        ov = result["overall"]
        grades_counter[ov["grade"]] += 1
        if ov["fail_count"] > 0:
            fail_files.append((result["file"], ov["fails"]))

        if VERBOSE:
            print(f"📄 {result['file']}")
            print(f"   Grade: {ov['grade']:>4s}  Score: {ov['overall_score']:>5.1f}")
            print(f"   Language: {result['language']}  "
                  f"Segments: {result['num_segments']}  "
                  f"Duration: {result['temporal']['duration_seconds']:.0f}s")
            print(f"   Text: {result['text_quality']['score']:.0f}  "
                  f"Diar: {result['diarization']['score']:.0f}  "
                  f"Temp: {result['temporal']['score']:.0f}  "
                  f"Anom: {result['anomaly']['score']:.0f}")

            # 详情 (flags 和关键指标)
            details = []
            sp_count = result["diarization"]["speaker_count"]
            details.append(f"speakers={sp_count}")

            wps = result["text_quality"]["words_per_second"]
            details.append(f"wps={wps}")

            cov = result["temporal"]["coverage_ratio"]
            details.append(f"coverage={cov:.1%}")

            short_seg = result["text_quality"]["short_segment_ratio"]
            details.append(f"short_seg={short_seg:.1%}")

            filler = result["text_quality"]["filler_word_ratio"]
            details.append(f"filler={filler:.1%}")

            unknown = result["diarization"]["unknown_ratio"]
            details.append(f"unknown={unknown:.1%}")

            print(f"   {' | '.join(details)}")

            all_flags = (result["text_quality"].get("flags", []) +
                         result["diarization"].get("flags", []) +
                         result["temporal"].get("flags", []) +
                         result["anomaly"].get("flags", []))
            for fl in all_flags:
                print(f"   ⚑ {fl}")
            print()

    # ── 汇总统计 ──
    print(f"\n{'='*70}")
    print(f"  汇总统计")
    print(f"{'='*70}")
    print(f"    文件总数: {len(results)}")

    scores = [r["overall"]["overall_score"] for r in results]
    print(f"    综合评分: 均值={np.mean(scores):.1f}  中位数={np.median(scores):.1f}  "
          f"最低={min(scores):.1f}  最高={max(scores):.1f}")

    print(f"\n    等级分布:")
    for grade in ["GOOD", "FAIR", "POOR", "BAD"]:
        count = grades_counter.get(grade, 0)
        pct = count / len(results) * 100 if results else 0
        bar = "█" * count + "░" * max(0, len(results) - count)
        print(f"      {grade:>4s}: {count:>3d} ({pct:>5.1f}%)")

    if fail_files:
        print(f"\n    ⚠️ 需关注的文件 ({len(fail_files)}):")
        for fname, fails in fail_files[:10]:
            print(f"      {fname}: {'; '.join(fails)}")
        if len(fail_files) > 10:
            print(f"      ... 还有 {len(fail_files) - 10} 个")

    # ── 各维度平均分 ──
    print(f"\n    各维度平均分:")
    for dim in ["text_quality", "diarization", "temporal", "anomaly"]:
        dim_scores = [r[dim]["score"] for r in results]
        print(f"      {dim:>20s}: {np.mean(dim_scores):.1f}")

    # ── 保存报告 ──
    with open(REPORT_FILE, "w") as f:
        json.dump({
            "total_files": len(results),
            "summary": {
                "mean_score": round(np.mean(scores), 1),
                "median_score": round(np.median(scores), 1),
                "min_score": round(min(scores), 1),
                "max_score": round(max(scores), 1),
                "grade_distribution": dict(grades_counter),
                "dimension_averages": {
                    "text_quality": round(np.mean([r["text_quality"]["score"] for r in results]), 1),
                    "diarization": round(np.mean([r["diarization"]["score"] for r in results]), 1),
                    "temporal": round(np.mean([r["temporal"]["score"] for r in results]), 1),
                    "anomaly": round(np.mean([r["anomaly"]["score"] for r in results]), 1),
                },
            },
            "files": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n    报告已保存: {REPORT_FILE}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
