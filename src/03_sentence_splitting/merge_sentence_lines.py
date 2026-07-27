#!/usr/bin/env python3
"""
Merge broken sentence lines in audio_diarized/ files.

Pattern detected: many files have natural sentence breaks at commas or
prepositions that were split across lines during ASR export:
  "Well, I'm in the hands of a master class interviewer,"
  "so I hope it works out that way."

Rules:
  - Line ends with . ! ?        → keep as-is (sentence boundary)
  - Line ends with , or no punct
    AND next line starts with lowercase → merge with space
  - Empty lines are dropped
  - Common abbreviations (Mr., Dr., U.S., etc.) are NOT treated as boundaries

Usage:
  python3 src/utils/merge_sentence_lines.py              # process all
  python3 src/utils/merge_sentence_lines.py --dry-run     # preview only
  python3 src/utils/merge_sentence_lines.py --files <vid1> <vid2>  # specific files
"""

import os, re, sys

DIARIZED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "audio_diarized")
)

# Common abbreviations that end with period but are NOT sentence boundaries
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "ave.", "blvd.",
    "dept.", "co.", "inc.", "ltd.", "corp.", "gen.", "sgt.", "capt.", "lt.",
    "gov.", "sen.", "rep.", "hon.", "rev.", "fr.", "jr.", "sr.", "esq.",
    "etc.", "vs.", "a.m.", "p.m.", "u.s.", "u.k.", "e.u.",
}

SENTENCE_END = re.compile(r'[.!?…]$')
NEXT_LOWERCASE = re.compile(r'^[a-z"\'‘“(]')
COMMA_END = re.compile(r',$')
DASH_END = re.compile(r'[-–—]+$')
ELLIPSIS_END = re.compile(r'\.{2,}$')
QUOTE_END = re.compile(r'[.!?…][\'"’”)]?$')

def is_sentence_boundary(line: str) -> bool:
    """Check if a line ends with sentence-final punctuation."""
    line = line.rstrip()
    if not line:
        return True  # empty line = boundary
    # Check if it's an abbreviation (not a sentence end)
    lower = line.strip().lower()
    for abbr in ABBREVIATIONS:
        if lower.endswith(abbr):
            # Make sure it's not at the very end of actual sentence content
            # E.g. "Dr." in "Dr. Smith said." — the "." after "said" IS a boundary
            # "I saw Dr. Smith." → ends with "Smith." not "dr."
            return False
    # Check for end-of-sentence punctuation
    # Match . ! ? optionally followed by closing quote/paren
    return bool(QUOTE_END.search(line.rstrip()))


def starts_lowercase(line: str) -> bool:
    """Check if the next line starts with lowercase (likely continuation)."""
    line = line.lstrip()
    if not line:
        return False
    return bool(NEXT_LOWERCASE.match(line))


def should_merge(current: str, next_line: str) -> bool:
    """Determine if current line should merge with next line."""
    if not next_line:
        return False
    next_line = next_line.strip()
    if not next_line:
        return False
    
    # If current line is very short (1-2 words) and next line is a continuation
    # This catches cases like "everything.\neverything. And yet..."
    # But "everything." alone is hard to distinguish from a real sentence
    
    # Core rule: current line ends a sentence → keep separate
    if is_sentence_boundary(current):
        # Exception: next line's first word starts lowercase → fragment
        # e.g. "...decades.\nright?" "...paramilitary.\nfighting..."
        if starts_lowercase(next_line):
            pass  # merge fragment into current line
        else:
            return False
    
    # If current line ends with comma or dash → always merge
    if COMMA_END.search(current.rstrip()):
        return True
    if DASH_END.search(current.rstrip()):
        return True
    
    # If current line ends with a function word (article, preposition, etc.)
    # e.g. "threat to the\nUnited States" — "the" at line end
    last_word = current.rstrip().split()[-1].lower().strip('.,!?;:\'"()[]-') if current.rstrip().split() else ''
    FUNCTION_WORDS = {'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by',
                      'that', 'this', 'these', 'those', 'which', 'who', 'whom', 'whose',
                      'and', 'or', 'but', 'nor', 'yet', 'so', 'if', 'as', 'than',
                      'from', 'into', 'through', 'during', 'before', 'after',
                      'is', 'are', 'was', 'were', 'be', 'been', 'being',
                      'has', 'have', 'had', 'do', 'does', 'did',
                      'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might'}
    if last_word in FUNCTION_WORDS:
        return True
    
    # If next line starts with lowercase → merge
    if starts_lowercase(next_line):
        return True
    
    # If current line is very short (< 4 words) and doesn't end with punct
    words = current.split()
    if len(words) <= 3 and not SENTENCE_END.search(current.rstrip()):
        return True
    
    return False


def merge_file(filepath: str, dry_run: bool = False) -> tuple:
    """Merge broken lines in a file. Returns (original_lines, merged_lines, changes)."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    orig_count = len(lines)
    merged = []
    
    i = 0
    while i < len(lines):
        current = lines[i].rstrip()
        
        # Skip empty lines
        if not current.strip():
            i += 1
            if not dry_run:
                merged.append('')
            continue
        
        # Collect continuation lines
        j = i + 1
        joined = current
        while j < len(lines):
            next_line = lines[j].rstrip()
            if not next_line.strip():
                j += 1
                continue  # skip empty lines between fragments
            if should_merge(joined, next_line):
                # Join with appropriate spacing
                if joined.endswith('-') or joined.endswith('–') or joined.endswith('—'):
                    joined = joined + next_line  # no space after hyphen
                elif joined and joined[-1] in ',;:':
                    joined = joined + ' ' + next_line
                else:
                    joined = joined + ' ' + next_line
                j += 1
            else:
                break
        
        merged.append(joined)
        i = j
    
    # Count changes
    changes = sum(1 for orig, new in zip(
        [l.rstrip() for l in lines if l.strip()],
        [l for l in merged if l.strip()]
    ) if orig != new)
    
    return orig_count, len(merged), changes, merged


def main():
    dry_run = '--dry-run' in sys.argv
    
    # Get specific files if requested
    if '--files' in sys.argv:
        idx = sys.argv.index('--files')
        specific = sys.argv[idx + 1:]
        file_list = []
        for vid in specific:
            path = os.path.join(DIARIZED_DIR, f"{vid}.txt")
            if os.path.exists(path):
                file_list.append((vid, path))
            else:
                print(f"⚠  Not found: {vid}")
    else:
        file_list = sorted([
            (f.replace('.txt', ''), os.path.join(DIARIZED_DIR, f))
            for f in os.listdir(DIARIZED_DIR) if f.endswith('.txt')
        ])
    
    print(f"Audio dir: {DIARIZED_DIR}")
    print(f"Files: {len(file_list)}")
    print(f"Mode: {'DRY RUN (no write)' if dry_run else 'WRITE'}")
    print()
    
    total_fixed = 0
    total_merged_lines = 0
    
    for vid, path in file_list:
        orig_count, merged_count, changes, merged_lines = merge_file(path, dry_run)
        
        if changes > 0:
            total_fixed += 1
            total_merged_lines += changes
            diff = orig_count - merged_count
            print(f"  {vid:<30} {orig_count:>5}→{merged_count:>5} lines  ({changes} changes, {diff} fewer lines)")
            
            if not dry_run:
                # Write back
                with open(path, 'w', encoding='utf-8') as f:
                    for l in merged_lines:
                        if l:
                            f.write(l + '\n')
                        else:
                            f.write('\n')
    
    print(f"\n{'='*60}")
    print(f"Summary: {total_fixed}/{len(file_list)} files modified")
    print(f"Total line merges: {total_merged_lines}")
    if dry_run:
        print(f"Run without --dry-run to apply changes.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
