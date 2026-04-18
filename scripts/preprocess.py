"""
preprocess.py
-------------
Step 1 of 2 in the data pipeline.

Loads all raw label CSV files from data/raw/, standardises column names,
cleans and validates each row, removes duplicates, and saves the unified
dataset to data/elder_qa_master.csv.

Usage:
    python scripts/preprocess.py

Then run Step 2:
    python scripts/build_index.py

──────────────────────────────────────────────
Expected raw files in data/raw/
──────────────────────────────────────────────
  Label3_RudrangGade.csv                  → Timing
  Elder_Psychology.csv                    → Psychology
  Elder_Personal_life_2.csv              → Personal Life
  Elder_Label4_RiskManagement_200QA.csv  → Risk Management
  Elder_Adaptability.csv                  → Adaptability  (add when available)
──────────────────────────────────────────────
"""

import os
import re
import sys
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "elder_qa_master.csv")

# ── File registry ──────────────────────────────────────────────────────────
# (filename, label, question_col, answer_col)
FILE_REGISTRY = [
    ("Label3_RudrangGade.csv",               "Timing",           "Questions", "Answers"),
    ("Elder_Psychology.csv",                  "Psychology",       "Questions", "Answers"),
    ("Elder_Personal_life_2.csv",             "Personal Life",    "Question",  "Answer"),
    ("Elder_Label4_RiskManagement_200QA.csv", "Risk Management",  "Questions", "Answers"),
    ("Elder_Adaptability.csv",              "Adaptability",     "Question",  "Answer"),
]

# ── Quality thresholds ──────────────────────────────────────────────────────
MIN_QUESTION_WORDS = 3    # drop questions shorter than this
MIN_ANSWER_WORDS   = 5    # drop answers shorter than this
MAX_ANSWER_WORDS   = 300  # drop answers longer than this (likely parsing errors)


# ── Text cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise a single text field:
      - Strip leading / trailing whitespace
      - Collapse internal whitespace (newlines, tabs, multiple spaces)
      - Remove non-printable / non-ASCII characters
      - Fix common encoding artefacts (curly quotes → straight)
    """
    if not isinstance(text, str):
        return ""

    # Fix curly quotes and common encoding artefacts
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Collapse whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E]", "", text)

    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


# ── Loader ─────────────────────────────────────────────────────────────────

def load_file(filename: str, label: str, q_col: str, a_col: str) -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        print(f"  [SKIP]  {filename:<48} — not found in data/raw/")
        return None

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception as e:
            print(f"  [ERROR] {filename}: could not read — {e}")
            return None
    except Exception as e:
        print(f"  [ERROR] {filename}: {e}")
        return None

    # Validate expected columns
    if q_col not in df.columns or a_col not in df.columns:
        print(f"  [ERROR] {filename}: expected columns '{q_col}' / '{a_col}', "
              f"found {list(df.columns)}")
        return None

    df = df[[q_col, a_col]].copy()
    df.columns = ["question", "answer"]
    df["label"] = label
    return df


# ── Validation & cleaning pipeline ─────────────────────────────────────────

def clean_and_validate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    original = len(df)
    log = []

    # 1. Apply text cleaning
    df["question"] = df["question"].apply(clean_text)
    df["answer"]   = df["answer"].apply(clean_text)

    # 2. Drop rows that look like header rows accidentally included as data
    header_mask = (df["question"].str.lower() == "question") | \
                  (df["answer"].str.lower()   == "answer")
    if header_mask.sum():
        log.append(f"{header_mask.sum()} header row(s)")
    df = df[~header_mask]

    # 3. Drop empty rows
    empty_mask = (df["question"] == "") | (df["answer"] == "")
    if empty_mask.sum():
        log.append(f"{empty_mask.sum()} empty row(s)")
    df = df[~empty_mask]

    # 4. Drop rows with questions that are too short
    short_q = df["question"].apply(word_count) < MIN_QUESTION_WORDS
    if short_q.sum():
        log.append(f"{short_q.sum()} question(s) < {MIN_QUESTION_WORDS} words")
    df = df[~short_q]

    # 5. Drop rows with answers that are too short or too long
    ans_wc = df["answer"].apply(word_count)
    bad_ans = (ans_wc < MIN_ANSWER_WORDS) | (ans_wc > MAX_ANSWER_WORDS)
    if bad_ans.sum():
        log.append(f"{bad_ans.sum()} answer(s) outside word-count range "
                   f"[{MIN_ANSWER_WORDS}–{MAX_ANSWER_WORDS}]")
    df = df[~bad_ans]

    kept    = len(df)
    dropped = original - kept
    suffix  = f" (dropped {dropped}: {', '.join(log)})" if log else ""
    print(f"  [OK]    {source:<48} → {kept:>3} rows{suffix}")
    return df.reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────

def preprocess():
    os.makedirs(RAW_DIR, exist_ok=True)

    all_dfs  = []
    skipped  = []

    for filename, label, q_col, a_col in FILE_REGISTRY:
        raw = load_file(filename, label, q_col, a_col)
        if raw is None:
            skipped.append(filename)
            continue
        cleaned = clean_and_validate(raw, filename)
        if len(cleaned):
            all_dfs.append(cleaned)

    if not all_dfs:
        print("\n[ERROR] No data loaded. Place your CSV files in data/raw/ and retry.")
        sys.exit(1)

    # ── Merge ──────────────────────────────────────────────────────────────
    master = pd.concat(all_dfs, ignore_index=True)
    master = master[["question", "answer", "label"]]

    # ── Deduplication ──────────────────────────────────────────────────────
    before_dedup = len(master)
    master = master.drop_duplicates(subset=["question", "answer"])
    dupes_removed = before_dedup - len(master)

    # ── Save ───────────────────────────────────────────────────────────────
    master.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # ── Summary report ─────────────────────────────────────────────────────
    bar = "=" * 58
    print(f"\n{bar}")
    print(f"  Preprocessing complete")
    print(f"{bar}")
    print(f"  Total rows saved  : {len(master)}")
    if dupes_removed:
        print(f"  Duplicates removed: {dupes_removed}")
    print(f"\n  Rows per label:")
    for label, count in master["label"].value_counts().items():
        pct = count / len(master) * 100
        bar_fill = "█" * int(pct / 3)
        print(f"    {label:<25} {count:>3} rows  {bar_fill}")
    print(f"\n  Output: {OUTPUT_CSV}")
    if skipped:
        print(f"\n  Skipped ({len(skipped)} files not found in data/raw/):")
        for f in skipped:
            print(f"    · {f}")
        print("  Add them to data/raw/ and re-run to include.")
    print(f"\n  Next → python scripts/build_index.py")
    print(f"{bar}\n")


if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  Step 1/2 — Data preprocessing")
    print("=" * 58 + "\n")
    preprocess()
