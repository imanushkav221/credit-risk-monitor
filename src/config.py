"""Paths and constants shared by the pipeline and the dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Snapshot date for the data pack. The latest date in any table is a bureau
# line opened on 2025-12-29, so the extract is treated as taken at 2025-12-31.
AS_OF_DATE = "2025-12-31"

RANDOM_STATE = 42

# Income bands use fixed edges (roughly the portfolio quintiles) so that the
# labels stay stable when new data arrives.
INCOME_BINS = [0, 58_000, 73_000, 86_000, 101_000, float("inf")]
INCOME_LABELS = ["<58k", "58-73k", "73-86k", "86-101k", "101k+"]

AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["21-25", "26-35", "36-45", "46-55", "56+"]

# Risk bands on the model probability of default (PD).
PD_BINS = [0, 0.08, 0.13, 0.20, 0.30, 1.0001]
PD_LABELS = ["A (<8%)", "B (8-13%)", "C (13-20%)", "D (20-30%)", "E (30%+)"]
