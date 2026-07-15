"""Central path constants (single source of truth for data/artifact locations)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_KB = ROOT / "data" / "kb"

OPENI_DIR = DATA_RAW / "openi"
# Official tarball extracts PNGs flat into OPENI_DIR (not a NLMCXR_png/ subfolder).
OPENI_IMAGES = OPENI_DIR
OPENI_REPORTS = OPENI_DIR / "ecgen-radiology"
OPENI_CAPTIONS = DATA_PROC / "openi_captions.csv"

NIH14_DIR = DATA_RAW / "nih14"
NIH14_IMAGES = NIH14_DIR / "images"
NIH14_LABELS = NIH14_DIR / "Data_Entry_2017_v2020.csv"

CHECKPOINTS = ROOT / "checkpoints"
ARTIFACTS = ROOT / "artifacts"

for p in (DATA_RAW, DATA_PROC, DATA_KB, CHECKPOINTS, ARTIFACTS):
    p.mkdir(parents=True, exist_ok=True)
