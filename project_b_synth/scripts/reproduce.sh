#!/usr/bin/env bash
# One-command reproduction of Flagship B's utility study.
# Assumes data has been downloaded (see data/scripts/*.py) and LoRA has been trained.
set -euo pipefail

LORA="${LORA:-checkpoints/sd_lora_openi/final/unet_lora}"
REAL_DIR="${REAL_DIR:-data/raw/openi}"
SYNTH_DIR="${SYNTH_DIR:-artifacts/synth_images}"

echo "[1/5] Generate balanced synthetic set"
python -m project_b_synth.scripts.generate --lora "$LORA" --per-class 200 --out "$SYNTH_DIR"

echo "[2/5] Quality gate"
python -m project_b_synth.scripts.quality_gate --synth-dir "$SYNTH_DIR" --real-dir "$REAL_DIR" --threshold 0.4

echo "[3/5] Memorization audit"
python -m project_b_synth.scripts.memorization_audit \
    --train-images "$REAL_DIR" --holdout-csv data/processed/openi_splits.csv --synth-dir "$SYNTH_DIR"

echo "[4/5] Train downstream classifiers (real / real+synth)"
python -m project_b_synth.scripts.train_downstream --regime real
python -m project_b_synth.scripts.train_downstream --regime real_synth --kept-csv "$SYNTH_DIR/kept.csv"

echo "[5/5] Utility study"
python -m project_b_synth.eval.utility_study

echo "Done. See artifacts/utility_report.json, artifacts/memorization.json, artifacts/quality_report.json."
