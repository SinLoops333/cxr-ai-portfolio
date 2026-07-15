"""Deterministic train/val/test split for the OpenI captions CSV.

Grouped by ``report_id`` so multiple images from the same study never leak across splits.

Usage:
    python data/scripts/build_openi_splits.py --train 0.8 --val 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.logging_utils import get_logger
from common.paths import DATA_PROC, OPENI_CAPTIONS
from common.seed import set_seed

log = get_logger("data.openi.splits")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", type=Path, default=OPENI_CAPTIONS)
    ap.add_argument("--out-dir", type=Path, default=DATA_PROC)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.captions)
    reports = df["report_id"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(reports)

    n = len(reports)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train_ids = set(reports[:n_train])
    val_ids = set(reports[n_train : n_train + n_val])

    def _mark(rid: str) -> str:
        if rid in train_ids:
            return "train"
        if rid in val_ids:
            return "val"
        return "test"

    df["split"] = df["report_id"].map(_mark)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "openi_splits.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s: train=%d val=%d test=%d",
             out,
             (df.split == "train").sum(),
             (df.split == "val").sum(),
             (df.split == "test").sum())


if __name__ == "__main__":
    main()
