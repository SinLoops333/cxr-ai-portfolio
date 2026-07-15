"""Compute AUROC and 95% bootstrap CIs for both regimes and report the lift.

Loads the two ``*_probs_test.npz`` produced by ``train_downstream.py``, computes
macro + per-class AUROC with bootstrap CIs, and writes a JSON report highlighting
the AUROC delta on rare classes (defined by prevalence < ``rare_threshold``).

Usage:
    python project_b_synth/eval/utility_study.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.logging_utils import get_logger
from common.metrics import bootstrap_ci
from common.paths import ARTIFACTS

log = get_logger("synth.utility")


def _auroc_per_class(y_true: np.ndarray, y_score: np.ndarray, cls_idx: int, n_boot: int) -> dict:
    from sklearn.metrics import roc_auc_score

    def m(yt, ys, idx=cls_idx):
        if len(np.unique(yt[:, idx])) < 2:
            raise ValueError
        return float(roc_auc_score(yt[:, idx], ys[:, idx]))

    p, lo, hi = bootstrap_ci(m, y_true, y_score, n_boot=n_boot)
    return {"auroc": p, "ci_lo": lo, "ci_hi": hi}


def _macro_auroc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int) -> dict:
    from sklearn.metrics import roc_auc_score

    def m(yt, ys):
        aucs = []
        for i in range(yt.shape[1]):
            if len(np.unique(yt[:, i])) < 2:
                continue
            aucs.append(roc_auc_score(yt[:, i], ys[:, i]))
        return float(np.mean(aucs)) if aucs else float("nan")

    p, lo, hi = bootstrap_ci(m, y_true, y_score, n_boot=n_boot)
    return {"auroc": p, "ci_lo": lo, "ci_hi": hi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, default=ARTIFACTS / "downstream" / "real_probs_test.npz")
    ap.add_argument("--real-synth", type=Path, default=ARTIFACTS / "downstream" / "real_synth_probs_test.npz")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--rare-threshold", type=float, default=0.05, help="Prevalence below this is 'rare'.")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "utility_report.json")
    args = ap.parse_args()

    real = np.load(args.real, allow_pickle=True)
    real_synth = np.load(args.real_synth, allow_pickle=True)
    y = real["y_true"]
    classes = list(real["class_names"])
    assert np.array_equal(y, real_synth["y_true"]), "test-set labels must match across regimes"

    prevalence = y.mean(0)
    rare_idx = [i for i, p in enumerate(prevalence) if p < args.rare_threshold]

    report = {"n_test": int(len(y)), "class_names": classes, "prevalence": prevalence.tolist(), "rare_classes": [classes[i] for i in rare_idx]}
    for name, arr in [("real", real["y_score"]), ("real_synth", real_synth["y_score"])]:
        macro = _macro_auroc(y, arr, args.n_boot)
        per = {classes[i]: _auroc_per_class(y, arr, i, args.n_boot) for i in range(y.shape[1]) if len(np.unique(y[:, i])) >= 2}
        rare_scores = [per[classes[i]]["auroc"] for i in rare_idx if classes[i] in per]
        report[name] = {"macro": macro, "per_class": per, "rare_macro": float(np.mean(rare_scores)) if rare_scores else None}

    if report["real"]["macro"]["auroc"] is not None and report["real_synth"]["macro"]["auroc"] is not None:
        report["macro_lift"] = report["real_synth"]["macro"]["auroc"] - report["real"]["macro"]["auroc"]
    if report["real"].get("rare_macro") is not None and report["real_synth"].get("rare_macro") is not None:
        report["rare_macro_lift"] = report["real_synth"]["rare_macro"] - report["real"]["rare_macro"]

    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)
    log.info("macro AUROC: real=%.4f real+synth=%.4f (lift %+.4f)",
             report["real"]["macro"]["auroc"], report["real_synth"]["macro"]["auroc"], report.get("macro_lift", 0.0))


if __name__ == "__main__":
    main()
