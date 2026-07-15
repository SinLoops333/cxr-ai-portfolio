"""Shared eval metrics for the CXR portfolio.

- ``multilabel_auroc``: macro/per-class AUROC for pathology classifiers.
- ``bootstrap_ci``: nonparametric bootstrap confidence interval for any scalar metric.
- ``chexbert_f1_placeholder``: hook for CheXbert-labeler-based F1 (loaded lazily; falls
  back to a naive keyword overlap when unavailable so the eval harness always runs).

The CheXbert labeler weights are not included; when the model isn't installed we use
a keyword-based fallback so results are still reproducible offline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CHEXPERT_KEYWORDS: dict[str, list[str]] = {
    "atelectasis": ["atelectasis", "atelectatic"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
    "consolidation": ["consolidation", "consolidative"],
    "edema": ["edema", "pulmonary edema"],
    "effusion": ["effusion", "pleural effusion"],
    "emphysema": ["emphysema"],
    "fibrosis": ["fibrosis", "fibrotic"],
    "hernia": ["hernia"],
    "infiltration": ["infiltrate", "infiltration"],
    "mass": ["mass"],
    "nodule": ["nodule", "nodular"],
    "pleural_thickening": ["pleural thickening"],
    "pneumonia": ["pneumonia"],
    "pneumothorax": ["pneumothorax"],
}


def multilabel_auroc(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, dict[int, float]]:
    """Macro AUROC + per-class AUROC. Skips classes with only one label present."""
    from sklearn.metrics import roc_auc_score

    per_class: dict[int, float] = {}
    for i in range(y_true.shape[1]):
        y = y_true[:, i]
        if len(np.unique(y)) < 2:
            continue
        per_class[i] = float(roc_auc_score(y, y_score[:, i]))
    macro = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return macro, per_class


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) using nonparametric bootstrap over samples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            stats.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    point = metric_fn(y_true, y_score)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def keyword_labels(text: str, labels: Sequence[str] = tuple(CHEXPERT_KEYWORDS)) -> np.ndarray:
    """Fallback labeler: 1 if any keyword for the label appears in text (lowercased)."""
    t = text.lower()
    out = np.zeros(len(labels), dtype=np.int32)
    for i, lab in enumerate(labels):
        for kw in CHEXPERT_KEYWORDS.get(lab, [lab]):
            if kw in t:
                out[i] = 1
                break
    return out


def chexbert_f1_placeholder(
    pred_reports: Sequence[str], ref_reports: Sequence[str]
) -> dict[str, float]:
    """Micro/macro F1 over 14 CheXpert-style labels using the keyword fallback.

    Swap this for the real CheXbert labeler when available; the interface is stable.
    """
    from sklearn.metrics import f1_score

    y_pred = np.stack([keyword_labels(t) for t in pred_reports])
    y_true = np.stack([keyword_labels(t) for t in ref_reports])
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
