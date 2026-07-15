"""Basic sanity tests for shared utilities."""

from __future__ import annotations

import numpy as np
import pytest

from common.config import load_config
from common.metrics import (
    bootstrap_ci,
    chexbert_f1_placeholder,
    keyword_labels,
    multilabel_auroc,
)
from common.seed import set_seed


def test_set_seed_is_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("x: 1\nnested:\n  y: 2\n")
    cfg = load_config(p)
    assert cfg["x"] == 1
    assert cfg.nested.y == 2


def test_multilabel_auroc_perfect():
    y_true = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_score = y_true.astype(float)
    macro, per_class = multilabel_auroc(y_true, y_score)
    assert macro == pytest.approx(1.0)
    assert set(per_class) == {0, 1}


def test_bootstrap_ci_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(100, 1))
    s = rng.random((100, 1))

    def m(yt, ys):
        return multilabel_auroc(yt, ys)[0]

    point, lo, hi = bootstrap_ci(m, y, s, n_boot=50, seed=0)
    assert lo <= point <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_keyword_labels_detects():
    v = keyword_labels("Findings: mild cardiomegaly and small pleural effusion.")
    assert v.sum() >= 2


def test_chexbert_placeholder_perfect():
    txts = [
        "cardiomegaly present",
        "no acute findings",
        "pneumothorax on the right side",
    ]
    scores = chexbert_f1_placeholder(txts, txts)
    assert scores["micro_f1"] == pytest.approx(1.0)
    assert scores["macro_f1"] > 0
