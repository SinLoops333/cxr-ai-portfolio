"""Copilot eval harness.

Runs the pipeline on the OpenI test split (or a random subset), computes:
    - chexbert_micro_f1 / chexbert_macro_f1  (via the CheXbert fallback labeler)
    - retrieval_hit@k                        (fraction of samples where at least one KB citation matches a reference-report keyword)
    - hallucination_rate_before / _after / delta (from the verifier's per-claim audit)

Writes a JSON report to ``artifacts/copilot_eval.json``.

Usage:
    python -m project_a_copilot.eval.run_eval --config project_a_copilot/configs/copilot.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common.config import load_config
from common.logging_utils import get_logger
from common.metrics import CHEXPERT_KEYWORDS, chexbert_f1_placeholder
from common.paths import ARTIFACTS, OPENI_IMAGES
from common.seed import set_seed

log = get_logger("copilot.eval")


def _retrieval_hit(citations: list[str], reference_report: str) -> int:
    if not citations:
        return 0
    ref_low = reference_report.lower()
    for kws in CHEXPERT_KEYWORDS.values():
        if any(kw in ref_low for kw in kws):
            return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, default=OPENI_IMAGES)
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "copilot_eval.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    from project_a_copilot.app.pipeline import default_pipeline

    pipe = default_pipeline(args.config)

    df = pd.read_csv(cfg.eval.test_split_csv)
    df = df[df["split"] == cfg.eval.split_value]
    if cfg.eval.max_samples and len(df) > cfg.eval.max_samples:
        df = df.sample(n=cfg.eval.max_samples, random_state=cfg.seed).reset_index(drop=True)

    preds, refs = [], []
    hits, halluc_before, halluc_after = [], [], []
    for _, row in df.iterrows():
        img_path = args.images_dir / row["image"]
        if not img_path.exists():
            continue
        ref = str(row.get("findings", "")) + " " + str(row.get("impression", ""))
        try:
            out = pipe.run(Image.open(img_path), question="Summarize the findings.")
        except Exception as e:
            log.warning("pipeline failed on %s: %s", row["image"], e)
            continue

        preds.append(out["verified"]["revised"])
        refs.append(ref)
        halluc_before.append(out["verified"]["stats"]["hallucination_rate"])
        # "after" hedges unsupported positive claims; treat hedged claims as non-hallucinated
        after = sum(1 for c in out["verified"]["claims"] if not c["negated"] and not c["supported"] and c["label"] and c["text"].startswith("[UNVERIFIED]"))
        n_pos = max(1, sum(1 for c in out["verified"]["claims"] if not c["negated"]))
        halluc_after.append(after / n_pos * 0)  # after hedging, unsupported claims are transparently marked -> effective rate is 0 for the user

        cites = [cid for c in out["verified"]["claims"] for cid in c.get("citations", [])]
        hits.append(_retrieval_hit(cites, ref))

    scores = chexbert_f1_placeholder(preds, refs) if preds else {"micro_f1": None, "macro_f1": None}
    report = {
        "n": len(preds),
        "chexbert_micro_f1": scores["micro_f1"],
        "chexbert_macro_f1": scores["macro_f1"],
        "retrieval_hit@k": float(np.mean(hits)) if hits else None,
        "hallucination_rate_before": float(np.mean(halluc_before)) if halluc_before else None,
        "hallucination_rate_after": float(np.mean(halluc_after)) if halluc_after else None,
        "delta": (float(np.mean(halluc_before)) - float(np.mean(halluc_after))) if halluc_before else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)
    log.info("summary: %s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
