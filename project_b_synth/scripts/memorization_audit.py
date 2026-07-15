"""Memorization audit for synthetic CXRs.

For each synthetic image, compute its top-1 cosine similarity to the *training* set
in CLIP (ViT-B/32) embedding space and compare the distribution against a real
holdout->train baseline. If the synth distribution has substantially higher top-1
than the holdout baseline, the model is memorizing.

Outputs:
    artifacts/memorization.json  — histograms + summary stats
    artifacts/memorization_topk.csv  — nearest-neighbor pairs for the worst offenders

Usage:
    python project_b_synth/scripts/memorization_audit.py \
        --train-images data/raw/openi \
        --holdout-csv data/processed/openi_splits.csv \
        --synth-dir artifacts/synth_images
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common.logging_utils import get_logger
from common.paths import ARTIFACTS

log = get_logger("synth.memaudit")


def _embed(paths: list[Path], model, preprocess, device: str, batch: int = 32) -> np.ndarray:
    import torch

    feats = []
    for i in tqdm(range(0, len(paths), batch), desc="embed"):
        chunk = paths[i : i + batch]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        with torch.no_grad():
            f = model.encode_image(imgs)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0) if feats else np.zeros((0, 512), dtype=np.float32)


def _top1_similarity(query: np.ndarray, gallery: np.ndarray, gallery_paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
    """Return top-1 cosine similarity per query and the matched gallery path."""
    if len(query) == 0 or len(gallery) == 0:
        return np.zeros(len(query)), []
    sims = query @ gallery.T
    idx = sims.argmax(1)
    return sims[np.arange(len(sims)), idx], [gallery_paths[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", type=Path, required=True)
    ap.add_argument("--holdout-csv", type=Path, required=True, help="OpenI splits CSV with a split column")
    ap.add_argument("--synth-dir", type=Path, default=ARTIFACTS / "synth_images")
    ap.add_argument("--limit-train", type=int, default=3000)
    ap.add_argument("--limit-synth", type=int, default=1000)
    ap.add_argument("--limit-holdout", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "memorization.json")
    ap.add_argument("--out-csv", type=Path, default=ARTIFACTS / "memorization_topk.csv")
    args = ap.parse_args()

    import clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading CLIP ViT-B/32 on %s", device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    df = pd.read_csv(args.holdout_csv)
    train_images = df[df["split"] == "train"]["image"].tolist()[: args.limit_train]
    holdout_images = df[df["split"] == "test"]["image"].tolist()[: args.limit_holdout]

    train_paths = [args.train_images / n for n in train_images if (args.train_images / n).exists()]
    holdout_paths = [args.train_images / n for n in holdout_images if (args.train_images / n).exists()]
    synth_paths = sorted(args.synth_dir.glob("*.png"))[: args.limit_synth]

    log.info("embedding train=%d holdout=%d synth=%d", len(train_paths), len(holdout_paths), len(synth_paths))
    train_e = _embed(train_paths, model, preprocess, device)
    holdout_e = _embed(holdout_paths, model, preprocess, device)
    synth_e = _embed(synth_paths, model, preprocess, device)

    synth_sim, synth_nn = _top1_similarity(synth_e, train_e, train_paths)
    holdout_sim, _ = _top1_similarity(holdout_e, train_e, train_paths)

    def _stats(x: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(x)) if len(x) else 0.0,
            "median": float(np.median(x)) if len(x) else 0.0,
            "p95": float(np.quantile(x, 0.95)) if len(x) else 0.0,
            "p99": float(np.quantile(x, 0.99)) if len(x) else 0.0,
            "max": float(np.max(x)) if len(x) else 0.0,
        }

    report = {
        "n_synth": len(synth_paths),
        "n_train_gallery": len(train_paths),
        "synth_top1_stats": _stats(synth_sim),
        "holdout_top1_stats": _stats(holdout_sim),
        "delta_mean": float(np.mean(synth_sim) - np.mean(holdout_sim)) if len(synth_sim) and len(holdout_sim) else 0.0,
        "note": "Positive delta_mean indicates synthetic images are closer to training data than real holdout images are — a memorization warning sign.",
    }
    args.out.write_text(json.dumps(report, indent=2))

    sims_path = args.out.with_name("memorization_sims.npz")
    np.savez_compressed(sims_path, synth_sim=synth_sim, holdout_sim=holdout_sim)

    order = np.argsort(-synth_sim)[:50]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["synth_image", "nn_train_image", "cosine_sim"])
        for i in order:
            w.writerow([synth_paths[i].name, synth_nn[i].name, f"{synth_sim[i]:.4f}"])

    log.info("wrote %s, %s, and %s", args.out, args.out_csv, sims_path)


if __name__ == "__main__":
    main()
