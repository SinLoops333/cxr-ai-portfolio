"""Quality gate for synthetic CXRs.

Two independent filters:
    1. Classifier-plausibility: run a pretrained torchxrayvision DenseNet on each
       synthetic image and require the intended pathology probability to exceed
       a threshold (default 0.4). Rejects images that don't visually express the
       requested finding.
    2. Distributional sanity: compute Inception-based FID between the accepted
       synthetic set and a real reference set. Reports the delta between raw and
       filtered FID so the gate's effect is measurable.

Writes ``artifacts/synth_images/kept.csv`` and ``artifacts/quality_report.json``.

Usage:
    python project_b_synth/scripts/quality_gate.py \
        --synth-dir artifacts/synth_images \
        --real-dir data/raw/openi \
        --threshold 0.4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common.logging_utils import get_logger
from common.paths import ARTIFACTS

log = get_logger("synth.qgate")

# Map our prompt-side pathology names to torchxrayvision's canonical labels.
PATHOLOGY_ALIAS = {
    "cardiomegaly": "Cardiomegaly",
    "effusion": "Effusion",
    "consolidation": "Consolidation",
    "pneumothorax": "Pneumothorax",
    "edema": "Edema",
    "atelectasis": "Atelectasis",
    "nodule": "Nodule",
    "mass": "Mass",
    "no_finding": None,  # skip probability check
}


def _load_classifier():
    import torch
    import torchxrayvision as xrv

    m = xrv.models.DenseNet(weights="densenet121-res224-all")
    m.eval()
    if torch.cuda.is_available():
        m = m.cuda()
    return m


def _preprocess(img_path: Path):
    import torch
    import torchxrayvision as xrv
    from skimage.io import imread

    img = imread(img_path)
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(-1)
    img = img[None, ...]
    from torchvision import transforms

    tfm = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    img = tfm(img)
    t = torch.from_numpy(img).unsqueeze(0).float()
    if torch.cuda.is_available():
        t = t.cuda()
    return t


def classifier_filter(synth_dir: Path, manifest: Path, threshold: float, out_csv: Path) -> dict:
    import torch

    model = _load_classifier()
    label_names = model.pathologies

    rows = list(csv.DictReader(open(manifest)))
    kept, dropped = 0, 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "prompt", "pathology", "score", "kept"])
        for row in tqdm(rows, desc="classifier gate"):
            img_path = synth_dir / row["image"]
            target = PATHOLOGY_ALIAS.get(row["pathology"])
            score = float("nan")
            keep = True
            if target is not None and target in label_names:
                idx = label_names.index(target)
                with torch.no_grad():
                    out = model(_preprocess(img_path))
                score = float(out[0, idx].cpu())
                keep = score >= threshold
            w.writerow([row["image"], row["prompt"], row["pathology"], f"{score:.4f}", int(keep)])
            kept += int(keep)
            dropped += int(not keep)
    return {"kept": kept, "dropped": dropped, "threshold": threshold, "total": kept + dropped}


def _pil_dir_to_uint8_tensor(paths: list[Path], size: int = 299):
    """Load a batch of PNGs as (N, 3, size, size) uint8 tensor for torch-fidelity."""
    import torch
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        imgs.append(np.array(img))
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()


def compute_fid(real_dir: Path, synth_dir: Path, kept_csv: Path | None = None) -> dict:
    """FID between real reference set and (a) all synth, (b) accepted subset."""
    try:
        import torch_fidelity
    except ImportError:
        log.warning("torch-fidelity not installed; skipping FID")
        return {}

    real_paths = sorted(real_dir.glob("*.png"))[:2000]  # cap for speed
    real_dump = ARTIFACTS / "_fid_real"
    real_dump.mkdir(exist_ok=True)
    for p in real_paths:
        dst = real_dump / p.name
        if not dst.exists():
            dst.symlink_to(p.resolve())

    def _fid(synth_root: Path) -> float:
        r = torch_fidelity.calculate_metrics(input1=str(real_dump), input2=str(synth_root), fid=True, cuda=True, verbose=False)
        return float(r["frechet_inception_distance"])

    result = {"fid_all": _fid(synth_dir)}
    if kept_csv:
        kept_only = ARTIFACTS / "_fid_kept"
        kept_only.mkdir(exist_ok=True)
        for row in csv.DictReader(open(kept_csv)):
            if int(row["kept"]):
                src = synth_dir / row["image"]
                dst = kept_only / row["image"]
                if not dst.exists() and src.exists():
                    dst.symlink_to(src.resolve())
        result["fid_kept"] = _fid(kept_only)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, default=ARTIFACTS / "synth_images")
    ap.add_argument("--real-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--out-report", type=Path, default=ARTIFACTS / "quality_report.json")
    args = ap.parse_args()

    manifest = args.synth_dir / "manifest.csv"
    kept_csv = args.synth_dir / "kept.csv"

    log.info("running classifier filter (threshold=%.2f)", args.threshold)
    counts = classifier_filter(args.synth_dir, manifest, args.threshold, kept_csv)
    log.info("kept %d / %d", counts["kept"], counts["total"])

    log.info("computing FID (all vs. kept)")
    try:
        fid = compute_fid(args.real_dir, args.synth_dir, kept_csv)
    except Exception as e:
        # torch-fidelity + DataLoader workers can fail on some torch/CUDA combos;
        # classifier filter results are still valid without FID.
        log.warning("FID computation failed (%s); continuing without FID", e)
        fid = {"error": str(e)}

    report = {"classifier_filter": counts, "fid": fid}
    args.out_report.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out_report)


if __name__ == "__main__":
    main()
