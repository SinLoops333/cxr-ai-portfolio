"""Train a DenseNet-121 CXR classifier under two regimes and dump probabilities for eval.

Regimes:
    real:      NIH14 train split only
    real+synth: NIH14 train + accepted synthetic images from Flagship B's generator

The synthetic labels are derived from the ``pathology`` column of ``manifest.csv`` /
``kept.csv`` mapped onto the 14-class NIH label space (only classes present in the
synthetic bank are set; others are 0).

Outputs (per regime):
    checkpoints/downstream/<regime>/best.pt
    artifacts/downstream/<regime>_probs_test.npz  (y_true, y_score, class_names)

Usage:
    python project_b_synth/scripts/train_downstream.py --regime real
    python project_b_synth/scripts/train_downstream.py --regime real_synth --kept-csv artifacts/synth_images/kept.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common.logging_utils import get_logger
from common.paths import ARTIFACTS, CHECKPOINTS, NIH14_DIR
from common.seed import set_seed

log = get_logger("synth.downstream")

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NIH_LABEL_TO_IDX = {n: i for i, n in enumerate(NIH_LABELS)}

# Map our synthetic pathology tags to NIH14 columns (lowercased alias -> official).
SYNTH_TO_NIH = {
    "cardiomegaly": "Cardiomegaly",
    "effusion": "Effusion",
    "consolidation": "Consolidation",
    "pneumothorax": "Pneumothorax",
    "edema": "Edema",
    "atelectasis": "Atelectasis",
    "nodule": "Nodule",
    "mass": "Mass",
    "no_finding": None,
}


class MultiLabelCXR:
    """Simple multi-label dataset yielding (tensor, target) pairs."""

    def __init__(self, records: list[tuple[Path, np.ndarray]], size: int = 224, train: bool = False):
        import torch
        from torchvision import transforms

        self.records = records
        self.train = train
        base = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(size + 16),
            transforms.CenterCrop(size),
        ]
        if train:
            base = [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(size + 32),
                transforms.RandomResizedCrop(size, scale=(0.85, 1.0)),
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
            ]
        base += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.tfm = transforms.Compose(base)
        self._torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        p, y = self.records[i]
        img = Image.open(p)
        return self.tfm(img), self._torch.from_numpy(y).float()


def _load_nih_records(images_dir: Path, labels_csv: Path, split_list: Path) -> list[tuple[Path, np.ndarray]]:
    labels = pd.read_csv(labels_csv)
    keep = set(split_list.read_text().split())
    labels = labels[labels["Image Index"].isin(keep)]
    on_disk = {p.name for p in images_dir.glob("*.png")}
    labels = labels[labels["Image Index"].isin(on_disk)]

    records = []
    for _, row in labels.iterrows():
        y = np.zeros(len(NIH_LABELS), dtype=np.float32)
        for finding in str(row["Finding Labels"]).split("|"):
            if finding in NIH_LABEL_TO_IDX:
                y[NIH_LABEL_TO_IDX[finding]] = 1.0
        records.append((images_dir / row["Image Index"], y))
    return records


def _load_synth_records(kept_csv: Path, synth_dir: Path) -> list[tuple[Path, np.ndarray]]:
    df = pd.read_csv(kept_csv) if kept_csv.exists() else pd.read_csv(synth_dir / "manifest.csv").assign(kept=1)
    df = df[df["kept"] == 1]
    records = []
    for _, row in df.iterrows():
        y = np.zeros(len(NIH_LABELS), dtype=np.float32)
        nih = SYNTH_TO_NIH.get(row["pathology"])
        if nih is not None and nih in NIH_LABEL_TO_IDX:
            y[NIH_LABEL_TO_IDX[nih]] = 1.0
        records.append((synth_dir / row["image"], y))
    return records


def _build_model():
    import torch.nn as nn
    from torchvision.models import DenseNet121_Weights, densenet121

    m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = nn.Linear(m.classifier.in_features, len(NIH_LABELS))
    return m


def _train_one(regime: str, train_recs, val_recs, test_recs, out_dir: Path, epochs: int, lr: float, bs: int):
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr = DataLoader(MultiLabelCXR(train_recs, train=True), batch_size=bs, shuffle=True, num_workers=2)
    va = DataLoader(MultiLabelCXR(val_recs), batch_size=bs, num_workers=2)

    best_val = float("inf")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        for x, y in tqdm(tr, desc=f"[{regime}] epoch {ep}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(device), y.to(device)
                vloss += loss_fn(model(x), y).item() * len(x)
                n += len(x)
        vloss /= max(1, n)
        log.info("[%s] epoch %d val_loss %.4f", regime, ep, vloss)
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), out_dir / "best.pt")

    # test-set probabilities
    te = DataLoader(MultiLabelCXR(test_recs), batch_size=bs, num_workers=2)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for x, y in te:
            x = x.to(device)
            ss.append(torch.sigmoid(model(x)).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["real", "real_synth"], required=True)
    ap.add_argument("--nih-images", type=Path, default=NIH14_DIR / "images")
    ap.add_argument("--labels-csv", type=Path, default=NIH14_DIR / "Data_Entry_2017_v2020.csv")
    ap.add_argument("--train-list", type=Path, default=NIH14_DIR / "train_val_list.txt")
    ap.add_argument("--test-list", type=Path, default=NIH14_DIR / "test_list.txt")
    ap.add_argument("--synth-dir", type=Path, default=ARTIFACTS / "synth_images")
    ap.add_argument("--kept-csv", type=Path, default=ARTIFACTS / "synth_images" / "kept.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    all_train = _load_nih_records(args.nih_images, args.labels_csv, args.train_list)
    test_recs = _load_nih_records(args.nih_images, args.labels_csv, args.test_list)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(all_train))
    n_val = int(len(all_train) * args.val_frac)
    val_recs = [all_train[i] for i in idx[:n_val]]
    train_recs = [all_train[i] for i in idx[n_val:]]

    if args.regime == "real_synth":
        synth_recs = _load_synth_records(args.kept_csv, args.synth_dir)
        train_recs = train_recs + synth_recs
        log.info("appended %d synthetic samples", len(synth_recs))

    out_dir = CHECKPOINTS / "downstream" / args.regime
    y, s = _train_one(args.regime, train_recs, val_recs, test_recs, out_dir, args.epochs, args.lr, args.bs)

    art_dir = ARTIFACTS / "downstream"
    art_dir.mkdir(parents=True, exist_ok=True)
    np.savez(art_dir / f"{args.regime}_probs_test.npz", y_true=y, y_score=s, class_names=np.array(NIH_LABELS))
    log.info("saved probs to %s", art_dir / f"{args.regime}_probs_test.npz")


if __name__ == "__main__":
    main()
