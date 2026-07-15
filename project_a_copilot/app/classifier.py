"""CXR pathology classifier wrapper.

Two modes:
    - default: use torchxrayvision's pretrained DenseNet121-res224-all (no training needed).
    - finetune: warm-start from those weights and fine-tune the classifier head on NIH14
      (uses the same records loader as Flagship B).

The wrapper exposes ``predict_from_pil(img) -> dict[label, prob]`` for the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common.logging_utils import get_logger

log = get_logger("copilot.clf")


class CXRClassifier:
    def __init__(self, weights: str = "densenet121-res224-all", device: str = "auto", labels: list[str] | None = None):
        import torch
        import torchxrayvision as xrv

        self.torch = torch
        self.xrv = xrv
        self.device = self._resolve_device(device)
        self.model = xrv.models.DenseNet(weights=weights).to(self.device)
        self.model.eval()
        self.all_labels: list[str] = list(self.model.pathologies)
        self.labels: list[str] = [lab for lab in (labels or self.all_labels) if lab in self.all_labels]
        self._label_idx = [self.all_labels.index(lab) for lab in self.labels]

    def _resolve_device(self, spec: str) -> str:
        if spec == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        return spec

    def _preprocess(self, img: Image.Image):
        from torchvision import transforms

        arr = np.array(img.convert("L"))
        arr = self.xrv.datasets.normalize(arr, 255)[None, ...]
        tfm = transforms.Compose(
            [self.xrv.datasets.XRayCenterCrop(), self.xrv.datasets.XRayResizer(224)]
        )
        arr = tfm(arr)
        t = self.torch.from_numpy(arr).unsqueeze(0).float().to(self.device)
        return t

    def predict_from_pil(self, img: Image.Image) -> dict[str, float]:
        with self.torch.no_grad():
            out = self.model(self._preprocess(img))
        probs = out[0].detach().cpu().numpy()
        return {lab: float(probs[self.all_labels.index(lab)]) for lab in self.labels}

    def assertions(self, probs: dict[str, float], threshold: float) -> list[str]:
        return [k for k, v in probs.items() if v >= threshold]


def _cli(cfg_path: Path, image: Path | None) -> None:
    from common.config import load_config

    cfg = load_config(cfg_path)
    clf = CXRClassifier(cfg.classifier.weights, cfg.classifier.device, cfg.classifier.labels)
    if image is None:
        log.info("classifier initialized on %s with %d labels", clf.device, len(clf.labels))
        return
    p = clf.predict_from_pil(Image.open(image))
    for k, v in sorted(p.items(), key=lambda kv: -kv[1]):
        log.info("%s: %.3f", k, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--image", type=Path, default=None)
    args = ap.parse_args()
    _cli(args.config, args.image)


if __name__ == "__main__":
    main()
