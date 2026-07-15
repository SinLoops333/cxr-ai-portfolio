"""Download the Indiana University (OpenI) chest X-ray dataset.

Fetches the two official tarballs from the NLM (7,470 PNG images + 3,955 XML reports),
extracts them under ``data/raw/openi/``. Idempotent: skips existing files.

Usage:
    python data/scripts/download_openi.py
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from common.logging_utils import get_logger
from common.paths import OPENI_DIR

log = get_logger("data.openi")

URLS = {
    "NLMCXR_png.tgz": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
    "NLMCXR_reports.tgz": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
}


def _download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        log.info("skip download (exists): %s", dst.name)
        return
    log.info("downloading %s -> %s", url, dst)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with (
            open(dst, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar,
        ):
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
                    bar.update(len(c))


def _extract(tar_path: Path, out_dir: Path) -> None:
    marker = out_dir / f".{tar_path.name}.extracted"
    if marker.exists():
        log.info("skip extract (marker exists): %s", tar_path.name)
        return
    log.info("extracting %s -> %s", tar_path, out_dir)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    marker.touch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OPENI_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dst = args.out / name
        _download(url, dst)
        _extract(dst, args.out)
    log.info("done. images at %s (flat PNGs), reports at %s", args.out, args.out / "ecgen-radiology")


if __name__ == "__main__":
    main()
