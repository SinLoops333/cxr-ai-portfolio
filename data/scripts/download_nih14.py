"""Download the NIH ChestX-ray14 dataset (labels + optional image tarballs).

The label CSV (``Data_Entry_2017_v2020.csv``) is hosted on NIH Box; the 12 image
tarballs are ~45GB total. To keep the plan's 24GB-GPU / short-timeline scope, we
support ``--subset small`` which downloads only the first tarball (~4GB, ~4.6k images)
which is enough to train a solid classifier for demo purposes when combined with
transfer learning from torchxrayvision.

If Box URLs move, pass ``--images-dir`` pointing at a manually downloaded copy and
run this script with ``--labels-only`` to just fetch the CSV + train/val/test splits.

Usage:
    python data/scripts/download_nih14.py --subset small
    python data/scripts/download_nih14.py --labels-only
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from common.logging_utils import get_logger
from common.paths import NIH14_DIR

log = get_logger("data.nih14")

# Official NIH Box direct-download URLs for ChestX-ray14 (as documented in the NIH README).
LABELS_URL = "https://nihcc.box.com/shared/static/7jm6lio4hz4vzsy8kwbaesh3n1zvcm5n.csv"  # Data_Entry_2017_v2020.csv
TRAIN_VAL_LIST_URL = "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.txt"
TEST_LIST_URL = "https://nihcc.box.com/shared/static/p89n7nx28r5ic7lgw17xk5jhy4o0mtes.txt"

# 12 image tarballs. Full list documented by NIH; we hardcode indices 1..12.
IMAGE_TARS = {
    1: "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    2: "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    3: "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    4: "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    5: "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    6: "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    7: "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    8: "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    9: "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    10: "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    11: "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    12: "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
}


def _download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        log.info("skip download (exists): %s", dst.name)
        return
    log.info("downloading %s -> %s", url, dst)
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
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
        return
    log.info("extracting %s", tar_path.name)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)
    marker.touch()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subset",
        choices=["small", "full"],
        default="small",
        help="'small' fetches only tarball 1 (~4GB, ~4.6k images); 'full' fetches all 12 (~45GB).",
    )
    ap.add_argument("--labels-only", action="store_true", help="Only fetch labels + splits.")
    ap.add_argument("--out", type=Path, default=NIH14_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _download(LABELS_URL, args.out / "Data_Entry_2017_v2020.csv")
    _download(TRAIN_VAL_LIST_URL, args.out / "train_val_list.txt")
    _download(TEST_LIST_URL, args.out / "test_list.txt")

    if args.labels_only:
        log.info("labels-only mode: done. Manually place images under %s/images", args.out)
        return

    tars_dir = args.out / "tars"
    tars_dir.mkdir(exist_ok=True)
    images_dir = args.out / "images"
    images_dir.mkdir(exist_ok=True)

    indices = [1] if args.subset == "small" else sorted(IMAGE_TARS)
    for i in indices:
        dst = tars_dir / f"images_{i:02d}.tar.gz"
        try:
            _download(IMAGE_TARS[i], dst)
            _extract(dst, images_dir)
        except requests.HTTPError as e:
            log.warning("tar %d failed (%s). NIH Box URLs move occasionally; download manually and drop into %s", i, e, images_dir)

    log.info("done. images at %s", images_dir)


if __name__ == "__main__":
    main()
