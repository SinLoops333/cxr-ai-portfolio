"""Parse OpenI XML radiology reports into a captions CSV.

Extracts Findings + Impression sections and links each report to its associated PNG(s).
Also generates a compact "prompt-style" caption for text-to-image training (Flagship B)
and keeps the full findings text for retrieval / report-generation eval (Flagship A).

Output CSV columns:
    image, report_id, findings, impression, caption_prompt, mesh_tags

Usage:
    python data/scripts/parse_openi_reports.py
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from common.logging_utils import get_logger
from common.paths import OPENI_CAPTIONS, OPENI_DIR, OPENI_IMAGES

log = get_logger("data.openi.parse")

_WS = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", text).strip()


def _abstract_section(root: ET.Element, label: str) -> str:
    for a in root.iter("AbstractText"):
        if (a.get("Label") or "").lower() == label.lower():
            return _clean(a.text)
    return ""


def _make_prompt(findings: str, impression: str, mesh: list[str]) -> str:
    """Build a short T2I prompt: 'frontal chest x-ray, <top mesh terms>, <shortened impression>'."""
    base = "frontal chest x-ray"
    parts = [base]
    if mesh:
        parts.append(", ".join(mesh[:4]))
    text = impression or findings
    if text:
        text = text.lower().strip(". ")
        if len(text) > 180:
            text = text[:180].rsplit(" ", 1)[0]
        parts.append(text)
    return ", ".join(parts)


def parse_report(xml_path: Path) -> list[dict]:
    """Return a list of rows (one per associated image)."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        log.warning("parse error %s: %s", xml_path.name, e)
        return []
    root = tree.getroot()

    findings = _abstract_section(root, "FINDINGS")
    impression = _abstract_section(root, "IMPRESSION")
    mesh = [_clean(m.text) for m in root.iter("major") if _clean(m.text)]

    image_ids = [img.get("id") for img in root.iter("parentImage") if img.get("id")]
    if not image_ids:
        return []

    prompt = _make_prompt(findings, impression, mesh)
    rid = xml_path.stem
    rows = []
    for img_id in image_ids:
        rows.append(
            {
                "image": f"{img_id}.png",
                "report_id": rid,
                "findings": findings,
                "impression": impression,
                "caption_prompt": prompt,
                "mesh_tags": "|".join(mesh),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, default=OPENI_DIR / "ecgen-radiology")
    ap.add_argument("--images-dir", type=Path, default=OPENI_IMAGES)
    ap.add_argument("--out", type=Path, default=OPENI_CAPTIONS)
    args = ap.parse_args()

    if not args.reports_dir.exists():
        raise SystemExit(f"reports dir not found: {args.reports_dir}. Run download_openi.py first.")

    xmls = sorted(args.reports_dir.glob("*.xml"))
    log.info("parsing %d reports from %s", len(xmls), args.reports_dir)

    rows: list[dict] = []
    for x in xmls:
        rows.extend(parse_report(x))

    df = pd.DataFrame(rows)
    if args.images_dir.exists():
        existing = {p.name for p in args.images_dir.glob("*.png")}
        before = len(df)
        df = df[df["image"].isin(existing)].reset_index(drop=True)
        log.info("kept %d/%d rows with matching PNGs on disk", len(df), before)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("wrote %s (%d rows)", args.out, len(df))


if __name__ == "__main__":
    main()
