"""Tests for the OpenI XML report parser."""

from __future__ import annotations

from pathlib import Path

from data.scripts.parse_openi_reports import parse_report

SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<eCitation>
  <MedlineCitation>
    <Article>
      <Abstract>
        <AbstractText Label="COMPARISON">None.</AbstractText>
        <AbstractText Label="INDICATION">Cough.</AbstractText>
        <AbstractText Label="FINDINGS">Mild cardiomegaly. Small left pleural effusion.</AbstractText>
        <AbstractText Label="IMPRESSION">Cardiomegaly with small effusion.</AbstractText>
      </Abstract>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName><major>Cardiomegaly</major></DescriptorName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName><major>Pleural Effusion</major></DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </Article>
  </MedlineCitation>
  <parentImage id="CXR1_001-0001">
    <panel><url>x.png</url></panel>
  </parentImage>
  <parentImage id="CXR1_001-0002">
    <panel><url>y.png</url></panel>
  </parentImage>
</eCitation>
"""


def test_parse_report(tmp_path: Path):
    p = tmp_path / "1.xml"
    p.write_text(SAMPLE)
    rows = parse_report(p)
    assert len(rows) == 2
    r = rows[0]
    assert r["image"].endswith(".png")
    assert "cardiomegaly" in r["findings"].lower()
    assert "cardiomegaly" in r["caption_prompt"].lower()
    assert "cardiomegaly" in r["mesh_tags"].lower()


def test_parse_report_no_images(tmp_path: Path):
    p = tmp_path / "2.xml"
    p.write_text("<eCitation><MedlineCitation></MedlineCitation></eCitation>")
    assert parse_report(p) == []
