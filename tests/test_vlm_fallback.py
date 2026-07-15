"""The TemplateReporter fallback must always work without any model weights."""

from __future__ import annotations

from PIL import Image

from project_a_copilot.app.vlm import TemplateReporter


def test_template_report_positive_findings():
    r = TemplateReporter()
    img = Image.new("L", (64, 64))
    out = r.draft_report(img, "Summarize", {"Cardiomegaly": 0.9, "Pneumothorax": 0.02, "Edema": 0.4})
    assert "cardiomegaly" in out.text.lower()
    # borderline (0.3-0.5) hedged, low-prob absent
    assert "edema" in out.text.lower()
    assert out.meta["backend"] == "template"


def test_template_report_no_findings():
    r = TemplateReporter()
    img = Image.new("L", (64, 64))
    out = r.draft_report(img, "", {"Cardiomegaly": 0.1, "Pneumothorax": 0.02})
    assert "No dominant pathology" in out.text
