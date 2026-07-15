"""VLM wrapper around MedGemma 1.5 4B with a robust deterministic fallback.

Design:
    - Primary: HuggingFace ``google/medgemma-1.5-4b-it`` loaded in 4-bit (bitsandbytes)
      so it fits comfortably on a 24GB consumer GPU.
    - Fallback: ``TemplateReporter`` that composes a report string from the classifier's
      structured probabilities. This means the whole pipeline (including tests, CI, and
      the demo on hosts without a HF token or a GPU) always works end-to-end.

Both implementations satisfy the same protocol ``draft_report(image, question, findings)``
returning ``ReportDraft(text=..., meta=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from common.logging_utils import get_logger

log = get_logger("copilot.vlm")

SYSTEM = (
    "You are a careful assistant that drafts chest X-ray impressions. "
    "Base every claim on the provided image and the structured findings signal. "
    "Prefer short, atomic sentences (one finding per sentence). "
    "If unsure, say 'not clearly demonstrated'. Never invent measurements or numbers. "
    "This is a research prototype, not medical advice."
)


@dataclass
class ReportDraft:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


class TemplateReporter:
    """Deterministic classifier-driven report. Zero external dependencies at runtime."""

    def draft_report(self, image: Image.Image, question: str, findings: dict[str, float]) -> ReportDraft:
        pos = [(k, v) for k, v in findings.items() if v >= 0.5]
        pos.sort(key=lambda kv: -kv[1])
        borderline = [(k, v) for k, v in findings.items() if 0.3 <= v < 0.5]
        borderline.sort(key=lambda kv: -kv[1])
        parts: list[str] = []
        if pos:
            for name, p in pos:
                parts.append(f"{name.lower().replace('_', ' ')} is present (model probability {p:.2f}).")
        else:
            parts.append("No dominant pathology asserted by the classifier.")
        for name, p in borderline:
            parts.append(f"{name.lower().replace('_', ' ')} is borderline (probability {p:.2f}) and not clearly demonstrated.")
        if question:
            parts.append(f"Question addressed: {question}")
        return ReportDraft(text=" ".join(parts), meta={"backend": "template", "positive": len(pos), "borderline": len(borderline)})


class MedGemmaReporter:
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", load_in_4bit: bool = True, max_new_tokens: int = 384, temperature: float = 0.2):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_in_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        log.info("loading %s (4bit=%s)", model_id, load_in_4bit)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model_id = model_id

    def draft_report(self, image: Image.Image, question: str, findings: dict[str, float]) -> ReportDraft:
        top = sorted(findings.items(), key=lambda kv: -kv[1])[:8]
        findings_str = ", ".join(f"{k} p={v:.2f}" for k, v in top)
        user = (
            f"Draft a brief chest X-ray impression as short atomic sentences. "
            f"Structured classifier signal: {findings_str}. "
            f"User question: {question or 'summarize findings'}."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user}]},
        ]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with self._no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0, temperature=self.temperature)
        gen = out[0][inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(gen, skip_special_tokens=True).strip()
        return ReportDraft(text=text, meta={"backend": "medgemma", "model": self.model_id})

    def _no_grad(self):
        import torch
        return torch.inference_mode()


def load_reporter(cfg) -> Any:
    """Try MedGemma; fall back to the deterministic template on any failure."""
    try:
        return MedGemmaReporter(cfg.vlm.model, cfg.vlm.load_in_4bit, cfg.vlm.max_new_tokens, cfg.vlm.temperature)
    except Exception as e:
        log.warning("MedGemma unavailable (%s). Falling back to TemplateReporter.", e)
        return TemplateReporter()
