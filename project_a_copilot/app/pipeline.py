"""End-to-end pipeline: image + question -> verified report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from common.config import load_config
from common.logging_utils import get_logger

from .classifier import CXRClassifier
from .rag import FaissRetriever
from .verifier import verify
from .vlm import load_reporter

log = get_logger("copilot.pipeline")


class CopilotPipeline:
    def __init__(self, cfg_path: str | Path):
        cfg = load_config(cfg_path)
        self.cfg = cfg
        self.classifier = CXRClassifier(cfg.classifier.weights, cfg.classifier.device, cfg.classifier.labels)
        self.reporter = load_reporter(cfg)
        try:
            self.retriever: FaissRetriever | None = FaissRetriever(Path(cfg.rag.index_path), Path(cfg.rag.meta_path), cfg.rag.embedder, cfg.rag.top_k)
        except Exception as e:
            log.warning("retriever unavailable (%s). Continuing without RAG.", e)
            self.retriever = None

    def run(self, image: Image.Image, question: str = "") -> dict[str, Any]:
        probs = self.classifier.predict_from_pil(image)
        draft = self.reporter.draft_report(image, question, probs)
        verified = verify(
            draft.text,
            probs,
            self.retriever,
            classifier_support_threshold=self.cfg.verifier.classifier_support_threshold,
            min_kb_hits=self.cfg.verifier.min_kb_hits,
            hedge_unsupported=self.cfg.verifier.hedge_unsupported,
            top_k=self.cfg.rag.top_k,
        )
        return {
            "classifier": probs,
            "asserted": self.classifier.assertions(probs, self.cfg.classifier.assertion_threshold),
            "draft": draft.text,
            "draft_meta": draft.meta,
            "verified": verified.to_dict(),
        }


def default_pipeline(cfg_path: str | Path = "project_a_copilot/configs/copilot.yaml") -> CopilotPipeline:
    return CopilotPipeline(cfg_path)
