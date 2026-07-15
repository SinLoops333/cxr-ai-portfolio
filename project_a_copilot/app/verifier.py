"""Claim decomposition + faithfulness scoring.

For each atomic claim in the VLM's drafted report:
    1. Detect which pathology (if any) the claim asserts (positive or negated).
    2. Cross-check it against
         (a) the structured classifier probabilities, and
         (b) the top-k retrieved KB passages.
    3. Return a per-claim ``VerifiedClaim`` with a supported/unsupported flag,
       the two independent signals, and a citation.
    4. Optionally rewrite the final report to hedge unsupported claims.

This is the project's differentiator: we surface a measurable
"hallucination rate before/after verification" in the eval harness.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from common.metrics import CHEXPERT_KEYWORDS
from project_a_copilot.app.rag import FaissRetriever, RetrievedPassage

# Map torchxrayvision label names to lowercased keyword group keys used above.
LABEL_TO_KEY = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Effusion": "effusion",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Hernia": "hernia",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pleural_Thickening": "pleural_thickening",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}

NEGATION_CUES = re.compile(r"\b(no|without|absent|denies|negative for|not\s+demonstrat\w*|not\s+seen|clear of)\b", re.IGNORECASE)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class VerifiedClaim:
    text: str
    label: str | None
    negated: bool
    classifier_prob: float | None
    kb_hits: int
    citations: list[str] = field(default_factory=list)
    supported: bool = False
    reason: str = ""


@dataclass
class VerifiedReport:
    original: str
    revised: str
    claims: list[VerifiedClaim]

    def stats(self) -> dict[str, float]:
        n = len(self.claims)
        # Only claims that assert a *detectable pathology* positively are eligible to be
        # counted as hallucinations. Claims with no known pathology (e.g. "lungs are
        # clear") are excluded because we have no independent signal to verify them.
        eligible = [c for c in self.claims if c.label is not None and not c.negated]
        n_supported = sum(1 for c in eligible if c.supported)
        return {
            "n_claims": float(n),
            "n_positive_claims": float(len(eligible)),
            "n_supported": float(n_supported),
            "hallucination_rate": float((len(eligible) - n_supported) / max(1, len(eligible))) if eligible else 0.0,
        }

    def to_dict(self) -> dict:
        return {"original": self.original, "revised": self.revised, "claims": [asdict(c) for c in self.claims], "stats": self.stats()}


def split_claims(text: str) -> list[str]:
    if not text:
        return []
    parts = SENT_SPLIT.split(text.strip())
    out: list[str] = []
    for p in parts:
        for sub in re.split(r"[;\n]|(?:,\s+and\s+)", p):
            s = sub.strip(" .")
            if len(s) >= 4:
                out.append(s + ".")
    return out


def detect_label(claim: str) -> tuple[str | None, bool]:
    """Return (canonical_label, negated)."""
    low = claim.lower()
    negated = bool(NEGATION_CUES.search(low))
    for label, key in LABEL_TO_KEY.items():
        for kw in CHEXPERT_KEYWORDS.get(key, [key]):
            if kw in low:
                return label, negated
    return None, negated


def _kb_hits_for(label_key: str, passages: list[RetrievedPassage]) -> tuple[int, list[str]]:
    kws = CHEXPERT_KEYWORDS.get(label_key, [label_key])
    hits, cites = 0, []
    for p in passages:
        t = p.text.lower()
        if any(kw in t for kw in kws):
            hits += 1
            cites.append(p.id)
    return hits, cites


def verify(
    report: str,
    classifier_probs: dict[str, float],
    retriever: FaissRetriever | None,
    classifier_support_threshold: float = 0.5,
    min_kb_hits: int = 1,
    hedge_unsupported: bool = True,
    top_k: int = 5,
) -> VerifiedReport:
    claims_text = split_claims(report)
    claims: list[VerifiedClaim] = []
    revised_parts: list[str] = []
    for ct in claims_text:
        label, negated = detect_label(ct)
        cprob: float | None = None
        kb_hits, cites = 0, []
        supported = False
        reason = "no known pathology mentioned"

        if label is not None:
            cprob = classifier_probs.get(label)
            if retriever is not None:
                passages = retriever.query(f"{label} chest x-ray", k=top_k)
                kb_hits, cites = _kb_hits_for(LABEL_TO_KEY[label], passages)
            if negated:
                # Negated claim: supported if classifier is confidently LOW.
                if cprob is not None and cprob < (1 - classifier_support_threshold):
                    supported = True
                    reason = f"classifier prob {cprob:.2f} < {1 - classifier_support_threshold:.2f}"
            else:
                clf_ok = cprob is not None and cprob >= classifier_support_threshold
                kb_ok = kb_hits >= min_kb_hits
                supported = clf_ok or kb_ok
                if clf_ok and kb_ok:
                    reason = f"classifier prob {cprob:.2f} and {kb_hits} KB citations"
                elif clf_ok:
                    reason = f"classifier prob {cprob:.2f}"
                elif kb_ok:
                    reason = f"{kb_hits} KB citations (no classifier signal)"
                else:
                    reason = f"classifier prob {cprob if cprob is None else round(cprob,2)}, {kb_hits} KB hits"

        vc = VerifiedClaim(text=ct, label=label, negated=negated, classifier_prob=cprob, kb_hits=kb_hits, citations=cites, supported=supported, reason=reason)
        claims.append(vc)

        if not supported and not negated and label is not None and hedge_unsupported:
            revised_parts.append(f"[UNVERIFIED] {ct}")
        else:
            revised_parts.append(ct)

    revised = " ".join(revised_parts)
    return VerifiedReport(original=report, revised=revised, claims=claims)
