"""Tests for the verifier / claim-decomposition module (no model weights required)."""

from __future__ import annotations

from project_a_copilot.app.verifier import detect_label, split_claims, verify


def test_split_claims_basic():
    text = "Mild cardiomegaly is present. No pneumothorax; small effusion noted."
    parts = split_claims(text)
    assert len(parts) >= 3
    assert all(p.endswith(".") for p in parts)


def test_detect_label_positive_and_negation():
    lab, neg = detect_label("Moderate cardiomegaly present.")
    assert lab == "Cardiomegaly"
    assert neg is False
    lab, neg = detect_label("No pneumothorax.")
    assert lab == "Pneumothorax"
    assert neg is True
    lab, _ = detect_label("Lungs are clear.")
    assert lab is None


def test_verify_supports_high_prob_positive_claim():
    probs = {"Cardiomegaly": 0.9, "Pneumothorax": 0.05, "Effusion": 0.1}
    vr = verify(
        "Cardiomegaly is present. No pneumothorax.",
        probs,
        retriever=None,
        classifier_support_threshold=0.5,
        min_kb_hits=1,
        hedge_unsupported=True,
    )
    labels = [c.label for c in vr.claims]
    assert "Cardiomegaly" in labels
    assert "Pneumothorax" in labels
    # positive high-prob claim is supported
    c = next(c for c in vr.claims if c.label == "Cardiomegaly" and not c.negated)
    assert c.supported is True
    # negated low-prob claim is supported
    c = next(c for c in vr.claims if c.label == "Pneumothorax" and c.negated)
    assert c.supported is True
    # revised report should not hedge those
    assert "[UNVERIFIED]" not in vr.revised


def test_verify_flags_unsupported_positive_claim():
    probs = {"Pneumothorax": 0.02}  # classifier disagrees
    vr = verify("Pneumothorax is present.", probs, retriever=None, hedge_unsupported=True)
    assert vr.claims[0].supported is False
    assert "[UNVERIFIED]" in vr.revised
    stats = vr.stats()
    assert stats["hallucination_rate"] == 1.0


def test_verify_no_positive_claims_gives_zero_hallucination():
    vr = verify("Lungs are clear.", {"Pneumothorax": 0.01}, retriever=None)
    stats = vr.stats()
    assert stats["hallucination_rate"] == 0.0
