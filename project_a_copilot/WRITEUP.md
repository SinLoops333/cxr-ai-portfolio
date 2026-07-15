# CXR Copilot — Portfolio Write-up

## Problem
Medical VLMs (MedGemma, LLaVA-Med, etc.) can draft plausible-sounding chest X-ray reports, but they routinely assert findings the image does not actually show. The failure mode is dangerous precisely because the writing is fluent. Existing "VLM + RAG" demos rarely try to measure or reduce this hallucination rate.

## What I built
An end-to-end web app (FastAPI + Streamlit, Dockerized) where:
1. A DenseNet-121 classifier (torchxrayvision) produces structured pathology probabilities.
2. MedGemma 1.5 4B (loaded in 4-bit) drafts a short report conditioned on the image, the user's question, and the classifier's top findings.
3. A FAISS retriever over PMC OA + StatPearls returns evidence per queried pathology.
4. A **verifier agent** splits the draft into atomic claims, detects which pathology each claim asserts (with negation cues), and marks a claim as *supported* only when either the classifier probability exceeds a threshold or the retrieved passages contain relevant keywords. Unsupported positive claims are transparently rewritten as `[UNVERIFIED] ...`.

The eval harness reports **CheXbert-style F1**, **retrieval hit@k**, and — the punchline — **hallucination rate before vs. after verification**.

## Engineering decisions worth mentioning in interviews
- **Graceful degradation.** If MedGemma or a GPU isn't available, the pipeline falls back to a deterministic template reporter so the whole system (and CI) still runs. This is why the test suite passes without model weights.
- **Measurable differentiator.** Every threshold that matters (`classifier_support_threshold`, `min_kb_hits`, `hedge_unsupported`) is in one YAML. Change them, rerun `run_eval.py`, and the hallucination-rate delta moves.
- **No label leakage.** OpenI split by `report_id`, not by image, so studies with multiple images can't cross the train/test boundary.
- **License-safe KB.** RAG corpus is PMC OA + StatPearls stubs via E-utilities — no scraping of restricted sites.

## Limits and honest gaps
- The CheXbert labeler used in eval is a **keyword fallback**; swapping in the real CheXbert weights would tighten the F1 numbers. The interface is stable.
- Retrieval is dense-only (sentence-transformers MiniLM). A BM25 hybrid or MedCPT embedder would help clinical recall.
- The classifier is off-the-shelf. Fine-tuning on NIH14 (path in the same config) would likely lift verifier accuracy on borderline findings.

## Portfolio talking points
- "I built a VLM-based medical assistant, then built the honest thing on top: a verifier that measurably reduces hallucinations."
- "It's Dockerized, has CI, and every claim has a citation."
- "The whole thing works on one 24GB GPU with no credentialed datasets."
