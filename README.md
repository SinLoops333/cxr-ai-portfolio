# Chest X-ray AI

Two chest X-ray systems sharing one codebase and data pipeline: a grounded report copilot and a text-to-image synthetic data engine with downstream utility evaluation.

> **Research use only.** Not a medical device. Not intended for clinical decision-making.

## Overview

**A) Report Copilot.** Given a chest X-ray and an optional question, the system produces structured pathology probabilities (DenseNet-121 via torchxrayvision), a drafted report from a medical VLM (MedGemma 1.5 4B in 4-bit, with a deterministic template fallback), and retrieved evidence from a FAISS index over PMC Open Access + StatPearls stubs. A verifier agent splits the draft into atomic claims, checks each claim against the classifier and retrieved passages, and flags unsupported findings as `[UNVERIFIED]`. Evaluation reports CheXbert-style F1 (keyword fallback), retrieval hit@k, and hallucination rate before vs. after verification.

**B) Synthetic Data Engine.** Stable Diffusion v1.5 is LoRA-fine-tuned on OpenI image–caption pairs to generate chest X-rays from findings text. Generations are filtered by a classifier-based quality gate, audited for memorization with CLIP nearest-neighbor similarity against the training set, and used to train a downstream DenseNet under two regimes (real vs. real+synth). Utility is measured as per-class AUROC with 95% bootstrap confidence intervals, including a rare-class subset.

Both systems use the same OpenI preprocessing, shared utilities under `common/`, and the same repository layout.

## Datasets

All datasets below are openly downloadable without PhysioNet/CITI credentialing.

| Dataset | Size | Used by | Source |
|---|---|---|---|
| OpenI (Indiana University) CXR + reports | 7,470 images + 3,955 reports (~2 GB) | A + B | https://openi.nlm.nih.gov (CC BY-NC-ND 4.0) |
| NIH ChestX-ray14 (subset OK) | 4.6k–112k images, 14 labels | Downstream classifier for B | NIH Box (public) |
| PMC OA + StatPearls stubs | ~few thousand passages | RAG knowledge base for A | NCBI E-utilities |

## Installation and Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Shared data
python -m data.scripts.download_openi
python -m data.scripts.parse_openi_reports
python -m data.scripts.build_openi_splits
python -m data.scripts.download_nih14 --subset small     # ~4 GB, single tar
python -m data.scripts.build_rag_corpus                   # ~few minutes
# Note: OpenI PNGs extract flat into data/raw/openi/ (not a NLMCXR_png/ subfolder).
```

## Usage

### A) Report Copilot

```bash
# Build the FAISS index
python -m project_a_copilot.app.rag --config project_a_copilot/configs/copilot.yaml --build

# Docker (API + Streamlit UI)
docker compose -f docker/docker-compose.yml up copilot
# API:  http://localhost:8000  (GET /health, POST /predict)
# UI:   http://localhost:8501

# Or run locally
uvicorn project_a_copilot.app.api:app --port 8000
streamlit run project_a_copilot/frontend/streamlit_app.py
```

Optional eval harness:

```bash
python -m project_a_copilot.eval.run_eval --config project_a_copilot/configs/copilot.yaml
```

### B) Synthetic Data Engine

```bash
# Fine-tune LoRA on OpenI (single GPU; ~8–10 GB VRAM at 512px)
accelerate launch project_b_synth/scripts/train_sd_lora.py \
    --config project_b_synth/configs/lora.yaml

# Generate → quality gate → memorization audit → downstream classifiers → utility report
bash project_b_synth/scripts/reproduce.sh

# Figures for the Results section
python project_b_synth/scripts/generate_results_plots.py

# Gradio demo (findings text → synthetic CXR)
python -m project_b_synth.demo.app --lora checkpoints/sd_lora_openi/final/unet_lora
```

## Project Structure

```
.
├── common/                config, logging, metrics, seeding, paths
├── data/scripts/          OpenI / NIH14 download + parse; RAG corpus builder
├── project_a_copilot/
│   ├── app/               classifier, rag, vlm, verifier, pipeline, FastAPI
│   ├── frontend/          Streamlit UI
│   ├── eval/              report / retrieval / hallucination metrics
│   └── configs/           copilot.yaml
├── project_b_synth/
│   ├── scripts/           LoRA train, generate, quality gate, audit, plots, reproduce.sh
│   ├── eval/              utility study (AUROC + bootstrap CIs)
│   ├── demo/              Gradio app
│   └── configs/           lora.yaml
├── docker/                Dockerfiles + compose
├── tests/                 unit tests (no model weights required)
└── .github/workflows/     ruff + pytest
```

## Results

Measured on the NIH ChestX-ray14 small subset (n_test = 460) after LoRA training on OpenI (5,979 train images). Synthetic set: 1,800 generations; quality gate kept 339 / 1,800 at classifier threshold 0.4.

![Per-class AUROC: real vs real+synth](artifacts/utility_comparison.png)

![Memorization audit: CLIP top-1 similarity](artifacts/memorization_histogram.png)

| Regime | Macro AUROC (95% CI) | Rare-class Macro AUROC |
|---|---|---|
| Real only | 0.727 (0.689, 0.764) | 0.731 |
| Real + synth (filtered) | 0.737 (0.695, 0.772) | 0.760 |
| Lift | **+0.010** | **+0.029** |

| Population | Mean | Median | P95 |
|---|---|---|---|
| Real holdout → train | 0.982 | 0.985 | 0.992 |
| Synth → train | 0.950 | 0.952 | 0.968 |
| Δ mean (synth − holdout) | **−0.032** | | |

On this subset, adding filtered synthetic data improves macro AUROC and rare-class macro AUROC. The memorization Δ mean is negative, so synthetic images are not closer to the training gallery than real holdout images under CLIP top-1 similarity.

Raw reports: `artifacts/utility_report.json`, `artifacts/memorization.json`, `artifacts/quality_report.json`.

## Key Design Decisions

- **No horizontal flips** on chest X-rays during LoRA training (anatomical handedness / situs solitus).
- **Parameter-efficient LoRA** on UNet attention projections only (rank 16 / α 16), so training fits on a single consumer GPU.
- **VLM graceful fallback**: if MedGemma cannot load, the copilot uses a classifier-templated reporter so the pipeline and CI still run end-to-end.
- **Verifier thresholds** (`classifier_support_threshold`, `min_kb_hits`) are config-driven; hallucination rate is measured before vs. after hedging.
- **Utility evaluation** uses nonparametric bootstrap CIs over the test set, with a rare-class subset defined by prevalence &lt; 0.05.
- **Memorization audit** compares synth→train vs. real-holdout→train CLIP top-1 similarity; a positive Δ mean is treated as a warning.
- **Quality gate** drops generations whose intended pathology score falls below threshold before they enter downstream training.
- **OpenI splits** are grouped by `report_id` so multi-image studies do not leak across train/val/test.

## License

Code: MIT.
