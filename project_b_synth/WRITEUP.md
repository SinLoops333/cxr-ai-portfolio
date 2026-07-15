# Text-to-CXR Synth Engine — Portfolio Write-up

## Problem
Rare pathologies in chest X-ray datasets have too few labeled examples for a downstream classifier to learn well. Generative models can produce "more data" — but most demos stop at "the images look X-ray-ish" and don't answer the actually useful questions: does it help downstream, and is it just memorizing training data?

## What I built
A three-stage pipeline that (a) fine-tunes SD v1.5 on OpenI with LoRA, (b) filters the generations through a quality gate, and (c) proves — or disproves — that the accepted synthetic data lifts a downstream classifier on rare pathologies, with confidence intervals.

## The three checks most demos skip
1. **Utility, not vibes.** Two identical DenseNet-121 pipelines are trained on identical NIH14 splits, one with and one without the synthetic augmentation. `eval/utility_study.py` reports macro + per-class AUROC with 95% bootstrap CIs, plus a rare-class subset defined by prevalence < 5%.
2. **Memorization audit.** For each synth image, CLIP ViT-B/32 top-1 cosine similarity to the training set is computed and compared to the same statistic on a real holdout → training-set baseline. A positive `delta_mean` (synth closer to train than real holdout) is a warning that the model is copying.
3. **Quality gate.** A pretrained torchxrayvision classifier is run on every synth image; ones that don't visually express the requested pathology (probability below threshold) are dropped. FID is computed before and after filtering so the gate's effect is measurable.

## Engineering decisions worth mentioning in interviews
- **`random_flip: false`.** Chest X-rays have handedness (heart on the left in situs solitus); flipping healthy CXRs corrupts anatomy and training signal.
- **LoRA target set.** Only UNet cross-attention projections (Q, K, V, out) — smallest set that still learns radiographic style/content at r=16 α=16.
- **8–10 GB VRAM at 512px, fp16, bs=2.** Fits a 24GB consumer card with room for the rest of the pipeline.
- **Prompt bank oversamples rare classes.** The utility study is designed to move the needle where it matters, not on classes that are already easy.
- **One-shot repro.** [`scripts/reproduce.sh`](scripts/reproduce.sh) runs generate → gate → audit → downstream × 2 → utility report.

## Limits and honest gaps
- Prompts are hand-designed from a small pathology bank, not learned. A stronger version would sample captions directly from OpenI reports.
- Downstream classifier is trained for only a handful of epochs to keep the loop fast on a single GPU; scaling epochs and hyperparameter search would give tighter CIs.
- FID is a coarse distributional metric; a CXR-specific realism scorer would be a nice follow-up.

## Portfolio talking points
- "I fine-tuned Stable Diffusion on chest X-rays with LoRA — then proved with confidence intervals that the synthetic data actually helped downstream on rare classes."
- "And I ran a memorization audit so I'd know if it was just copying."
- "It ships with a Gradio demo, a one-command repro, and a quality gate that quantifies FID with and without filtering."
