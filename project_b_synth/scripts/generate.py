"""Generate a pathology-balanced synthetic CXR set from the fine-tuned LoRA.

Given a manifest of (pathology, prompt) tuples and a target count per class, samples
uniformly and writes PNGs + a manifest.csv (image, prompt, pathology) into
``artifacts/synth_images/``.

Usage:
    python project_b_synth/scripts/generate.py --lora checkpoints/sd_lora_openi/final/unet_lora \
        --per-class 200 --out artifacts/synth_images
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from common.logging_utils import get_logger
from common.paths import ARTIFACTS
from common.seed import set_seed

log = get_logger("synth.generate")

# Pathology-focused prompts oversampling rare classes for the utility study.
PROMPT_BANK: dict[str, list[str]] = {
    "cardiomegaly": [
        "frontal chest x-ray, cardiomegaly",
        "frontal chest x-ray, moderate cardiomegaly, no effusion",
    ],
    "effusion": [
        "frontal chest x-ray, small left pleural effusion",
        "frontal chest x-ray, moderate bilateral pleural effusions",
    ],
    "consolidation": [
        "frontal chest x-ray, right lower lobe consolidation",
        "frontal chest x-ray, left lower lobe consolidation with air bronchograms",
    ],
    "pneumothorax": [
        "frontal chest x-ray, right apical pneumothorax",
        "frontal chest x-ray, left tension pneumothorax",
    ],
    "edema": [
        "frontal chest x-ray, pulmonary edema with cephalization",
        "frontal chest x-ray, bilateral perihilar edema",
    ],
    "atelectasis": [
        "frontal chest x-ray, left lower lobe atelectasis",
        "frontal chest x-ray, right middle lobe atelectasis",
    ],
    "nodule": [
        "frontal chest x-ray, solitary pulmonary nodule right upper lobe",
        "frontal chest x-ray, small left mid lung nodule",
    ],
    "mass": [
        "frontal chest x-ray, right hilar mass",
        "frontal chest x-ray, left upper lobe mass",
    ],
    "no_finding": [
        "frontal chest x-ray, no acute cardiopulmonary abnormality",
        "frontal chest x-ray, normal heart size, clear lungs",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora", type=Path, required=True, help="Path to LoRA weights (unet_lora dir)")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--neg", default="blurry, low quality, watermark, text, natural photograph, color image")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "synth_images")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)

    import torch
    from diffusers import StableDiffusionPipeline
    from peft import PeftModel

    pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16, safety_checker=None)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "prompt", "pathology"])
        idx = 0
        for pathology, prompts in PROMPT_BANK.items():
            for i in tqdm(range(args.per_class), desc=pathology):
                prompt = prompts[i % len(prompts)]
                g = torch.Generator(device="cuda").manual_seed(args.seed + idx)
                img = pipe(
                    prompt,
                    negative_prompt=args.neg,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=g,
                ).images[0]
                fname = f"{pathology}_{i:05d}.png"
                img.save(args.out / fname)
                w.writerow([fname, prompt, pathology])
                idx += 1
    log.info("wrote %d images + %s", idx, manifest_path)


if __name__ == "__main__":
    main()
