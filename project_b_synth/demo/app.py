"""Gradio demo: type a chest-X-ray findings prompt, get a synthetic CXR back.

Loads the trained LoRA weights on top of SD v1.5 and exposes a minimal UI. Always
displays a research-only disclaimer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DISCLAIMER = (
    "**Research use only.** Images are synthetic and are NOT valid for any clinical "
    "decision-making. Model may hallucinate anatomy or findings."
)


def _load_pipe(base: str, lora: Path):
    import torch
    from diffusers import StableDiffusionPipeline
    from peft import PeftModel

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=dtype, safety_checker=None)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build(base: str, lora: Path):
    import gradio as gr

    pipe = _load_pipe(base, lora)

    def generate(prompt: str, guidance: float, steps: int, seed: int):
        import torch

        gen = torch.Generator(device=pipe.device).manual_seed(int(seed))
        img = pipe(
            prompt or "frontal chest x-ray, no acute abnormality",
            negative_prompt="blurry, low quality, watermark, text, color image, natural photograph",
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=gen,
        ).images[0]
        return img

    with gr.Blocks(title="CXR Synth Demo") as demo:
        gr.Markdown(f"# Text-to-CXR Synth Demo\n{DISCLAIMER}")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Findings prompt", value="frontal chest x-ray, cardiomegaly, mild pulmonary edema")
                guidance = gr.Slider(1.0, 12.0, value=6.0, label="Guidance scale")
                steps = gr.Slider(10, 60, value=30, step=1, label="Inference steps")
                seed = gr.Number(value=1234, label="Seed")
                btn = gr.Button("Generate")
            out = gr.Image(label="Synthetic CXR", type="pil")
        btn.click(generate, inputs=[prompt, guidance, steps, seed], outputs=out)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--lora", type=Path, default=Path("checkpoints/sd_lora_openi/final/unet_lora"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build(args.base, args.lora).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
