"""LoRA fine-tune Stable Diffusion v1.5 on OpenI chest X-rays.

Runs on a single 24GB consumer GPU (fits in ~8-10GB VRAM at 512px, bs=2, fp16).

Design notes:
- LoRA only on UNet cross-attention projections (Q,K,V, out) — smallest set that still
  learns radiographic style/content. Rank 16 / alpha 16 (aggressive but stable at 512px).
- ``random_flip=False`` because chest X-rays have handedness (situs solitus); flipping
  a healthy CXR would place the heart on the wrong side and pollute training.
- Uses HuggingFace ``diffusers`` + ``peft`` + ``accelerate``. Keeps things dependency-light
  (no bespoke training loop framework).

Usage:
    accelerate launch project_b_synth/scripts/train_sd_lora.py \
        --config project_b_synth/configs/lora.yaml
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common.config import load_config
from common.logging_utils import get_logger, init_wandb
from common.seed import set_seed

log = get_logger("synth.lora")


class OpenICaptionsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        image_col: str,
        caption_col: str,
        resolution: int,
        center_crop: bool,
    ):
        images_dir = Path(images_dir)
        # Filter rows where the image file exists (CSV may reference missing PNGs).
        before = len(df)
        mask = df[image_col].map(lambda name: (images_dir / name).exists())
        df = df[mask].reset_index(drop=True)
        if len(df) == 0:
            raise FileNotFoundError(
                f"No training images found under {images_dir} "
                f"(checked {before} CSV rows). Check data.images_dir in the config."
            )
        if len(df) < before:
            log.warning("dropped %d/%d rows with missing image files", before - len(df), before)
        log.info("dataset size: %d images under %s", len(df), images_dir)
        self.df = df
        self.images_dir = images_dir
        self.image_col = image_col
        self.caption_col = caption_col
        tfm = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
        tfm.append(
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
        )
        tfm += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.tfm = transforms.Compose(tfm)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = self.images_dir / row[self.image_col]
        img = Image.open(img_path).convert("RGB")  # SD expects 3-channel input
        return {"pixel_values": self.tfm(img), "caption": str(row[self.caption_col])}


def _collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # Lazy-import heavy libs so tests/CI don't need them installed
    from accelerate import Accelerator
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        UNet2DConditionModel,
    )
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        init_wandb("cxr-synth", "sd-lora-openi", dict(cfg))

    base = cfg.model.base
    log.info("loading base model %s", base)
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    log.info("trainable params: %.2fM", sum(p.numel() for p in trainable) / 1e6)

    df = pd.read_csv(cfg.data.splits_csv)
    df = df[df[cfg.data.split_col] == cfg.data.train_split]
    ds = OpenICaptionsDataset(
        df,
        Path(cfg.data.images_dir),
        cfg.data.image_col,
        cfg.data.caption_col,
        cfg.data.resolution,
        cfg.data.center_crop,
    )
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=1, collate_fn=_collate)

    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.learning_rate)
    total_steps = cfg.train.max_steps
    warmup = cfg.train.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    unet, optimizer, dl, scheduler = accelerator.prepare(unet, optimizer, dl, scheduler)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=total_steps, disable=not accelerator.is_main_process)
    unet.train()
    while step < total_steps:
        for batch in dl:
            with accelerator.accumulate(unet):
                pixels = batch["pixel_values"].to(accelerator.device)
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                    tokens = tokenizer(
                        batch["captions"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    enc_hidden = text_encoder(tokens.input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                pred = unet(noisy, timesteps, encoder_hidden_states=enc_hidden).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                pbar.update(1)
                if accelerator.is_main_process and step % cfg.output.log_every == 0:
                    log.info("step %d loss %.4f lr %.2e", step, loss.item(), scheduler.get_last_lr()[0])
                    try:
                        import wandb
                        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)
                    except Exception:
                        pass
                if step % cfg.train.checkpoint_every == 0 and accelerator.is_main_process:
                    _save(accelerator, unet, tokenizer, text_encoder, vae, out_dir / f"step_{step}", cfg)
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    if accelerator.is_main_process:
        _save(accelerator, unet, tokenizer, text_encoder, vae, out_dir / "final", cfg)
        log.info("training complete. weights at %s/final", out_dir)


def _save(accelerator, unet, tokenizer, text_encoder, vae, out_dir: Path, cfg) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(unet)
    unwrapped.save_pretrained(out_dir / "unet_lora")
    log.info("saved LoRA weights -> %s", out_dir / "unet_lora")
    # quick validation grid
    try:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(cfg.model.base, torch_dtype=torch.float16, safety_checker=None).to(accelerator.device)
        pipe.unet = unwrapped
        for i, prompt in enumerate(cfg.train.validation_prompts):
            img = pipe(prompt, num_inference_steps=25, guidance_scale=6.0).images[0]
            img.save(out_dir / f"val_{i}.png")
        del pipe
        torch.cuda.empty_cache()
    except Exception as e:
        log.warning("validation grid failed: %s", e)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
