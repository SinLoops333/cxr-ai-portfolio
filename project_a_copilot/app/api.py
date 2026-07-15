"""FastAPI service for the CXR Copilot.

Endpoints:
    GET  /health           -> {status, backend}
    POST /predict          -> multipart/form-data with 'image' and optional 'question'
"""

from __future__ import annotations

import io
import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from common.logging_utils import get_logger

log = get_logger("copilot.api")

app = FastAPI(title="CXR Copilot", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_PIPELINE: Any | None = None


def _pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from .pipeline import default_pipeline
        cfg = os.environ.get("COPILOT_CONFIG", "project_a_copilot/configs/copilot.yaml")
        _PIPELINE = default_pipeline(cfg)
    return _PIPELINE


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "config": os.environ.get("COPILOT_CONFIG", "project_a_copilot/configs/copilot.yaml")}


@app.post("/predict")
async def predict(image: UploadFile = File(...), question: str = Form("")):
    try:
        raw = await image.read()
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}") from e
    return _pipeline().run(img, question)
