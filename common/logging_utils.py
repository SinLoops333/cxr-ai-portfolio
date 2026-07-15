"""Logging + optional W&B initialization."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(h)
    logger.propagate = False
    return logger


def init_wandb(project: str, run_name: str | None = None, config: dict[str, Any] | None = None):
    """Initialize wandb if WANDB_MODE is not 'disabled' and wandb is installed."""
    if os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(project=project, name=run_name, config=config or {})
