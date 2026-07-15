"""YAML config loader with light dot-access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    def __getattr__(self, k: str) -> Any:
        if k not in self:
            raise AttributeError(k)
        v = self[k]
        return DotDict(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def load_config(path: str | Path) -> DotDict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return DotDict(raw)
