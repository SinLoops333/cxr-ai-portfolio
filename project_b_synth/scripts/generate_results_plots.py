"""Generate portfolio result figures for Flagship B.

Reads ``artifacts/utility_report.json`` and ``artifacts/memorization.json``
(plus ``artifacts/memorization_sims.npz`` when available) and writes:

- ``artifacts/utility_comparison.png``
- ``artifacts/memorization_histogram.png``

Usage (from repo root)::

    python project_b_synth/scripts/generate_results_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
UTILITY_JSON = ARTIFACTS / "utility_report.json"
MEM_JSON = ARTIFACTS / "memorization.json"
MEM_SIMS = ARTIFACTS / "memorization_sims.npz"


def _load_utility(path: Path = UTILITY_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_memorization(path: Path = MEM_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_utility_comparison(report: dict, out: Path) -> None:
    sns.set_theme(style="whitegrid")
    class_names = list(report["class_names"])
    rare = set(report.get("rare_classes", []))
    prevalence = {c: p for c, p in zip(class_names, report["prevalence"], strict=False)}

    # Prefer rare_classes list; fall back to prevalence < 0.05.
    def is_rare(name: str) -> bool:
        return name in rare or prevalence.get(name, 1.0) < 0.05

    rows = []
    for name in sorted(class_names):
        if name not in report["real"]["per_class"] or name not in report["real_synth"]["per_class"]:
            continue
        label = f"{name}*" if is_rare(name) else name
        for regime, key in [("real", "real"), ("real+synth", "real_synth")]:
            d = report[key]["per_class"][name]
            rows.append(
                {
                    "class": label,
                    "regime": regime,
                    "auroc": d["auroc"],
                    "yerr_lo": max(0.0, d["auroc"] - d["ci_lo"]),
                    "yerr_hi": max(0.0, d["ci_hi"] - d["auroc"]),
                }
            )

    classes = sorted({r["class"] for r in rows})
    x = np.arange(len(classes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))

    for offset, regime, color in [(-width / 2, "real", "#4C72B0"), (width / 2, "real+synth", "#DD8452")]:
        subset = {r["class"]: r for r in rows if r["regime"] == regime}
        means = [subset[c]["auroc"] for c in classes]
        yerr = np.array([[subset[c]["yerr_lo"] for c in classes], [subset[c]["yerr_hi"] for c in classes]])
        ax.bar(
            x + offset,
            means,
            width=width,
            label=regime,
            color=color,
            yerr=yerr,
            capsize=3,
            error_kw={"elinewidth": 1.0},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-class AUROC: real vs real+synth (95% bootstrap CI)\n* = rare class (prevalence < 0.05)")
    ax.legend(title="Regime")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def plot_memorization(report: dict, out: Path, sims_path: Path = MEM_SIMS) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    if sims_path.exists():
        data = np.load(sims_path)
        synth = np.asarray(data["synth_sim"], dtype=float)
        holdout = np.asarray(data["holdout_sim"], dtype=float)
        sns.kdeplot(holdout, ax=ax, fill=True, alpha=0.35, color="#4C72B0", label="Real holdout → train", cut=0)
        sns.kdeplot(synth, ax=ax, fill=True, alpha=0.35, color="#DD8452", label="Synth → train", cut=0)
        ax.axvline(float(np.mean(holdout)), color="#4C72B0", linestyle="--", linewidth=1.5)
        ax.axvline(float(np.mean(synth)), color="#DD8452", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Top-1 CLIP cosine similarity to training gallery")
        ax.set_ylabel("Density")
    else:
        # Fallback: plot summary markers when raw sims are unavailable.
        h = report["holdout_top1_stats"]
        s = report["synth_top1_stats"]
        ax.bar(
            ["Real holdout → train", "Synth → train"],
            [h["mean"], s["mean"]],
            yerr=[[h["mean"] - h.get("median", h["mean"]), s["mean"] - s.get("median", s["mean"])],
                  [h["p95"] - h["mean"], s["p95"] - s["mean"]]],
            color=["#4C72B0", "#DD8452"],
            capsize=6,
        )
        ax.set_ylabel("Top-1 CLIP cosine similarity")
        ax.set_ylim(0.0, 1.05)
        print(f"warning: {sims_path} missing — plotted summary bars instead of KDE")

    delta = report.get("delta_mean", float("nan"))
    ax.set_title(f"Memorization audit (CLIP ViT-B/32)\nΔ mean (synth − holdout) = {delta:+.4f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"memorization delta_mean = {delta:+.6f}")
    if delta > 0:
        print("interpretation: positive Δ — synth closer to train than real holdout (memorization warning)")
    else:
        print("interpretation: non-positive Δ — no memorization flag vs. real-holdout baseline")


def main() -> None:
    utility = _load_utility()
    mem = _load_memorization()
    plot_utility_comparison(utility, ARTIFACTS / "utility_comparison.png")
    plot_memorization(mem, ARTIFACTS / "memorization_histogram.png")


if __name__ == "__main__":
    main()
