#!/usr/bin/env python
"""Plot a smoothed Square comparison figure from embedded curve data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_SCRIPT = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/"
    "resfit/rl_finetuning/scripts/plot_square_real_wandb_curves_from_csv.py"
)
OUTPUT_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/"
    "outputs/figures/square_macro_horizon_comparison_smoothed"
)
PREFIX = "square_real_wandb_comparison_smoothed"


def load_source_module():
    spec = importlib.util.spec_from_file_location("square_real_curves_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smooth_preserve_start(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) <= 2:
        return values
    smoothed = values.copy()
    smoothed[1:] = (
        pd.Series(values[1:])
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    return smoothed


def main() -> None:
    source = load_source_module()
    df = source.build_dataframe()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    styles = {
        "fixed2": {"color": "#72b7d2", "order": 0},
        "fixed4": {"color": "#f2b36d", "order": 1},
        "fixed10": {"color": "#8e6bbf", "order": 2},
        "adaptive": {"color": "#e45756", "order": 3},
    }

    smoothed_rows = []
    for variant in sorted(source.CURVES, key=lambda item: styles[item]["order"]):
        part = df[df["variant"].eq(variant)].sort_values("step").copy()
        y_smooth = smooth_preserve_start(part["success_rate"].to_numpy(dtype=float), window=5)
        part["success_rate_smoothed"] = y_smooth
        smoothed_rows.append(part)

    smoothed_df = pd.concat(smoothed_rows, ignore_index=True)
    data_path = OUTPUT_DIR / f"{PREFIX}_data.csv"
    smoothed_df.to_csv(data_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.1, 4.0), dpi=320)
    ax.set_facecolor("#fbfaf7")

    for variant in sorted(source.CURVES, key=lambda item: styles[item]["order"]):
        part = smoothed_df[smoothed_df["variant"].eq(variant)].sort_values("step")
        x = part["step"].to_numpy(dtype=float) / 1000.0
        y = part["success_rate_smoothed"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=styles[variant]["color"],
            linestyle="-",
            linewidth=1.6,
            label="Adaptive (Ours)" if variant == "adaptive" else source.CURVES[variant]["label"],
            alpha=0.98,
            zorder=3 if variant == "adaptive" else 2,
        )

    ax.set_title("Square", fontsize=24, pad=10)
    ax.set_xlabel("Environment Steps (x1000)", fontsize=18)
    ax.set_ylabel("Success Rate", fontsize=18)
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.grid(True, color="#cfcfcf", linewidth=1.0, alpha=0.45)
    ax.tick_params(labelsize=14, width=1.0, length=5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    legend = ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.92)
    legend.get_frame().set_edgecolor("#dedede")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{PREFIX}.png"
    pdf_path = OUTPUT_DIR / f"{PREFIX}.pdf"
    fig.savefig(png_path, dpi=320)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(png_path)
    print(pdf_path)
    print(data_path)


if __name__ == "__main__":
    main()
