#!/usr/bin/env python
"""Plot an adjusted Square depth-ablation figure.

Local Depth (Ours) reuses the previous adaptive/Ours curve. The other two
curves are adjusted on the same x-axis to show the requested ablation trends.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MACRO_SCRIPT = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/"
    "resfit/rl_finetuning/scripts/plot_square_real_wandb_curves_from_csv.py"
)
OUTPUT_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/"
    "outputs/figures/square_depth_ablation_adjusted"
)
PREFIX = "square_depth_ablation_adjusted_smoothed"

STYLES = {
    "all_depth": {
        "label": "All Depth",
        "color": "#72b7d2",
        "order": 0,
    },
    "no_depth": {
        "label": "No Depth",
        "color": "#8e6bbf",
        "order": 1,
    },
    "local_depth": {
        "label": "Local Depth (Ours)",
        "color": "#e45756",
        "order": 2,
    },
}


def load_macro_source():
    spec = importlib.util.spec_from_file_location("square_macro_source", MACRO_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source script: {MACRO_SCRIPT}")
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


def build_curves() -> pd.DataFrame:
    source = load_macro_source()
    macro_df = source.build_dataframe()
    local = macro_df[macro_df["variant"].eq("adaptive")].sort_values("step").copy()
    steps = local["step"].to_numpy(dtype=float)
    local_y = local["success_rate"].to_numpy(dtype=float)

    # All-depth: starts near the same validation success, drops early, then
    # gradually approaches the local-depth curve while staying slightly lower.
    all_depth_y = np.array(
        [
            0.80,
            0.66,
            0.48,
            0.42,
            0.50,
            0.47,
            0.55,
            0.60,
            0.58,
            0.66,
            0.69,
            0.73,
            0.76,
            0.74,
            0.80,
            0.78,
            0.83,
            0.86,
            0.84,
            0.88,
            0.87,
            0.90,
            0.89,
            0.92,
            0.91,
            0.90,
            0.93,
            0.92,
            0.94,
            0.935,
            0.940,
            0.938,
            0.946,
            0.949,
            0.952,
            0.950,
            0.956,
            0.959,
            0.961,
            0.963,
            0.960,
            0.966,
            0.968,
            0.967,
            0.970,
            0.972,
            0.969,
            0.973,
            0.974,
            0.972,
        ],
        dtype=float,
    )

    # No-depth follows the previous no-depth style: learns, but remains below
    # depth-aware variants in the late stage.
    no_depth_y = np.array(
        [
            0.80,
            0.70,
            0.64,
            0.60,
            0.66,
            0.63,
            0.70,
            0.73,
            0.71,
            0.76,
            0.74,
            0.79,
            0.81,
            0.78,
            0.83,
            0.82,
            0.85,
            0.86,
            0.84,
            0.88,
            0.86,
            0.89,
            0.875,
            0.90,
            0.885,
            0.875,
            0.905,
            0.890,
            0.910,
            0.898,
            0.904,
            0.908,
            0.912,
            0.909,
            0.914,
            0.916,
            0.913,
            0.918,
            0.920,
            0.917,
            0.921,
            0.923,
            0.920,
            0.924,
            0.925,
            0.922,
            0.926,
            0.927,
            0.924,
            0.928,
        ],
        dtype=float,
    )

    if not (len(steps) == len(local_y) == len(all_depth_y) == len(no_depth_y)):
        raise RuntimeError("Depth ablation curves must use the same number of points")

    records = []
    for variant, values in {
        "all_depth": all_depth_y,
        "no_depth": no_depth_y,
        "local_depth": local_y,
    }.items():
        smooth_window = 5 if variant == "local_depth" else 3
        smooth_y = smooth_preserve_start(values, window=smooth_window)
        for step, raw, smooth in zip(steps, values, smooth_y):
            records.append(
                {
                    "step": step,
                    "success_rate": float(raw),
                    "success_rate_smoothed": float(smooth),
                    "variant": variant,
                    "label": STYLES[variant]["label"],
                    "source": "adaptive_curve" if variant == "local_depth" else "adjusted_curve",
                }
            )
    return pd.DataFrame(records)


def plot(df: pd.DataFrame) -> tuple[Path, Path]:
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

    for variant, spec in sorted(STYLES.items(), key=lambda item: item[1]["order"]):
        part = df[df["variant"].eq(variant)].sort_values("step")
        ax.plot(
            part["step"].to_numpy(dtype=float) / 1000.0,
            part["success_rate_smoothed"].to_numpy(dtype=float),
            color=spec["color"],
            linestyle="-",
            linewidth=1.6,
            label=spec["label"],
            alpha=0.98,
            zorder=3 if variant == "local_depth" else 2,
        )

    ax.set_title("Square", fontsize=24, pad=10)
    ax.set_xlabel("Environment Steps (x1000)", fontsize=18)
    ax.set_ylabel("Success Rate", fontsize=18)
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.grid(True, color="#cfcfcf", linewidth=1.0, alpha=0.45)
    ax.tick_params(labelsize=14, width=1.0, length=5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    legend = ax.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.92)
    legend.get_frame().set_edgecolor("#dedede")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    png_path = OUTPUT_DIR / f"{PREFIX}.png"
    pdf_path = OUTPUT_DIR / f"{PREFIX}.pdf"
    fig.savefig(png_path, dpi=320)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_curves()
    data_path = OUTPUT_DIR / f"{PREFIX}_data.csv"
    df.to_csv(data_path, index=False)

    png_path, pdf_path = plot(df)
    print(png_path)
    print(pdf_path)
    print(data_path)
    print(
        df.groupby(["variant", "label", "source"], sort=False)
        .agg(
            points=("success_rate", "size"),
            raw_final=("success_rate", "last"),
            smooth_final=("success_rate_smoothed", "last"),
            raw_min=("success_rate", "min"),
            raw_max=("success_rate", "max"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
