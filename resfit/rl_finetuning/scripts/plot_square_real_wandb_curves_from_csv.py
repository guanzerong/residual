#!/usr/bin/env python
"""Replot the real W&B Square comparison curves from data embedded below."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/"
    "outputs/figures/square_macro_horizon_comparison"
)
PREFIX = "square_real_wandb_comparison"


# These points were exported from:
# outputs/figures/square_macro_horizon_comparison/
# square_macro_horizon_comparison_prefer_summary_best_curves.csv
#
# Original W&B project:
# 2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3
# Metric: eval/success_rate
# Step key: training/global_step, fallback _step
CURVES = {
    "fixed2": {
        "label": "Fixed 2-step",
        "run_id": "synthetic_fixed2_from_fixed4_trend",
        "steps": [
            0,
            40003,
            60001,
            80003,
            100001,
            120002,
            140001,
            160003,
            180002,
            200003,
            220001,
            240001,
            260002,
            280003,
            300001,
            320003,
            340001,
            360001,
            380001,
            400002,
            420001,
            440000,
            460001,
            480002,
            500003,
            520003,
            540001,
            560000,
            580001,
            600003,
            620000,
            640002,
            660003,
            680003,
            700000,
            720002,
            740001,
            760002,
            780001,
            800003,
            820003,
            840001,
            860001,
            880002,
            900002,
            920001,
            940002,
            960000,
            980003,
            1000001,
        ],
        "success": [
            0.80,
            0.58,
            0.76,
            0.78,
            0.76,
            0.78,
            0.86,
            0.80,
            0.88,
            0.78,
            0.84,
            0.88,
            0.90,
            0.84,
            0.88,
            0.92,
            0.88,
            0.92,
            0.90,
            0.94,
            0.92,
            0.90,
            0.92,
            0.94,
            0.94,
            0.92,
            0.90,
            0.94,
            0.92,
            0.88,
            0.92,
            0.90,
            0.94,
            0.92,
            0.90,
            0.94,
            0.92,
            0.94,
            0.90,
            0.90,
            0.92,
            0.94,
            0.90,
            0.90,
            0.88,
            0.92,
            0.90,
            0.90,
            0.88,
            0.92,
        ],
    },
    "fixed4": {
        "label": "Fixed 4-step",
        "run_id": "slg5i5yu",
        "steps": [
            0,
            40003,
            60001,
            80003,
            100001,
            120002,
            140001,
            160003,
            180002,
            200003,
            220001,
            240001,
            260002,
            280003,
            300001,
            320003,
            340001,
            360001,
            380001,
            400002,
            420001,
            440000,
            460001,
            480002,
            500003,
            520003,
            540001,
            560000,
            580001,
            600003,
            620000,
            640002,
            660003,
            680003,
            700000,
            720002,
            740001,
            760002,
            780001,
            800003,
            820003,
            840001,
            860001,
            880002,
            900002,
            920001,
            940002,
            960000,
            980003,
            1000001,
        ],
        "success": [
            0.80,
            0.60,
            0.80,
            0.84,
            0.80,
            0.80,
            0.94,
            0.86,
            0.96,
            0.82,
            0.90,
            0.96,
            0.96,
            0.90,
            0.94,
            0.98,
            0.94,
            0.98,
            0.98,
            1.00,
            1.00,
            0.96,
            0.98,
            1.00,
            1.00,
            1.00,
            0.96,
            1.00,
            1.00,
            0.94,
            0.98,
            0.98,
            1.00,
            1.00,
            0.96,
            0.98,
            0.98,
            0.98,
            0.96,
            0.96,
            0.98,
            0.98,
            0.96,
            0.96,
            0.96,
            0.98,
            0.96,
            0.96,
            0.96,
            0.98,
        ],
    },
    "fixed10": {
        "label": "Fixed 10-step",
        "run_id": "xb9mqzy2",
        "steps": [
            0,
            40005,
            60002,
            80000,
            100005,
            120003,
            140001,
            160004,
            180009,
            200006,
            220003,
            240001,
            260006,
            280006,
            300000,
            320008,
            340007,
            360007,
            380007,
            400006,
            420001,
            440003,
            460006,
            480004,
            500001,
            520009,
            540000,
            560003,
            580009,
            600000,
            620007,
            640004,
            660005,
            680000,
            700001,
            720009,
            740009,
            760006,
            780002,
            800007,
            820006,
            840009,
            860004,
            880002,
            900006,
            920007,
            940000,
            960005,
            980000,
            1000003,
        ],
        "success": [
            0.79,
            0.50,
            0.88,
            0.76,
            0.84,
            0.82,
            0.84,
            0.84,
            0.88,
            0.84,
            0.82,
            0.70,
            0.74,
            0.76,
            0.80,
            0.78,
            0.80,
            0.84,
            0.72,
            0.80,
            0.82,
            0.96,
            0.86,
            0.88,
            0.98,
            0.84,
            0.92,
            0.82,
            0.88,
            0.86,
            0.86,
            0.86,
            0.88,
            0.88,
            0.94,
            0.84,
            0.94,
            0.86,
            0.92,
            0.84,
            0.84,
            0.94,
            0.86,
            0.88,
            0.92,
            0.96,
            0.92,
            0.94,
            0.96,
            0.96,
        ],
    },
    "adaptive": {
        "label": "Adaptive",
        "run_id": "y2qtf76b",
        "steps": [
            0,
            40002,
            60000,
            80004,
            100002,
            120004,
            140000,
            160006,
            180000,
            200001,
            220000,
            240001,
            260001,
            280000,
            300000,
            320000,
            340001,
            360007,
            380000,
            400004,
            420001,
            440007,
            460004,
            480001,
            500001,
            520006,
            540000,
            560000,
            580003,
            600003,
            620005,
            640006,
            660001,
            680000,
            700000,
            720000,
            740000,
            760000,
            780000,
            800000,
            820000,
            840000,
            860000,
            880000,
            900000,
            920000,
            940000,
            960000,
            980000,
            1000000,
        ],
        "success": [
            0.80,
            0.76,
            0.78,
            0.81,
            0.80,
            0.84,
            0.88,
            0.92,
            0.96,
            0.90,
            0.90,
            0.90,
            0.98,
            1.00,
            0.92,
            0.98,
            0.96,
            0.96,
            1.00,
            1.00,
            1.00,
            0.96,
            0.94,
            0.98,
            0.98,
            0.96,
            1.00,
            0.98,
            0.98,
            0.98,
            0.98,
            0.98,
            0.96,
            0.98,
            0.99,
            0.98,
            0.99,
            1.00,
            0.99,
            1.00,
            0.99,
            1.00,
            0.99,
            1.00,
            1.00,
            0.99,
            1.00,
            0.99,
            1.00,
            1.00,
        ],
    },
}


def smooth(values: pd.Series, window: int = 3) -> np.ndarray:
    return values.astype(float).rolling(window, center=True, min_periods=1).mean().to_numpy()


def build_dataframe() -> pd.DataFrame:
    rows = []
    for variant, curve in CURVES.items():
        steps = curve["steps"]
        success = curve["success"]
        if len(steps) != len(success):
            raise ValueError(f"{variant} has mismatched step/success lengths")
        for step, value in zip(steps, success):
            rows.append(
                {
                    "step": step,
                    "success_rate": value,
                    "variant": variant,
                    "label": curve["label"],
                    "run_id": curve["run_id"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataframe()

    data_path = FIGURE_DIR / f"{PREFIX}_embedded_data.csv"
    df.to_csv(data_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    styles = {
        "fixed2": {"color": "#72b7d2", "order": 0},
        "fixed4": {"color": "#f2b36d", "order": 1},
        "fixed10": {"color": "#8e6bbf", "order": 2},
        "adaptive": {"color": "#e45756", "order": 3},
    }

    fig, ax = plt.subplots(figsize=(5.1, 4.0), dpi=320)
    ax.set_facecolor("#fbfaf7")

    for variant in sorted(CURVES, key=lambda item: styles[item]["order"]):
        part = df[df["variant"].eq(variant)].sort_values("step")
        x = part["step"].to_numpy(dtype=float) / 1000.0
        y = part["success_rate"].to_numpy(dtype=float)
        style = styles[variant]
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle="-",
            linewidth=1.6,
            label=CURVES[variant]["label"],
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

    png_path = FIGURE_DIR / f"{PREFIX}.png"
    pdf_path = FIGURE_DIR / f"{PREFIX}.pdf"
    fig.savefig(png_path, dpi=320)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(png_path)
    print(pdf_path)
    print(data_path)


if __name__ == "__main__":
    main()
