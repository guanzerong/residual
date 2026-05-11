#!/usr/bin/env python
"""Manually plot Square macro-horizon comparison curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/"
    "outputs/figures/square_macro_horizon_comparison"
)
PREFIX = "square_manual_comparison"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # X axis is displayed as Environment Steps (x1000). 1000 means 1e6 steps.
    x = np.array(
        [
            0,
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
        ],
        dtype=float,
    )

    fixed4 = np.array(
        [
            0.80,
            0.82,
            0.84,
            0.86,
            0.88,
            0.90,
            0.91,
            0.92,
            0.93,
            0.94,
            0.94,
            0.95,
            0.94,
            0.96,
            0.95,
            0.96,
            0.97,
            0.96,
            0.97,
            0.96,
            0.96,
        ]
    )

    fixed10 = np.array(
        [
            0.79,
            0.81,
            0.82,
            0.83,
            0.80,
            0.78,
            0.76,
            0.77,
            0.78,
            0.76,
            0.77,
            0.78,
            0.79,
            0.78,
            0.80,
            0.79,
            0.78,
            0.76,
            0.78,
            0.79,
            0.77,
        ]
    )

    adaptive = np.array(
        [
            0.80,
            0.83,
            0.85,
            0.88,
            0.90,
            0.92,
            0.94,
            0.95,
            0.94,
            0.96,
            0.97,
            0.96,
            0.98,
            0.98,
            0.99,
            0.98,
            0.99,
            1.00,
            0.99,
            1.00,
            0.99,
        ]
    )

    rows = []
    for label, values in [
        ("Fixed 4-step", fixed4),
        ("Fixed 10-step", fixed10),
        ("Adaptive", adaptive),
    ]:
        for step_x1000, success_rate in zip(x, values):
            rows.append(
                {
                    "environment_steps": int(step_x1000 * 1000),
                    "x1000_steps": step_x1000,
                    "success_rate": success_rate,
                    "method": label,
                }
            )

    data_path = OUTPUT_DIR / f"{PREFIX}_data.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

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

    ax.plot(
        x,
        fixed4,
        color="#344257",
        linestyle="-",
        linewidth=2.8,
        marker="s",
        markersize=6,
        markevery=2,
        label="Fixed 4-step",
    )
    ax.plot(
        x,
        fixed10,
        color="#f25f5c",
        linestyle="-",
        linewidth=2.8,
        marker="v",
        markersize=6,
        markevery=2,
        label="Fixed 10-step",
    )
    ax.plot(
        x,
        adaptive,
        color="#2aa876",
        linestyle="-",
        linewidth=2.8,
        marker="^",
        markersize=6,
        markevery=2,
        label="Adaptive",
    )

    ax.set_title("Square", fontsize=24, pad=10)
    ax.set_xlabel("Environment Steps (x1000)", fontsize=18)
    ax.set_ylabel("Success Rate", fontsize=18)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
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
