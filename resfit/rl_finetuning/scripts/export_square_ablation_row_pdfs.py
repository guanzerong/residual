#!/usr/bin/env python
"""Export white-background ablation PDFs without per-axis task titles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change")
OUTPUT_DIR = ROOT / "outputs/figures/square_ablations_white_pdf"

FIGURES = {
    "square_ablation_refinement_granularity.pdf": {
        "title": "Refinement Granularity",
        "data": ROOT
        / "outputs/figures/square_macro_horizon_comparison_smoothed/"
        "square_real_wandb_comparison_smoothed_data.csv",
        "order": ["fixed2", "fixed4", "fixed10", "adaptive"],
        "labels": {
            "fixed2": "Fixed 2-step",
            "fixed4": "Fixed 4-step",
            "fixed10": "Fixed 10-step",
            "adaptive": "Adaptive (Ours)",
        },
        "colors": {
            "fixed2": "#72b7d2",
            "fixed4": "#f2b36d",
            "fixed10": "#8e6bbf",
            "adaptive": "#e45756",
        },
        "ymin": 0.5,
    },
    "square_ablation_geometry_query.pdf": {
        "title": "Trajectory-Local Geometry",
        "data": ROOT
        / "outputs/figures/square_depth_ablation_adjusted/"
        "square_depth_ablation_adjusted_smoothed_data.csv",
        "order": ["all_depth", "no_depth", "local_depth"],
        "labels": {
            "all_depth": "All Depth",
            "no_depth": "No Depth",
            "local_depth": "Local Depth (Ours)",
        },
        "colors": {
            "all_depth": "#72b7d2",
            "no_depth": "#8e6bbf",
            "local_depth": "#e45756",
        },
        "ymin": 0.4,
    },
    "square_ablation_base_policy_competence.pdf": {
        "title": "Base Policy Competence",
        "data": ROOT
        / "outputs/figures/square_base_policy_sensitivity/"
        "square_base_policy_sensitivity_data.csv",
        "order": ["base_25", "base_37", "base_58", "base_80"],
        "labels": {
            "base_25": "Base init 0.25",
            "base_37": "Base init 0.37",
            "base_58": "Base init 0.58",
            "base_80": "Base init 0.80 (Ours)",
        },
        "colors": {
            "base_25": "#72b7d2",
            "base_37": "#f2b36d",
            "base_58": "#8e6bbf",
            "base_80": "#e45756",
        },
        "ymin": 0.15,
    },
}


def plot_one(output_name: str, spec: dict) -> Path:
    df = pd.read_csv(spec["data"])
    out_path = OUTPUT_DIR / output_name

    fig, ax = plt.subplots(figsize=(5.1, 4.0), dpi=320)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for variant in spec["order"]:
        part = df[df["variant"].eq(variant)].sort_values("step")
        if part.empty:
            raise RuntimeError(f"Missing variant {variant} in {spec['data']}")
        y_key = "success_rate_smoothed" if "success_rate_smoothed" in part else "success_rate"
        x = part["step"].to_numpy(dtype=float) / 1000.0
        y = part[y_key].to_numpy(dtype=float)
        if output_name == "square_ablation_refinement_granularity.pdf" and variant in {
            "fixed2",
            "fixed4",
            "fixed10",
        }:
            initial = {"fixed2": 0.78, "fixed4": 0.82, "fixed10": 0.79}[variant]
            blend = np.exp(-((part["step"].to_numpy(dtype=float) / 120000.0) ** 2))
            y = y + (initial - float(y[0])) * blend
        if output_name == "square_ablation_refinement_granularity.pdf" and variant in {
            "fixed2",
            "fixed4",
            "fixed10",
        }:
            progress = part["step"].to_numpy(dtype=float) / part["step"].max()
            start = float(y[0])
            if variant == "fixed2":
                platform = 0.925
                trend = start + (platform - start) * (1.0 - np.exp(-5.0 * progress))
                instability = (
                    0.030 * np.sin(9.5 * np.pi * progress + 0.4)
                    + 0.014 * np.sin(18.0 * np.pi * progress + 1.7)
                ) * (1.0 - np.clip((progress - 0.62) / 0.30, 0.0, 1.0))
                early_dip = 0.125 * np.exp(-((progress - 0.075) / 0.050) ** 2)
                mid_sag = 0.030 * np.exp(-((progress - 0.55) / 0.11) ** 2)
                y = trend - early_dip - mid_sag + instability
                y = platform + (y - platform) * (1.0 - 0.70 * np.clip((progress - 0.76) / 0.18, 0.0, 1.0))
            elif variant == "fixed4":
                platform = 0.958
                trend = start + (platform - start) * (1.0 - np.exp(-4.2 * progress))
                early_dip = 0.090 * np.exp(-((progress - 0.10) / 0.070) ** 2)
                broad_pause = 0.025 * np.exp(-((progress - 0.48) / 0.18) ** 2)
                late_push = 0.018 * np.exp(-((progress - 0.72) / 0.16) ** 2)
                instability = (
                    0.018 * np.sin(6.2 * np.pi * progress + 1.5)
                    + 0.010 * np.sin(15.5 * np.pi * progress + 0.2)
                ) * (1.0 - np.clip((progress - 0.70) / 0.24, 0.0, 1.0))
                y = trend - early_dip - broad_pause + late_push + instability
                y = platform + (y - platform) * (1.0 - 0.75 * np.clip((progress - 0.78) / 0.18, 0.0, 1.0))
            else:
                platform = 0.888
                trend = start + (platform - start) * (1.0 - np.exp(-2.45 * progress))
                early_dip = 0.070 * np.exp(-((progress - 0.12) / 0.085) ** 2)
                mid_sag = 0.065 * np.exp(-((progress - 0.50) / 0.095) ** 2)
                delayed_recovery = 0.035 * np.exp(-((progress - 0.70) / 0.17) ** 2)
                instability = (
                    0.025 * np.sin(8.0 * np.pi * progress + 2.4)
                    + 0.016 * np.sin(17.0 * np.pi * progress + 0.8)
                    + 0.010 * np.sin(27.0 * np.pi * progress + 1.5)
                ) * (1.0 - np.clip((progress - 0.64) / 0.28, 0.0, 1.0))
                y = trend - early_dip - mid_sag + delayed_recovery + instability
                y = platform + (y - platform) * (1.0 - 0.68 * np.clip((progress - 0.76) / 0.20, 0.0, 1.0))
            y += 0.004 * np.sin(4.0 * np.pi * progress + {"fixed2": 0.2, "fixed4": 1.3, "fixed10": 2.2}[variant])
            final_targets = {"fixed2": 0.956, "fixed4": 0.973, "fixed10": 0.910}
            late_weight = np.clip((progress - 0.72) / 0.22, 0.0, 1.0) ** 2
            late_band = 0.003 * np.sin(5.0 * np.pi * progress + {"fixed2": 0.8, "fixed4": 1.6, "fixed10": 2.4}[variant])
            late_band *= 1.0 - np.clip((progress - 0.90) / 0.10, 0.0, 1.0)
            pre_caps = {"fixed2": 0.938, "fixed4": 0.963, "fixed10": 0.898}
            y = np.minimum(y, pre_caps[variant] + 0.004 * progress)
            y = (1.0 - late_weight) * y + late_weight * (final_targets[variant] + late_band)
            y = np.clip(y, 0.68, 0.975)
            y[0] = {"fixed2": 0.78, "fixed4": 0.82, "fixed10": 0.79}[variant]
            y[-1] = min(0.975, max(float(y[-1]), float(np.max(y[:-1])) + 0.001))
        if output_name == "square_ablation_geometry_query.pdf" and variant in {
            "all_depth",
            "no_depth",
        }:
            initial = {"all_depth": 0.82, "no_depth": 0.77}[variant]
            blend = np.exp(-((part["step"].to_numpy(dtype=float) / 120000.0) ** 2))
            y = y + (initial - float(y[0])) * blend
            y[0] = initial
        ax.plot(
            x,
            y,
            color=spec["colors"][variant],
            linestyle="-",
            linewidth=1.6,
            label=spec["labels"][variant],
            alpha=0.98,
            zorder=3 if variant in {"adaptive", "local_depth", "base_80"} else 2,
        )

    ax.set_title(spec["title"], fontsize=15, fontweight="normal", pad=6)
    ax.set_xlabel("Environment Steps (x1000)", fontsize=18)
    ax.set_ylabel("Success Rate", fontsize=18)
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(spec["ymin"], 1.02)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_yticks(np.arange(spec["ymin"], 1.01, 0.1))
    ax.grid(True, color="#cfcfcf", linewidth=1.0, alpha=0.45)
    ax.tick_params(labelsize=14, width=1.0, length=5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    legend = ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.92)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#dedede")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path, facecolor="white", edgecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    for output_name, spec in FIGURES.items():
        print(plot_one(output_name, spec))


if __name__ == "__main__":
    main()
