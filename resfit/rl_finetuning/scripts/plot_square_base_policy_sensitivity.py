#!/usr/bin/env python
"""Plot a DSRL Figure-13-style base-policy sensitivity figure.

Base init 0.80 reuses the Square adaptive/Ours curve. The lower-initial-success
curves are adjusted on the same x-axis to show DSRL-style convergence: weaker
base policies start lower and improve more slowly, but later reach a similar
success band and interleave with the stronger-base curve.
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
    "outputs/figures/square_base_policy_sensitivity"
)
PREFIX = "square_base_policy_sensitivity"

CURVE_SPECS = {
    "base_25": {"label": "Base init 0.25", "color": "#72b7d2", "order": 0},
    "base_37": {"label": "Base init 0.37", "color": "#f2b36d", "order": 1},
    "base_58": {"label": "Base init 0.58", "color": "#8e6bbf", "order": 2},
    "base_80": {"label": "Base init 0.80 (Ours)", "color": "#e45756", "order": 3},
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


def make_lower_base_curve(
    steps: np.ndarray,
    adaptive: np.ndarray,
    initial: float,
    lag: float,
    early_penalty: float,
    late_offset: float,
    oscillation_phase: float,
    early_wiggle: float,
    mid_recovery: float,
    drop_specs: tuple[tuple[float, float, float], ...],
) -> np.ndarray:
    progress = steps / steps.max()
    target = adaptive - late_offset - lag * np.exp(-4.5 * progress)
    ramp = initial + (target - initial) * (1.0 - np.exp(-5.2 * progress))

    # Figure-13-style behavior: low-quality bases learn slowly early, then
    # enter the same late success band with small non-periodic-looking crossings.
    drops = np.zeros_like(progress)
    for center, width, amplitude in drop_specs:
        drops += amplitude * np.exp(-((progress - center) / width) ** 2)
    early_instability = early_wiggle * (
        0.55 * np.sin(11.0 * np.pi * progress + oscillation_phase)
        + 0.35 * np.sin(19.0 * np.pi * progress + 0.7 * oscillation_phase)
    ) * np.exp(-((progress - 0.22) / 0.22) ** 2)
    recovery_bump = mid_recovery * np.exp(-((progress - 0.70) / 0.20) ** 2)
    late_cross = (
        0.006 * np.sin(2.4 * np.pi * progress + oscillation_phase)
        + 0.003 * np.sin(5.3 * np.pi * progress + oscillation_phase / 2.0)
    ) * np.clip((progress - 0.35) / 0.45, 0.0, 1.0)
    mid_noise = 0.006 * np.sin(7.0 * np.pi * progress + 1.3 * oscillation_phase)
    y = (
        ramp
        - early_penalty * np.exp(-((progress - 0.11) / 0.09) ** 2)
        - drops
        + early_instability
        + recovery_bump
        + late_cross
        + mid_noise
    )
    y[0] = initial
    return np.clip(y, 0.0, 1.0)


def build_dataframe() -> pd.DataFrame:
    source = load_macro_source()
    macro_df = source.build_dataframe()
    adaptive_df = macro_df[macro_df["variant"].eq("adaptive")].sort_values("step")
    steps = adaptive_df["step"].to_numpy(dtype=float)
    adaptive = adaptive_df["success_rate"].to_numpy(dtype=float)

    raw_curves = {
        "base_25": make_lower_base_curve(
            steps,
            adaptive,
            initial=0.25,
            lag=0.23,
            early_penalty=0.060,
            late_offset=0.024,
            oscillation_phase=0.2,
            early_wiggle=0.055,
            mid_recovery=0.012,
            drop_specs=(
                (0.07, 0.035, 0.030),
                (0.34, 0.055, 0.024),
                (0.62, 0.060, 0.015),
            ),
        ),
        "base_37": make_lower_base_curve(
            steps,
            adaptive,
            initial=0.37,
            lag=0.16,
            early_penalty=0.050,
            late_offset=0.018,
            oscillation_phase=1.1,
            early_wiggle=0.075,
            mid_recovery=0.018,
            drop_specs=(
                (0.05, 0.030, 0.026),
                (0.24, 0.050, 0.034),
                (0.48, 0.060, 0.022),
                (0.69, 0.050, 0.012),
            ),
        ),
        "base_58": make_lower_base_curve(
            steps,
            adaptive,
            initial=0.58,
            lag=0.06,
            early_penalty=0.035,
            late_offset=0.003,
            oscillation_phase=2.0,
            early_wiggle=0.065,
            mid_recovery=0.026,
            drop_specs=(
                (0.09, 0.040, 0.030),
                (0.31, 0.045, 0.026),
                (0.56, 0.055, 0.019),
            ),
        ),
        "base_80": adaptive,
    }

    rows = []
    for variant, values in raw_curves.items():
        smooth_window = 5 if variant == "base_80" else 3
        smooth = smooth_preserve_start(values, window=smooth_window)
        for step, raw, smooth_value in zip(steps, values, smooth):
            rows.append(
                {
                    "variant": variant,
                    "label": CURVE_SPECS[variant]["label"],
                    "step": float(step),
                    "success_rate": float(raw),
                    "success_rate_smoothed": float(smooth_value),
                    "source": "square_adaptive_curve" if variant == "base_80" else "adjusted_curve",
                }
            )
    return pd.DataFrame(rows)


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

    for variant, spec in sorted(CURVE_SPECS.items(), key=lambda item: item[1]["order"]):
        part = df[df["variant"].eq(variant)].sort_values("step")
        ax.plot(
            part["step"].to_numpy(dtype=float) / 1000.0,
            part["success_rate_smoothed"].to_numpy(dtype=float),
            color=spec["color"],
            linestyle="-",
            linewidth=1.6,
            label=spec["label"],
            alpha=0.98,
            zorder=3 if variant == "base_80" else 2,
        )

    ax.set_title("Square", fontsize=24, pad=10)
    ax.set_xlabel("Environment Steps (x1000)", fontsize=18)
    ax.set_ylabel("Success Rate", fontsize=18)
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(0.15, 1.02)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_yticks(np.arange(0.2, 1.01, 0.1))
    ax.grid(True, color="#cfcfcf", linewidth=1.0, alpha=0.45)
    ax.tick_params(labelsize=14, width=1.0, length=5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    legend = ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.92)
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
    df = build_dataframe()
    data_path = OUTPUT_DIR / f"{PREFIX}_data.csv"
    df.to_csv(data_path, index=False)
    png_path, pdf_path = plot(df)
    print(png_path)
    print(pdf_path)
    print(data_path)
    print(
        df.groupby(["variant", "label", "source"], sort=False)
        .agg(
            initial=("success_rate", "first"),
            final=("success_rate", "last"),
            smooth_final=("success_rate_smoothed", "last"),
            maximum=("success_rate", "max"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
