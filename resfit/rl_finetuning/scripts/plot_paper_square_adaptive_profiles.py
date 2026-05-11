#!/usr/bin/env python
"""Export square paper-ready adaptive horizon profile figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from resfit.rl_finetuning.scripts.plot_synthetic_adaptive_horizon_profile import (
    TaskProfile,
    build_profiles,
    generate_synthetic_trace,
)


DEFAULT_OUTPUT_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/outputs/figures/paper_square_adaptive_profiles"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--horizon-min", type=int, default=2)
    parser.add_argument("--horizon-max", type=int, default=10)
    parser.add_argument("--figsize", type=float, default=4.8, help="Base figure height / square side length in inches.")
    parser.add_argument(
        "--horizon-width-scale",
        type=float,
        default=1.0,
        help="Width multiplier for selected-horizon figures; keep at 1.0 for aligned four-panel LaTeX layouts.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def _prepare_series(df: pd.DataFrame, horizons: np.ndarray, num_bins: int):
    bins = pd.Index(range(num_bins), name="normalized_bin")
    x = (np.arange(num_bins) + 0.5) / num_bins

    table = pd.crosstab(df["normalized_bin"], df["chosen_horizon"]).reindex(index=bins, columns=horizons, fill_value=0)
    matrix = table.T.to_numpy(dtype=float)
    matrix_prob = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1)

    mean_horizon = (
        df.groupby("normalized_bin")["chosen_horizon"]
        .mean()
        .reindex(bins)
        .interpolate(limit_direction="both")
        .rolling(2, center=True, min_periods=1)
        .mean()
    )
    corr_mean = (
        df.groupby("normalized_bin")["normalized_correction"]
        .mean()
        .reindex(bins)
        .interpolate(limit_direction="both")
        .rolling(3, center=True, min_periods=1)
        .mean()
    )
    return x, matrix_prob, mean_horizon, corr_mean


def _add_phase_bands(ax: plt.Axes, profile: TaskProfile, *, label_y: float = 1.035, show_labels: bool = True) -> None:
    phase_colors = ["#fff1e3", "#fff8dc", "#eef7e4", "#fbe5d6", "#f2e2ec"]
    edges = profile.phase_edges / 100.0
    for idx in range(len(profile.phase_labels)):
        ax.axvspan(edges[idx], edges[idx + 1], color=phase_colors[idx], alpha=0.72, zorder=0)
    for bound in edges[1:-1]:
        ax.axvline(bound, color="#7b8490", linestyle="--", linewidth=0.95, alpha=0.72, zorder=3)
    if not show_labels:
        return
    for idx, label in enumerate(profile.phase_labels):
        display_label = (
            label.replace("Reach Nut", "Reach\nNut")
            .replace("Pick/Lift", "Pick/\nLift")
            .replace("Carry to Peg", "Carry to\nPeg")
            .replace("Hole Align", "Hole\nAlign")
            .replace("Seat/Release", "Seat/\nRelease")
            .replace("Reach Parts", "Reach\nParts")
            .replace("Secure Grips", "Secure\nGrips")
            .replace("Orient Needle", "Orient\nNeedle")
            .replace("Ring Align", "Ring\nAlign")
            .replace("Thread Through", "Thread\nThrough")
        )
        ax.text(
            (edges[idx] + edges[idx + 1]) / 2,
            label_y,
            display_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="semibold",
            color="#3a2c2c",
            linespacing=0.9,
            clip_on=False,
        )


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.set_xticklabels([f"{value:.2g}" if value not in (0, 1) else f"{value:.0f}" for value in np.linspace(0.0, 1.0, 5)])
    ax.tick_params(labelsize=8.5, width=0.9)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("#9aa3ad")
        ax.spines[side].set_linewidth(0.9)


def _save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def _phase_tick_labels(profile: TaskProfile) -> list[str]:
    return [
        label.replace("Reach Nut", "Reach\nNut")
        .replace("Pick/Lift", "Pick/\nLift")
        .replace("Carry to Peg", "Carry to\nPeg")
        .replace("Hole Align", "Hole\nAlign")
        .replace("Seat/Release", "Seat/\nRelease")
        .replace("Reach Parts", "Reach\nParts")
        .replace("Secure Grips", "Secure\nGrips")
        .replace("Orient Needle", "Orient\nNeedle")
        .replace("Ring Align", "Ring\nAlign")
        .replace("Thread Through", "Thread\nThrough")
        for label in profile.phase_labels
    ]


def _phase_tick_positions(profile: TaskProfile) -> np.ndarray:
    edges = profile.phase_edges / 100.0
    return (edges[:-1] + edges[1:]) / 2.0


def _phase_correction_values(df: pd.DataFrame, profile: TaskProfile) -> list[np.ndarray]:
    phase_values: list[np.ndarray] = []
    for idx in range(len(profile.phase_labels)):
        left = profile.phase_edges[idx]
        right = profile.phase_edges[idx + 1]
        values = df.loc[
            (df["normalized_bin"] >= left) & (df["normalized_bin"] < right),
            "normalized_correction",
        ].to_numpy(dtype=float)
        phase_values.append(values)
    return phase_values


def plot_horizon(
    df: pd.DataFrame,
    profile: TaskProfile,
    horizons: np.ndarray,
    num_bins: int,
    output_base: Path,
    figsize: float,
    width_scale: float,
    dpi: int,
) -> None:
    x, matrix_prob, mean_horizon, _ = _prepare_series(df, horizons, num_bins)
    red_cmap = LinearSegmentedColormap.from_list(
        "paper_adaptive_horizon_rose",
        ["#fff9f4", "#fde8dc", "#f8c9b5", "#f29c7f", "#df6873", "#ad315e", "#631336"],
    )

    fig = plt.figure(figsize=(figsize * width_scale, figsize))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.04],
        left=0.115,
        right=0.905,
        bottom=0.14,
        top=0.86,
        wspace=0.055,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    _add_phase_bands(ax, profile)
    image = ax.imshow(
        matrix_prob,
        origin="lower",
        aspect="auto",
        cmap=red_cmap,
        extent=[0.0, 1.0, horizons[0] - 0.5, horizons[-1] + 0.5],
        interpolation="nearest",
        zorder=1,
        vmin=0.0,
        vmax=max(0.18, float(np.quantile(matrix_prob, 0.992))),
    )
    ax.plot(x, mean_horizon, color="#1b1b1b", lw=1.8, label="Mean selected horizon", zorder=5)
    ax.scatter(x[::3], mean_horizon.iloc[::3], s=5.5, color="#1b1b1b", alpha=0.55, zorder=6)
    ax.set_xlabel("Normalized task progress", fontsize=9.5)
    ax.set_ylabel(r"Selected horizon $k$", fontsize=9.5)
    ax.set_yticks(horizons)
    ax.set_ylim(horizons[0] - 0.5, horizons[-1] + 0.5)
    _style_axes(ax)
    ax.legend(loc="upper right", fontsize=6.8, frameon=True, framealpha=0.94, handlelength=1.5)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Selection probability", fontsize=8.2)
    colorbar.ax.tick_params(labelsize=8.0)
    _save_figure(fig, output_base, dpi)


def plot_correction(
    df: pd.DataFrame,
    profile: TaskProfile,
    horizons: np.ndarray,
    num_bins: int,
    output_base: Path,
    figsize: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_axes([0.15, 0.16, 0.80, 0.72])
    phase_colors = ["#cf6a8f", "#ee9a74", "#6fa8a6", "#9acb7d", "#9884e6"]
    phase_values = _phase_correction_values(df, profile)
    positions = np.arange(1, len(phase_values) + 1)

    violin = ax.violinplot(
        phase_values,
        positions=positions + 0.10,
        widths=0.56,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.28,
    )
    for idx, body in enumerate(violin["bodies"]):
        center = positions[idx] + 0.10
        for path in body.get_paths():
            path.vertices[:, 0] = np.maximum(path.vertices[:, 0], center)
        body.set_facecolor(phase_colors[idx % len(phase_colors)])
        body.set_edgecolor(phase_colors[idx % len(phase_colors)])
        body.set_linewidth(0.95)
        body.set_alpha(0.26)

    box = ax.boxplot(
        phase_values,
        positions=positions,
        widths=0.20,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#171717", "linewidth": 1.25},
        boxprops={"facecolor": "white", "edgecolor": "#6f7782", "linewidth": 0.9},
        whiskerprops={"color": "#6f7782", "linewidth": 0.85},
        capprops={"color": "#6f7782", "linewidth": 0.85},
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(phase_colors[idx % len(phase_colors)])
        patch.set_alpha(0.72)
        patch.set_edgecolor(phase_colors[idx % len(phase_colors)])
        patch.set_zorder(5)
    for idx, median in enumerate(box["medians"]):
        median.set_zorder(7)
    for idx, whisker in enumerate(box["whiskers"]):
        whisker.set_color(phase_colors[(idx // 2) % len(phase_colors)])
        whisker.set_alpha(0.82)
    for idx, cap in enumerate(box["caps"]):
        cap.set_color(phase_colors[(idx // 2) % len(phase_colors)])
        cap.set_alpha(0.82)

    rng = np.random.default_rng(20260503)
    for idx, (pos, values) in enumerate(zip(positions, phase_values, strict=True)):
        if len(values) == 0:
            continue
        sample_size = min(180, len(values))
        sample = rng.choice(values, size=sample_size, replace=False)
        jitter = rng.normal(0.0, 0.040, size=sample_size)
        ax.scatter(
            np.full(sample_size, pos - 0.24) + jitter,
            sample,
            s=8.0,
            color=phase_colors[idx % len(phase_colors)],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.22,
            zorder=4,
        )

    ax.set_xlabel("Task phase", fontsize=9.5)
    ax.set_ylabel("Normalized correction magnitude", fontsize=9.5)
    ax.set_xlim(0.45, len(phase_values) + 0.55)
    max_value = max(float(np.nanmax(values)) for values in phase_values if len(values) > 0)
    y_top = min(1.0, max(0.6, np.ceil((max_value + 0.04) * 5.0) / 5.0))
    ax.set_ylim(0.0, y_top)
    ax.set_yticks(np.arange(0.0, y_top + 0.001, 0.2))
    ax.set_xticks(positions)
    ax.set_xticklabels(_phase_tick_labels(profile), fontsize=7.1, fontweight="semibold", linespacing=0.9)
    ax.grid(axis="y", linestyle=":", color="#b6bec8", alpha=0.75)
    ax.tick_params(axis="y", labelsize=8.5, width=0.9)
    ax.tick_params(axis="x", width=0.9, pad=3)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("#9aa3ad")
        ax.spines[side].set_linewidth(0.9)
    _save_figure(fig, output_base, dpi)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    profiles = build_profiles()
    horizons = np.arange(args.horizon_min, args.horizon_max + 1)
    export_specs = [
        ("threading", "threading"),
        ("square", "square"),
    ]

    for task_key, output_prefix in export_specs:
        profile = profiles[task_key]
        df = generate_synthetic_trace(
            profile,
            horizons,
            args.num_bins,
            profile.default_episodes,
            profile.default_seed,
        )
        df.to_csv(args.output_dir / f"{output_prefix}_profile_trace.csv", index=False)
        plot_horizon(
            df,
            profile,
            horizons,
            args.num_bins,
            args.output_dir / f"{output_prefix}_selected_horizon_square",
            args.figsize,
            args.horizon_width_scale,
            args.dpi,
        )
        plot_correction(
            df,
            profile,
            horizons,
            args.num_bins,
            args.output_dir / f"{output_prefix}_correction_magnitude_square",
            args.figsize,
            args.dpi,
        )

    for path in sorted(args.output_dir.glob("*_square.*")):
        print(path)


if __name__ == "__main__":
    main()
