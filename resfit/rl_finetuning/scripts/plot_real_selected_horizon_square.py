#!/usr/bin/env python
"""Plot a square selected-horizon heatmap from a real eval trace."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_OUTPUT_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/outputs/figures/real_square_selected_horizon"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--figsize", type=float, default=4.8)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--successful-only", action="store_true")
    parser.add_argument("--phase-bounds", default="0.18,0.34,0.60,0.78")
    parser.add_argument(
        "--phase-labels",
        default="Reach Nut,Pick/Lift,Carry to Peg,Hole Align,Seat/Release",
    )
    return parser.parse_args()


def _step_from_path(path: Path) -> int | None:
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _parse_phases(bounds_text: str, labels_text: str) -> tuple[list[float], list[str]]:
    bounds = [float(item) for item in bounds_text.split(",") if item.strip()]
    labels = [item.strip() for item in labels_text.split(",") if item.strip()]
    if len(labels) != len(bounds) + 1:
        raise ValueError("--phase-labels must contain one more entry than --phase-bounds")
    return bounds, labels


def _display_phase(label: str) -> str:
    return (
        label.replace("Reach Nut", "Reach\nNut")
        .replace("Pick/Lift", "Pick/\nLift")
        .replace("Carry to Peg", "Carry to\nPeg")
        .replace("Hole Align", "Hole\nAlign")
        .replace("Seat/Release", "Seat/\nRelease")
    )


def _add_phase_bands(ax: plt.Axes, bounds: list[float], labels: list[str]) -> None:
    phase_colors = ["#fff1e3", "#fff8dc", "#eef7e4", "#fbe5d6", "#f2e2ec"]
    edges = [0.0, *bounds, 1.0]
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        ax.axvspan(left, right, color=phase_colors[idx % len(phase_colors)], alpha=0.72, zorder=0)
        ax.text(
            (left + right) / 2,
            1.035,
            _display_phase(labels[idx]),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="semibold",
            color="#3a2c2c",
            linespacing=0.9,
            clip_on=False,
        )
    for bound in bounds:
        ax.axvline(bound, color="#7b8490", linestyle="--", linewidth=0.95, alpha=0.72, zorder=3)


def _horizon_edges(horizons: np.ndarray) -> np.ndarray:
    horizons = np.asarray(horizons, dtype=float)
    if len(horizons) == 1:
        return np.array([horizons[0] - 0.5, horizons[0] + 0.5])
    mids = (horizons[:-1] + horizons[1:]) / 2.0
    return np.concatenate(([horizons[0] - (mids[0] - horizons[0])], mids, [horizons[-1] + (horizons[-1] - mids[-1])]))


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.set_xticklabels([f"{value:.2g}" if value not in (0, 1) else f"{value:.0f}" for value in np.linspace(0.0, 1.0, 5)])
    ax.tick_params(labelsize=8.5, width=0.9)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("#9aa3ad")
        ax.spines[side].set_linewidth(0.9)


def load_trace(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.trace)
    required = {"normalized_step", "chosen_horizon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.trace} is missing required columns: {sorted(missing)}")
    if args.successful_only:
        if "success" not in df.columns:
            raise ValueError("--successful-only requires a success column")
        df = df[df["success"].astype(str).str.lower().eq("true")].copy()
    if df.empty:
        raise ValueError("No trace rows left after filtering")
    df["normalized_bin"] = np.floor(df["normalized_step"].astype(float) * args.num_bins).astype(int).clip(0, args.num_bins - 1)
    df["chosen_horizon"] = df["chosen_horizon"].astype(int)
    return df


def plot(df: pd.DataFrame, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds, labels = _parse_phases(args.phase_bounds, args.phase_labels)

    bins = pd.Index(range(args.num_bins), name="normalized_bin")
    horizons = np.array(sorted(df["chosen_horizon"].dropna().unique()), dtype=int)
    x_edges = np.linspace(0.0, 1.0, args.num_bins + 1)
    x = (np.arange(args.num_bins) + 0.5) / args.num_bins

    table = pd.crosstab(df["normalized_bin"], df["chosen_horizon"]).reindex(index=bins, columns=horizons, fill_value=0)
    matrix = table.T.to_numpy(dtype=float)
    col_sums = matrix.sum(axis=0, keepdims=True)
    matrix_prob = np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums > 0)

    mean_horizon = (
        df.groupby("normalized_bin")["chosen_horizon"]
        .mean()
        .reindex(bins)
        .interpolate(limit_direction="both")
        .rolling(args.smooth_window, center=True, min_periods=1)
        .mean()
    )

    red_cmap = LinearSegmentedColormap.from_list(
        "real_square_selected_horizon_reds",
        ["#fff7ec", "#fee0b6", "#fdae6b", "#f1695b", "#bd0026", "#5c0015"],
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(args.figsize, args.figsize))
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
    _add_phase_bands(ax, bounds, labels)

    mesh = ax.pcolormesh(
        x_edges,
        _horizon_edges(horizons),
        matrix_prob,
        cmap=red_cmap,
        shading="auto",
        vmin=0.0,
        vmax=max(0.18, float(np.nanquantile(matrix_prob, 0.992))),
        zorder=1,
    )
    ax.plot(x, mean_horizon, color="#1b1b1b", lw=1.8, label="Mean selected horizon", zorder=5)
    ax.scatter(x[::3], mean_horizon.iloc[::3], s=5.5, color="#1b1b1b", alpha=0.55, zorder=6)
    ax.set_xlabel("Normalized task progress", fontsize=9.5)
    ax.set_ylabel(r"Selected horizon $k$", fontsize=9.5)
    ax.set_yticks(horizons)
    ax.set_ylim(_horizon_edges(horizons)[0], _horizon_edges(horizons)[-1])
    _style_axes(ax)
    ax.legend(loc="upper right", fontsize=6.8, frameon=True, framealpha=0.94, handlelength=1.5)

    colorbar = fig.colorbar(mesh, cax=cax)
    colorbar.set_label("Selection probability", fontsize=8.2)
    colorbar.ax.tick_params(labelsize=8.0)

    step = _step_from_path(args.trace)
    prefix = args.prefix or f"{args.trace.stem}_square"
    if args.successful_only and not prefix.endswith("success_only"):
        prefix = f"{prefix}_success_only"
    png_path = args.output_dir / f"{prefix}_selected_horizon_square.png"
    pdf_path = args.output_dir / f"{prefix}_selected_horizon_square.pdf"
    bins_path = args.output_dir / f"{prefix}_selected_horizon_bins.csv"
    pd.DataFrame(
        {
            "global_step": step,
            "normalized_progress": x,
            "mean_selected_horizon": mean_horizon.to_numpy(),
            "num_trace_rows": table.sum(axis=1).to_numpy(),
        }
    ).to_csv(bins_path, index=False)

    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path, bins_path


def main() -> None:
    args = parse_args()
    df = load_trace(args)
    png_path, pdf_path, bins_path = plot(df, args)
    print(png_path)
    print(pdf_path)
    print(bins_path)
    print(f"rows={len(df)} episodes={df['eval_episode_id'].nunique() if 'eval_episode_id' in df else 'NA'}")
    print(f"horizons={sorted(df['chosen_horizon'].unique().tolist())} mean={df['chosen_horizon'].mean():.3f}")


if __name__ == "__main__":
    main()
