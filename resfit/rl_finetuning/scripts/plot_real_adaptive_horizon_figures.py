#!/usr/bin/env python
"""Plot real adaptive-horizon evaluation traces as separate publication figures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_OUTPUT_DIR = Path("/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/outputs/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Path to eval_horizon_trace_step_*.csv.")
    parser.add_argument("--summary", type=Path, help="Optional eval_episode_summary_step_*.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="", help="Output filename prefix. Defaults to run-like trace stem.")
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--successful-only", action="store_true", help="Plot only rows from successful episodes.")
    parser.add_argument(
        "--phase-bounds",
        default="0.18,0.38,0.60,0.78",
        help="Comma-separated normalized phase boundaries in [0, 1].",
    )
    parser.add_argument(
        "--phase-labels",
        default="Reach,Grasp/Lift,Transport,Align,Insert/Contact",
        help="Comma-separated phase labels; must be one more than --phase-bounds.",
    )
    parser.add_argument(
        "--correction-column",
        default="normalized_correction",
        help="Trace column for correction magnitude.",
    )
    return parser.parse_args()


def step_from_trace(path: Path) -> int | None:
    match = re.search(r"step_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def parse_phases(bounds_text: str, labels_text: str) -> tuple[list[float], list[str]]:
    bounds = [float(item) for item in bounds_text.split(",") if item.strip()]
    labels = [item.strip() for item in labels_text.split(",") if item.strip()]
    if any(bound <= 0.0 or bound >= 1.0 for bound in bounds):
        raise ValueError("--phase-bounds must be normalized values inside (0, 1)")
    if len(labels) != len(bounds) + 1:
        raise ValueError("--phase-labels must have exactly len(--phase-bounds) + 1 labels")
    return bounds, labels


def smooth(series: pd.Series, window: int) -> pd.Series:
    values = series.astype(float).interpolate(limit_direction="both")
    if window <= 1:
        return values
    return values.rolling(window=window, center=True, min_periods=1).mean()


def horizon_edges(horizons: np.ndarray) -> np.ndarray:
    horizons = np.asarray(horizons, dtype=float)
    if len(horizons) == 1:
        return np.array([horizons[0] - 0.5, horizons[0] + 0.5])
    mids = (horizons[:-1] + horizons[1:]) / 2.0
    return np.concatenate(
        [
            [horizons[0] - (mids[0] - horizons[0])],
            mids,
            [horizons[-1] + (horizons[-1] - mids[-1])],
        ]
    )


def add_phase_background(ax: plt.Axes, bounds: list[float], labels: list[str], labels_on_top: bool) -> None:
    colors = ["#fff1e6", "#fff7dc", "#eef7e8", "#fde8db", "#f3e5ee"]
    edges = [0.0, *bounds, 1.0]
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        ax.axvspan(left, right, color=colors[idx % len(colors)], alpha=0.58, zorder=0)
        if labels_on_top:
            ax.text(
                (left + right) / 2.0,
                1.045,
                labels[idx],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#3b2e2e",
                clip_on=False,
            )
    for bound in bounds:
        ax.axvline(bound, color="#7d8a92", linestyle="--", linewidth=1.05, alpha=0.78, zorder=4)


def load_trace(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.trace)
    required = {"normalized_step", "chosen_horizon", args.correction_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.trace} is missing required columns: {sorted(missing)}")
    if args.successful_only:
        if "success" not in df.columns:
            raise ValueError("--successful-only requires a success column")
        df = df[df["success"].astype(str).str.lower().eq("true")].copy()
    if df.empty:
        raise ValueError("No trace rows left after filtering")

    df["normalized_bin"] = np.floor(df["normalized_step"].astype(float) * args.num_bins).astype(int)
    df["normalized_bin"] = df["normalized_bin"].clip(0, args.num_bins - 1)
    df["chosen_horizon"] = df["chosen_horizon"].astype(int)
    return df


def save_figures(df: pd.DataFrame, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    bounds, labels = parse_phases(args.phase_bounds, args.phase_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step = step_from_trace(args.trace)
    prefix = args.prefix or f"{args.trace.stem}_real"
    if args.successful_only and not prefix.endswith("success_only"):
        prefix = f"{prefix}_success_only"

    bins = pd.Index(range(args.num_bins), name="normalized_bin")
    x_edges = np.linspace(0.0, 1.0, args.num_bins + 1)
    x = (np.arange(args.num_bins) + 0.5) / args.num_bins
    horizons = np.array(sorted(df["chosen_horizon"].dropna().unique()), dtype=int)

    table = pd.crosstab(df["normalized_bin"], df["chosen_horizon"]).reindex(index=bins, columns=horizons, fill_value=0)
    matrix = table.T.to_numpy(dtype=float)
    col_sums = matrix.sum(axis=0, keepdims=True)
    matrix_prob = np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums > 0)

    horizon_mean = smooth(df.groupby("normalized_bin")["chosen_horizon"].mean().reindex(bins), args.smooth_window)
    corr_mean = smooth(df.groupby("normalized_bin")[args.correction_column].mean().reindex(bins), args.smooth_window)

    profile = pd.DataFrame(
        {
            "normalized_progress": x,
            "mean_selected_horizon": horizon_mean.to_numpy(),
            "mean_correction_magnitude": corr_mean.to_numpy(),
            "num_trace_rows": table.sum(axis=1).to_numpy(),
        }
    )
    profile_path = args.output_dir / f"{prefix}_profile_bins.csv"
    profile.to_csv(profile_path, index=False)

    red_cmap = LinearSegmentedColormap.from_list(
        "real_horizon_reds",
        ["#fff7ec", "#fee6ce", "#fdae6b", "#fb6a4a", "#cb181d", "#67000d"],
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    # Selected horizon heatmap.
    fig_h, ax_h = plt.subplots(figsize=(9.2, 3.75), dpi=220)
    add_phase_background(ax_h, bounds, labels, labels_on_top=True)
    mesh = ax_h.pcolormesh(
        x_edges,
        horizon_edges(horizons),
        matrix_prob,
        cmap=red_cmap,
        shading="auto",
        vmin=0.0,
        vmax=max(0.50, float(np.nanmax(matrix_prob))),
        zorder=2,
    )
    ax_h.plot(x, horizon_mean, color="#1f2933", linewidth=2.4, label="Mean selected horizon", zorder=5)
    ax_h.set_xlim(0.0, 1.0)
    ax_h.set_ylim(horizon_edges(horizons)[0], horizon_edges(horizons)[-1])
    ax_h.set_yticks(horizons)
    ax_h.set_xlabel("Normalized task progress")
    ax_h.set_ylabel(r"Selected horizon $k$")
    ax_h.grid(False)
    ax_h.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#d0d4d8")
    for spine in ax_h.spines.values():
        spine.set_color("#98a2ad")
        spine.set_linewidth(1.0)
    cbar = fig_h.colorbar(mesh, ax=ax_h, pad=0.012, fraction=0.045)
    cbar.set_label("Selection probability")
    horizon_png = args.output_dir / f"{prefix}_selected_horizon.png"
    horizon_pdf = args.output_dir / f"{prefix}_selected_horizon.pdf"
    fig_h.tight_layout()
    fig_h.savefig(horizon_png, bbox_inches="tight")
    fig_h.savefig(horizon_pdf, bbox_inches="tight")
    plt.close(fig_h)

    # Correction magnitude.
    fig_c, ax_c = plt.subplots(figsize=(9.2, 3.25), dpi=220)
    add_phase_background(ax_c, bounds, labels, labels_on_top=True)
    ax_c.plot(x, corr_mean, color="#a82222", linewidth=2.4, label="Mean correction magnitude", zorder=5)
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_xlabel("Normalized task progress")
    ax_c.set_ylabel("Normalized correction magnitude")
    ax_c.grid(axis="y", linestyle=":", color="#aab4bf", linewidth=0.9, alpha=0.75)
    ax_c.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#d0d4d8")
    for spine in ax_c.spines.values():
        spine.set_color("#98a2ad")
        spine.set_linewidth(1.0)
    correction_png = args.output_dir / f"{prefix}_correction_magnitude.png"
    correction_pdf = args.output_dir / f"{prefix}_correction_magnitude.pdf"
    fig_c.tight_layout()
    fig_c.savefig(correction_png, bbox_inches="tight")
    fig_c.savefig(correction_pdf, bbox_inches="tight")
    plt.close(fig_c)

    success_rate = None
    if args.summary and args.summary.exists():
        summary = pd.read_csv(args.summary)
        if "success" in summary.columns:
            success_rate = float(summary["success"].astype(bool).mean())
    print(f"Trace: {args.trace}")
    if args.summary:
        print(f"Summary: {args.summary}")
    if step is not None:
        print(f"Step: {step}")
    if success_rate is not None:
        print(f"Episode success rate: {success_rate:.4f}")
    print(f"Rows plotted: {len(df)}")
    print(f"Horizons: {horizons.tolist()}")
    print(f"Saved: {horizon_png}")
    print(f"Saved: {horizon_pdf}")
    print(f"Saved: {correction_png}")
    print(f"Saved: {correction_pdf}")
    print(f"Saved: {profile_path}")
    return horizon_png, correction_png, profile_path


def main() -> None:
    args = parse_args()
    df = load_trace(args)
    save_figures(df, args)


if __name__ == "__main__":
    main()
