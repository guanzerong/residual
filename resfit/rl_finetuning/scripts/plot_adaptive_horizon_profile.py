#!/usr/bin/env python
"""Plot adaptive horizon frequency and correction magnitude from eval traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_WANDB_CACHE = Path("/data_all/gzr1/.wandb/runs/wandb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a two-panel paper-style plot from eval_horizon_trace_step_*.csv: "
            "selected horizon heatmap plus normalized correction magnitude."
        )
    )
    parser.add_argument("--trace", type=Path, help="Path to eval_horizon_trace_step_*.csv.")
    parser.add_argument("--run-id", help="W&B run id, e.g. j70d5wg5. Used when --trace is omitted.")
    parser.add_argument(
        "--wandb-cache",
        type=Path,
        default=DEFAULT_WANDB_CACHE,
        help=f"Local W&B cache root. Default: {DEFAULT_WANDB_CACHE}",
    )
    parser.add_argument(
        "--step",
        default="latest",
        help="Evaluation step to plot: latest, best, or an integer. Used with --run-id.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG/PDF/SVG path.")
    parser.add_argument("--num-bins", type=int, default=100, help="Number of normalized decision-step bins.")
    parser.add_argument(
        "--heatmap-mode",
        choices=("count", "probability"),
        default="count",
        help="Use raw counts or per-bin probabilities for the horizon heatmap.",
    )
    parser.add_argument(
        "--horizon-axis",
        choices=("full", "observed"),
        default="full",
        help="full uses every integer horizon from min to max; observed uses only horizons in the CSV.",
    )
    parser.add_argument(
        "--correction-column",
        default="normalized_correction",
        help="Column used for the correction curve, e.g. normalized_correction or relative_correction.",
    )
    parser.add_argument(
        "--correction-norm",
        choices=("none", "max", "p95"),
        default="none",
        help="Optional display normalization for the correction curve.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Centered rolling window over normalized bins for displayed mean curves.",
    )
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Use only trace rows from successful episodes.",
    )
    parser.add_argument(
        "--phase-bounds",
        default="25,55,78",
        help='Comma-separated phase boundaries in percent, or "none" to disable.',
    )
    parser.add_argument(
        "--phase-labels",
        default="Approach,Lift,Align,Contact",
        help="Comma-separated labels for the phase bands.",
    )
    parser.add_argument("--title", default="", help="Optional figure title.")
    return parser.parse_args()


def step_from_trace_path(path: Path) -> int | None:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def find_run_files(run_id: str, cache_root: Path) -> Path:
    matches = sorted(cache_root.glob(f"run-*-{run_id}/files"))
    if not matches:
        raise FileNotFoundError(f"No local W&B cache found for run id {run_id!r} under {cache_root}")
    if len(matches) > 1:
        print(f"Found multiple cache dirs for {run_id}; using {matches[-1]}")
    return matches[-1]


def select_trace_from_run(run_id: str, cache_root: Path, step: str) -> tuple[Path, Path | None]:
    files_dir = find_run_files(run_id, cache_root)
    trace_paths = sorted(files_dir.glob("eval_horizon_trace_step_*.csv"), key=lambda p: step_from_trace_path(p) or -1)
    if not trace_paths:
        raise FileNotFoundError(f"No eval_horizon_trace_step_*.csv found in {files_dir}")

    selected_step: int
    if step == "latest":
        selected = trace_paths[-1]
        selected_step = step_from_trace_path(selected) or 0
    elif step == "best":
        summary_rows: list[dict[str, float | int | Path]] = []
        for summary_path in files_dir.glob("eval_episode_summary_step_*.csv"):
            summary_step = step_from_trace_path(summary_path)
            if summary_step is None:
                continue
            df = pd.read_csv(summary_path)
            if "success" not in df.columns:
                continue
            success = df["success"].astype(str).str.lower().eq("true").mean()
            summary_rows.append({"step": summary_step, "success_rate": float(success), "path": summary_path})
        if not summary_rows:
            raise FileNotFoundError(f"No usable eval_episode_summary_step_*.csv found in {files_dir}")
        best = max(summary_rows, key=lambda row: (row["success_rate"], row["step"]))
        selected_step = int(best["step"])
        selected = files_dir / f"eval_horizon_trace_step_{selected_step}.csv"
        if not selected.exists():
            raise FileNotFoundError(f"Best summary step has no matching trace CSV: {selected}")
    else:
        selected_step = int(step)
        selected = files_dir / f"eval_horizon_trace_step_{selected_step}.csv"
        if not selected.exists():
            raise FileNotFoundError(f"No trace CSV for step {selected_step}: {selected}")

    summary = files_dir / f"eval_episode_summary_step_{selected_step}.csv"
    return selected, summary if summary.exists() else None


def smooth(values: pd.Series, window: int) -> pd.Series:
    values = values.astype(float).interpolate(limit_direction="both")
    if window <= 1:
        return values
    return values.rolling(window=window, center=True, min_periods=1).mean()


def parse_phase_spec(bounds_text: str, labels_text: str) -> tuple[list[float], list[str]]:
    if bounds_text.strip().lower() in {"", "none", "off", "false"}:
        return [], []
    bounds = [float(item) for item in bounds_text.split(",") if item.strip()]
    if any(bound <= 0 or bound >= 100 for bound in bounds):
        raise ValueError("--phase-bounds must be inside (0, 100)")
    labels = [item.strip() for item in labels_text.split(",") if item.strip()]
    if labels and len(labels) != len(bounds) + 1:
        raise ValueError("--phase-labels must have exactly len(--phase-bounds) + 1 labels")
    return bounds, labels


def add_phase_bands(ax: plt.Axes, bounds: list[float], labels: list[str], show_labels: bool) -> None:
    if not bounds:
        return
    edges = [0.0, *bounds, 100.0]
    colors = ["#d9ecff", "#e7f4df", "#fff0d8", "#eadff8"]
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        ax.axvspan(left, right, color=colors[idx % len(colors)], alpha=0.35, zorder=0)
        if show_labels and labels:
            ax.text(
                (left + right) / 2,
                1.02,
                labels[idx],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color="#2f3542",
            )
    for bound in bounds:
        ax.axvline(bound, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.8, zorder=3)


def ensure_normalized_bins(df: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    df = df.copy()
    if "normalized_bin" not in df.columns:
        if "normalized_step" not in df.columns:
            raise ValueError("Trace must contain normalized_bin or normalized_step.")
        df["normalized_bin"] = np.floor(df["normalized_step"].astype(float) * num_bins)
    df["normalized_bin"] = df["normalized_bin"].astype(int).clip(0, num_bins - 1)
    return df


def load_and_prepare(trace_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(trace_path)
    required = {"chosen_horizon", args.correction_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{trace_path} is missing required columns: {sorted(missing)}")
    df = ensure_normalized_bins(df, args.num_bins)
    if args.successful_only:
        if "success" not in df.columns:
            raise ValueError("--successful-only requires a success column in the trace.")
        df = df[df["success"].astype(str).str.lower().eq("true")].copy()
    if df.empty:
        raise ValueError("No rows left after filtering.")
    return df


def build_horizon_matrix(df: pd.DataFrame, num_bins: int, axis_mode: str, heatmap_mode: str) -> tuple[np.ndarray, list[int]]:
    observed = sorted(int(h) for h in df["chosen_horizon"].dropna().unique())
    if axis_mode == "full":
        horizons = list(range(min(observed), max(observed) + 1))
        if min(observed) > 1 and max(observed) <= 32:
            horizons = list(range(1, max(observed) + 1))
    else:
        horizons = observed

    table = pd.crosstab(df["normalized_bin"], df["chosen_horizon"].astype(int))
    table = table.reindex(index=range(num_bins), columns=horizons, fill_value=0)
    matrix = table.T.to_numpy(dtype=float)
    if heatmap_mode == "probability":
        col_sums = matrix.sum(axis=0, keepdims=True)
        matrix = np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums > 0)
    return matrix, horizons


def summarize_trace(df: pd.DataFrame, summary_path: Path | None) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {
        "rows": int(len(df)),
        "episodes": int(df["eval_episode_id"].nunique()) if "eval_episode_id" in df.columns else None,
        "row_success_rate": None,
        "episode_success_rate": None,
    }
    if "success" in df.columns:
        summary["row_success_rate"] = float(df["success"].astype(str).str.lower().eq("true").mean())
    if summary_path is not None and summary_path.exists():
        ep = pd.read_csv(summary_path)
        if "success" in ep.columns:
            summary["episode_success_rate"] = float(ep["success"].astype(str).str.lower().eq("true").mean())
        summary["episodes"] = int(len(ep))
    return summary


def plot_profile(
    df: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    trace_path: Path,
    summary_path: Path | None,
) -> None:
    bins = pd.Index(range(args.num_bins), name="normalized_bin")
    x = np.arange(args.num_bins, dtype=float) + 0.5

    matrix, horizons = build_horizon_matrix(df, args.num_bins, args.horizon_axis, args.heatmap_mode)

    horizon_mean = df.groupby("normalized_bin")["chosen_horizon"].mean().reindex(bins)
    horizon_mean = smooth(horizon_mean, args.smooth_window)

    corr_group = df.groupby("normalized_bin")[args.correction_column].agg(["mean", "std", "count"]).reindex(bins)
    corr_mean = corr_group["mean"].astype(float)
    corr_ci = 1.96 * corr_group["std"].fillna(0).astype(float) / np.sqrt(corr_group["count"].fillna(0).clip(lower=1))

    if args.correction_norm != "none":
        values = df[args.correction_column].dropna().astype(float)
        if args.correction_norm == "max":
            denom = values.max()
        else:
            denom = values.quantile(0.95)
        if denom > 0:
            corr_mean = corr_mean / denom
            corr_ci = corr_ci / denom

    corr_mean = smooth(corr_mean, args.smooth_window)
    corr_ci = smooth(corr_ci, args.smooth_window).fillna(0)

    bounds, labels = parse_phase_spec(args.phase_bounds, args.phase_labels)

    fig, (ax_h, ax_c) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.28},
    )

    if args.title:
        fig.suptitle(args.title, fontsize=12, y=0.99)

    add_phase_bands(ax_h, bounds, labels, show_labels=True)
    add_phase_bands(ax_c, bounds, labels, show_labels=False)

    image = ax_h.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0, 100, min(horizons) - 0.5, max(horizons) + 0.5],
        interpolation="nearest",
        zorder=1,
    )
    ax_h.plot(x, horizon_mean, color="#ff2d2d", linewidth=2.0, label="Mean horizon", zorder=4)
    ax_h.set_ylabel(r"Selected horizon $k_t$")
    ax_h.set_xlim(0, 100)
    ax_h.set_ylim(min(horizons) - 0.5, max(horizons) + 0.5)
    ax_h.set_yticks(horizons if len(horizons) <= 12 else np.linspace(min(horizons), max(horizons), 6).round())
    ax_h.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8)
    ax_h.grid(False)

    cbar = fig.colorbar(image, ax=ax_h, pad=0.015, fraction=0.046)
    cbar.set_label("Frequency" if args.heatmap_mode == "count" else "Probability")

    lower = corr_mean - corr_ci
    upper = corr_mean + corr_ci
    ax_c.fill_between(x, lower.to_numpy(), upper.to_numpy(), color="#4f7ed8", alpha=0.22, linewidth=0)
    ax_c.plot(x, corr_mean, color="#356ac3", linewidth=2.0, label="Mean (+/-95% CI)")
    ax_c.set_ylabel("Normalized correction magnitude")
    ax_c.set_xlabel("Normalized decision step")
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8)
    ax_c.grid(axis="y", linestyle=":", color="#bdc3c7", alpha=0.7)

    for ax in (ax_h, ax_c):
        ax.spines["top"].set_color("#aab2bd")
        ax.spines["right"].set_color("#aab2bd")
        ax.spines["bottom"].set_color("#aab2bd")
        ax.spines["left"].set_color("#aab2bd")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    info = summarize_trace(df, summary_path)
    print(f"Trace: {trace_path}")
    if summary_path is not None:
        print(f"Episode summary: {summary_path}")
    print(f"Rows: {info['rows']}, episodes: {info['episodes']}")
    if info["episode_success_rate"] is not None:
        print(f"Episode success rate: {info['episode_success_rate']:.3f}")
    print(f"Saved figure: {output_path}")


def main() -> None:
    args = parse_args()

    if args.trace is not None:
        trace_path = args.trace
        step = step_from_trace_path(trace_path)
        summary_path = trace_path.with_name(f"eval_episode_summary_step_{step}.csv") if step is not None else None
        if summary_path is not None and not summary_path.exists():
            summary_path = None
    else:
        if not args.run_id:
            raise SystemExit("Pass either --trace or --run-id.")
        trace_path, summary_path = select_trace_from_run(args.run_id, args.wandb_cache, args.step)

    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    output_path = args.output
    if output_path is None:
        stem = trace_path.stem.replace("eval_horizon_trace_", "adaptive_horizon_profile_")
        output_path = trace_path.with_name(f"{stem}.png")

    df = load_and_prepare(trace_path, args)
    plot_profile(df, output_path, args, trace_path, summary_path)


if __name__ == "__main__":
    main()
