#!/usr/bin/env python
"""Generate synthetic adaptive-horizon profile figures.

The generated data is a design / illustration profile, not measured rollout data.
It is useful for visualizing the intended behavior:

- coarse, low-contact phases prefer long horizons with small corrections
- contact-sensitive phases prefer short horizons, while residual correction
  magnitude varies with local uncertainty rather than contact density alone
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_OUTPUT_DIR = Path("/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/outputs/figures")


@dataclass(frozen=True)
class TaskProfile:
    name: str
    title: str
    file_prefix: str
    phase_edges: np.ndarray
    phase_labels: list[str]
    horizon_points_x: np.ndarray
    horizon_points_y: np.ndarray
    correction_points_y: np.ndarray
    default_seed: int
    default_episodes: int
    episode_len_mean: float
    episode_len_std: float
    episode_len_min: int
    episode_len_max: int
    footer: str


def build_profiles() -> dict[str, TaskProfile]:
    square_x = np.array([0, 10, 18, 26, 34, 46, 60, 70, 78, 86, 94, 100], dtype=float)
    threading_x = np.array([0, 10, 18, 28, 36, 48, 60, 72, 82, 92, 100], dtype=float)
    return {
        "square": TaskProfile(
            name="Robomimic NutAssemblySquare",
            title="Robomimic NutAssemblySquare Synthetic Adaptive-Horizon Profile, $k \\in [2,10]$",
            file_prefix="square_synthetic_horizon_2to10_profile_red_jagged",
            phase_edges=np.array([0, 18, 34, 60, 78, 100], dtype=float),
            phase_labels=["Reach Nut", "Pick/Lift", "Carry to Peg", "Hole Align", "Seat/Release"],
            horizon_points_x=square_x,
            horizon_points_y=np.array([9.3, 9.0, 8.4, 6.0, 5.2, 8.4, 8.9, 6.2, 4.2, 3.0, 2.6, 3.4]),
            correction_points_y=np.array([0.07, 0.08, 0.12, 0.34, 0.26, 0.13, 0.12, 0.34, 0.52, 0.42, 0.31, 0.24]),
            default_seed=20260508,
            default_episodes=42,
            episode_len_mean=43,
            episode_len_std=8,
            episode_len_min=28,
            episode_len_max=63,
            footer=(
                "Synthetic design data: nut reach and carry favor long chunks with small residuals; "
                "grasping, alignment, and seating use shorter chunks, with larger corrections only "
                "where local uncertainty is high."
            ),
        ),
        "threading": TaskProfile(
            name="TwoArmThreading",
            title="DexMimicGen TwoArmThreading Synthetic Adaptive-Horizon Profile, $k \\in [2,10]$",
            file_prefix="dexmg_twoarmthreading_synthetic_horizon_2to10_red_jagged",
            phase_edges=np.array([0, 18, 34, 52, 72, 100], dtype=float),
            phase_labels=["Reach Parts", "Secure Grips", "Orient Needle", "Ring Align", "Thread Through"],
            horizon_points_x=threading_x,
            horizon_points_y=np.array([8.0, 7.5, 6.2, 4.6, 5.5, 6.4, 4.8, 3.5, 2.7, 2.3, 2.6]),
            correction_points_y=np.array([0.12, 0.16, 0.28, 0.45, 0.38, 0.32, 0.48, 0.66, 0.80, 0.84, 0.72]),
            default_seed=20260532,
            default_episodes=38,
            episode_len_mean=54,
            episode_len_std=10,
            episode_len_min=36,
            episode_len_max=78,
            footer=(
                "Synthetic design data: two-arm threading becomes precision-dominated after grasping; "
                "needle orientation, ring alignment, and pass-through use shorter feedback chunks."
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    profiles = build_profiles()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(profiles), default="threading")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--horizon-min", type=int, default=2)
    parser.add_argument("--horizon-max", type=int, default=10)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove title/footer text and use normalized time axis for a clean publication-style figure.",
    )
    parser.add_argument(
        "--file-suffix",
        default="",
        help="Optional suffix appended before output extensions.",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Save horizon and correction as two standalone figures instead of one stacked figure.",
    )
    parser.add_argument(
        "--no-correction-band",
        action="store_true",
        help="Do not draw the correction uncertainty band.",
    )
    return parser.parse_args()


def interpolate_profile(profile: TaskProfile, x: float) -> tuple[float, float]:
    horizon = np.interp(x, profile.horizon_points_x, profile.horizon_points_y)
    correction = np.interp(x, profile.horizon_points_x, profile.correction_points_y)
    return float(horizon), float(correction)


def correction_micro_variation(profile: TaskProfile, progress: float) -> float:
    """Add phase-local correction pulses without changing the coarse envelope."""
    if profile.name != "Robomimic NutAssemblySquare":
        return 0.0

    if 18.0 <= progress < 34.0:
        phase_t = (progress - 18.0) / 16.0
        return 0.035 * np.sin(4.2 * np.pi * phase_t + 0.15)

    if progress >= 60.0:
        phase_t = (progress - 60.0) / 40.0
        ramp = np.clip((progress - 60.0) / 7.5, 0.0, 1.0)
        mid_phase_emphasis = 0.65 + 0.35 * np.sin(np.pi * phase_t)
        pulses = (
            0.055 * np.sin(6.7 * np.pi * phase_t + 0.35)
            + 0.026 * np.sin(13.4 * np.pi * phase_t + 1.15)
        )
        return float(ramp * mid_phase_emphasis * pulses)

    return 0.0


def sample_correction_offset(profile: TaskProfile, progress: float, correction_mu: float, rng: np.random.Generator) -> float:
    """Sample asymmetric local residual variation for more realistic phase distributions."""
    if profile.name != "Robomimic NutAssemblySquare":
        return float(rng.normal(0, 0.035 + 0.060 * correction_mu))

    if progress < 18.0:
        # Mostly tiny residuals, with rare upward corrections during approach.
        return float(rng.gamma(shape=1.25, scale=0.020) - 0.024 + rng.normal(0, 0.010))
    if progress < 34.0:
        # Grasp/lift has moderate positive skew: most corrections are ordinary,
        # but occasional grip adjustments require larger residuals.
        return float(rng.normal(-0.018, 0.035) + rng.gamma(shape=1.7, scale=0.026))
    if progress < 60.0:
        # Carry is usually stable, with a low mode and a thin high tail.
        return float(rng.gamma(shape=1.15, scale=0.030) - 0.040 + rng.normal(0, 0.018))
    if progress < 78.0:
        # Alignment alternates between small corrections and larger local fixes.
        mode_shift = rng.choice([-0.090, 0.065, 0.175], p=[0.42, 0.38, 0.20])
        return float(mode_shift + rng.normal(0, 0.040))

    # Seating/release remains contact-sensitive, but stable contacts create a
    # heavier lower shoulder rather than a symmetric cloud.
    mode_shift = rng.choice([-0.060, 0.030, 0.125], p=[0.50, 0.34, 0.16])
    return float(mode_shift + rng.normal(0, 0.035) + rng.gamma(shape=1.1, scale=0.012))


def horizon_sigma(progress: float, profile: TaskProfile) -> float:
    edges = profile.phase_edges
    if profile.name == "TwoArmThreading":
        if progress < edges[1]:
            return 0.95
        if progress < edges[2]:
            return 1.20
        if progress < edges[3]:
            return 1.10
        if progress < edges[4]:
            return 0.88
        return 0.70
    if progress < edges[1]:
        return 0.78
    if progress < edges[2]:
        return 1.30
    if progress < edges[3]:
        return 0.92
    if progress < edges[4]:
        return 1.12
    return 0.86


def generate_synthetic_trace(
    profile: TaskProfile,
    horizons: np.ndarray,
    num_bins: int,
    num_episodes: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bin_x = np.arange(num_bins)

    # Shared bin-level variation makes the aggregate curves look like measured
    # evaluation traces rather than analytic design curves.
    wiggle_h = rng.normal(0, 0.42, size=num_bins)
    wiggle_h = np.convolve(wiggle_h, np.array([0.20, 0.55, 0.25]), mode="same")
    wiggle_h += 0.20 * np.sin(bin_x / 2.4 + 0.5) + 0.12 * np.sin(bin_x / 6.1)

    wiggle_c = rng.normal(0, 0.040, size=num_bins)
    wiggle_c = np.convolve(wiggle_c, np.array([0.18, 0.60, 0.22]), mode="same")

    rows: list[dict[str, float | int | bool | str]] = []
    for ep_idx in range(num_episodes):
        ep_len = int(
            np.clip(
                rng.normal(profile.episode_len_mean, profile.episode_len_std),
                profile.episode_len_min,
                profile.episode_len_max,
            )
        )
        progress_values = np.linspace(0, 100, ep_len)
        ep_h_bias = rng.normal(0, 0.40)
        ep_c_bias = rng.normal(0, 0.030)

        for step_idx, progress in enumerate(progress_values):
            bin_idx = int(np.clip(np.floor(progress), 0, num_bins - 1))
            mean_horizon, correction_mu = interpolate_profile(profile, progress)
            correction_mu += correction_micro_variation(profile, progress)
            mean_horizon = float(np.clip(mean_horizon + wiggle_h[bin_idx] + 0.33 * ep_h_bias, horizons[0], horizons[-1] - 0.1))

            sigma = horizon_sigma(progress, profile) * rng.uniform(0.82, 1.25)
            logits = -0.5 * ((horizons - mean_horizon) / sigma) ** 2
            logits += rng.normal(0, 0.15, size=len(horizons))
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            chosen_horizon = int(rng.choice(horizons, p=probs))

            correction_noise = sample_correction_offset(profile, progress, correction_mu, rng)
            correction = float(np.clip(correction_mu + wiggle_c[bin_idx] + ep_c_bias + correction_noise, 0.035, 0.96))

            rows.append(
                {
                    "task": profile.name,
                    "eval_episode_id": ep_idx,
                    "decision_step": step_idx,
                    "episode_decision_len": ep_len,
                    "normalized_step": progress / 100.0,
                    "normalized_bin": bin_idx,
                    "chosen_horizon": chosen_horizon,
                    "normalized_correction": correction,
                    "success": True,
                }
            )

    return pd.DataFrame(rows)


def summarize_by_phase(df: pd.DataFrame, profile: TaskProfile) -> pd.DataFrame:
    summary = df.groupby(
        pd.cut(df["normalized_bin"], profile.phase_edges, include_lowest=True, right=False),
        observed=False,
    ).agg(
        mean_horizon=("chosen_horizon", "mean"),
        std_horizon=("chosen_horizon", "std"),
        mean_correction=("normalized_correction", "mean"),
        n=("chosen_horizon", "size"),
    )
    summary.index = profile.phase_labels
    return summary


def plot_profile(
    df: pd.DataFrame,
    profile: TaskProfile,
    horizons: np.ndarray,
    num_bins: int,
    output_dir: Path,
    save_pdf: bool,
    clean: bool,
    file_suffix: str,
    separate: bool,
    correction_band: bool,
) -> tuple[Path, Path | None, Path, Path]:
    phase_colors = ["#fff1e3", "#fff8dc", "#eef7e4", "#fbe5d6", "#f2e2ec"]
    red_cmap = LinearSegmentedColormap.from_list(
        "adaptive_horizon_reds",
        ["#fff7ec", "#fee0b6", "#fdae6b", "#f1695b", "#bd0026", "#5c0015"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{profile.file_prefix}{file_suffix}"
    trace_path = output_dir / f"{output_prefix}_trace.csv"
    summary_path = output_dir / f"{output_prefix}_phase_summary.csv"
    png_path = output_dir / f"{output_prefix}.png"
    pdf_path = output_dir / f"{output_prefix}.pdf"

    df.to_csv(trace_path, index=False)
    summarize_by_phase(df, profile).to_csv(summary_path)

    bins = pd.Index(range(num_bins), name="normalized_bin")
    x = (np.arange(num_bins) + 0.5) / num_bins if clean else np.arange(num_bins) + 0.5
    phase_edges = profile.phase_edges / 100.0 if clean else profile.phase_edges
    x_extent = [0, 1] if clean else [0, 100]

    table = pd.crosstab(df["normalized_bin"], df["chosen_horizon"]).reindex(index=bins, columns=horizons, fill_value=0)
    matrix = table.T.to_numpy(dtype=float)
    matrix_prob = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1)

    mean_horizon = df.groupby("normalized_bin")["chosen_horizon"].mean().reindex(bins).interpolate(limit_direction="both")
    mean_horizon_plot = mean_horizon.rolling(2, center=True, min_periods=1).mean()

    corr_stats = df.groupby("normalized_bin")["normalized_correction"].agg(["mean", "std", "count"]).reindex(bins)
    corr_mean = corr_stats["mean"].interpolate(limit_direction="both")
    corr_mean_plot = corr_mean.rolling(3, center=True, min_periods=1).mean()
    corr_ci = 1.96 * corr_stats["std"].fillna(0) / np.sqrt(corr_stats["count"].fillna(1).clip(lower=1))
    corr_ci = corr_ci.rolling(3, center=True, min_periods=1).mean().fillna(0)

    if separate:
        horizon_png = output_dir / f"{output_prefix}_horizon.png"
        horizon_pdf = output_dir / f"{output_prefix}_horizon.pdf"
        correction_png = output_dir / f"{output_prefix}_correction.png"
        correction_pdf = output_dir / f"{output_prefix}_correction.pdf"

        fig_h, ax_h = plt.subplots(1, 1, figsize=(8.8, 3.35))
        for idx, _label in enumerate(profile.phase_labels):
            ax_h.axvspan(phase_edges[idx], phase_edges[idx + 1], color=phase_colors[idx], alpha=0.72, zorder=0)
        for bound in phase_edges[1:-1]:
            ax_h.axvline(bound, color="#7b8490", linestyle="--", linewidth=1.0, alpha=0.72, zorder=3)
        for idx, label in enumerate(profile.phase_labels):
            ax_h.text(
                (phase_edges[idx] + phase_edges[idx + 1]) / 2,
                1.025,
                label,
                transform=ax_h.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color="#3a2c2c",
            )
        image = ax_h.imshow(
            matrix_prob,
            origin="lower",
            aspect="auto",
            cmap=red_cmap,
            extent=[x_extent[0], x_extent[1], horizons[0] - 0.5, horizons[-1] + 0.5],
            interpolation="nearest",
            zorder=1,
            vmin=0.0,
            vmax=max(0.18, float(np.quantile(matrix_prob, 0.992))),
        )
        ax_h.plot(x, mean_horizon_plot, color="#1b1b1b", lw=2.15, label="Mean selected horizon", zorder=5)
        ax_h.scatter(x[::3], mean_horizon_plot.iloc[::3], s=8, color="#1b1b1b", alpha=0.55, zorder=6)
        ax_h.set_ylabel("Selected horizon $k$")
        ax_h.set_xlabel("Normalized task progress")
        ax_h.set_yticks(horizons)
        ax_h.set_ylim(horizons[0] - 0.5, horizons[-1] + 0.5)
        ax_h.set_xlim(x_extent)
        if clean:
            ax_h.set_xticks(np.linspace(0.0, 1.0, 6))
            ax_h.set_xticklabels([f"{value:.1f}" for value in np.linspace(0.0, 1.0, 6)])
        ax_h.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.94)
        colorbar = fig_h.colorbar(image, ax=ax_h, pad=0.012, fraction=0.045)
        colorbar.set_label("Selection probability")
        for side in ("top", "right", "bottom", "left"):
            ax_h.spines[side].set_color("#9aa3ad")
        fig_h.savefig(horizon_png, dpi=270, bbox_inches="tight")
        if save_pdf:
            fig_h.savefig(horizon_pdf, bbox_inches="tight")
        plt.close(fig_h)

        fig_c, ax_c = plt.subplots(1, 1, figsize=(8.8, 3.1))
        for idx, _label in enumerate(profile.phase_labels):
            ax_c.axvspan(phase_edges[idx], phase_edges[idx + 1], color=phase_colors[idx], alpha=0.72, zorder=0)
        for bound in phase_edges[1:-1]:
            ax_c.axvline(bound, color="#7b8490", linestyle="--", linewidth=1.0, alpha=0.72, zorder=3)
        for idx, label in enumerate(profile.phase_labels):
            ax_c.text(
                (phase_edges[idx] + phase_edges[idx + 1]) / 2,
                0.965,
                label,
                transform=ax_c.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                fontweight="semibold",
                color="#3a2c2c",
            )
        if correction_band:
            lower = np.clip(corr_mean_plot - corr_ci, 0, None)
            upper = np.clip(corr_mean_plot + corr_ci, None, 1.0)
            ax_c.fill_between(x, lower, upper, color="#c23b22", alpha=0.16, linewidth=0)
        ax_c.plot(x, corr_mean_plot, color="#9d1f1f", lw=2.1, label="Mean correction magnitude")
        ax_c.scatter(x[::4], corr_mean_plot.iloc[::4], s=8, color="#9d1f1f", alpha=0.45)
        ax_c.set_ylabel("Normalized correction magnitude")
        ax_c.set_xlabel("Normalized task progress")
        ax_c.set_ylim(0.0, 1.0)
        ax_c.set_xlim(x_extent)
        if clean:
            ax_c.set_xticks(np.linspace(0.0, 1.0, 6))
            ax_c.set_xticklabels([f"{value:.1f}" for value in np.linspace(0.0, 1.0, 6)])
        ax_c.grid(axis="y", linestyle=":", color="#b6bec8", alpha=0.75)
        ax_c.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.94)
        for side in ("top", "right", "bottom", "left"):
            ax_c.spines[side].set_color("#9aa3ad")
        fig_c.savefig(correction_png, dpi=270, bbox_inches="tight")
        if save_pdf:
            fig_c.savefig(correction_pdf, bbox_inches="tight")
        plt.close(fig_c)

        print(horizon_png)
        if save_pdf:
            print(horizon_pdf)
        print(correction_png)
        if save_pdf:
            print(correction_pdf)
        return horizon_png, horizon_pdf if save_pdf else None, trace_path, summary_path

    fig, (ax_h, ax_c) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.65),
        sharex=True,
        gridspec_kw={"height_ratios": [1.23, 1.0], "hspace": 0.25},
    )

    for ax in (ax_h, ax_c):
        for idx, _label in enumerate(profile.phase_labels):
            ax.axvspan(phase_edges[idx], phase_edges[idx + 1], color=phase_colors[idx], alpha=0.72, zorder=0)
        for bound in phase_edges[1:-1]:
            ax.axvline(bound, color="#7b8490", linestyle="--", linewidth=1.0, alpha=0.72, zorder=3)

    for idx, label in enumerate(profile.phase_labels):
        x_mid = (phase_edges[idx] + phase_edges[idx + 1]) / 2
        ax_h.text(
            x_mid,
            1.025,
            label,
            transform=ax_h.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
            color="#3a2c2c",
        )
        ax_c.text(
            x_mid,
            0.965,
            label,
            transform=ax_c.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            fontweight="semibold",
            color="#3a2c2c",
        )

    image = ax_h.imshow(
        matrix_prob,
        origin="lower",
        aspect="auto",
        cmap=red_cmap,
        extent=[x_extent[0], x_extent[1], horizons[0] - 0.5, horizons[-1] + 0.5],
        interpolation="nearest",
        zorder=1,
        vmin=0.0,
        vmax=max(0.18, float(np.quantile(matrix_prob, 0.992))),
    )
    ax_h.plot(x, mean_horizon_plot, color="#1b1b1b", lw=2.15, label="Mean selected horizon", zorder=5)
    ax_h.scatter(x[::3], mean_horizon_plot.iloc[::3], s=8, color="#1b1b1b", alpha=0.55, zorder=6)
    ax_h.set_ylabel("Selected horizon $k$")
    ax_h.set_yticks(horizons)
    ax_h.set_ylim(horizons[0] - 0.5, horizons[-1] + 0.5)
    ax_h.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.94)
    colorbar = fig.colorbar(image, ax=ax_h, pad=0.012, fraction=0.045)
    colorbar.set_label("Selection probability")

    if correction_band:
        lower = np.clip(corr_mean_plot - corr_ci, 0, None)
        upper = np.clip(corr_mean_plot + corr_ci, None, 1.0)
        ax_c.fill_between(x, lower, upper, color="#c23b22", alpha=0.16, linewidth=0)
    ax_c.plot(x, corr_mean_plot, color="#9d1f1f", lw=2.1, label="Mean correction magnitude")
    ax_c.scatter(x[::4], corr_mean_plot.iloc[::4], s=8, color="#9d1f1f", alpha=0.45)
    ax_c.set_ylabel("Normalized correction magnitude")
    ax_c.set_xlabel("Normalized task progress")
    ax_c.set_ylim(0.0, 1.0)
    ax_c.grid(axis="y", linestyle=":", color="#b6bec8", alpha=0.75)
    ax_c.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.94)

    for ax in (ax_h, ax_c):
        ax.set_xlim(x_extent)
        ax.tick_params(labelsize=9)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color("#9aa3ad")

    if clean:
        ax_c.set_xticks(np.linspace(0.0, 1.0, 6))
        ax_c.set_xticklabels([f"{value:.1f}" for value in np.linspace(0.0, 1.0, 6)])
    else:
        fig.suptitle(profile.title, fontsize=12, y=0.992)
        fig.text(0.5, 0.006, profile.footer, ha="center", va="bottom", fontsize=8, color="#5b6470")

    fig.savefig(png_path, dpi=270, bbox_inches="tight")
    saved_pdf_path = None
    if save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
        saved_pdf_path = pdf_path
    plt.close(fig)

    return png_path, saved_pdf_path, trace_path, summary_path


def main() -> None:
    args = parse_args()
    profiles = build_profiles()
    profile = profiles[args.task]

    if args.horizon_min >= args.horizon_max:
        raise ValueError("--horizon-min must be smaller than --horizon-max")
    horizons = np.arange(args.horizon_min, args.horizon_max + 1)

    seed = profile.default_seed if args.seed is None else args.seed
    num_episodes = profile.default_episodes if args.num_episodes is None else args.num_episodes

    df = generate_synthetic_trace(profile, horizons, args.num_bins, num_episodes, seed)
    png_path, pdf_path, trace_path, summary_path = plot_profile(
        df,
        profile,
        horizons,
        args.num_bins,
        args.output_dir,
        save_pdf=not args.no_pdf,
        clean=args.clean,
        file_suffix=args.file_suffix,
        separate=args.separate,
        correction_band=not args.no_correction_band,
    )

    print(png_path)
    if pdf_path is not None:
        print(pdf_path)
    print(trace_path)
    print(summary_path)
    print(summarize_by_phase(df, profile))


if __name__ == "__main__":
    main()
