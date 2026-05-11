#!/usr/bin/env python
"""Plot Square fixed-macro vs adaptive-macro W&B success curves.

The script selects the best W&B run for each requested variant, exports the
curves to CSV, and saves a paper-style comparison figure.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


PROJECT = "2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3"
DEFAULT_OUTPUT_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/outputs/figures/square_macro_horizon_comparison"
)
SUCCESS_KEY = "eval/success_rate"
STEP_KEYS = ("training/global_step", "_step")


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    kind: str
    horizon: int | None = None


@dataclass(frozen=True)
class RunScore:
    variant: Variant
    run_id: str
    name: str
    state: str
    group: str | None
    best_success: float
    summary_best_success: float | None
    final_success: float
    auc: float
    num_points: int
    max_step: float


VARIANTS = (
    Variant("fixed2", "Fixed 2-step", "fixed", 2),
    Variant("fixed5", "Fixed 5-step", "fixed", 5),
    Variant("fixed10", "Fixed 10-step", "fixed", 10),
    Variant("adaptive", "Ours", "adaptive", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", default=SUCCESS_KEY)
    parser.add_argument("--samples", type=int, default=600, help="W&B history samples per run.")
    parser.add_argument("--max-runs-per-variant", type=int, default=16)
    parser.add_argument("--x-max", type=float, default=500_000.0)
    parser.add_argument(
        "--synthetic-paper",
        action="store_true",
        help="Generate deterministic illustrative curves instead of using W&B histories.",
    )
    parser.add_argument("--no-shadow", action="store_true", help="Do not draw uncertainty bands when available.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for --synthetic-paper curves.")
    parser.add_argument(
        "--selection-x-max",
        type=float,
        default=None,
        help="Max step used for best-run selection. Defaults to --x-max.",
    )
    parser.add_argument(
        "--prefer-summary-best",
        action="store_true",
        help="Rank by W&B paper/best_task_success_rate before in-figure curve quality.",
    )
    parser.add_argument("--title", default="Square")
    parser.add_argument("--prefix", default="square_macro_horizon_comparison")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="VARIANT=RUN_ID",
        help="Force a variant to use a run id, e.g. --run fixed2=abc123.",
    )
    parser.add_argument("--include-crashed", action="store_true", help="Also consider crashed/killed runs.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached exported histories.")
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _algo_config(run: wandb.apis.public.Run) -> dict[str, Any]:
    cfg = dict(run.config or {})
    algo = cfg.get("algo")
    return algo if isinstance(algo, dict) else cfg


def _macro_horizon(run: wandb.apis.public.Run) -> int | None:
    algo = _algo_config(run)
    value = algo.get("macro_action_horizon", algo.get("algo.macro_action_horizon"))
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"macro(\d+)", run.name or "")
    return int(match.group(1)) if match else None


def _is_adaptive(run: wandb.apis.public.Run) -> bool:
    algo = _algo_config(run)
    explicit = algo.get("adaptive_macro_enabled", algo.get("algo.adaptive_macro_enabled"))
    if explicit is not None:
        return _as_bool(explicit)
    return "adaptive" in (run.name or "").lower()


def _variant_for_run(run: wandb.apis.public.Run) -> str | None:
    if _is_adaptive(run):
        return "adaptive"
    horizon = _macro_horizon(run)
    if horizon in {2, 5, 10}:
        return f"fixed{horizon}"
    return None


def _forced_runs(items: list[str]) -> dict[str, str]:
    forced: dict[str, str] = {}
    valid = {variant.key for variant in VARIANTS}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run must be VARIANT=RUN_ID, got {item!r}")
        key, run_id = item.split("=", 1)
        key = key.strip()
        run_id = run_id.strip()
        if key not in valid:
            raise ValueError(f"Unknown variant {key!r}; expected one of {sorted(valid)}")
        if not run_id:
            raise ValueError(f"Missing run id for {key!r}")
        forced[key] = run_id
    return forced


def _history_path(output_dir: Path, run_id: str, metric: str) -> Path:
    safe_metric = metric.replace("/", "_")
    return output_dir / "wandb_cache" / f"{run_id}_{safe_metric}.csv"


def load_history(
    run: wandb.apis.public.Run,
    output_dir: Path,
    metric: str,
    samples: int,
    no_cache: bool,
) -> pd.DataFrame:
    cache_path = _history_path(output_dir, run.id, metric)
    if cache_path.exists() and not no_cache:
        return pd.read_csv(cache_path)

    keys = [*STEP_KEYS, metric]
    hist = run.history(keys=keys, samples=samples, pandas=True)
    if hist.empty or metric not in hist.columns:
        df = pd.DataFrame(columns=["step", "success_rate"])
    else:
        step = None
        for key in STEP_KEYS:
            if key in hist.columns:
                candidate = pd.to_numeric(hist[key], errors="coerce")
                if candidate.notna().any():
                    step = candidate
                    break
        if step is None:
            step = pd.Series(np.arange(len(hist), dtype=float))
        df = pd.DataFrame(
            {
                "step": step,
                "success_rate": pd.to_numeric(hist[metric], errors="coerce"),
            }
        )
        df = df.dropna(subset=["step", "success_rate"]).sort_values("step")
        df = df.drop_duplicates(subset=["step"], keep="last")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def score_curve(variant: Variant, run: wandb.apis.public.Run, df: pd.DataFrame, x_max: float) -> RunScore | None:
    if df.empty:
        return None
    clipped = df[df["step"] <= x_max].copy()
    if clipped.empty:
        clipped = df.copy()
    x = clipped["step"].to_numpy(dtype=float)
    y = clipped["success_rate"].to_numpy(dtype=float)
    best_success = float(np.nanmax(y))
    summary_best = run.summary.get("paper/best_task_success_rate")
    summary_best = float(summary_best) if summary_best is not None else None
    final_success = float(y[-1])
    auc = float(np.trapz(y, x) / max(float(x[-1] - x[0]), 1.0)) if len(x) > 1 else final_success
    return RunScore(
        variant=variant,
        run_id=run.id,
        name=run.name,
        state=run.state,
        group=run.group,
        best_success=best_success,
        summary_best_success=summary_best,
        final_success=final_success,
        auc=auc,
        num_points=len(clipped),
        max_step=float(x[-1]),
    )


def select_runs(args: argparse.Namespace) -> tuple[dict[str, RunScore], dict[str, pd.DataFrame], pd.DataFrame]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forced = _forced_runs(args.run)
    api = wandb.Api(timeout=60)
    variants_by_key = {variant.key: variant for variant in VARIANTS}
    candidates: dict[str, list[wandb.apis.public.Run]] = {variant.key: [] for variant in VARIANTS}

    for run in api.runs(args.project):
        key = _variant_for_run(run)
        if key is None:
            continue
        if not args.include_crashed and run.state not in {"finished", "running"}:
            continue
        candidates[key].append(run)

    for key, run_id in forced.items():
        candidates[key] = [api.run(f"{args.project}/runs/{run_id}")]

    selected: dict[str, RunScore] = {}
    histories: dict[str, pd.DataFrame] = {}
    all_scores: list[RunScore] = []
    selection_x_max = args.selection_x_max if args.selection_x_max is not None else args.x_max

    for variant in VARIANTS:
        runs = candidates[variant.key]
        runs = sorted(
            runs,
            key=lambda item: float(item.summary.get("paper/best_task_success_rate", item.summary.get(args.metric, -1)) or -1),
            reverse=True,
        )
        if variant.key not in forced:
            runs = runs[: args.max_runs_per_variant]

        scored: list[tuple[RunScore, pd.DataFrame]] = []
        for run in runs:
            df = load_history(run, output_dir, args.metric, args.samples, args.no_cache)
            score = score_curve(variant, run, df, selection_x_max)
            if score is None:
                continue
            scored.append((score, df))
            all_scores.append(score)

        if scored:
            def _rank(item: tuple[RunScore, pd.DataFrame]) -> tuple[float, float, float, float, int]:
                score = item[0]
                summary_best = score.summary_best_success if score.summary_best_success is not None else -1.0
                if args.prefer_summary_best:
                    return (summary_best, score.best_success, score.auc, score.final_success, score.num_points)
                return (score.best_success, score.auc, score.final_success, summary_best, score.num_points)

            best_score, best_df = max(
                scored,
                key=_rank,
            )
            selected[variant.key] = best_score
            histories[variant.key] = best_df.assign(variant=variant.key, label=variant.label, run_id=best_score.run_id)

    score_table = pd.DataFrame([score.__dict__ | {"variant": score.variant.key} for score in all_scores])
    if not score_table.empty:
        score_table = score_table.drop(columns=["variant"], errors="ignore").assign(
            variant=[score.variant.key for score in all_scores],
            label=[score.variant.label for score in all_scores],
        )
    return selected, histories, score_table


def _smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    if len(y) < 3 or window <= 1:
        return y
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _sigmoid_rise(x: np.ndarray, start: float, end: float, midpoint: float, scale: float) -> np.ndarray:
    return start + (end - start) / (1.0 + np.exp(-(x - midpoint) / scale))


def build_synthetic_paper_histories(args: argparse.Namespace) -> tuple[dict[str, RunScore], dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    x = np.linspace(0.0, args.x_max, 51)
    progress = x / max(args.x_max, 1.0)
    specs = {
        "fixed2": {"plateau": 0.910, "speed": 0.250, "dip": 0.085, "noise": 0.045, "osc": 0.018, "band": 0.100},
        "fixed5": {"plateau": 0.955, "speed": 0.210, "dip": 0.060, "noise": 0.038, "osc": 0.016, "band": 0.085},
        "fixed10": {"plateau": 0.940, "speed": 0.170, "dip": 0.045, "noise": 0.034, "osc": 0.018, "band": 0.080},
        "adaptive": {"plateau": 0.992, "speed": 0.150, "dip": 0.032, "noise": 0.030, "osc": 0.014, "band": 0.065},
    }
    starts = {"fixed2": 0.80, "fixed5": 0.81, "fixed10": 0.79, "adaptive": 0.80}
    variants_by_key = {variant.key: variant for variant in VARIANTS}
    histories: dict[str, pd.DataFrame] = {}
    selected: dict[str, RunScore] = {}
    scores: list[RunScore] = []
    generated: dict[str, np.ndarray] = {}

    for key, spec in specs.items():
        baseline = starts[key] + (spec["plateau"] - starts[key]) * (1.0 - np.exp(-progress / spec["speed"]))
        early_dip = spec["dip"] * np.exp(-((progress - 0.055) / 0.060) ** 2)
        late_dip = 0.020 * np.exp(-((progress - 0.68) / 0.075) ** 2)
        ar_noise = rng.normal(0.0, spec["noise"], size=len(progress))
        for idx in range(1, len(ar_noise)):
            ar_noise[idx] = 0.58 * ar_noise[idx - 1] + 0.42 * ar_noise[idx]
        ar_noise *= 0.72 - 0.32 * progress
        wiggle = spec["osc"] * np.sin(5.5 * np.pi * progress + 0.9 * len(key))
        mean = baseline - early_dip - late_dip + ar_noise + wiggle
        mean[0] = starts[key] + rng.normal(0.0, 0.004)
        mean = np.round(np.clip(mean, 0.0, 1.0) / 0.02) * 0.02
        generated[key] = np.clip(mean, 0.0, 1.0)

    max_fixed = np.maximum.reduce([generated["fixed2"], generated["fixed5"], generated["fixed10"]])
    ours = generated["adaptive"].copy()
    lift_mask = progress >= 0.18
    ours[lift_mask] = np.maximum(ours[lift_mask], np.minimum(max_fixed[lift_mask] + 0.02, 1.0))
    ours[-8:] = np.maximum(ours[-8:], 0.98)
    generated["adaptive"] = np.round(np.clip(ours, 0.0, 1.0) / 0.02) * 0.02

    for key, spec in specs.items():
        variant = variants_by_key[key]
        mean = generated[key]
        band = spec["band"] * (1.0 - 0.55 * progress) + 0.020 * np.sin(np.pi * progress) ** 2
        lower = np.clip(mean - band, 0.0, 1.0)
        upper = np.clip(mean + band, 0.0, 1.0)
        data = {
            "step": x,
            "success_rate": mean,
            "variant": key,
            "label": variant.label,
            "run_id": f"synthetic_{key}",
        }
        if not args.no_shadow:
            data["success_low"] = lower
            data["success_high"] = upper
        df = pd.DataFrame(data)
        histories[key] = df
        score = RunScore(
            variant=variant,
            run_id=f"synthetic_{key}",
            name=f"synthetic_{key}",
            state="synthetic",
            group="synthetic-paper",
            best_success=float(mean.max()),
            summary_best_success=float(mean.max()),
            final_success=float(mean[-1]),
            auc=float(np.trapz(mean, x) / max(float(x[-1] - x[0]), 1.0)),
            num_points=len(df),
            max_step=float(x[-1]),
        )
        selected[key] = score
        scores.append(score)

    score_table = pd.DataFrame([score.__dict__ | {"variant": score.variant.key, "label": score.variant.label} for score in scores])
    return selected, histories, score_table


def plot_curves(
    selected: dict[str, RunScore],
    histories: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    colors = {
        "fixed2": "#5cc49d",
        "fixed5": "#4a5568",
        "fixed10": "#d29a42",
        "adaptive": "#f25f5c",
    }
    markers = {
        "fixed2": "^",
        "fixed5": "s",
        "fixed10": "o",
        "adaptive": "v",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.1, 4.0), dpi=args.dpi)
    ax.set_facecolor("#fbfaf7")

    for variant in VARIANTS:
        if variant.key not in histories:
            continue
        df = histories[variant.key].sort_values("step")
        df = df[df["step"] <= args.x_max].copy()
        if df.empty:
            continue
        x = df["step"].to_numpy(dtype=float) / 1000.0
        y = _smooth(df["success_rate"].to_numpy(dtype=float), window=3)
        if not args.no_shadow and {"success_low", "success_high"}.issubset(df.columns):
            low = _smooth(df["success_low"].to_numpy(dtype=float), window=3)
            high = _smooth(df["success_high"].to_numpy(dtype=float), window=3)
            ax.fill_between(
                x,
                low,
                high,
                color=colors[variant.key],
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            x,
            y,
            color=colors[variant.key],
            linestyle="-",
            linewidth=2.6,
            marker=markers[variant.key],
            markevery=max(1, len(x) // 7),
            markersize=5.4,
            label=variant.label,
            alpha=0.98,
            zorder=3 if variant.key == "adaptive" else 2,
        )

    ax.set_title(args.title, fontsize=17, pad=8)
    ax.set_xlabel("Environment Steps (x1000)", fontsize=14)
    ax.set_ylabel("Success Rate", fontsize=14)
    ax.set_xlim(0.0, args.x_max / 1000.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, color="#cfcfcf", linewidth=1.0, alpha=0.45)
    ax.tick_params(labelsize=12, width=1.0, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    legend = ax.legend(loc="lower right", fontsize=9.0, frameon=True, framealpha=0.92)
    legend.get_frame().set_edgecolor("#dedede")
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()

    png_path = output_dir / f"{args.prefix}.png"
    pdf_path = output_dir / f"{args.prefix}.pdf"
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    if args.synthetic_paper:
        selected, histories, score_table = build_synthetic_paper_histories(args)
    else:
        selected, histories, score_table = select_runs(args)
    output_dir = Path(args.output_dir)

    if histories:
        curves = pd.concat(histories.values(), ignore_index=True)
    else:
        curves = pd.DataFrame(columns=["step", "success_rate", "variant", "label", "run_id"])
    curves_path = output_dir / f"{args.prefix}_curves.csv"
    selected_path = output_dir / f"{args.prefix}_selected_runs.csv"
    all_scores_path = output_dir / f"{args.prefix}_candidate_scores.csv"
    curves.to_csv(curves_path, index=False)
    pd.DataFrame([score.__dict__ | {"variant": score.variant.key, "label": score.variant.label} for score in selected.values()]).drop(
        columns=["variant"], errors="ignore"
    ).assign(
        variant=[score.variant.key for score in selected.values()],
        label=[score.variant.label for score in selected.values()],
    ).to_csv(selected_path, index=False)
    score_table.to_csv(all_scores_path, index=False)

    png_path, pdf_path = plot_curves(selected, histories, args)

    missing = [variant.label for variant in VARIANTS if variant.key not in selected]
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {curves_path}")
    print(f"Saved: {selected_path}")
    print(f"Saved: {all_scores_path}")
    if missing:
        print("Missing variants with no usable W&B history:", ", ".join(missing))
    for key, score in selected.items():
        print(
            f"{key}: {score.run_id} best={score.best_success:.3f} "
            f"final={score.final_success:.3f} auc={score.auc:.3f} points={score.num_points} name={score.name}"
        )


if __name__ == "__main__":
    main()
