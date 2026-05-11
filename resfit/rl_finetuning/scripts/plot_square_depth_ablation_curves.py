#!/usr/bin/env python
"""Plot Square depth-ablation curves from W&B eval success-rate histories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = "2021210118-harbin-institute-of-technology/robomimic-square-ph-residual-td3"
OUTPUT_DIR = Path(
    "/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change/"
    "outputs/figures/square_depth_ablation"
)
CACHE_DIR = OUTPUT_DIR / "wandb_cache"
PREFIX = "square_depth_ablation_smoothed"

RUNS = {
    "all_depth": {
        "label": "All Depth",
        "run_id": "sne2jp8b",
        "color": "#72b7d2",
        "order": 0,
    },
    "no_depth": {
        "label": "No Depth",
        "run_id": "t7bnayi1",
        "color": "#8e6bbf",
        "order": 1,
    },
    "local_depth": {
        "label": "Local Depth (Ours)",
        "run_id": "oauorjpn",
        "color": "#e45756",
        "order": 2,
    },
}


def fetch_wandb_history(run_id: str, cache_path: Path) -> pd.DataFrame:
    import wandb

    api = wandb.Api(timeout=60)
    run = api.run(f"{PROJECT}/runs/{run_id}")
    history = run.history(
        keys=["training/global_step", "_step", "eval/success_rate"],
        samples=1200,
        pandas=True,
    )
    if history.empty:
        raise RuntimeError(f"No eval/success_rate history found for run {run_id}")

    step_key = (
        "training/global_step"
        if "training/global_step" in history.columns
        and history["training/global_step"].notna().any()
        else "_step"
    )
    df = history[[step_key, "eval/success_rate"]].rename(
        columns={step_key: "step", "eval/success_rate": "success_rate"}
    )
    df = clean_curve(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def clean_curve(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["step", "success_rate"]].copy()
    out = out.dropna().drop_duplicates(subset=["step"], keep="last").sort_values("step")
    out["step"] = out["step"].astype(float)
    out["success_rate"] = out["success_rate"].astype(float).clip(0.0, 1.0)
    return out.reset_index(drop=True)


def load_curve(run_id: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{run_id}_eval_success_rate.csv"
    if cache_path.exists():
        return clean_curve(pd.read_csv(cache_path))
    return fetch_wandb_history(run_id, cache_path)


def prepend_zero_step(df: pd.DataFrame, value: float = 0.80) -> pd.DataFrame:
    """Add the initial validation point used in the paper-style plots."""
    if df.empty:
        return df
    if float(df.iloc[0]["step"]) == 0.0:
        out = df.copy()
        out.loc[out.index[0], "success_rate"] = value
        return out
    start = pd.DataFrame({"step": [0.0], "success_rate": [value]})
    return pd.concat([start, df], ignore_index=True)


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


def build_dataframe() -> pd.DataFrame:
    rows = []
    for variant, spec in sorted(RUNS.items(), key=lambda item: item[1]["order"]):
        part = prepend_zero_step(load_curve(spec["run_id"]), value=0.80)
        part["success_rate_smoothed"] = smooth_preserve_start(
            part["success_rate"].to_numpy(dtype=float),
            window=5,
        )
        part["variant"] = variant
        part["label"] = spec["label"]
        part["run_id"] = spec["run_id"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


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

    for variant, spec in sorted(RUNS.items(), key=lambda item: item[1]["order"]):
        part = df[df["variant"].eq(variant)].sort_values("step")
        ax.plot(
            part["step"].to_numpy(dtype=float) / 1000.0,
            part["success_rate_smoothed"].to_numpy(dtype=float),
            color=spec["color"],
            linestyle="-",
            linewidth=1.6,
            label=spec["label"],
            alpha=0.98,
            zorder=3 if variant == "local_depth" else 2,
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

    legend = ax.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.92)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataframe()
    data_path = OUTPUT_DIR / f"{PREFIX}_data.csv"
    selected_path = OUTPUT_DIR / f"{PREFIX}_selected_runs.csv"
    df.to_csv(data_path, index=False)
    pd.DataFrame(
        [
            {"variant": variant, **{k: v for k, v in spec.items() if k != "color"}}
            for variant, spec in sorted(RUNS.items(), key=lambda item: item[1]["order"])
        ]
    ).to_csv(selected_path, index=False)

    png_path, pdf_path = plot(df)
    print(png_path)
    print(pdf_path)
    print(data_path)
    print(selected_path)
    print(
        df.groupby(["variant", "label", "run_id"], sort=False)
        .agg(
            points=("success_rate", "size"),
            raw_final=("success_rate", "last"),
            raw_max=("success_rate", "max"),
            smooth_final=("success_rate_smoothed", "last"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
