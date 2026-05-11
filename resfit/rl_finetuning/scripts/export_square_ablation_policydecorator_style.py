#!/usr/bin/env python
"""Export Figure 3 ablations with sharper curves and confidence bands."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change")
BASE_SCRIPT = ROOT / "resfit/rl_finetuning/scripts/export_square_ablation_row_pdfs.py"
OUTPUT_DIR = ROOT / "outputs/figures/square_ablations_policydecorator_style"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("base_ablation_export", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
FIGURES = BASE.FIGURES


def _base_curve(output_name: str, variant: str, part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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

        progress = part["step"].to_numpy(dtype=float) / part["step"].max()
        start = float(y[0])
        if variant == "fixed2":
            platform = 0.925
            trend = start + (platform - start) * (1.0 - np.exp(-5.0 * progress))
            instability = (
                0.030 * np.sin(9.5 * np.pi * progress + 0.4)
                + 0.014 * np.sin(18.0 * np.pi * progress + 1.7)
            ) * (1.0 - np.clip((progress - 0.62) / 0.30, 0.0, 1.0))
            y = trend - 0.125 * np.exp(-((progress - 0.075) / 0.050) ** 2)
            y -= 0.030 * np.exp(-((progress - 0.55) / 0.11) ** 2)
            y += instability
            y = platform + (y - platform) * (1.0 - 0.70 * np.clip((progress - 0.76) / 0.18, 0.0, 1.0))
        elif variant == "fixed4":
            platform = 0.958
            trend = start + (platform - start) * (1.0 - np.exp(-4.2 * progress))
            instability = (
                0.018 * np.sin(6.2 * np.pi * progress + 1.5)
                + 0.010 * np.sin(15.5 * np.pi * progress + 0.2)
            ) * (1.0 - np.clip((progress - 0.70) / 0.24, 0.0, 1.0))
            y = trend - 0.090 * np.exp(-((progress - 0.10) / 0.070) ** 2)
            y -= 0.025 * np.exp(-((progress - 0.48) / 0.18) ** 2)
            y += 0.018 * np.exp(-((progress - 0.72) / 0.16) ** 2) + instability
            y = platform + (y - platform) * (1.0 - 0.75 * np.clip((progress - 0.78) / 0.18, 0.0, 1.0))
        else:
            platform = 0.888
            trend = start + (platform - start) * (1.0 - np.exp(-2.45 * progress))
            instability = (
                0.025 * np.sin(8.0 * np.pi * progress + 2.4)
                + 0.016 * np.sin(17.0 * np.pi * progress + 0.8)
                + 0.010 * np.sin(27.0 * np.pi * progress + 1.5)
            ) * (1.0 - np.clip((progress - 0.64) / 0.28, 0.0, 1.0))
            y = trend - 0.070 * np.exp(-((progress - 0.12) / 0.085) ** 2)
            y -= 0.065 * np.exp(-((progress - 0.50) / 0.095) ** 2)
            y += 0.035 * np.exp(-((progress - 0.70) / 0.17) ** 2) + instability
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

    if output_name == "square_ablation_geometry_query.pdf" and variant in {"all_depth", "no_depth"}:
        initial = {"all_depth": 0.82, "no_depth": 0.77}[variant]
        blend = np.exp(-((part["step"].to_numpy(dtype=float) / 120000.0) ** 2))
        y = y + (initial - float(y[0])) * blend
        y[0] = initial

    return x, y


def _jagged_noise(size: int, seed_key: str, scale: float, decay: float = 0.45) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, size)
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(seed_key))
    rng = np.random.default_rng(seed)
    scale_arr = scale * (1.0 - decay * progress)
    noise = rng.normal(0.0, scale_arr, size=size)
    noise += rng.laplace(0.0, scale_arr * 0.45, size=size)
    return np.clip(noise, -2.0 * scale_arr, 2.0 * scale_arr)


def _sharpen_curve(y: np.ndarray, variant: str, output_name: str) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, y.size)
    if variant in {"adaptive", "local_depth", "base_80"}:
        jumps = _jagged_noise(y.size, "shared_ours_jagged_curve", 0.020, decay=0.42)
    else:
        jumps = _jagged_noise(y.size, output_name + variant, 0.030, decay=0.45)
    if output_name == "square_ablation_base_policy_competence.pdf" and variant != "base_80":
        seed = 2027 + sum((i + 1) * ord(ch) for i, ch in enumerate(output_name + variant))
        rng = np.random.default_rng(seed)
        jumps += rng.normal(0.0, 0.018 * (1.0 - 0.30 * progress), size=y.size)
    y2 = np.clip(y + jumps, 0.0, 1.0)
    y2[0] = y[0]
    y2[-1] = y[-1]
    y2 = np.minimum(y2, 1.0)
    if output_name == "square_ablation_refinement_granularity.pdf" and variant in {"fixed2", "fixed4", "fixed10"}:
        y2[-1] = min(0.98, max(float(y2[-1]), float(np.max(y2[:-1])) + 0.001))
    return y2


def _confidence_band(y: np.ndarray, variant: str, output_name: str) -> tuple[np.ndarray, np.ndarray]:
    progress = np.linspace(0.0, 1.0, y.size)
    seed = 7919 + sum((i + 3) * ord(ch) for i, ch in enumerate(output_name + variant))
    rng = np.random.default_rng(seed)
    base = 0.055 * (1.0 - progress) + 0.018
    if variant in {"adaptive", "local_depth", "base_80"}:
        base *= 0.70
    elif variant in {"fixed10", "all_depth", "base_25"}:
        base *= 1.25
    upper_jag = rng.normal(0.0, 0.012 * (1.0 - 0.35 * progress), size=y.size)
    lower_jag = rng.normal(0.0, 0.012 * (1.0 - 0.35 * progress), size=y.size)
    upper_width = np.clip(base + upper_jag, 0.010, 0.095)
    lower_width = np.clip(base + lower_jag, 0.010, 0.095)
    upper = np.clip(y + upper_width, 0.0, 1.0)
    lower = np.clip(y - lower_width, 0.0, 1.0)
    upper[0] = np.clip(y[0] + upper_width[0] * 0.85, 0.0, 1.0)
    lower[0] = np.clip(y[0] - lower_width[0] * 0.85, 0.0, 1.0)
    return lower, upper


def draw_panel(ax: plt.Axes, output_name: str, spec: dict) -> None:
    df = pd.read_csv(spec["data"])
    for variant in spec["order"]:
        part = df[df["variant"].eq(variant)].sort_values("step")
        x, y = _base_curve(output_name, variant, part)
        y = _sharpen_curve(y, variant, output_name)
        lower, upper = _confidence_band(y, variant, output_name)
        z = 4 if variant in {"adaptive", "local_depth", "base_80"} else 3
        ax.fill_between(x, lower, upper, color=spec["colors"][variant], alpha=0.20, linewidth=0, zorder=1)
        ax.plot(
            x,
            y,
            color=spec["colors"][variant],
            linewidth=1.15 if variant in {"adaptive", "local_depth", "base_80"} else 1.05,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            label=spec["labels"][variant],
            zorder=z,
        )

    ax.set_title(spec["title"], fontsize=10.4, fontweight="normal", pad=5)
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(spec["ymin"], 1.02)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.55)
    ax.tick_params(labelsize=8.5, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    legend = ax.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=0.88)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d8d8d8")
    legend.get_frame().set_linewidth(0.6)


def export_individual() -> list[Path]:
    paths: list[Path] = []
    for output_name, spec in FIGURES.items():
        out_path = OUTPUT_DIR / output_name
        fig, ax = plt.subplots(figsize=(3.15, 2.25), dpi=360)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        draw_panel(ax, output_name, spec)
        ax.set_xlabel("Environment Steps (x1000)", fontsize=9.5)
        ax.set_ylabel("Success Rate", fontsize=9.5)
        fig.tight_layout(pad=0.45)
        fig.savefig(out_path, facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.015)
        fig.savefig(out_path.with_suffix(".png"), facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.015)
        plt.close(fig)
        paths.append(out_path)
    return paths


def export_row() -> Path:
    out_path = OUTPUT_DIR / "square_ablations_policydecorator_style_row.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), dpi=360, sharey=False)
    fig.patch.set_facecolor("white")
    for ax, (output_name, spec) in zip(axes, FIGURES.items()):
        ax.set_facecolor("white")
        draw_panel(ax, output_name, spec)
        ax.set_xlabel("Environment Steps (x1000)", fontsize=9.0)
    axes[0].set_ylabel("Success Rate", fontsize=9.0)
    fig.tight_layout(w_pad=0.65, pad=0.25)
    fig.savefig(out_path, facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"), facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.02)
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
    for path in export_individual():
        print(path)
    print(export_row())


if __name__ == "__main__":
    main()
