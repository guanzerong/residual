#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "eval/success_rate",
    "eval/mean_episode_primitive_steps",
    "eval/residual_decisions_per_episode",
    "eval/proposal_queries_per_episode",
    "eval/depth_refreshes_per_episode",
    "eval/depth_encoder_forwards_per_episode",
    "eval/mean_selected_horizon",
    "eval/runtime_depth_encoder_forward_mean_ms",
    "eval/runtime_proposal_query_mean_ms",
    "eval/policy_latency_mean_ms",
    "eval/policy_latency_p95_ms",
    "eval/control_cycle_ms_per_primitive_step",
)


def identify_run(record: dict) -> tuple[str, int] | None:
    run_name = str(record.get("run_name", ""))
    if "reactive" in run_name or "fixed10" in run_name:
        method = "Per-Step Reactive (Fixed-10)"
    elif "ahr" in run_name or "adaptive4-7-10" in run_name or "macro4710" in run_name:
        method = "AHR"
    else:
        return None

    seed_match = re.search(r"seed(\d+)", run_name)
    if seed_match is None:
        return None
    return method, int(seed_match.group(1))


def load_final_records(result_dir: Path) -> dict[tuple[str, int], dict]:
    candidates: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for path in sorted(result_dir.glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                identity = identify_run(record)
                if identity is not None:
                    candidates[identity].append(record)

    final_records: dict[tuple[str, int], dict] = {}
    for identity, records in candidates.items():
        final_records[identity] = max(
            records,
            key=lambda record: (
                int(record.get("num_episodes", 0)),
                int(record.get("global_step", 0)),
                float(record.get("recorded_at_unix_s", 0.0)),
            ),
        )
    return final_records


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    records = load_final_records(args.result_dir)
    if not records:
        raise SystemExit(f"No Square comparison JSONL records found in {args.result_dir}")

    print("Final record per seed")
    print("method\tseed\tglobal_step\trollouts\tsuccesses\tsuccess_rate")
    for (method, seed), record in sorted(records.items()):
        episodes = record.get("episodes", [])
        successes = sum(bool(episode.get("success")) for episode in episodes)
        success_rate = float(record["metrics"]["eval/success_rate"])
        print(
            f"{method}\t{seed}\t{record.get('global_step')}\t{record.get('num_episodes')}\t"
            f"{successes}\t{success_rate:.4f}"
        )

    print("\nAggregate across seeds (sample standard deviation)")
    for method in ("Per-Step Reactive (Fixed-10)", "AHR"):
        method_records = [record for (record_method, _), record in records.items() if record_method == method]
        if not method_records:
            continue
        print(f"\n{method}: n={len(method_records)} seeds")
        for metric_name in METRICS:
            values = [
                float(record["metrics"][metric_name])
                for record in method_records
                if metric_name in record.get("metrics", {})
            ]
            print(f"  {metric_name}: {format_mean_std(values)}")


if __name__ == "__main__":
    main()
