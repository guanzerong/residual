#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METHODS = ("fixed4", "fixed14", "reactive", "ahr")
METRICS = (
    "eval/success_rate",
    "eval/mean_episode_primitive_steps",
    "eval/proposal_queries_per_episode",
    "eval/depth_encoder_forwards_per_episode",
    "eval/runtime_proposal_query_mean_ms",
    "eval/runtime_depth_encoder_forward_mean_ms",
    "eval/policy_latency_mean_ms",
    "eval/policy_latency_p95_ms",
)


def final_record(metrics_dir: Path) -> dict | None:
    records = []
    for path in metrics_dir.glob("*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    if not records:
        return None
    return max(
        records,
        key=lambda item: (
            int(item.get("num_episodes", 0)),
            int(item.get("global_step", 0)),
            float(item.get("recorded_at_unix_s", 0.0)),
        ),
    )


def mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.4f} +/- {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    collected: dict[str, list[dict]] = {method: [] for method in METHODS}
    print("method\tseed\trollouts\tsuccesses\tsuccess_rate\twall_clock_s")
    for method in METHODS:
        for seed_dir in sorted((args.result_root / method).glob("seed*")):
            record = final_record(seed_dir / "metrics")
            if record is None:
                continue
            status_path = seed_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            episodes = record.get("episodes", [])
            successes = sum(bool(item.get("success")) for item in episodes)
            seed = int(seed_dir.name.removeprefix("seed"))
            print(
                f"{method}\t{seed}\t{record.get('num_episodes')}\t{successes}\t"
                f"{record['metrics']['eval/success_rate']:.4f}\t{status.get('wall_clock_seconds', 'n/a')}"
            )
            collected[method].append(record)

    print("\nAggregate across seeds (sample standard deviation)")
    for method in METHODS:
        print(f"\n{method}: n={len(collected[method])}")
        for metric in METRICS:
            values = [float(item["metrics"][metric]) for item in collected[method] if metric in item["metrics"]]
            print(f"  {metric}: {mean_std(values)}")


if __name__ == "__main__":
    main()
