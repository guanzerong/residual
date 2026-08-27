from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path


WARMUP_RE = re.compile(
    r"Warm-up summary: primitive_steps=(?P<steps>\d+), decisions=(?P<decisions>\d+), "
    r"wall_clock_seconds=(?P<seconds>[0-9.]+), candidate_counts=(?P<counts>\{.*\})"
)
OFFLINE_RE = re.compile(
    r"(?:Added (?P<added>\d+) offline transitions to buffer|"
    r"Loaded offline buffer .*?\(size=(?P<loaded>\d+)\))"
)


def _last_jsonl_record(metrics_dir: Path, *, run_name_hint: str | None = None) -> dict:
    records: list[dict] = []
    if not metrics_dir.is_dir():
        return {}
    for path in sorted(metrics_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if run_name_hint is None or run_name_hint in str(record.get("run_name", "")):
                    records.append(record)
    return max(records, key=lambda record: float(record.get("recorded_at_unix_s", 0.0))) if records else {}


def _normalized_entropy(counts: dict[int, int]) -> float | None:
    positive = [count for count in counts.values() if count > 0]
    total = sum(positive)
    if total == 0 or len(counts) <= 1:
        return None
    entropy = -sum((count / total) * math.log(count / total) for count in positive)
    return entropy / math.log(len(counts))


def summarize_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "running"}
    trainer_log = (run_dir / "trainer.log").read_text(errors="replace")

    warmup_match = WARMUP_RE.search(trainer_log)
    warmup: dict = {}
    if warmup_match:
        counts = {int(key): int(value) for key, value in ast.literal_eval(warmup_match["counts"]).items()}
        decisions = int(warmup_match["decisions"])
        warmup = {
            "primitive_steps": int(warmup_match["steps"]),
            "decisions": decisions,
            "wall_clock_seconds": float(warmup_match["seconds"]),
            "candidate_counts": counts,
            "mean_exposures_per_candidate": decisions / len(counts) if counts else None,
            "normalized_candidate_entropy": _normalized_entropy(counts),
        }

    offline_match = OFFLINE_RE.search(trainer_log)
    # New launchers keep metrics under the result root.  Fall back to the
    # historical per-run directory and the legacy shared metrics directory so
    # runs started before this fix remain summarizable.
    method_seed_hint = f"square_{manifest['method']}_seed{manifest['seed']}__"
    candidate_metric_dirs = [
        run_dir / "metrics",
        run_dir.parent.parent / "metrics",
        Path("/data_all/gzr1/experiment_results/q2_groot_n15_square/metrics"),
    ]
    evaluation = {}
    for metrics_dir in candidate_metric_dirs:
        evaluation = _last_jsonl_record(metrics_dir, run_name_hint=method_seed_hint)
        if evaluation:
            break
    return {
        "method": manifest["method"],
        "seed": manifest["seed"],
        "candidate_horizons": manifest.get("candidate_horizons", []),
        "status": status.get("status", "unknown"),
        "process_wall_clock_seconds": status.get("wall_clock_seconds"),
        "offline_transitions": (
            int(offline_match["added"] or offline_match["loaded"])
            if offline_match
            else None
        ),
        "warmup": warmup,
        "evaluation": {
            "global_step": evaluation.get("global_step"),
            "success_rate": evaluation.get("metrics", {}).get("eval/success_rate"),
            "chosen_horizon_counts": evaluation.get("chosen_horizon_counts", {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runs = []
    for manifest_path in sorted(args.result_root.glob("*/seed*/manifest.json")):
        runs.append(summarize_run(manifest_path.parent))

    if args.json:
        print(json.dumps(runs, indent=2, sort_keys=True))
        return

    print("method\tseed\tstatus\tsteps\tdecisions\texposure/candidate\twarmup_s\toffline_transitions\tsuccess")
    for run in runs:
        warmup = run["warmup"]
        evaluation = run["evaluation"]
        print(
            f"{run['method']}\t{run['seed']}\t{run['status']}\t"
            f"{warmup.get('primitive_steps', '')}\t{warmup.get('decisions', '')}\t"
            f"{warmup.get('mean_exposures_per_candidate', '')}\t"
            f"{warmup.get('wall_clock_seconds', '')}\t"
            f"{run['offline_transitions'] if run['offline_transitions'] is not None else ''}\t"
            f"{evaluation.get('success_rate', '')}"
        )


if __name__ == "__main__":
    main()
