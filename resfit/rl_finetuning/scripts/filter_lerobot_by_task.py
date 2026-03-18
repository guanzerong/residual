from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _extract_stat_count(record: dict[str, Any], feature_key: str = "state") -> int:
    return int(record["stats"][feature_key]["count"][0])


def _replace_int64_column(table: pa.Table, column_name: str, values: list[int]) -> pa.Table:
    column_idx = table.schema.get_field_index(column_name)
    if column_idx < 0:
        raise KeyError(f"Column {column_name!r} not found in parquet table.")
    return table.set_column(column_idx, column_name, pa.array(values, type=pa.int64()))


def _rewrite_episode_parquet(
    src_path: Path,
    dst_path: Path,
    *,
    new_episode_index: int,
    global_frame_offset: int,
    task_index: int,
) -> int:
    table = pq.read_table(src_path)
    length = table.num_rows
    table = _replace_int64_column(table, "episode_index", [new_episode_index] * length)
    table = _replace_int64_column(table, "frame_index", list(range(length)))
    table = _replace_int64_column(table, "index", list(range(global_frame_offset, global_frame_offset + length)))
    table = _replace_int64_column(table, "task_index", [task_index] * length)
    pq.write_table(table, dst_path)
    return length


def _rewrite_episode_stats(
    src_record: dict[str, Any],
    *,
    new_episode_index: int,
    global_frame_offset: int,
    task_index: int,
) -> dict[str, Any]:
    record = deepcopy(src_record)
    length = _extract_stat_count(record)

    record["episode_index"] = new_episode_index
    record["stats"]["episode_index"] = {
        "min": [new_episode_index],
        "max": [new_episode_index],
        "mean": [float(new_episode_index)],
        "std": [0.0],
        "count": [length],
    }
    record["stats"]["index"] = {
        "min": [global_frame_offset],
        "max": [global_frame_offset + length - 1],
        "mean": [global_frame_offset + (length - 1) / 2.0],
        "std": [(((length**2) - 1) / 12.0) ** 0.5 if length > 1 else 0.0],
        "count": [length],
    }
    record["stats"]["task_index"] = {
        "min": [task_index],
        "max": [task_index],
        "mean": [float(task_index)],
        "std": [0.0],
        "count": [length],
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one task from a local LeRobot dataset.")
    parser.add_argument("--src", required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--dst", required=True, help="Destination dataset root. Must not already exist.")
    parser.add_argument("--task-index", type=int, required=True, help="Original task_index to keep.")
    parser.add_argument(
        "--new-task-index",
        type=int,
        default=0,
        help="Task index to write into the filtered dataset. Default: 0.",
    )
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")
    if dst_root.exists():
        raise FileExistsError(f"Destination already exists: {dst_root}")

    src_meta = src_root / "meta"
    src_data = src_root / "data"
    if not src_meta.exists() or not src_data.exists():
        raise FileNotFoundError(f"Source dataset is missing meta/ or data/: {src_root}")

    tasks_records = _load_jsonl(src_meta / "tasks.jsonl")
    episodes_records = _load_jsonl(src_meta / "episodes.jsonl")
    episode_stats_records = _load_jsonl(src_meta / "episodes_stats.jsonl")
    info = json.loads((src_meta / "info.json").read_text())

    original_task_name = None
    for record in tasks_records:
        if int(record["task_index"]) == args.task_index:
            original_task_name = record["task"]
            break
    if original_task_name is None:
        raise ValueError(f"task_index={args.task_index} not found in {src_meta / 'tasks.jsonl'}")

    selected = []
    for stats_record in episode_stats_records:
        record_task_index = int(stats_record["stats"]["task_index"]["min"][0])
        if record_task_index != args.task_index:
            continue
        episode_index = int(stats_record["episode_index"])
        episode_record = next(r for r in episodes_records if int(r["episode_index"]) == episode_index)
        selected.append((episode_index, episode_record, stats_record))

    if not selected:
        raise ValueError(f"No episodes found for task_index={args.task_index}.")

    selected.sort(key=lambda item: item[0])

    (dst_root / "meta").mkdir(parents=True, exist_ok=False)
    (dst_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=False)

    new_episodes_records: list[dict[str, Any]] = []
    new_episode_stats_records: list[dict[str, Any]] = []
    global_frame_offset = 0

    for new_episode_index, (old_episode_index, episode_record, stats_record) in enumerate(selected):
        src_parquet = src_root / "data" / "chunk-000" / f"episode_{old_episode_index:06d}.parquet"
        dst_parquet = dst_root / "data" / "chunk-000" / f"episode_{new_episode_index:06d}.parquet"
        if not src_parquet.exists():
            raise FileNotFoundError(f"Missing source parquet: {src_parquet}")

        length = _rewrite_episode_parquet(
            src_parquet,
            dst_parquet,
            new_episode_index=new_episode_index,
            global_frame_offset=global_frame_offset,
            task_index=args.new_task_index,
        )
        new_episodes_records.append(
            {
                "episode_index": new_episode_index,
                "tasks": [original_task_name],
                "length": length,
            }
        )
        new_episode_stats_records.append(
            _rewrite_episode_stats(
                stats_record,
                new_episode_index=new_episode_index,
                global_frame_offset=global_frame_offset,
                task_index=args.new_task_index,
            )
        )
        global_frame_offset += length

    new_info = deepcopy(info)
    new_info["total_episodes"] = len(new_episodes_records)
    new_info["total_frames"] = global_frame_offset
    new_info["total_tasks"] = 1
    new_info["splits"] = {"train": f"0:{len(new_episodes_records)}"}

    _write_jsonl(dst_root / "meta" / "tasks.jsonl", [{"task_index": args.new_task_index, "task": original_task_name}])
    _write_jsonl(dst_root / "meta" / "episodes.jsonl", new_episodes_records)
    _write_jsonl(dst_root / "meta" / "episodes_stats.jsonl", new_episode_stats_records)
    (dst_root / "meta" / "info.json").write_text(json.dumps(new_info, indent=4, ensure_ascii=True) + "\n")

    for extra_meta_name in ("stats.json",):
        extra_meta = src_meta / extra_meta_name
        if extra_meta.exists():
            shutil.copy2(extra_meta, dst_root / "meta" / extra_meta_name)

    print(
        json.dumps(
            {
                "src": str(src_root),
                "dst": str(dst_root),
                "task_index": args.task_index,
                "new_task_index": args.new_task_index,
                "task_name": original_task_name,
                "episodes": len(new_episodes_records),
                "frames": global_frame_offset,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
