"""Join MGR SMG multi-agent outcomes with paired solo travel times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", nargs="+", type=Path, required=True)
    parser.add_argument("--solo", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(paths):
    records = []
    for path in paths:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def main():
    args = parse_args()
    multi = read_jsonl(args.multi)
    solo = read_jsonl(args.solo)
    solo_index = {
        (
            record["parent_fingerprint"],
            int(record["robot_index"]),
        ): record
        for record in solo
        if "error" not in record
    }
    output_records = []
    for record in multi:
        if "error" in record:
            output_records.append(record)
            continue
        count = int(record["num_agents"])
        isolated = np.full(count, np.inf)
        for robot in range(count):
            solo_record = solo_index.get(
                (record["scenario_fingerprint"], robot)
            )
            if solo_record is None:
                continue
            value = solo_record.get("first_arrival_makespan_s")
            if value is not None:
                isolated[robot] = float(value)
        multi_times = np.asarray(
            [
                np.inf if value is None else float(value)
                for value in record["first_arrival_times_s"]
            ]
        )
        valid = (
            np.isfinite(multi_times)
            & np.isfinite(isolated)
            & (isolated > 0.0)
        )
        delays = multi_times[valid] - isolated[valid]
        normalized = delays / isolated[valid]
        combined = dict(record)
        combined.update(
            {
                "algorithm": "MGR",
                "family": record["scenario_family"],
                "n_robots": count,
                "seed": int(record["scenario_seed"]),
                "fingerprint": record["scenario_fingerprint"],
                "mission_success": bool(
                    record["success"]
                    and not record["physical_collision"]
                ),
                "safe": not bool(record["physical_collision"]),
                "robot_arrival_rate": float(record["arrival_rate"]),
                "delay_evaluated_robots": int(np.sum(valid)),
                "average_interference_delay_s": (
                    float(np.mean(delays)) if len(delays) else None
                ),
                "p95_interference_delay_s": (
                    float(np.quantile(delays, 0.95))
                    if len(delays)
                    else None
                ),
                "normalized_average_interference_delay": (
                    float(np.mean(normalized))
                    if len(normalized)
                    else None
                ),
                "isolated_ttg_s": [
                    None if not np.isfinite(value) else float(value)
                    for value in isolated
                ],
            }
        )
        output_records.append(combined)
    payload = {
        "algorithm": "MGR",
        "records": output_records,
        "multi_record_count": len(multi),
        "solo_record_count": len(solo),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {len(output_records)} paired MGR SMG records")


if __name__ == "__main__":
    main()

