"""Merge disjoint SMG JSON shards while rejecting conflicting trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def record_key(record):
    method = record.get("method", record.get("algorithm"))
    family = record.get("family", record.get("scenario_family"))
    count = record.get("n_robots", record.get("num_agents"))
    seed = record.get("seed", record.get("scenario_seed"))
    return str(method), str(family), int(count), int(seed)


def main():
    args = parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    merged = dict(payloads[0])
    merged["source_shards"] = [str(path) for path in args.inputs]
    records = {}
    for payload in payloads:
        for record in payload["records"]:
            key = record_key(record)
            if key in records and records[key] != record:
                raise ValueError(f"conflicting duplicate SMG trial: {key}")
            records[key] = record
    merged["records"] = [
        records[key]
        for key in sorted(
            records,
            key=lambda value: (
                value[1],
                value[2],
                value[3],
                value[0],
            ),
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"merged {len(args.inputs)} shards into {len(records)} trials")


if __name__ == "__main__":
    main()
