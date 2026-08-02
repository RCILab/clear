#!/usr/bin/env python3
"""Merge incrementally written benchmark JSON shards with key validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FAMILY_ORDER = {"free": 0, "swap": 1, "circ15": 2, "rect15": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    algorithms = {payload["algorithm"] for payload in payloads}
    if len(algorithms) != 1:
        raise ValueError(f"mixed algorithms: {sorted(algorithms)}")

    records: dict[tuple[str, int, int], dict] = {}
    for payload in payloads:
        for record in payload["records"]:
            key = (
                str(record["family"]),
                int(record["n_robots"]),
                int(record["seed"]),
            )
            previous = records.get(key)
            if previous is not None:
                if previous != record:
                    raise ValueError(f"conflicting duplicate record: {key}")
                continue
            records[key] = record

    merged = dict(payloads[0])
    merged["records"] = sorted(
        records.values(),
        key=lambda record: (
            FAMILY_ORDER[str(record["family"])],
            int(record["n_robots"]),
            int(record["seed"]),
        ),
    )
    merged["merged_from"] = [str(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {len(merged['records'])} unique records",
        flush=True,
    )


if __name__ == "__main__":
    main()
