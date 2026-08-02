"""Combine checkpointed CLEAR result files without mixing configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robots", nargs="+", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata: dict | None = None
    records: dict[str, dict] = {}
    allowed = set(args.robots) if args.robots else None

    for path in args.files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_metadata = {
            key: value for key, value in payload.items() if key != "records"
        }
        if metadata is None:
            metadata = current_metadata
        elif current_metadata != metadata:
            raise ValueError(f"mixed configurations: {path}")

        for record in payload["records"]:
            if allowed is not None and record["n_robots"] not in allowed:
                continue
            fingerprint = record["fingerprint"]
            if fingerprint in records:
                raise ValueError(f"duplicate scenario fingerprint: {fingerprint}")
            records[fingerprint] = record

    if metadata is None:
        raise ValueError("no input files")
    ordered = sorted(
        records.values(),
        key=lambda record: (
            record["n_robots"],
            record["family"],
            record["seed"],
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**metadata, "records": ordered}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {len(ordered)} records to {args.output}")


if __name__ == "__main__":
    main()
