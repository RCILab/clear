"""Merge deterministic split result files into one duplicate-checked payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FAMILY_ORDER = {"free": 0, "circ15": 1, "rect15": 2, "swap": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata: dict | None = None
    records_by_fingerprint: dict[str, dict] = {}
    keys: dict[tuple[str, int, int], str] = {}
    duplicate_count = 0

    for path in args.files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_metadata = {
            key: value for key, value in payload.items() if key != "records"
        }
        if metadata is None:
            metadata = candidate_metadata
        elif candidate_metadata != metadata:
            raise ValueError(f"metadata mismatch: {path}")
        for record in payload["records"]:
            fingerprint = record["fingerprint"]
            key = (
                record["family"],
                int(record["n_robots"]),
                int(record["seed"]),
            )
            if fingerprint in records_by_fingerprint:
                if records_by_fingerprint[fingerprint] != record:
                    raise ValueError(
                        f"nonidentical duplicate fingerprint: {fingerprint}"
                    )
                duplicate_count += 1
                continue
            if key in keys:
                raise ValueError(
                    f"scenario key has multiple fingerprints: {key}"
                )
            records_by_fingerprint[fingerprint] = record
            keys[key] = fingerprint

    records = sorted(
        records_by_fingerprint.values(),
        key=lambda record: (
            int(record["n_robots"]),
            FAMILY_ORDER[record["family"]],
            int(record["seed"]),
        ),
    )
    if (
        args.expected_records is not None
        and len(records) != args.expected_records
    ):
        raise ValueError(
            f"expected {args.expected_records} unique records, got "
            f"{len(records)}"
        )
    if metadata is None:
        raise ValueError("at least one input file is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                **metadata,
                "canonicalization": {
                    "input_files": [str(path) for path in args.files],
                    "identical_duplicates_removed": duplicate_count,
                },
                "records": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {len(records)} unique records, "
        f"{duplicate_count} identical duplicates removed"
    )


if __name__ == "__main__":
    main()
