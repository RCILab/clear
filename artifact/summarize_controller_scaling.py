"""Aggregate the controller-only scaling benchmark across seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("validation/controller_scaling.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/controller_scaling_summary.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("validation/CONTROLLER_SCALING.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in payload["records"]:
        grouped[(record["family"], record["n_robots"])].append(record)

    groups: list[dict] = []
    for (family, count), records in sorted(
        grouped.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        means = np.asarray(
            [
                item["command_after_warmup"]["mean_ms"]
                for item in records
            ]
        )
        summary = {
            "family": family,
            "n_robots": count,
            "seed_count": len(records),
            "mean_controller_ms": float(np.mean(means)),
            "std_seed_mean_ms": float(
                np.std(means, ddof=1) if len(means) > 1 else 0.0
            ),
            "maximum_seed_p95_ms": float(
                max(
                    item["command_after_warmup"]["p95_ms"]
                    for item in records
                )
            ),
            "maximum_observed_ms": float(
                max(
                    item["command_after_warmup"]["maximum_ms"]
                    for item in records
                )
            ),
            "maximum_cbf_sweeps": int(
                max(
                    item.get(
                        "maximum_native_projection_sweeps",
                        item.get("maximum_cbf_sweeps", 0),
                    )
                    for item in records
                )
            ),
            "nonconverged_steps": int(
                sum(item["nonconverged_steps"] for item in records)
            ),
        }
        groups.append(summary)

    worst_by_count: list[dict] = []
    for count in sorted({group["n_robots"] for group in groups}):
        candidates = [
            group for group in groups if group["n_robots"] == count
        ]
        worst = max(
            candidates,
            key=lambda group: group["maximum_seed_p95_ms"],
        )
        worst_by_count.append(
            {
                "n_robots": count,
                "family": worst["family"],
                "mean_controller_ms": worst["mean_controller_ms"],
                "maximum_seed_p95_ms": worst["maximum_seed_p95_ms"],
                "within_30ms_at_p95": (
                    worst["maximum_seed_p95_ms"] <= 30.0
                ),
            }
        )

    output = {
        "measurement_scope": payload["measurement_scope"],
        "environment": payload["environment"],
        "groups": groups,
        "worst_family_by_count": worst_by_count,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Controller computation-time scaling",
        "",
        payload["measurement_scope"],
        "",
        f"CPU: {payload['environment']['cpu_model']}",
        "",
        "| N | Family | Mean command (ms) | Worst seed p95 (ms) | Max (ms) |",
        "|---:|:---|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            "| {n_robots} | {family} | "
            "{mean_controller_ms:.3f} | "
            "{maximum_seed_p95_ms:.3f} | "
            "{maximum_observed_ms:.3f} |".format(**group)
        )
    lines.extend(
        [
            "",
            "The p95 column is the largest per-seed p95 among the three "
            "paired state streams. The maximum is retained as an outlier "
            "diagnostic, not as the real-time claim.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
