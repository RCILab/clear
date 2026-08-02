"""Aggregate the dedicated behavior-preserving MGR timing audit."""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    records = []
    for filename in sorted(
        glob.glob("results/mgr_timing_cvxopt_n*.jsonl")
    ):
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "error" in record:
                    raise RuntimeError(record["error"])
                records.append(record)
    if len(records) != 48:
        raise RuntimeError(f"expected 48 timing records, found {len(records)}")
    fingerprints = [
        record["scenario_fingerprint"] for record in records
    ]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("duplicate timing scenario fingerprints")
    records.sort(
        key=lambda record: (
            int(record["num_agents"]),
            record["scenario_family"],
            int(record["scenario_seed"]),
        )
    )
    combined = Path("results/mgr_timing_cvxopt_48.jsonl")
    combined.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    groups = defaultdict(list)
    for record in records:
        groups[(record["scenario_family"], record["num_agents"])].append(
            record
        )
    summaries = []
    for (family, robots), group in sorted(groups.items()):
        if len(group) != 3:
            raise RuntimeError(
                f"expected 3 timing seeds for {family} N={robots}"
            )
        summaries.append(
            {
                "family": family,
                "n_robots": robots,
                "mean_ms": float(
                    1.0e3
                    * np.mean(
                        [record["mean_iteration_wall_s"] for record in group]
                    )
                ),
                "std_seed_mean_ms": float(
                    1.0e3
                    * np.std(
                        [record["mean_iteration_wall_s"] for record in group],
                        ddof=1,
                    )
                ),
                "worst_seed_p95_ms": float(
                    1.0e3
                    * max(
                        record["p95_iteration_wall_s"] for record in group
                    )
                ),
            }
        )
    worst_by_size = []
    for robots in (20, 40, 60, 80):
        candidates = [
            summary for summary in summaries
            if summary["n_robots"] == robots
        ]
        worst_by_size.append(
            max(candidates, key=lambda item: item["worst_seed_p95_ms"])
        )
    payload = {
        "protocol": {
            "qp_backend": "cvxopt",
            "optimization": (
                "behavior-preserving vectorization, caching, and "
                "constant-QP-data reuse"
            ),
            "control_period_s": 0.03,
            "horizon_s": 60.0,
            "seeds": [0, 7, 13],
            "warmup_steps": 40,
            "parallel_processes": 4,
        },
        "groups": summaries,
        "worst_by_size": worst_by_size,
    }
    output = Path("results/mgr_timing_summary.json")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_lines = [
        "# Behavior-preserving optimized MGR timing",
        "",
        "- Public CVXOPT numerical path retained.",
        "- Three seeds per family and team size; 40 warm-up steps excluded.",
        "- Mean is the mean of seed means; p95 is the largest seed p95.",
        "",
        "| N | Worst task | Mean (ms) | Seed std. (ms) | "
        "Worst-seed p95 (ms) | p95 < 30 ms |",
        "|---:|---|---:|---:|---:|:---:|",
    ]
    for summary in worst_by_size:
        report_lines.append(
            f"| {summary['n_robots']} | {summary['family']} | "
            f"{summary['mean_ms']:.3f} | "
            f"{summary['std_seed_mean_ms']:.3f} | "
            f"{summary['worst_seed_p95_ms']:.3f} | "
            f"{'yes' if summary['worst_seed_p95_ms'] < 30.0 else 'no'} |"
        )
    report = Path("results/MGR_TIMING_REPORT.md")
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(output)
    print(report)
    print(combined)


if __name__ == "__main__":
    main()
