#!/usr/bin/env python3
"""Aggregate canonical navigation-baseline result files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


FAMILY_NAMES = {
    "free": "Free",
    "swap": "Swap",
    "circ15": "Circ15",
    "circle": "Circ15",
    "rect15": "Rect15",
    "rect": "Rect15",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Additional input glob; repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    records = payload.get("records")
    if records is None:
        raise ValueError(f"{path} has no records array")
    algorithm = payload.get("algorithm")
    if algorithm is not None:
        for record in records:
            record.setdefault("algorithm", algorithm)
    return records


def first(record: dict, *names: str, default=None):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def normalize(record: dict) -> dict:
    algorithm = str(first(record, "algorithm", default="")).upper()
    if algorithm == "CLEAR":
        algorithm = "CLEAR"
    elif algorithm == "VANILLA CBF-QP":
        algorithm = "Vanilla CBF-QP"
    elif algorithm == "MGR":
        algorithm = "MGR"
    elif algorithm == "GCBF+":
        algorithm = "GCBF+"
    elif algorithm == "NH-ORCA":
        algorithm = "NH-ORCA"
    elif algorithm == "ORCA":
        algorithm = "ORCA"
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")

    family_raw = str(
        first(record, "family", "scenario_family", "environment")
    ).lower()
    family = FAMILY_NAMES.get(family_raw)
    if family is None:
        raise ValueError(f"unknown family {family_raw!r}")

    n_robots = int(first(record, "n_robots", "num_agents"))
    seed = int(first(record, "seed", "scenario_seed"))
    fingerprint = str(
        first(record, "fingerprint", "scenario_fingerprint")
    )

    if algorithm in ("CLEAR", "Vanilla CBF-QP"):
        arrived = bool(record["mission_success"])
        pair_safe = (
            float(record["minimum_physical_pair_distance_m"]) >= 0.44 - 1e-6
        )
        obstacle_safe = (
            float(record["minimum_physical_obstacle_clearance_m"]) >= -1e-6
        )
        safe = pair_safe and obstacle_safe
        success = arrived and safe
    elif algorithm == "MGR":
        arrived = float(record["arrival_rate"]) >= 1.0 - 1e-12
        pair_safe = (
            float(record["min_pair_declared_safety_margin_m"]) >= -1e-6
        )
        obstacle_margin = record.get(
            "min_obstacle_declared_safety_margin_m"
        )
        obstacle_safe = (
            obstacle_margin is None or float(obstacle_margin) >= -1e-6
        )
        workspace_safe = (
            float(record["min_workspace_safe_radius_margin_m"]) >= -1e-6
        )
        safe = (
            not bool(record["physical_collision"])
            and pair_safe
            and obstacle_safe
            and workspace_safe
        )
        success = arrived and safe
    else:
        arrived = bool(
            first(
                record,
                "final_arrival_success",
                "arrival_success",
                default=False,
            )
        )
        safe = bool(record["safe"])
        success = bool(record["success"])

    if algorithm == "MGR":
        makespan = first(
            record,
            "first_arrival_makespan_s",
            "makespan_s",
        )
    else:
        makespan = first(
            record,
            "makespan_s",
            "first_arrival_makespan_s",
        )
    control_ms = first(record, "control_time_mean_ms")
    if control_ms is None and record.get("mean_iteration_wall_s") is not None:
        control_ms = 1000.0 * float(record["mean_iteration_wall_s"])

    return {
        "algorithm": algorithm,
        "family": family,
        "n_robots": n_robots,
        "seed": seed,
        "fingerprint": fingerprint,
        "arrived": arrived,
        "safe": safe,
        "success": success,
        "robot_arrival_rate": float(
            first(record, "robot_arrival_rate", "arrival_rate", default=0.0)
        ),
        "makespan_s": None if makespan is None else float(makespan),
        "control_time_mean_ms": (
            None if control_ms is None else float(control_ms)
        ),
    }


def exact_two_sided_binomial(left: int, right: int) -> float:
    total = left + right
    if total == 0:
        return 1.0
    tail = min(left, right)
    probability = sum(math.comb(total, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2.0**total))


def bootstrap_mean_interval(
    values: list[float],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=float)
    estimates = np.empty(samples, dtype=float)
    batch_size = 2_000
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(
            0,
            len(array),
            size=(stop - start, len(array)),
        )
        estimates[start:stop] = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def paired_summary(
    clear: dict[tuple, dict],
    baseline: dict[tuple, dict],
    horizon: float,
    rng: np.random.Generator,
    samples: int,
    n_robots: int | None,
    outcome: str,
) -> dict:
    common = sorted(set(clear) & set(baseline))
    if n_robots is not None:
        common = [key for key in common if key[1] == n_robots]

    both = clear_only = baseline_only = neither = 0
    joint_deltas = []
    penalized_deltas = []
    for key in common:
        lhs = clear[key]
        rhs = baseline[key]
        if lhs[outcome] and rhs[outcome]:
            both += 1
            joint_deltas.append(lhs["makespan_s"] - rhs["makespan_s"])
        elif lhs[outcome]:
            clear_only += 1
        elif rhs[outcome]:
            baseline_only += 1
        else:
            neither += 1
        lhs_time = lhs["makespan_s"] if lhs[outcome] else horizon
        rhs_time = rhs["makespan_s"] if rhs[outcome] else horizon
        penalized_deltas.append(lhs_time - rhs_time)

    joint_est, joint_lo, joint_hi = bootstrap_mean_interval(
        joint_deltas, rng, samples
    )
    penalized_est, penalized_lo, penalized_hi = bootstrap_mean_interval(
        penalized_deltas, rng, samples
    )
    clear_faster = sum(value < -1e-12 for value in joint_deltas)
    baseline_faster = sum(value > 1e-12 for value in joint_deltas)
    ties = len(joint_deltas) - clear_faster - baseline_faster
    return {
        "outcome": outcome,
        "n_robots": n_robots,
        "instances": len(common),
        "both_success": both,
        "clear_only": clear_only,
        "baseline_only": baseline_only,
        "neither": neither,
        "mcnemar_exact_p": exact_two_sided_binomial(
            clear_only, baseline_only
        ),
        "joint_success_delta_s": {
            "count": len(joint_deltas),
            "mean": joint_est,
            "ci95": [joint_lo, joint_hi],
            "clear_faster": clear_faster,
            "baseline_faster": baseline_faster,
            "ties": ties,
            "sign_test_p": exact_two_sided_binomial(
                clear_faster, baseline_faster
            ),
        },
        "timeout_penalized_delta_s": {
            "mean": penalized_est,
            "ci95": [penalized_lo, penalized_hi],
        },
    }


def main() -> None:
    args = parse_args()
    for pattern in args.input_glob:
        args.inputs.extend(Path(path) for path in sorted(glob.glob(pattern)))
    if not args.inputs:
        raise ValueError("at least one input is required")
    records = [
        normalize(record)
        for path in args.inputs
        for record in load_records(path)
    ]

    seen: set[tuple] = set()
    for record in records:
        key = (
            record["algorithm"],
            record["family"],
            record["n_robots"],
            record["seed"],
        )
        if key in seen:
            raise ValueError(f"duplicate algorithm instance: {key}")
        seen.add(key)

    fingerprints: dict[tuple, set[str]] = defaultdict(set)
    for record in records:
        fingerprints[
            (record["family"], record["n_robots"], record["seed"])
        ].add(record["fingerprint"])
    mismatches = {
        str(key): sorted(values)
        for key, values in fingerprints.items()
        if len(values) > 1
    }
    if mismatches:
        raise ValueError(f"fingerprint mismatch: {mismatches}")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        groups[
            (record["algorithm"], record["n_robots"], record["family"])
        ].append(record)

    rows = []
    for (algorithm, n_robots, family), group in sorted(groups.items()):
        arrived_times = [
            record["makespan_s"]
            for record in group
            if record["arrived"] and record["makespan_s"] is not None
        ]
        certified_times = [
            record["makespan_s"]
            for record in group
            if record["success"] and record["makespan_s"] is not None
        ]
        arrival_penalized_times = [
            record["makespan_s"]
            if record["arrived"] and record["makespan_s"] is not None
            else args.horizon
            for record in group
        ]
        certified_penalized_times = [
            record["makespan_s"]
            if record["success"] and record["makespan_s"] is not None
            else args.horizon
            for record in group
        ]
        timing = [
            record["control_time_mean_ms"]
            for record in group
            if record["control_time_mean_ms"] is not None
        ]
        rows.append(
            {
                "algorithm": algorithm,
                "n_robots": n_robots,
                "family": family,
                "trials": len(group),
                "arrived": sum(record["arrived"] for record in group),
                "safe": sum(record["safe"] for record in group),
                "successes": sum(record["success"] for record in group),
                "robot_arrival_rate": mean(
                    record["robot_arrival_rate"] for record in group
                ),
                "arrived_makespan_mean_s": (
                    mean(arrived_times) if arrived_times else None
                ),
                "certified_makespan_mean_s": (
                    mean(certified_times) if certified_times else None
                ),
                "arrival_penalized_time_mean_s": mean(
                    arrival_penalized_times
                ),
                "certified_penalized_time_mean_s": mean(
                    certified_penalized_times
                ),
                "run_mean_control_time_ms": (
                    mean(timing) if timing else None
                ),
            }
        )

    by_algorithm: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for record in records:
        key = (
            record["family"],
            record["n_robots"],
            record["seed"],
        )
        by_algorithm[record["algorithm"]][key] = record

    pairwise = []
    pairwise_certified = []
    if "CLEAR" in by_algorithm:
        for algorithm in sorted(by_algorithm):
            if algorithm == "CLEAR":
                continue
            rng = np.random.default_rng(
                args.bootstrap_seed
                + sum(ord(character) for character in algorithm)
            )
            comparisons = [
                paired_summary(
                    by_algorithm["CLEAR"],
                    by_algorithm[algorithm],
                    args.horizon,
                    rng,
                    args.bootstrap_samples,
                    n_robots,
                    "arrived",
                )
                for n_robots in (None, 20, 40, 60, 80)
            ]
            pairwise.append(
                {
                    "baseline": algorithm,
                    "comparisons": [
                        value for value in comparisons if value["instances"]
                    ],
                }
            )
            certified_comparisons = [
                paired_summary(
                    by_algorithm["CLEAR"],
                    by_algorithm[algorithm],
                    args.horizon,
                    rng,
                    args.bootstrap_samples,
                    n_robots,
                    "success",
                )
                for n_robots in (None, 20, 40, 60, 80)
            ]
            pairwise_certified.append(
                {
                    "baseline": algorithm,
                    "comparisons": [
                        value
                        for value in certified_comparisons
                        if value["instances"]
                    ],
                }
            )

    totals = []
    for algorithm in sorted(by_algorithm):
        group = list(by_algorithm[algorithm].values())
        totals.append(
            {
                "algorithm": algorithm,
                "trials": len(group),
                "all_arrived": sum(record["arrived"] for record in group),
                "safe_runs": sum(record["safe"] for record in group),
                "certified_completions": sum(
                    record["success"] for record in group
                ),
                "mean_robot_arrival_rate": mean(
                    record["robot_arrival_rate"] for record in group
                ),
            }
        )

    payload = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "horizon_s": args.horizon,
        "record_count": len(records),
        "fingerprints_verified": True,
        "primary_outcome": "simultaneous all-robot arrival",
        "certified_outcome": (
            "arrival plus common pair, obstacle, and workspace margins"
        ),
        "pairwise_clear_comparison": pairwise,
        "pairwise_clear_certified_comparison": pairwise_certified,
        "totals": totals,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"wrote {args.output}: {len(records)} records, {len(rows)} rows")


if __name__ == "__main__":
    main()
