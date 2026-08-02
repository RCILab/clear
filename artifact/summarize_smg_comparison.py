"""Aggregate paired SMG success, flow, and method-specific delay metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import comb
from pathlib import Path
from statistics import mean

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--expected-trials",
        type=int,
        help="require this many distinct seeds in every method/scenario/N row",
    )
    return parser.parse_args()


def load(path):
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload["records"]


def method_name(record):
    raw = record.get("method", record.get("algorithm", ""))
    aliases = {
        "clear": "CLEAR",
        "vanilla cbf-qp": "Vanilla CBF-QP",
        "mgr": "MGR",
        "orca": "ORCA",
        "nh-orca": "NH-ORCA",
        "gcbf+": "GCBF+",
        "impc-dr": "IMPC-DR",
    }
    key = str(raw).strip().lower()
    if key not in aliases:
        raise ValueError(f"unknown SMG method {raw!r}")
    return aliases[key]


def family_name(record):
    raw = str(
        record.get(
            "family",
            record.get("scenario_family", ""),
        )
    ).lower()
    if raw.startswith("doorway"):
        return "Doorway"
    if raw.startswith("intersection"):
        return "Intersection"
    raise ValueError(f"unknown SMG family {raw!r}")


def normalized(record):
    method = method_name(record)
    success = bool(
        record.get(
            "mission_success",
            record.get("success", False),
        )
    )
    if "safe" in record:
        safe = bool(record["safe"])
    elif (
        "minimum_physical_pair_distance_m" in record
        and "minimum_physical_obstacle_clearance_m" in record
    ):
        safe = bool(
            float(record["minimum_physical_pair_distance_m"])
            >= 0.44 - 1e-6
            and float(record["minimum_physical_obstacle_clearance_m"])
            >= -1e-6
        )
    else:
        safe = bool(record.get("physical_mission_success", success))
    multi_times = record.get("first_arrival_times_s", [])
    solo_times = record.get("isolated_ttg_s", [])
    all_arrived = bool(
        record.get(
            "arrival_success",
            (
                len(multi_times)
                == int(record.get("n_robots", record.get("num_agents", 0)))
                and all(value is not None for value in multi_times)
            ),
        )
    )
    delays = []
    if len(multi_times) == len(solo_times):
        for multi, solo in zip(multi_times, solo_times):
            if multi is not None and solo is not None and solo > 0.0:
                delays.append(float(multi) - float(solo))
    return {
        "method": method,
        "family": family_name(record),
        "n_robots": int(
            record.get("n_robots", record.get("num_agents"))
        ),
        "seed": int(record.get("seed", record.get("scenario_seed", 0))),
        "fingerprint": str(
            record.get(
                "fingerprint",
                record.get("scenario_fingerprint", ""),
            )
        ),
        "success": success,
        "all_arrived": all_arrived,
        "safe": safe,
        "arrival_rate": float(
            record.get(
                "robot_arrival_rate",
                record.get("arrival_rate", 0.0),
            )
        ),
        "makespan_s": record.get(
            "makespan_s",
            record.get("first_arrival_makespan_s"),
        ),
        "gap_completed": int(record.get("gap_completed_robots", 0)),
        "throughput": float(
            record.get("parallel_throughput_robots_per_s", 0.0)
        ),
        "flow_rate": float(
            record.get("smg_flow_rate_robots_per_m_s", 0.0)
        ),
        "delays": delays,
        "mean_control_step_ms": record.get(
            "mean_control_step_ms",
            record.get("control_time_mean_ms"),
        ),
    }


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        comb(discordant, value)
        for value in range(min(left_only, right_only) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def main():
    args = parse_args()
    records = [
        normalized(record)
        for path in args.inputs
        for record in load(path)
        if "error" not in record
    ]
    record_keys = [
        (
            record["method"],
            record["family"],
            record["n_robots"],
            record["seed"],
        )
        for record in records
    ]
    if len(record_keys) != len(set(record_keys)):
        duplicates = sorted(
            key for key in set(record_keys) if record_keys.count(key) > 1
        )
        raise ValueError(f"duplicate SMG trials: {duplicates}")
    fingerprint_sets = defaultdict(set)
    for record in records:
        fingerprint_sets[
            (
                record["family"],
                record["n_robots"],
                record["seed"],
            )
        ].add(record["fingerprint"])
    mismatches = {
        str(key): sorted(values)
        for key, values in fingerprint_sets.items()
        if len(values) > 1
    }
    if mismatches:
        raise ValueError(f"paired fingerprint mismatch: {mismatches}")

    groups = defaultdict(list)
    for record in records:
        groups[
            (
                record["method"],
                record["family"],
                record["n_robots"],
            )
        ].append(record)
    if args.expected_trials is not None:
        bad_counts = {
            str(key): len(group)
            for key, group in groups.items()
            if len(group) != args.expected_trials
            or len({record["seed"] for record in group})
            != args.expected_trials
        }
        if bad_counts:
            raise ValueError(
                "unexpected SMG trial counts: "
                f"expected {args.expected_trials}, got {bad_counts}"
            )
    rows = []
    for (method, family, count), group in sorted(groups.items()):
        delays = [
            delay for record in group for delay in record["delays"]
        ]
        successful_flow = [
            record["flow_rate"]
            for record in group
            if record["gap_completed"] == count
        ]
        control = [
            float(record["mean_control_step_ms"])
            for record in group
            if record["mean_control_step_ms"] is not None
        ]
        rows.append(
            {
                "method": method,
                "family": family,
                "n_robots": count,
                "trials": len(group),
                "mission_successes": sum(
                    record["success"] for record in group
                ),
                "all_arrived": sum(
                    record["all_arrived"] for record in group
                ),
                "safe_runs": sum(record["safe"] for record in group),
                "mean_robot_arrival_rate": mean(
                    record["arrival_rate"] for record in group
                ),
                "resource_completion_rate": mean(
                    record["gap_completed"] == count for record in group
                ),
                "penalized_flow_rate_robots_per_m_s": mean(
                    record["flow_rate"] for record in group
                ),
                "successful_flow_rate_robots_per_m_s": (
                    mean(successful_flow) if successful_flow else None
                ),
                "delay_evaluated_robots": len(delays),
                "average_interference_delay_s": (
                    mean(delays) if delays else None
                ),
                "p95_interference_delay_s": (
                    float(np.quantile(delays, 0.95))
                    if delays
                    else None
                ),
                "mean_control_step_ms": (
                    mean(control) if control else None
                ),
            }
        )
    by_instance = {
        (
            record["method"],
            record["family"],
            record["n_robots"],
            record["seed"],
        ): record
        for record in records
    }
    pairwise = []
    baseline_groups = sorted(
        {
            (
                record["method"],
                record["family"],
                record["n_robots"],
            )
            for record in records
            if record["method"] != "CLEAR"
        }
    )
    for method, family, count in baseline_groups:
        baseline_seeds = {
            record["seed"]
            for record in groups[(method, family, count)]
        }
        clear_seeds = {
            record["seed"]
            for record in groups.get(("CLEAR", family, count), [])
        }
        seeds = sorted(baseline_seeds & clear_seeds)
        if not seeds:
            continue
        clear_only = baseline_only = both = neither = 0
        resource_clear_only = resource_baseline_only = 0
        resource_both = resource_neither = 0
        flow_delta = []
        delay_delta = []
        for seed in seeds:
            clear = by_instance[("CLEAR", family, count, seed)]
            baseline = by_instance[(method, family, count, seed)]
            outcomes = (clear["success"], baseline["success"])
            if outcomes == (True, True):
                both += 1
            elif outcomes == (True, False):
                clear_only += 1
            elif outcomes == (False, True):
                baseline_only += 1
            else:
                neither += 1
            clear_resource = clear["gap_completed"] == count
            baseline_resource = baseline["gap_completed"] == count
            if clear_resource and baseline_resource:
                resource_both += 1
            elif clear_resource:
                resource_clear_only += 1
            elif baseline_resource:
                resource_baseline_only += 1
            else:
                resource_neither += 1
            flow_delta.append(clear["flow_rate"] - baseline["flow_rate"])
            if clear["delays"] and baseline["delays"]:
                delay_delta.append(
                    mean(clear["delays"]) - mean(baseline["delays"])
                )
        pairwise.append(
            {
                "baseline": method,
                "family": family,
                "n_robots": count,
                "paired_trials": len(seeds),
                "mission": {
                    "both": both,
                    "clear_only": clear_only,
                    "baseline_only": baseline_only,
                    "neither": neither,
                    "mcnemar_exact_p": exact_mcnemar_p(
                        clear_only, baseline_only
                    ),
                },
                "resource": {
                    "both": resource_both,
                    "clear_only": resource_clear_only,
                    "baseline_only": resource_baseline_only,
                    "neither": resource_neither,
                    "mcnemar_exact_p": exact_mcnemar_p(
                        resource_clear_only, resource_baseline_only
                    ),
                },
                "mean_paired_penalized_flow_delta": mean(flow_delta),
                "mean_paired_delay_delta_s": (
                    mean(delay_delta) if delay_delta else None
                ),
                "paired_delay_trials": len(delay_delta),
            }
        )
    payload = {
        "record_count": len(records),
        "fingerprints_verified": True,
        "metric_definitions": {
            "penalized_flow_rate": (
                "resource flow rate with zero assigned to incomplete trials"
            ),
            "average_interference_delay": (
                "pooled per-robot TTG_multi - TTG_solo for the same method "
                "and paired start-goal instance"
            ),
        },
        "rows": rows,
        "paired_clear_comparisons": pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.markdown:
        lines = [
            "# Social Mini-Game comparison",
            "",
            "| Method | Scenario | N | Arr./cert. | Resource | Flow | Delay (s) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            delay = row["average_interference_delay_s"]
            lines.append(
                f"| {row['method']} | {row['family']} | {row['n_robots']} "
                f"| {row['all_arrived']}/{row['mission_successes']} "
                f"| {row['resource_completion_rate']:.1%} "
                f"| {row['penalized_flow_rate_robots_per_m_s']:.3f} "
                f"| {'--' if delay is None else f'{delay:.3f}'} |"
            )
        lines.extend(
            [
                "",
                "## Paired CLEAR comparison",
                "",
                "| Baseline | Scenario | N | C-only | B-only | "
                "Flow delta | Delay delta (s) |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in pairwise:
            delay = row["mean_paired_delay_delta_s"]
            lines.append(
                f"| {row['baseline']} | {row['family']} | "
                f"{row['n_robots']} | {row['mission']['clear_only']} | "
                f"{row['mission']['baseline_only']} | "
                f"{row['mean_paired_penalized_flow_delta']:.3f} | "
                f"{'--' if delay is None else f'{delay:.3f}'} |"
            )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"aggregated {len(records)} SMG records into {len(rows)} rows")


if __name__ == "__main__":
    main()
