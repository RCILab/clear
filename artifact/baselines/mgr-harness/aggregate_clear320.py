"""Validate and aggregate the canonical 320-trial CLEAR/MGR pairing."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import glob
import json
from math import comb
from pathlib import Path

import numpy as np


FAMILY_ORDER = ("free", "swap", "circ15", "rect15")
ROBOT_ORDER = (20, 40, 60, 80)
EXPECTED_COMMIT = "84aaefd705ab61748010ad3dd6a8e018a46e7713"


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-raw",
        type=Path,
        default=Path(
            "../../results/unicycle_clear_all_sizes_raw.json"
        ),
    )
    parser.add_argument(
        "--mgr-glob",
        default="results/mgr_clear320_final*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mgr_clear320_paired_summary.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/MGR_CLEAR320_PAIRED_REPORT.md"),
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help=(
            "Label the evaluated MGR implementation as computationally "
            "optimized with behavior-preserving vectorization and caching."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def bootstrap_mean(
    values: list[float],
    rng: np.random.Generator,
    samples: int,
) -> Interval:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = np.mean(array[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return Interval(float(np.mean(array)), float(lower), float(upper))


def mcnemar_exact(clear_only: int, mgr_only: int) -> float:
    discordant = clear_only + mgr_only
    if discordant == 0:
        return 1.0
    tail = min(clear_only, mgr_only)
    probability = sum(comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def sign_test_exact(negative: int, positive: int) -> float:
    non_ties = negative + positive
    if non_ties == 0:
        return 1.0
    tail = min(negative, positive)
    probability = sum(comb(non_ties, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**non_ties))


def load_mgr(pattern: str) -> tuple[list[Path], dict[str, dict]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise RuntimeError(f"no MGR shard files match {pattern!r}")
    records = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if "error" in record:
                raise RuntimeError(
                    f"evaluator error in {path}:{line_number}: "
                    f"{record['error']}"
                )
            fingerprint = record["scenario_fingerprint"]
            if fingerprint in records:
                raise RuntimeError(f"duplicate MGR fingerprint {fingerprint}")
            if record["upstream_commit"] != EXPECTED_COMMIT:
                raise RuntimeError(
                    f"unexpected MGR commit {record['upstream_commit']}"
                )
            records[fingerprint] = record
    return paths, records


def load_clear(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for record in payload["records"]:
        fingerprint = record["fingerprint"]
        if fingerprint in records:
            raise RuntimeError(f"duplicate CLEAR fingerprint {fingerprint}")
        records[fingerprint] = record
    return payload, records


def margin_safe(record: dict, tolerance: float = 1.0e-6) -> bool:
    obstacle = record["minimum_physical_obstacle_clearance_m"]
    return (
        record["minimum_physical_pair_distance_m"] >= 0.44 - tolerance
        and obstacle >= -tolerance
    )


def aggregate_group(
    pairs: list[tuple[dict, dict]],
    rng: np.random.Generator,
    bootstrap_samples: int,
    timeout: float,
) -> dict:
    clear_success = [bool(clear["mission_success"]) for clear, _ in pairs]
    mgr_success = [bool(mgr["success"]) for _, mgr in pairs]
    clear_safe = [margin_safe(clear) for clear, _ in pairs]
    mgr_physical_safe = [
        not bool(mgr["physical_collision"]) for _, mgr in pairs
    ]
    mgr_margin_safe = [
        not bool(mgr["declared_safety_margin_violation"]) for _, mgr in pairs
    ]
    clear_only = sum(c and not m for c, m in zip(clear_success, mgr_success))
    mgr_only = sum(m and not c for c, m in zip(clear_success, mgr_success))
    joint_differences = [
        float(clear["makespan_s"] - mgr["first_arrival_makespan_s"])
        for (clear, mgr), c, m in zip(pairs, clear_success, mgr_success)
        if c and m
    ]
    penalized_differences = [
        float(
            (clear["makespan_s"] if c else timeout)
            - (mgr["first_arrival_makespan_s"] if m else timeout)
        )
        for (clear, mgr), c, m in zip(pairs, clear_success, mgr_success)
    ]
    clear_success_times = [
        float(clear["makespan_s"])
        for (clear, _), success in zip(pairs, clear_success)
        if success
    ]
    mgr_success_times = [
        float(mgr["first_arrival_makespan_s"])
        for (_, mgr), success in zip(pairs, mgr_success)
        if success
    ]
    joint_interval = bootstrap_mean(
        joint_differences, rng, bootstrap_samples
    )
    clear_faster = sum(difference < -1.0e-9 for difference in joint_differences)
    mgr_faster = sum(difference > 1.0e-9 for difference in joint_differences)
    time_ties = len(joint_differences) - clear_faster - mgr_faster
    penalized_interval = bootstrap_mean(
        penalized_differences, rng, bootstrap_samples
    )
    iteration_means = [
        float(mgr["mean_iteration_wall_s"]) for _, mgr in pairs
    ]
    iteration_p95s = [
        float(mgr["p95_iteration_wall_s"])
        for _, mgr in pairs
        if "p95_iteration_wall_s" in mgr
    ]
    qp_solves = sum(int(mgr.get("qp_solve_count", 0)) for _, mgr in pairs)
    qp_failures = sum(
        int(mgr.get("qp_failure_events", 0)) for _, mgr in pairs
    )
    qp_iteration_weight = sum(
        float(mgr.get("qp_mean_iterations") or 0.0)
        * int(mgr.get("qp_solve_count", 0))
        for _, mgr in pairs
    )
    return {
        "instances": len(pairs),
        "clear": {
            "arrival_successes": sum(clear_success),
            "arrival_success_rate": float(np.mean(clear_success)),
            "declared_safe_runs": sum(clear_safe),
            "robot_arrival_rate": float(
                np.mean(
                    [clear["robot_arrival_rate"] for clear, _ in pairs]
                )
            ),
            "successful_makespan_mean_s": (
                float(np.mean(clear_success_times))
                if clear_success_times
                else None
            ),
            "successful_makespan_std_s": (
                float(np.std(clear_success_times, ddof=1))
                if clear_success_times
                else None
            ),
        },
        "mgr": {
            "arrival_successes": sum(mgr_success),
            "arrival_success_rate": float(np.mean(mgr_success)),
            "physical_safe_runs": sum(mgr_physical_safe),
            "declared_margin_safe_runs": sum(mgr_margin_safe),
            "arrival_and_declared_margin_successes": sum(
                success and safe
                for success, safe in zip(mgr_success, mgr_margin_safe)
            ),
            "robot_arrival_rate": float(
                np.mean([mgr["arrival_rate"] for _, mgr in pairs])
            ),
            "successful_makespan_mean_s": (
                float(np.mean(mgr_success_times))
                if mgr_success_times
                else None
            ),
            "successful_makespan_std_s": (
                float(np.std(mgr_success_times, ddof=1))
                if len(mgr_success_times) > 1
                else None
            ),
        },
        "paired_success": {
            "clear_only": clear_only,
            "mgr_only": mgr_only,
            "both": sum(c and m for c, m in zip(clear_success, mgr_success)),
            "neither": sum(
                not c and not m for c, m in zip(clear_success, mgr_success)
            ),
            "mcnemar_exact_p": mcnemar_exact(clear_only, mgr_only),
        },
        "joint_success_clear_minus_mgr_makespan_s": {
            "count": len(joint_differences),
            **asdict(joint_interval),
            "median": (
                float(np.median(joint_differences))
                if joint_differences
                else None
            ),
            "clear_faster": clear_faster,
            "mgr_faster": mgr_faster,
            "ties": time_ties,
            "exact_sign_p": sign_test_exact(clear_faster, mgr_faster),
        },
        "timeout_penalized_clear_minus_mgr_s": {
            "timeout_s": timeout,
            **asdict(penalized_interval),
            "median": float(np.median(penalized_differences)),
        },
        "minimum_safety": {
            "clear_pair_distance_m": min(
                clear["minimum_physical_pair_distance_m"]
                for clear, _ in pairs
            ),
            "clear_static_margin_m": min(
                clear["minimum_physical_obstacle_clearance_m"]
                for clear, _ in pairs
            ),
            "mgr_pair_distance_m": min(
                mgr["min_pair_distance_m"] for _, mgr in pairs
            ),
            "mgr_obstacle_physical_clearance_m": finite_or_none(
                min(
                    (
                        mgr["min_obstacle_physical_clearance_m"]
                        if mgr["min_obstacle_physical_clearance_m"] is not None
                        else float("inf")
                    )
                    for _, mgr in pairs
                )
            ),
            "mgr_pair_declared_margin_m": min(
                mgr["min_pair_declared_safety_margin_m"]
                for _, mgr in pairs
            ),
            "mgr_obstacle_declared_margin_m": finite_or_none(
                min(
                    (
                        mgr["min_obstacle_declared_safety_margin_m"]
                        if mgr["min_obstacle_declared_safety_margin_m"] is not None
                        else float("inf")
                    )
                    for _, mgr in pairs
                )
            ),
        },
        "computation": {
            "mean_of_run_mean_iteration_ms": float(
                1.0e3 * np.mean(iteration_means)
            ),
            "median_run_mean_iteration_ms": float(
                1.0e3 * np.median(iteration_means)
            ),
            "mean_of_run_p95_iteration_ms": float(
                1.0e3 * np.mean(iteration_p95s)
            ) if iteration_p95s else None,
            "median_run_p95_iteration_ms": (
                float(1.0e3 * np.median(iteration_p95s))
                if iteration_p95s
                else None
            ),
            "qp_solve_count": qp_solves,
            "qp_failure_events": qp_failures,
            "qp_mean_iterations": (
                qp_iteration_weight / qp_solves if qp_solves else None
            ),
            "qp_max_iterations": max(
                int(mgr.get("qp_max_iterations", 0)) for _, mgr in pairs
            ),
        },
    }


def fmt_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / total:.1f}%)"


def fmt_time(value: float | None, deviation: float | None) -> str:
    if value is None:
        return "--"
    if deviation is None:
        return f"{value:.2f}"
    return f"{value:.2f} $\\pm$ {deviation:.2f}"


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    args = parse_args()
    clear_payload, clear = load_clear(args.clear_raw)
    shard_paths, mgr = load_mgr(args.mgr_glob)
    if len(clear) != 320 or len(mgr) != 320:
        raise RuntimeError(
            f"pairing incomplete: CLEAR={len(clear)}, MGR={len(mgr)}"
        )
    if set(clear) != set(mgr):
        raise RuntimeError(
            "CLEAR/MGR fingerprint sets differ: "
            f"clear_only={len(set(clear)-set(mgr))}, "
            f"mgr_only={len(set(mgr)-set(clear))}"
        )
    for fingerprint in clear:
        c = clear[fingerprint]
        m = mgr[fingerprint]
        if (
            c["family"] != m["scenario_family"]
            or c["n_robots"] != m["num_agents"]
            or c["seed"] != m["scenario_seed"]
        ):
            raise RuntimeError(f"metadata mismatch for {fingerprint}")

    grouped = defaultdict(list)
    all_pairs = []
    for fingerprint, clear_record in clear.items():
        pair = (clear_record, mgr[fingerprint])
        grouped[(clear_record["family"], clear_record["n_robots"])].append(
            pair
        )
        all_pairs.append(pair)
    rng = np.random.default_rng(20260729)
    summaries = []
    for family in FAMILY_ORDER:
        for robots in ROBOT_ORDER:
            pairs = sorted(
                grouped[(family, robots)],
                key=lambda pair: pair[0]["seed"],
            )
            if len(pairs) != 20:
                raise RuntimeError(
                    f"expected 20 pairs for {family} N={robots}"
                )
            summaries.append(
                {
                    "family": family,
                    "n_robots": robots,
                    **aggregate_group(
                        pairs,
                        rng,
                        args.bootstrap_samples,
                        timeout=60.0,
                    ),
                }
            )
    overall = aggregate_group(
        all_pairs,
        rng,
        args.bootstrap_samples,
        timeout=60.0,
    )
    family_summaries = []
    for family in FAMILY_ORDER:
        pairs = sorted(
            [
                pair
                for (group_family, _), group_pairs in grouped.items()
                if group_family == family
                for pair in group_pairs
            ],
            key=lambda pair: (pair[0]["n_robots"], pair[0]["seed"]),
        )
        if len(pairs) != 80:
            raise RuntimeError(f"expected 80 pairs for {family}")
        family_summaries.append(
            {
                "family": family,
                **aggregate_group(
                    pairs,
                    rng,
                    args.bootstrap_samples,
                    timeout=60.0,
                ),
            }
        )
    robot_summaries = []
    for robots in ROBOT_ORDER:
        pairs = sorted(
            [
                pair
                for (_, group_robots), group_pairs in grouped.items()
                if group_robots == robots
                for pair in group_pairs
            ],
            key=lambda pair: (pair[0]["family"], pair[0]["seed"]),
        )
        if len(pairs) != 80:
            raise RuntimeError(f"expected 80 pairs for N={robots}")
        robot_summaries.append(
            {
                "n_robots": robots,
                **aggregate_group(
                    pairs,
                    rng,
                    args.bootstrap_samples,
                    timeout=60.0,
                ),
            }
        )
    implementation = (
        "computationally optimized public MGR implementation"
        if args.optimized
        else "public MGR implementation"
    )
    output = {
        "comparison": f"CLEAR versus {implementation}",
        "paired": True,
        "mgr_implementation": implementation,
        "mgr_controller_source_modified": bool(args.optimized),
        "mgr_computational_optimization": (
            "vectorized and cached geometry and communication operations; "
            "constant QP data reuse with the public CVXOPT path retained"
            if args.optimized
            else None
        ),
        "mgr_evaluator": (
            "headless wrapper preserving upstream control/update order; exact "
            "linear swept safety evaluation between control nodes"
        ),
        "clear_source": str(args.clear_raw),
        "mgr_shards": [str(path) for path in shard_paths],
        "physical_protocol": clear_payload["physical_protocol"],
        "bootstrap_seed": 20260729,
        "bootstrap_samples": args.bootstrap_samples,
        "summaries": summaries,
        "family_summaries": family_summaries,
        "robot_summaries": robot_summaries,
        "overall": overall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Direct paired evaluation: CLEAR vs public MGR",
        "",
        "- 320 identical scenario fingerprints; 20 seeds per family/size.",
        "- Common protocol: 16 m, r=0.20 m, safety margin=0.02 m, "
        "dt=0.03 s, horizon=60 s, arrival radius=0.22 m.",
        (
            "- Public MGR code computationally optimized using cached and "
            "vectorized geometry and communication operations plus constant "
            "QP-data reuse; controller equations, CVXOPT numerical path, and "
            "scenario protocol retained."
            if args.optimized
            else
            "- Public MGR controller evaluated through a headless loop "
            "preserving its control/update order."
        ),
        "",
        "| Family | N | CLEAR arrived | MGR arrived | CLEAR time (s) | "
        "MGR time (s) | penalized CLEAR-MGR (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        clear_summary = summary["clear"]
        mgr_summary = summary["mgr"]
        difference = summary["timeout_penalized_clear_minus_mgr_s"]
        lines.append(
            f"| {summary['family']} | {summary['n_robots']} | "
            f"{fmt_rate(clear_summary['arrival_successes'], 20)} | "
            f"{fmt_rate(mgr_summary['arrival_successes'], 20)} | "
            f"{fmt_time(clear_summary['successful_makespan_mean_s'], clear_summary['successful_makespan_std_s'])} | "
            f"{fmt_time(mgr_summary['successful_makespan_mean_s'], mgr_summary['successful_makespan_std_s'])} | "
            f"{difference['estimate']:+.2f} "
            f"[{difference['lower']:+.2f}, {difference['upper']:+.2f}] |"
        )
    paired_success = overall["paired_success"]
    joint = overall["joint_success_clear_minus_mgr_makespan_s"]
    penalized = overall["timeout_penalized_clear_minus_mgr_s"]
    lines.extend(
        [
            "",
            "## Family aggregates",
            "",
            "| Family | CLEAR arrived | MGR arrived | "
            "joint-success CLEAR-MGR (s) | penalized CLEAR-MGR (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for summary in family_summaries:
        clear_summary = summary["clear"]
        mgr_summary = summary["mgr"]
        family_joint = summary[
            "joint_success_clear_minus_mgr_makespan_s"
        ]
        family_penalized = summary["timeout_penalized_clear_minus_mgr_s"]
        lines.append(
            f"| {summary['family']} | "
            f"{fmt_rate(clear_summary['arrival_successes'], 80)} | "
            f"{fmt_rate(mgr_summary['arrival_successes'], 80)} | "
            f"{family_joint['estimate']:+.2f} "
            f"[{family_joint['lower']:+.2f}, {family_joint['upper']:+.2f}] | "
            f"{family_penalized['estimate']:+.2f} "
            f"[{family_penalized['lower']:+.2f}, "
            f"{family_penalized['upper']:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Team-size aggregates",
            "",
            "| N | CLEAR arrived | MGR arrived | "
            "joint-success CLEAR-MGR (s) | penalized CLEAR-MGR (s) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in robot_summaries:
        clear_summary = summary["clear"]
        mgr_summary = summary["mgr"]
        robot_joint = summary[
            "joint_success_clear_minus_mgr_makespan_s"
        ]
        robot_penalized = summary["timeout_penalized_clear_minus_mgr_s"]
        lines.append(
            f"| {summary['n_robots']} | "
            f"{fmt_rate(clear_summary['arrival_successes'], 80)} | "
            f"{fmt_rate(mgr_summary['arrival_successes'], 80)} | "
            f"{robot_joint['estimate']:+.2f} "
            f"[{robot_joint['lower']:+.2f}, {robot_joint['upper']:+.2f}] | "
            f"{robot_penalized['estimate']:+.2f} "
            f"[{robot_penalized['lower']:+.2f}, "
            f"{robot_penalized['upper']:+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- CLEAR arrivals: {fmt_rate(overall['clear']['arrival_successes'], 320)}",
            f"- MGR arrivals: {fmt_rate(overall['mgr']['arrival_successes'], 320)}",
            f"- Discordant successes (CLEAR-only / MGR-only): "
            f"{paired_success['clear_only']} / {paired_success['mgr_only']}; "
            f"McNemar exact p={paired_success['mcnemar_exact_p']:.4g}.",
            f"- Joint-success paired time difference, CLEAR-MGR: "
            f"{joint['estimate']:+.2f} s "
            f"(95% bootstrap CI [{joint['lower']:+.2f}, "
            f"{joint['upper']:+.2f}]); CLEAR faster / MGR faster / ties: "
            f"{joint['clear_faster']} / {joint['mgr_faster']} / "
            f"{joint['ties']}; exact sign p={joint['exact_sign_p']:.4g}.",
            f"- Timeout-penalized paired time difference, CLEAR-MGR: "
            f"{penalized['estimate']:+.2f} s "
            f"(95% bootstrap CI [{penalized['lower']:+.2f}, "
            f"{penalized['upper']:+.2f}]).",
            f"- MGR physical-safe runs: "
            f"{fmt_rate(overall['mgr']['physical_safe_runs'], 320)}; "
            f"declared-margin-safe runs: "
            f"{fmt_rate(overall['mgr']['declared_margin_safe_runs'], 320)}.",
            f"- CLEAR declared-safe runs: "
            f"{fmt_rate(overall['clear']['declared_safe_runs'], 320)}.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
