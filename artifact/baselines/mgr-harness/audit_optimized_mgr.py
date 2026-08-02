"""Run paired reference/optimized MGR trace and timing audits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


TIMING_FIELDS = {
    "max_iteration_wall_s",
    "mean_iteration_wall_s",
    "p95_iteration_wall_s",
    "sum_max_agent_wall_s",
    "wall_time_s",
    "qp_solve_count",
    "qp_failure_events",
    "qp_mean_iterations",
    "qp_max_iterations",
    "qp_status_counts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--optimized-repo", type=Path, required=True)
    parser.add_argument(
        "--run-headless",
        type=Path,
        default=Path(__file__).with_name("run_headless.py"),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["free", "swap", "circ15", "rect15"],
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=int,
        default=[20, 40, 60, 80],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 7, 13],
    )
    parser.add_argument("--timeout", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_process(
    repo: Path,
    run_headless: Path,
    yaml_path: Path,
    timeout: float,
) -> dict:
    command = [
        sys.executable,
        str(run_headless),
        "--repo",
        str(repo),
        "--instance",
        str(yaml_path),
        "--dt",
        "0.03",
        "--timeout",
        str(timeout),
        "--arrival-threshold",
        "0.22",
        "--termination-mode",
        "mission",
        "--audit-pruning",
        "--trace-digest",
        "--timing-warmup-steps",
        "40",
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"{repo} failed for {yaml_path}:\n{stderr}\n{stdout}"
        )
    records = [
        json.loads(line)
        for line in stdout.splitlines()
        if line.strip().startswith("{")
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"expected one record from {repo} for {yaml_path}, got "
            f"{len(records)}"
        )
    return records[0]


def audit_case(
    record: dict,
    base: Path,
    reference_repo: Path,
    optimized_repo: Path,
    run_headless: Path,
    timeout: float,
) -> dict:
    yaml_path = (base / record["relative_yaml"]).resolve()
    with ThreadPoolExecutor(max_workers=2) as executor:
        reference_future = executor.submit(
            run_process,
            reference_repo,
            run_headless,
            yaml_path,
            timeout,
        )
        optimized_future = executor.submit(
            run_process,
            optimized_repo,
            run_headless,
            yaml_path,
            timeout,
        )
        reference = reference_future.result()
        optimized = optimized_future.result()

    common_fields = sorted(
        (set(reference) & set(optimized)) - TIMING_FIELDS
    )
    mismatches = [
        field
        for field in common_fields
        if reference[field] != optimized[field]
    ]
    return {
        "family": record["family"],
        "n_robots": record["n_robots"],
        "seed": record["seed"],
        "fingerprint": record["fingerprint"],
        "exact_non_timing_match": not mismatches,
        "mismatched_fields": mismatches,
        "state_trace_match": (
            reference["state_trace_sha256"]
            == optimized["state_trace_sha256"]
        ),
        "command_trace_match": (
            reference["command_trace_sha256"]
            == optimized["command_trace_sha256"]
        ),
        "reference_mean_ms": (
            1000.0 * reference["mean_iteration_wall_s"]
        ),
        "optimized_mean_ms": (
            1000.0 * optimized["mean_iteration_wall_s"]
        ),
        "reference_p95_ms": (
            1000.0 * reference["p95_iteration_wall_s"]
        ),
        "optimized_p95_ms": (
            1000.0 * optimized["p95_iteration_wall_s"]
        ),
        "mean_speedup": (
            reference["mean_iteration_wall_s"]
            / optimized["mean_iteration_wall_s"]
        ),
        "p95_speedup": (
            reference["p95_iteration_wall_s"]
            / optimized["p95_iteration_wall_s"]
        ),
    }


def summarize(cases: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["family"], case["n_robots"])].append(case)
    summaries = []
    for (family, robots), group in sorted(grouped.items()):
        summaries.append(
            {
                "family": family,
                "n_robots": robots,
                "cases": len(group),
                "exact_matches": sum(
                    case["exact_non_timing_match"]
                    and case["state_trace_match"]
                    and case["command_trace_match"]
                    for case in group
                ),
                "reference_mean_ms": statistics.mean(
                    case["reference_mean_ms"] for case in group
                ),
                "optimized_mean_ms": statistics.mean(
                    case["optimized_mean_ms"] for case in group
                ),
                "reference_worst_p95_ms": max(
                    case["reference_p95_ms"] for case in group
                ),
                "optimized_worst_p95_ms": max(
                    case["optimized_p95_ms"] for case in group
                ),
                "median_mean_speedup": statistics.median(
                    case["mean_speedup"] for case in group
                ),
                "median_p95_speedup": statistics.median(
                    case["p95_speedup"] for case in group
                ),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [
        record
        for record in manifest["records"]
        if record["family"] in args.families
        and record["n_robots"] in args.robots
        and record["seed"] in args.seeds
    ]
    expected = len(args.families) * len(args.robots) * len(args.seeds)
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} selected records, got {len(selected)}"
        )

    cases = []
    base = args.manifest.parent.resolve()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                audit_case,
                record,
                base,
                args.reference_repo.resolve(),
                args.optimized_repo.resolve(),
                args.run_headless.resolve(),
                args.timeout,
            ): record
            for record in selected
        }
        for future in as_completed(futures):
            case = future.result()
            cases.append(case)
            state = "exact" if (
                case["exact_non_timing_match"]
                and case["state_trace_match"]
                and case["command_trace_match"]
            ) else "MISMATCH"
            print(
                f"{case['family']} N={case['n_robots']} "
                f"seed={case['seed']}: {state}, "
                f"mean {case['mean_speedup']:.3f}x, "
                f"p95 {case['p95_speedup']:.3f}x",
                flush=True,
            )

    cases.sort(key=lambda case: (
        case["family"], case["n_robots"], case["seed"]
    ))
    exact_cases = sum(
        case["exact_non_timing_match"]
        and case["state_trace_match"]
        and case["command_trace_match"]
        for case in cases
    )
    payload = {
        "comparison": "released MGR versus behavior-preserving optimized MGR",
        "timeout_s": args.timeout,
        "seeds": args.seeds,
        "cases": cases,
        "summaries": summarize(cases),
        "overall": {
            "cases": len(cases),
            "exact_cases": exact_cases,
            "median_mean_speedup": statistics.median(
                case["mean_speedup"] for case in cases
            ),
            "median_p95_speedup": statistics.median(
                case["p95_speedup"] for case in cases
            ),
        },
    }
    if exact_cases != len(cases):
        raise RuntimeError(
            f"optimization equivalence failed: {exact_cases}/{len(cases)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
