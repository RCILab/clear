"""Compare reference and optimized MGR state/control traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TIMING_FIELDS = {
    "max_iteration_wall_s",
    "mean_iteration_wall_s",
    "p95_iteration_wall_s",
    "sum_max_agent_wall_s",
    "wall_time_s",
    "trace_npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--optimized-prefix", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise RuntimeError(f"expected one JSON record in {path}")
    return json.loads(lines[0])


def first_different_step(reference: np.ndarray, optimized: np.ndarray) -> int | None:
    if reference.shape != optimized.shape:
        return 0
    different = np.any(reference != optimized, axis=tuple(range(1, reference.ndim)))
    indices = np.flatnonzero(different)
    return int(indices[0]) if indices.size else None


def main() -> None:
    args = parse_args()
    reference_record = load_jsonl(args.reference_prefix.with_suffix(".jsonl"))
    optimized_record = load_jsonl(args.optimized_prefix.with_suffix(".jsonl"))
    reference = np.load(args.reference_prefix.with_suffix(".npz"))
    optimized = np.load(args.optimized_prefix.with_suffix(".npz"))

    common_fields = sorted(
        (set(reference_record) & set(optimized_record)) - TIMING_FIELDS
    )
    mismatched_fields = [
        field
        for field in common_fields
        if reference_record[field] != optimized_record[field]
    ]
    result = {
        "reference": str(args.reference_prefix),
        "optimized": str(args.optimized_prefix),
        "common_non_timing_fields": len(common_fields),
        "mismatched_non_timing_fields": mismatched_fields,
        "states_shape_equal": (
            reference["states"].shape == optimized["states"].shape
        ),
        "commands_shape_equal": (
            reference["commands"].shape == optimized["commands"].shape
        ),
        "first_different_state_step": first_different_step(
            reference["states"], optimized["states"]
        ),
        "first_different_command_step": first_different_step(
            reference["commands"], optimized["commands"]
        ),
        "max_abs_state_difference": (
            float(
                np.max(
                    np.abs(reference["states"] - optimized["states"])
                )
            )
            if reference["states"].shape == optimized["states"].shape
            else None
        ),
        "max_abs_command_difference": (
            float(
                np.max(
                    np.abs(reference["commands"] - optimized["commands"])
                )
            )
            if reference["commands"].shape == optimized["commands"].shape
            else None
        ),
        "reference_mean_ms": (
            1000.0 * reference_record["mean_iteration_wall_s"]
        ),
        "optimized_mean_ms": (
            1000.0 * optimized_record["mean_iteration_wall_s"]
        ),
        "reference_p95_ms": (
            1000.0 * reference_record["p95_iteration_wall_s"]
        ),
        "optimized_p95_ms": (
            1000.0 * optimized_record["p95_iteration_wall_s"]
        ),
    }
    result["mean_speedup"] = (
        result["reference_mean_ms"] / result["optimized_mean_ms"]
    )
    result["p95_speedup"] = (
        result["reference_p95_ms"] / result["optimized_p95_ms"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
