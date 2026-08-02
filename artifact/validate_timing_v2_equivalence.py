"""Compare global and component CLEAR/Vanilla rollout endpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-output", type=Path, required=True)
    parser.add_argument("--component-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--tolerance-rationale",
        default=None,
        help="Recorded justification when a tolerance above 1e-6 is used.",
    )
    return parser.parse_args()


def records(path: Path) -> dict[tuple[str, str, int, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (
            record["variant"],
            record["family"],
            int(record["n_robots"]),
            int(record["seed"]),
        ): record
        for record in payload["records"]
    }


def main() -> None:
    args = parse_args()
    if args.tolerance > 1.0e-6 and not args.tolerance_rationale:
        raise ValueError(
            "--tolerance-rationale is required when tolerance exceeds 1e-6"
        )
    reference = records(args.global_output)
    candidate = records(args.component_output)
    if reference.keys() != candidate.keys():
        raise RuntimeError("global/component case matrices differ")
    comparisons = []
    maximum_state_error = 0.0
    for key in sorted(reference):
        left = reference[key]
        right = candidate[key]
        center_error = float(
            np.max(
                np.abs(
                    np.asarray(left["final_centers"])
                    - np.asarray(right["final_centers"])
                )
            )
        )
        heading_error = float(
            np.max(
                np.abs(
                    np.asarray(left["final_headings"])
                    - np.asarray(right["final_headings"])
                )
            )
        )
        maximum_state_error = max(
            maximum_state_error,
            center_error,
            heading_error,
        )
        status_equal = (
            left["nonconverged_steps"] == right["nonconverged_steps"]
            and left["feasibility_restoration_calls"]
            == right["feasibility_restoration_calls"]
        )
        comparisons.append(
            {
                "variant": key[0],
                "family": key[1],
                "n_robots": key[2],
                "seed": key[3],
                "scenario_fingerprint_equal": (
                    left["scenario_fingerprint"]
                    == right["scenario_fingerprint"]
                ),
                "maximum_center_error": center_error,
                "maximum_heading_error": heading_error,
                "solver_event_counts_equal": status_equal,
                "passed": bool(
                    center_error <= args.tolerance
                    and heading_error <= args.tolerance
                    and status_equal
                    and left["scenario_fingerprint"]
                    == right["scenario_fingerprint"]
                ),
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tolerance": args.tolerance,
        "tolerance_rationale": args.tolerance_rationale,
        "case_count": len(comparisons),
        "maximum_state_error": maximum_state_error,
        "passed": all(item["passed"] for item in comparisons),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise RuntimeError(
            f"equivalence failed; max error={maximum_state_error}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
