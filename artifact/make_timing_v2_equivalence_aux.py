"""Build compact parity certificates for timing-instrumented baselines."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "validation" / "timing_v2"
COMPARISON = ROOT / "baselines" / "comparison-harness" / "results" / "timing_v2"


def first_record(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    if len(records) != 1:
        raise RuntimeError(f"expected one smoke record in {path}")
    return records[0]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def nhorca_certificate() -> None:
    hook_path = COMPARISON / "smoke_nhorca_hook.json"
    legacy_path = COMPARISON / "smoke_nhorca_legacy.json"
    hook = first_record(hook_path)
    legacy = first_record(legacy_path)
    center_error = float(
        np.max(
            np.abs(
                np.asarray(hook["final_centers"], dtype=float)
                - np.asarray(legacy["final_centers"], dtype=float)
            )
        )
    )
    heading_error = float(
        np.max(
            np.abs(
                np.asarray(hook["final_headings"], dtype=float)
                - np.asarray(legacy["final_headings"], dtype=float)
            )
        )
    )
    fingerprint_equal = (
        hook["scenario_fingerprint"] == legacy["scenario_fingerprint"]
    )
    digest_equal = hook["final_state_sha256"] == legacy["final_state_sha256"]
    outcome_equal = all(
        hook[key] == legacy[key]
        for key in (
            "safe",
            "success",
            "arrival_success",
            "ever_arrival_success",
            "steps",
        )
    )
    passed = bool(
        fingerprint_equal
        and digest_equal
        and outcome_equal
        and center_error == 0.0
        and heading_error == 0.0
    )
    write_json(
        COMPARISON / "timing_v2_equivalence_nhorca.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference": str(legacy_path.relative_to(ROOT)).replace("\\", "/"),
            "candidate": str(hook_path.relative_to(ROOT)).replace("\\", "/"),
            "scenario_fingerprint_equal": fingerprint_equal,
            "final_state_sha256_equal": digest_equal,
            "maximum_center_error": center_error,
            "maximum_heading_error": heading_error,
            "outcome_fields_equal": outcome_equal,
            "passed": passed,
        },
    )
    if not passed:
        raise RuntimeError("NH-ORCA timing-hook parity failed")


def impc_certificate() -> None:
    persistent_path = VALIDATION / "smoke_impc_rect15.json"
    reference_path = VALIDATION / "smoke_impc_rect15_reference.json"
    persistent = first_record(persistent_path)
    reference = first_record(reference_path)
    position_error = float(
        np.max(
            np.abs(
                np.asarray(persistent["final_positions_m"], dtype=float)
                - np.asarray(reference["final_positions_m"], dtype=float)
            )
        )
    )
    fingerprint_equal = (
        persistent["scenario_fingerprint"]
        == reference["scenario_fingerprint"]
    )
    statuses_equal = (
        persistent["final_solve_status_counts"]
        == reference["final_solve_status_counts"]
        and persistent["final_solve_backend_counts"]
        == reference["final_solve_backend_counts"]
        and persistent["all_solves_accepted"]
        == reference["all_solves_accepted"]
    )
    tolerance = 1.0e-6
    passed = bool(
        fingerprint_equal
        and statuses_equal
        and position_error <= tolerance
    )
    write_json(
        VALIDATION / "timing_v2_equivalence_impc_persistent.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
            "candidate": str(persistent_path.relative_to(ROOT)).replace("\\", "/"),
            "scenario_fingerprint_equal": fingerprint_equal,
            "solver_status_and_acceptance_equal": statuses_equal,
            "maximum_position_error_m": position_error,
            "tolerance_m": tolerance,
            "passed": passed,
        },
    )
    if not passed:
        raise RuntimeError("IMPC persistent-workspace parity failed")


def gcbfplus_certificate() -> None:
    cpu_path = COMPARISON / "timing_v2_gcbfplus_cpu_rect15.json"
    gpu_probe_path = COMPARISON / "smoke_gcbfplus_gpu.json"
    cpu_payload = json.loads(cpu_path.read_text(encoding="utf-8"))
    records = cpu_payload["records"]
    if len(records) != 12:
        raise RuntimeError("expected 12 official GCBF+ CPU timing records")
    tolerance = {record["ego_action_parity_tolerance"] for record in records}
    if len(tolerance) != 1:
        raise RuntimeError("GCBF+ CPU parity tolerance changed across cases")
    maximum_error = max(
        float(record["ego_action_max_abs_error"]) for record in records
    )
    violations = sum(
        int(record["ego_action_parity_violations"]) for record in records
    )
    gpu_probe = first_record(gpu_probe_path)
    gpu_error = float(gpu_probe["ego_action_max_abs_error"])
    passed = bool(violations == 0 and maximum_error <= next(iter(tolerance)))
    write_json(
        COMPARISON / "timing_v2_equivalence_gcbfplus.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_case_count": len(records),
            "cpu_ego_action_tolerance": next(iter(tolerance)),
            "cpu_maximum_ego_action_error": maximum_error,
            "cpu_parity_violations": violations,
            "cpu_induced_ego_timing_accepted": passed,
            "tolerance_rationale": (
                "The induced graph preserves the one-hop messages, while its "
                "smaller padded shape changes float32 segment-reduction order. "
                "The accepted action tolerance maps to at most 0.001 rad/s "
                "under the physical yaw scaling."
            ),
            "gpu_probe_maximum_ego_action_error": gpu_error,
            "gpu_induced_ego_timing_accepted": False,
            "gpu_official_mode": (
                "synchronized full-graph batch only; no deployment "
                "critical-path claim"
            ),
            "passed": passed,
        },
    )
    if not passed:
        raise RuntimeError("GCBF+ CPU induced-ego parity failed")


def main() -> None:
    nhorca_certificate()
    impc_certificate()
    gcbfplus_certificate()
    print("wrote NH-ORCA, IMPC, and GCBF+ parity certificates")


if __name__ == "__main__":
    main()
