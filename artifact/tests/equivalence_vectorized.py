"""Step-stream parity audit for the vectorized CLEAR controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _reference_controller import ReferenceCLEARController
from clear_nav import (
    CLEARController,
    ControllerConfig,
    Protocol,
    make_scenario,
)
from clear_nav.guidance import GridPlanner


def clear_configuration() -> tuple[Protocol, ControllerConfig]:
    protocol = Protocol()
    config = ControllerConfig(
        handedness=1,
        boundary_progress_aligned=True,
        cluster_escape_gain=0.55,
        cluster_escape_hysteresis=True,
        tangent_band=0.08,
        terminal_capture_hysteresis=True,
    )
    return protocol, config


def audit_case(
    family: str,
    n_robots: int,
    seed: int,
    steps: int,
) -> dict:
    protocol, config = clear_configuration()
    reference = ReferenceCLEARController(protocol, config)
    vectorized = CLEARController(protocol, config)
    reference.reset()
    vectorized.reset()
    scenario = make_scenario(family, n_robots, seed, protocol)
    guidance = None
    if scenario.arena.obstacles:
        guidance = GridPlanner(
            scenario.arena,
            protocol,
        ).cost_field_plan(scenario.goals)
    positions = np.asarray(scenario.starts, dtype=float).copy()

    max_command_diff = 0.0
    max_structural_count_diff = 0
    max_field_diff = 0.0
    for _ in range(steps):
        control_goals = (
            guidance.targets(positions)
            if guidance is not None
            else scenario.goals
        )
        reference_audit = reference.command(
            positions,
            control_goals,
            scenario.arena,
        )
        vectorized_audit = vectorized.command(
            positions,
            control_goals,
            scenario.arena,
        )
        max_command_diff = max(
            max_command_diff,
            float(
                np.max(
                    np.abs(
                        reference_audit.executed_command
                        - vectorized_audit.executed_command
                    )
                )
            ),
        )
        for reference_field, vectorized_field in (
            (
                reference_audit.raw_circulation,
                vectorized_audit.raw_circulation,
            ),
            (
                reference_audit.cluster_circulation,
                vectorized_audit.cluster_circulation,
            ),
            (
                reference_audit.cone_circulation,
                vectorized_audit.cone_circulation,
            ),
            (
                reference_audit.nominal_command,
                vectorized_audit.nominal_command,
            ),
        ):
            max_field_diff = max(
                max_field_diff,
                float(np.max(np.abs(reference_field - vectorized_field))),
            )
        max_structural_count_diff = max(
            max_structural_count_diff,
            abs(
                reference_audit.active_pair_edges
                - vectorized_audit.active_pair_edges
            ),
            abs(
                reference_audit.active_boundary_edges
                - vectorized_audit.active_boundary_edges
            ),
            abs(
                reference_audit.active_cluster_escapes
                - vectorized_audit.active_cluster_escapes
            ),
            abs(
                reference_audit.tangent_constraints
                - vectorized_audit.tangent_constraints
            ),
            abs(reference_audit.cbf_sweeps - vectorized_audit.cbf_sweeps),
        )
        positions = (
            positions + protocol.dt * reference_audit.executed_command
        )

    return {
        "family": family,
        "n_robots": n_robots,
        "seed": seed,
        "steps": steps,
        "max_abs_command_diff": max_command_diff,
        "max_abs_intermediate_field_diff": max_field_diff,
        "max_structural_count_diff": max_structural_count_diff,
        "passed": (
            max_command_diff <= 1.0e-12
            and max_structural_count_diff == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [
        audit_case("swap", 20, 0, 400),
        audit_case("rect15", 20, 0, 400),
        audit_case("rect15", 40, 0, 300),
    ]
    for record in records:
        print(json.dumps(record, sort_keys=True))
    if not all(record["passed"] for record in records):
        raise SystemExit("vectorization parity gate failed")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"records": records}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
