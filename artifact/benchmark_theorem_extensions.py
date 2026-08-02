"""Validate CLEAR's new theorem audits and unicycle realization.

The script produces three artifacts in one JSON payload:

1. an exact straight-bridge benchmark inside the finite-exit theorem domain;
2. nearby guide/curvature perturbations that deliberately leave that domain;
3. a look-ahead-point unicycle evaluation on Swap and StraightBridge.

An optional main-suite audit reruns the N=20 benchmark to measure how often the
state-wise static-equilibrium certificate and the exact straight-bridge domain
occur in the existing task families.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from math import ceil, cos, pi, sin
from pathlib import Path

import numpy as np

from clear_nav.config import ControllerConfig, Protocol
from clear_nav.controller import CLEARController
from clear_nav.geometry import Arena, Circle, Rectangle
from clear_nav.scenarios import Scenario, make_scenario
from clear_nav.simulator import (
    _minimum_obstacle_clearance,
    _swept_pair_distance,
    simulate,
)
from clear_nav.theorem_audit import (
    audit_straight_bridge_tokens,
    static_certificate_holds,
)
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)

Array = np.ndarray


def report_progress(label: str, completed: int, total: int) -> None:
    """Emit sparse progress updates for the long 20-seed audits."""
    if completed == 1 or completed == total or completed % 5 == 0:
        print(f"[{label}] {completed}/{total}", flush=True)


def clear_config() -> ControllerConfig:
    return ControllerConfig(
        boundary_progress_aligned=True,
        cluster_escape_gain=0.55,
        cluster_escape_hysteresis=True,
        tangent_band=0.08,
        terminal_capture_hysteresis=True,
    )


def make_straight_bridge(
    physical_protocol: Protocol,
    n_robots: int,
    seed: int,
    *,
    design_protocol: Protocol | None = None,
    guide_angle_deg: float = 0.0,
    curved_radius: float | None = None,
) -> Scenario:
    """Construct a connected component between two parallel boundaries."""
    if n_robots < 2:
        raise ValueError("straight bridge needs at least two robots")
    design = design_protocol or physical_protocol
    rng = np.random.default_rng(seed)
    pair_gap = float(rng.uniform(0.006, 0.030))
    spacing = design.pair_clearance + pair_gap
    increments = spacing + rng.uniform(
        -0.15 * pair_gap,
        0.15 * pair_gap,
        size=n_robots - 1,
    )
    x = np.concatenate(([0.0], np.cumsum(increments)))
    x -= float(np.mean(x))
    x -= 2.4
    starts = np.stack((x, np.zeros(n_robots)), axis=1)

    angle = guide_angle_deg * pi / 180.0
    goal_shift = 4.0 * np.array([cos(angle), sin(angle)])
    goals = starts + goal_shift
    boundary_gap = float(rng.uniform(0.012, 0.035))
    half_gap = design.robot_clearance + boundary_gap

    if curved_radius is None:
        thickness = 0.60
        obstacles = (
            Rectangle(
                np.array([0.0, half_gap + 0.5 * thickness]),
                np.array([15.0, thickness]),
            ),
            Rectangle(
                np.array([0.0, -half_gap - 0.5 * thickness]),
                np.array([15.0, thickness]),
            ),
        )
        family = "straight_bridge"
    else:
        obstacles = (
            Circle(
                np.array([0.0, half_gap + curved_radius]),
                curved_radius,
            ),
            Circle(
                np.array([0.0, -half_gap - curved_radius]),
                curved_radius,
            ),
        )
        family = f"curved_bridge_r{curved_radius:g}"
    return Scenario(
        family=family,
        n_robots=n_robots,
        seed=seed,
        starts=starts,
        goals=goals,
        arena=Arena(physical_protocol.half_width, obstacles),
        protocol=physical_protocol,
    )


def _minimum_pair(positions: Array) -> float:
    if len(positions) < 2:
        return float("inf")
    left, right = np.triu_indices(len(positions), k=1)
    return float(
        np.min(np.linalg.norm(positions[left] - positions[right], axis=1))
    )


def run_bridge_event(
    scenario: Scenario,
    *,
    passage_length: float = 1.5,
) -> dict:
    """Run until longitudinal exit and audit every theorem-domain step."""
    controller = CLEARController(scenario.protocol, clear_config())
    controller.reset()
    positions = np.asarray(scenario.starts, dtype=float).copy()
    direction = np.array([1.0, 0.0])
    initial_progress = float(np.mean(positions @ direction))
    target_progress = initial_progress + passage_length

    token_created = False
    token_flips = 0
    prior_token: Array | None = None
    domain_steps = 0
    bound_violations = 0
    projection_violations = 0
    maximum_projection_error = 0.0
    minimum_step_slack = float("inf")
    minimum_rate_bound = float("inf")
    static_steps = 0
    static_violations = 0
    failure_reasons: Counter[str] = Counter()
    min_pair = _minimum_pair(positions)
    min_obstacle = _minimum_obstacle_clearance(scenario, positions)
    exit_reason = "timeout"
    exit_step = scenario.protocol.steps

    for step in range(scenario.protocol.steps):
        tokens_before = controller.cluster_tokens
        audit = controller.command(
            positions,
            scenario.goals,
            scenario.arena,
        )
        antecedent, conclusion = static_certificate_holds(audit)
        if antecedent:
            static_steps += 1
            static_violations += int(not conclusion)

        certificates = audit_straight_bridge_tokens(
            controller,
            positions,
            scenario.arena,
            audit,
        )
        certificate = next(
            (
                value
                for value in certificates
                if value.component == tuple(range(scenario.n_robots))
            ),
            None,
        )
        if certificate is not None:
            if certificate.component not in tokens_before:
                token_created = True
            if prior_token is not None and float(
                prior_token @ certificate.direction
            ) < 1.0 - 1.0e-8:
                token_flips += 1
            prior_token = certificate.direction.copy()
            for reason in certificate.failed_conditions:
                failure_reasons[reason] += 1

        velocity = audit.executed_command
        next_positions = positions + scenario.protocol.dt * velocity
        min_pair = min(
            min_pair,
            _swept_pair_distance(
                positions,
                velocity,
                scenario.protocol.dt,
            ),
        )
        min_obstacle = min(
            min_obstacle,
            _minimum_obstacle_clearance(
                scenario,
                positions + 0.5 * scenario.protocol.dt * velocity,
            ),
            _minimum_obstacle_clearance(scenario, next_positions),
        )

        if certificate is not None and certificate.domain_satisfied:
            actual_increment = float(
                np.mean(
                    (
                        next_positions[
                            np.asarray(certificate.component, dtype=int)
                        ]
                        - positions[
                            np.asarray(certificate.component, dtype=int)
                        ]
                    )
                    @ certificate.direction
                )
            )
            required_increment = (
                scenario.protocol.dt * certificate.progress_rate_bound
            )
            slack = actual_increment - required_increment
            domain_steps += 1
            bound_violations += int(slack < -1.0e-10)
            projection_violations += int(
                certificate.projection_axis_error
                > 10.0 * controller.config.projection_tolerance
            )
            maximum_projection_error = max(
                maximum_projection_error,
                certificate.projection_axis_error,
            )
            minimum_step_slack = min(minimum_step_slack, slack)
            minimum_rate_bound = min(
                minimum_rate_bound,
                certificate.progress_rate_bound,
            )

        positions = next_positions
        if float(np.mean(positions @ direction)) >= target_progress:
            exit_reason = "longitudinal_exit"
            exit_step = step + 1
            break

    if not np.isfinite(minimum_step_slack):
        minimum_step_slack = float("nan")
    predicted_bound = (
        int(ceil(passage_length / (scenario.protocol.dt * minimum_rate_bound)))
        if np.isfinite(minimum_rate_bound) and minimum_rate_bound > 0.0
        else None
    )
    return {
        "family": scenario.family,
        "n_robots": scenario.n_robots,
        "seed": scenario.seed,
        "exit_reason": exit_reason,
        "exit_steps": exit_step,
        "exit_time_s": exit_step * scenario.protocol.dt,
        "predicted_exit_bound_steps": predicted_bound,
        "token_created": token_created,
        "token_flips": token_flips,
        "domain_steps": domain_steps,
        "bound_violations": bound_violations,
        "projection_axis_violations": projection_violations,
        "maximum_projection_axis_error": maximum_projection_error,
        "minimum_step_slack_m": minimum_step_slack,
        "minimum_pair_distance_m": min_pair,
        "minimum_obstacle_clearance_m": min_obstacle,
        "static_certificate_steps": static_steps,
        "static_certificate_violations": static_violations,
        "failed_domain_conditions": dict(sorted(failure_reasons.items())),
    }


def aggregate_bridge(records: list[dict]) -> dict:
    in_domain = [record for record in records if record["domain_steps"] > 0]
    return {
        "runs": len(records),
        "longitudinal_exits": sum(
            record["exit_reason"] == "longitudinal_exit"
            for record in records
        ),
        "runs_with_domain_steps": len(in_domain),
        "theorem_domain_steps": sum(
            record["domain_steps"] for record in records
        ),
        "bound_violations": sum(
            record["bound_violations"] for record in records
        ),
        "projection_axis_violations": sum(
            record["projection_axis_violations"] for record in records
        ),
        "token_flips": sum(record["token_flips"] for record in records),
        "static_certificate_violations": sum(
            record["static_certificate_violations"] for record in records
        ),
        "maximum_projection_axis_error": max(
            (
                record["maximum_projection_axis_error"]
                for record in records
            ),
            default=float("nan"),
        ),
        "worst_exit_to_bound_ratio": max(
            (
                record["exit_steps"]
                / record["predicted_exit_bound_steps"]
                for record in in_domain
                if record["predicted_exit_bound_steps"]
            ),
            default=float("nan"),
        ),
        "minimum_pair_distance_m": min(
            record["minimum_pair_distance_m"] for record in records
        ),
        "minimum_obstacle_clearance_m": min(
            record["minimum_obstacle_clearance_m"] for record in records
        ),
    }


def aggregate_unicycle(records: list[dict]) -> dict:
    successful = [
        record for record in records if record["mission_success"]
    ]
    yaw_scales = [
        record["minimum_yaw_scale"]
        for record in records
        if record.get("minimum_yaw_scale") is not None
    ]
    yaw_retentions = [
        record["minimum_yaw_retention"]
        for record in records
        if record.get("minimum_yaw_retention") is not None
    ]
    return {
        "runs": len(records),
        "mission_successes": len(successful),
        "mean_successful_makespan_s": (
            float(np.mean([record["makespan_s"] for record in successful]))
            if successful
            else float("nan")
        ),
        "minimum_physical_pair_distance_m": min(
            record["minimum_physical_pair_distance_m"]
            for record in records
        ),
        "minimum_transferred_pair_lower_bound_m": min(
            record["minimum_transferred_pair_lower_bound_m"]
            for record in records
        ),
        "minimum_physical_obstacle_clearance_m": min(
            record["minimum_physical_obstacle_clearance_m"]
            for record in records
        ),
        "minimum_virtual_obstacle_clearance_m": min(
            record["minimum_virtual_obstacle_clearance_m"]
            for record in records
        ),
        "maximum_yaw_rate_rps": max(
            record["maximum_yaw_rate_rps"] for record in records
        ),
        "minimum_yaw_scale": (
            min(yaw_scales) if yaw_scales else None
        ),
        "minimum_yaw_retention": (
            min(yaw_retentions) if yaw_retentions else None
        ),
        "nonconverged_steps": sum(
            record["nonconverged_steps"] for record in records
        ),
        "feasibility_restoration_steps": sum(
            record.get("feasibility_restoration_steps", 0)
            for record in records
        ),
        "projection_iteration_cap_calls": sum(
            record.get("projection_iteration_cap_calls", 0)
            for record in records
        ),
        "static_certificate_violations": sum(
            record["static_certificate_violations"] for record in records
        ),
        "certified_bridge_steps": sum(
            record.get("certified_bridge_steps", 0)
            for record in records
        ),
        "certified_bridge_projection_calls": sum(
            record.get("certified_bridge_projection_calls", 0)
            for record in records
        ),
        "certified_bridge_row_rejections": sum(
            record.get("certified_bridge_row_rejections", 0)
            for record in records
        ),
        "certified_bridge_progress_violations": sum(
            record.get("certified_bridge_progress_violations", 0)
            for record in records
        ),
        "minimum_certified_bridge_progress_residual": min(
            (
                record["minimum_certified_bridge_progress_residual"]
                for record in records
                if record.get(
                    "minimum_certified_bridge_progress_residual"
                )
                is not None
            ),
            default=float("nan"),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("bridge", "bridge-scale", "unicycle", "main-audit"),
        default=("bridge", "unicycle"),
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/theorem_extension_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict = {
        "controller": asdict(clear_config()),
        "sections": list(args.sections),
    }

    if "bridge" in args.sections:
        bridge_protocol = Protocol(horizon=8.0, dt=0.03)
        exact_records = [
            run_bridge_event(
                make_straight_bridge(
                    bridge_protocol,
                    count,
                    seed,
                )
            )
            for count in (2, 4, 8)
            for seed in range(args.seeds)
        ]
        guide_records = [
            {
                **run_bridge_event(
                    make_straight_bridge(
                        bridge_protocol,
                        4,
                        seed,
                        guide_angle_deg=angle,
                    )
                ),
                "guide_angle_deg": angle,
            }
            for angle in (2.0, 5.0, 10.0)
            for seed in range(min(args.seeds, 10))
        ]
        curved_records = [
            {
                **run_bridge_event(
                    make_straight_bridge(
                        bridge_protocol,
                        4,
                        seed,
                        curved_radius=radius,
                    )
                ),
                "curved_radius_m": radius,
            }
            for radius in (200.0, 50.0)
            for seed in range(min(args.seeds, 10))
        ]
        payload["straight_bridge"] = {
            "protocol": asdict(bridge_protocol),
            "exact_domain": {
                "summary": aggregate_bridge(exact_records),
                "records": exact_records,
            },
            "guide_perturbation": {
                "summary": aggregate_bridge(guide_records),
                "records": guide_records,
            },
            "curvature_perturbation": {
                "summary": aggregate_bridge(curved_records),
                "records": curved_records,
            },
        }

    if "bridge-scale" in args.sections:
        exact_protocol = Protocol(horizon=8.0, dt=0.03)
        unicycle_protocol = Protocol(horizon=10.0, dt=0.03)
        unicycle_config = UnicycleConfig(
            lookahead=0.05,
            yaw_rate_limit=np.pi / 2.0,
            inner_substeps=3,
        )
        unicycle_design = inflated_unicycle_protocol(
            unicycle_protocol,
            unicycle_config.lookahead,
        )
        exact_by_size: dict[str, dict] = {}
        unicycle_by_size: dict[str, dict] = {}
        for count in (8, 16):
            exact_records = []
            native_records = []
            for seed in range(args.seeds):
                exact_records.append(
                    run_bridge_event(
                        make_straight_bridge(exact_protocol, count, seed)
                    )
                )
                native_records.append(
                    simulate_unicycle(
                        make_straight_bridge(
                            unicycle_protocol,
                            count,
                            seed,
                            design_protocol=unicycle_design,
                        ),
                        clear_config(),
                        unicycle_config,
                        initial_headings=np.zeros(count),
                    ).summary()
                )
                report_progress(
                    f"bridge-scale/n{count}",
                    seed + 1,
                    args.seeds,
                )
            exact_by_size[str(count)] = {
                "summary": aggregate_bridge(exact_records),
                "records": exact_records,
            }
            unicycle_by_size[str(count)] = {
                "summary": aggregate_unicycle(native_records),
                "records": native_records,
            }
        payload["straight_bridge_scale"] = {
            "exact_protocol": asdict(exact_protocol),
            "unicycle_protocol": asdict(unicycle_protocol),
            "unicycle_config": asdict(unicycle_config),
            "exact_by_size": exact_by_size,
            "unicycle_by_size": unicycle_by_size,
        }

    if "unicycle" in args.sections:
        unicycle_config = UnicycleConfig(
            lookahead=0.05,
            yaw_rate_limit=np.pi / 2.0,
            inner_substeps=3,
        )
        physical_protocol = Protocol(horizon=60.0, dt=0.03)
        design_protocol = inflated_unicycle_protocol(
            physical_protocol,
            unicycle_config.lookahead,
        )
        family_records: dict[str, list[dict]] = {}
        initial_axis = np.array([1.0, 0.0])
        for family in ("free", "swap", "circ15", "rect15"):
            records = []
            for seed in range(args.seeds):
                design_scenario = make_scenario(
                    family,
                    20,
                    seed,
                    design_protocol,
                )
                # Treat the generated safe sites as initial look-ahead points.
                # The physical centers are shifted back by ell along the
                # common initial heading, preserving the transfer certificate.
                scenario = Scenario(
                    family=family,
                    n_robots=20,
                    seed=seed,
                    starts=(
                        design_scenario.starts
                        - unicycle_config.lookahead * initial_axis
                    ),
                    goals=design_scenario.goals,
                    arena=design_scenario.arena,
                    protocol=physical_protocol,
                )
                records.append(
                    simulate_unicycle(
                        scenario,
                        clear_config(),
                        unicycle_config,
                        initial_headings=np.zeros(20),
                        guidance_mode="cost",
                    ).summary()
                )
                report_progress(
                    f"unicycle/{family}",
                    seed + 1,
                    args.seeds,
                )
            family_records[family] = records
        bridge_protocol = Protocol(horizon=10.0, dt=0.03)
        bridge_design = inflated_unicycle_protocol(
            bridge_protocol,
            unicycle_config.lookahead,
        )
        straight_records = []
        for seed in range(args.seeds):
            scenario = make_straight_bridge(
                bridge_protocol,
                4,
                seed,
                design_protocol=bridge_design,
            )
            straight_records.append(
                simulate_unicycle(
                    scenario,
                    clear_config(),
                    unicycle_config,
                    initial_headings=np.zeros(4),
                ).summary()
            )
            report_progress(
                "unicycle/straight_bridge",
                seed + 1,
                args.seeds,
            )
        payload["unicycle"] = {
            "config": asdict(unicycle_config),
            "n20_by_family": {
                family: {
                    "summary": aggregate_unicycle(records),
                    "records": records,
                }
                for family, records in family_records.items()
            },
            "straight_bridge_n4": {
                "summary": aggregate_unicycle(straight_records),
                "records": straight_records,
            },
        }

    if "main-audit" in args.sections:
        protocol = Protocol(horizon=60.0, dt=0.03)
        controller = CLEARController(protocol, clear_config())
        records = []
        for family in ("free", "swap", "circ15", "rect15"):
            for seed in range(args.seeds):
                rollout = simulate(
                    make_scenario(family, 20, seed, protocol),
                    controller,
                    guidance_mode="cost",
                )
                records.append(rollout.summary())
                report_progress(
                    f"main-audit/{family}",
                    seed + 1,
                    args.seeds,
                )
        payload["main_n20_audit"] = {
            "protocol": asdict(protocol),
            "runs": len(records),
            "mission_successes": sum(
                record["mission_success"] for record in records
            ),
            "global_tangent_audited_steps": sum(
                record["global_tangent_audited_steps"]
                for record in records
            ),
            "static_certificate_steps": sum(
                record["static_certificate_steps"] for record in records
            ),
            "static_certificate_violations": sum(
                record["static_certificate_violations"]
                for record in records
            ),
            "minimum_static_certified_speed_mps": min(
                (
                    record["minimum_static_certified_speed_mps"]
                    for record in records
                    if np.isfinite(
                        record["minimum_static_certified_speed_mps"]
                    )
                ),
                default=float("nan"),
            ),
            "straight_bridge_events": sum(
                record["straight_bridge_events"] for record in records
            ),
            "straight_bridge_domain_steps": sum(
                record["straight_bridge_domain_steps"]
                for record in records
            ),
            "straight_bridge_bound_violations": sum(
                record["straight_bridge_bound_violations"]
                for record in records
            ),
            "records": records,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
