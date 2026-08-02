"""Run the CLEAR evaluation matrix with look-ahead unicycle robots."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from clear_nav import (
    ControllerConfig,
    Protocol,
    REPORTED_CLEAR_CONTROLLER,
    Scenario,
    VanillaCBFController,
    make_scenario,
)
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="+",
        default=["free", "swap"],
        choices=["free", "circ15", "rect15", "swap"],
    )
    parser.add_argument("--robots", nargs="+", type=int, default=[20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--lookahead", type=float, default=0.05)
    parser.add_argument("--yaw-rate-limit", type=float, default=np.pi / 2.0)
    parser.add_argument("--inner-substeps", type=int, default=3)
    parser.add_argument(
        "--native-projection-max-sweeps",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--native-projection-backend",
        choices=("osqp", "hildreth"),
        default="osqp",
    )
    parser.add_argument(
        "--native-projection-tolerance",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--native-projection-extension-sweeps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--minimum-restoration-retention",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--hierarchical-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hqp-progress-retention",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--hqp-regularization",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--certified-bridge-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bridge-progress-retention",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--actuation-mode",
        choices=(
            "native-cbf",
            "common-scale",
            "component-scale",
            "robotarium-clip",
        ),
        default="native-cbf",
    )
    parser.add_argument("--record-stride", type=int, default=4)
    parser.add_argument("--handedness", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--variant",
        choices=("clear", "component-free", "vanilla-cbf-qp"),
        default="clear",
    )
    parser.add_argument(
        "--guidance",
        choices=("direct", "waypoint", "cost"),
        default="cost",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("fixed", "progress"),
        default="progress",
    )
    parser.add_argument("--terminal-capture-radius", type=float, default=0.22)
    parser.add_argument(
        "--terminal-open-capture-radius",
        type=float,
        default=0.60,
    )
    parser.add_argument("--terminal-release-radius", type=float, default=0.80)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def controller_config(args: argparse.Namespace) -> ControllerConfig:
    structured = args.variant == "clear"
    return replace(
        REPORTED_CLEAR_CONTROLLER,
        handedness=args.handedness,
        boundary_progress_aligned=args.boundary_mode == "progress",
        cluster_escape_gain=0.55 if structured else 0.0,
        cluster_escape_hysteresis=structured,
        # The component ablation changes only the component construction.
        # In particular, it retains CLEAR's structural tangent bundle so that
        # the paired comparison is not confounded by a wider projection band.
        tangent_band=0.08,
        terminal_capture_hysteresis=True,
        terminal_capture_radius=args.terminal_capture_radius,
        terminal_open_capture_radius=args.terminal_open_capture_radius,
        terminal_release_radius=args.terminal_release_radius,
    )


def make_unicycle_scenario(
    family: str,
    count: int,
    seed: int,
    physical: Protocol,
    unicycle: UnicycleConfig,
) -> Scenario:
    """Generate safe virtual sites, then map them to physical centers."""
    design = inflated_unicycle_protocol(physical, unicycle.lookahead)
    generated = make_scenario(family, count, seed, design)
    initial_axis = np.array([1.0, 0.0])
    return Scenario(
        family=family,
        n_robots=count,
        seed=seed,
        starts=generated.starts - unicycle.lookahead * initial_axis,
        goals=generated.goals,
        arena=generated.arena,
        protocol=physical,
    )


def main() -> None:
    args = parse_args()
    physical = Protocol(horizon=args.horizon, dt=args.dt)
    unicycle = UnicycleConfig(
        lookahead=args.lookahead,
        yaw_rate_limit=args.yaw_rate_limit,
        inner_substeps=args.inner_substeps,
        projection_backend=args.native_projection_backend,
        projection_tolerance=args.native_projection_tolerance,
        projection_max_sweeps=args.native_projection_max_sweeps,
        projection_extension_sweeps=(
            args.native_projection_extension_sweeps
        ),
        minimum_restoration_retention=(
            args.minimum_restoration_retention
        ),
        hierarchical_progress=args.hierarchical_progress,
        hqp_progress_retention=args.hqp_progress_retention,
        hqp_regularization=args.hqp_regularization,
        certified_bridge_progress=args.certified_bridge_progress,
        bridge_progress_retention=args.bridge_progress_retention,
        actuation_mode=args.actuation_mode,
    )
    config = controller_config(args)
    controller = None
    if args.variant == "vanilla-cbf-qp":
        controller = VanillaCBFController(
            inflated_unicycle_protocol(physical, unicycle.lookahead),
            config,
        )
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results") / f"unicycle_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": (
            "Vanilla CBF-QP"
            if args.variant == "vanilla-cbf-qp"
            else "CLEAR"
        ),
        "dynamics": "bounded-lookahead-unicycle",
        "variant": args.variant,
        "guidance": args.guidance,
        "boundary_mode": args.boundary_mode,
        "physical_protocol": asdict(physical),
        "control_protocol": asdict(
            inflated_unicycle_protocol(physical, unicycle.lookahead)
        ),
        "unicycle": asdict(unicycle),
        "native_projection": {
            "coordinates": "[v, lookahead*omega]",
            "hierarchy": (
                "maximize complete CLEAR-task progress, then track the CLEAR "
                "nominal command while retaining the first-level optimum"
                if unicycle.hierarchical_progress
                else "disabled"
            ),
            "feasibility_restoration": (
                "certified rigid witness when available; exact actuator "
                "contraction when it also restores primal feasibility; "
                "otherwise common contraction toward zero; an optional "
                "hierarchy may return its feasible first-level point"
            ),
            "status_semantics": {
                "solver_converged": (
                    "optimizer reached a solved termination status"
                ),
                "command_feasible": (
                    "executed command passes every CBF and input-bound row"
                ),
            },
        },
        "controller": asdict(config),
    }
    records: list[dict] = []
    total = len(args.families) * len(args.robots) * len(args.seeds)
    completed = 0
    for family in args.families:
        for count in args.robots:
            for seed in args.seeds:
                scenario = make_unicycle_scenario(
                    family,
                    count,
                    seed,
                    physical,
                    unicycle,
                )
                rollout = simulate_unicycle(
                    scenario,
                    config,
                    unicycle,
                    initial_headings=np.zeros(count),
                    record_stride=args.record_stride,
                    guidance_mode=args.guidance,
                    controller=controller,
                )
                record = rollout.summary()
                records.append(record)
                completed += 1
                output.write_text(
                    json.dumps(
                        {**metadata, "records": records},
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                print(json.dumps(record, sort_keys=True), flush=True)
                print(f"[progress] {completed}/{total}", flush=True)

    output.write_text(
        json.dumps(
            {**metadata, "records": records},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
