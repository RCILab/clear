"""Retain full-resolution traces for selected unicycle CLEAR cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clear_nav import Protocol, VanillaCBFController
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)
from run_unicycle import controller_config, make_unicycle_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="family:robot_count:seed",
    )
    parser.add_argument(
        "--variant",
        choices=("clear", "component-free", "vanilla-cbf-qp"),
        default="clear",
    )
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--lookahead", type=float, default=0.05)
    parser.add_argument("--yaw-rate-limit", type=float, default=np.pi / 2.0)
    parser.add_argument("--inner-substeps", type=int, default=3)
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
        "--actuation-mode",
        choices=(
            "native-cbf",
            "common-scale",
            "component-scale",
            "robotarium-clip",
        ),
        default="native-cbf",
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
    parser.add_argument("--handedness", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--terminal-capture-radius", type=float, default=0.22)
    parser.add_argument(
        "--terminal-open-capture-radius",
        type=float,
        default=0.60,
    )
    parser.add_argument("--terminal-release-radius", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_case(specification: str) -> tuple[str, int, int]:
    family, count, seed = specification.split(":")
    return family, int(count), int(seed)


def main() -> None:
    args = parse_args()
    physical = Protocol(horizon=args.horizon, dt=args.dt)
    unicycle = UnicycleConfig(
        lookahead=args.lookahead,
        yaw_rate_limit=args.yaw_rate_limit,
        inner_substeps=args.inner_substeps,
        hierarchical_progress=args.hierarchical_progress,
        hqp_progress_retention=args.hqp_progress_retention,
        actuation_mode=args.actuation_mode,
    )
    config = controller_config(args)
    controller = None
    if args.variant == "vanilla-cbf-qp":
        controller = VanillaCBFController(
            inflated_unicycle_protocol(physical, unicycle.lookahead),
            config,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    for specification in args.cases:
        family, count, seed = parse_case(specification)
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
            record_stride=1,
            guidance_mode=args.guidance,
            controller=controller,
        )
        report = rollout.summary()
        report.update(
            {
                "variant": args.variant,
                "guidance": args.guidance,
                "boundary_mode": args.boundary_mode,
                "actuation_mode": args.actuation_mode,
                "horizon_s": args.horizon,
                "terminal_capture_radius_m": args.terminal_capture_radius,
                "terminal_open_capture_radius_m": (
                    args.terminal_open_capture_radius
                ),
                "terminal_release_radius_m": args.terminal_release_radius,
                "hierarchical_progress": args.hierarchical_progress,
            }
        )
        stem = f"{family}_n{count}_seed{seed}"
        np.savez_compressed(
            args.output_dir / f"{stem}_trace.npz",
            times=rollout.times,
            positions=rollout.trajectory,
            headings=rollout.headings,
            starts=scenario.starts,
            goals=scenario.goals,
            first_arrival_times=rollout.first_arrival_times,
            final_goal_distances=rollout.final_goal_distances,
        )
        (args.output_dir / f"{stem}_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    all_reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output_dir.glob("*_report.json"))
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps({"reports": all_reports}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"wrote {len(reports)} traces ({len(all_reports)} retained) "
        f"to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
