"""Common-unicycle SMG benchmark for CLEAR and Vanilla CBF-QP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from clear_nav import (
    ControllerConfig,
    Protocol,
    SMGGeometry,
    VanillaCBFController,
    doorway_flow_metrics,
    inflated_unicycle_protocol,
    interference_delay_metrics,
    intersection_flow_metrics,
    make_smg_scenario,
    simulate_unicycle,
)
from clear_nav.controller import CLEARController
from clear_nav.scenarios import Scenario
from clear_nav.unicycle import UnicycleConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("clear", "vanilla-cbf-qp"),
        default=("clear", "vanilla-cbf-qp"),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("doorway", "intersection"),
        default=("doorway", "intersection"),
    )
    parser.add_argument("--robots", nargs="+", type=int, default=(8, 16))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument(
        "--doorway-widths",
        nargs="+",
        type=float,
        default=(0.8, 1.2),
    )
    parser.add_argument(
        "--intersection-width",
        type=float,
        default=2.4,
    )
    parser.add_argument("--wall-thickness", type=float, default=0.8)
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--lookahead", type=float, default=0.05)
    parser.add_argument("--yaw-rate-limit", type=float, default=np.pi / 2)
    parser.add_argument("--inner-substeps", type=int, default=3)
    parser.add_argument("--record-stride", type=int, default=1)
    parser.add_argument(
        "--isolated-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def clear_config() -> ControllerConfig:
    return ControllerConfig(
        boundary_progress_aligned=True,
        cluster_escape_gain=0.55,
        cluster_escape_hysteresis=True,
        tangent_band=0.08,
        terminal_capture_hysteresis=True,
    )


def make_controller(
    method: str,
    protocol: Protocol,
    unicycle: UnicycleConfig,
    config: ControllerConfig,
) -> CLEARController:
    control_protocol = inflated_unicycle_protocol(
        protocol,
        unicycle.lookahead,
    )
    if method == "clear":
        return CLEARController(control_protocol, config)
    return VanillaCBFController(control_protocol, config)


def run_one(
    method: str,
    scenario: Scenario,
    geometry: SMGGeometry,
    unicycle: UnicycleConfig,
    config: ControllerConfig,
    *,
    record_stride: int,
    isolated_reference: bool,
) -> dict:
    controller = make_controller(
        method,
        scenario.protocol,
        unicycle,
        config,
    )
    started = perf_counter()
    rollout = simulate_unicycle(
        scenario,
        config,
        unicycle,
        initial_headings=np.zeros(scenario.n_robots),
        record_stride=record_stride,
        guidance_mode="cost",
        controller=controller,
    )
    elapsed = perf_counter() - started
    record = rollout.summary()
    record.update(
        {
            "method": "CLEAR" if method == "clear" else "Vanilla CBF-QP",
            "controller_wall_time_s": elapsed,
            "mean_control_step_ms": (
                1.0e3
                * elapsed
                / scenario.protocol.steps
            ),
            "first_arrival_times_s": [
                None if not np.isfinite(value) else float(value)
                for value in rollout.first_arrival_times
            ],
            "final_positions_m": rollout.trajectory[-1].tolist(),
        }
    )
    if scenario.family.startswith("doorway"):
        record.update(
            doorway_flow_metrics(
                rollout,
                doorway_width=geometry.doorway_width,
                doorway_thickness=geometry.doorway_thickness,
            )
        )
    else:
        record.update(
            intersection_flow_metrics(
                rollout,
                corridor_width=geometry.intersection_corridor_width,
            )
        )

    if isolated_reference:
        solo_times = np.full(scenario.n_robots, np.inf)
        for index in range(scenario.n_robots):
            solo = Scenario(
                family=f"{scenario.family}_solo",
                n_robots=1,
                seed=scenario.seed,
                starts=scenario.starts[index : index + 1].copy(),
                goals=scenario.goals[index : index + 1].copy(),
                arena=scenario.arena,
                protocol=scenario.protocol,
            )
            solo_rollout = simulate_unicycle(
                solo,
                config,
                unicycle,
                initial_headings=np.zeros(1),
                record_stride=max(record_stride, 4),
                guidance_mode="cost",
                controller=make_controller(
                    method,
                    scenario.protocol,
                    unicycle,
                    config,
                ),
            )
            solo_times[index] = solo_rollout.first_arrival_times[0]
        record.update(
            interference_delay_metrics(
                rollout.first_arrival_times,
                solo_times,
            )
        )
        record["isolated_ttg_s"] = [
            None if not np.isfinite(value) else float(value)
            for value in solo_times
        ]
    return record


def main() -> None:
    args = parse_args()
    protocol = Protocol(horizon=args.horizon, dt=args.dt)
    unicycle = UnicycleConfig(
        lookahead=args.lookahead,
        yaw_rate_limit=args.yaw_rate_limit,
        inner_substeps=args.inner_substeps,
        projection_backend="osqp",
        projection_tolerance=1.0e-6,
        projection_max_sweeps=4000,
        certified_bridge_progress=True,
        actuation_mode="native-cbf",
    )
    config = clear_config()
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results") / f"smg_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "initial_heading_rad": 0.0,
        "protocol": asdict(protocol),
        "unicycle": asdict(unicycle),
        "controller": asdict(config),
        "metric_definitions": {
            "parallel_throughput": (
                "|S|/(t_end-t_start), with first resource entry and last "
                "post-resource exit"
            ),
            "smg_flow_rate": (
                "parallel throughput divided by physical gap width"
            ),
            "interference_delay": (
                "TTG_multi-TTG_solo using the same method, dynamics, map, "
                "start, and goal"
            ),
        },
        "records": [],
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("protocol") != payload["protocol"]:
            raise ValueError("existing SMG output uses a different protocol")
        if existing.get("unicycle") != payload["unicycle"]:
            raise ValueError("existing SMG output uses a different unicycle")
        payload["records"] = existing.get("records", [])
    completed_keys = {
        (
            str(record["method"]),
            str(record["family"]),
            int(record["n_robots"]),
            int(record["seed"]),
        )
        for record in payload["records"]
    }
    jobs: list[tuple[str, int, int, SMGGeometry]] = []
    for family in args.families:
        widths = (
            args.doorway_widths
            if family == "doorway"
            else (args.intersection_width,)
        )
        for width in widths:
            geometry = SMGGeometry(
                doorway_width=(
                    width if family == "doorway" else 0.8
                ),
                doorway_thickness=args.wall_thickness,
                intersection_corridor_width=(
                    width if family == "intersection" else 2.4
                ),
            )
            for count in args.robots:
                for seed in args.seeds:
                    jobs.append((family, count, seed, geometry))

    def job_key(method, family, count, seed, geometry):
        method_label = (
            "CLEAR" if method == "clear" else "Vanilla CBF-QP"
        )
        width = (
            geometry.doorway_width
            if family == "doorway"
            else geometry.intersection_corridor_width
        )
        return method_label, f"{family}_w{width:.2f}", count, seed

    requested_keys = {
        job_key(method, family, count, seed, geometry)
        for method in args.methods
        for family, count, seed, geometry in jobs
    }
    total = len(requested_keys)
    completed = len(requested_keys & completed_keys)
    for method in args.methods:
        for family, count, seed, geometry in jobs:
            key = job_key(method, family, count, seed, geometry)
            if key in completed_keys:
                continue
            scenario = make_smg_scenario(
                family,
                count,
                seed,
                protocol,
                geometry,
            )
            record = run_one(
                method,
                scenario,
                geometry,
                unicycle,
                config,
                record_stride=args.record_stride,
                isolated_reference=args.isolated_reference,
            )
            record["geometry"] = geometry.metadata(
                protocol,
                lookahead=unicycle.lookahead,
            )
            payload["records"].append(record)
            completed_keys.add(key)
            completed += 1
            output.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{completed}/{total}] {record['method']} "
                f"{record['family']} N={count}: "
                f"success={record['mission_success']}",
                flush=True,
            )
    print(output)


if __name__ == "__main__":
    main()
