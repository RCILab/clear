"""Evaluate public NH-ORCA on the canonical bounded-unicycle scenarios."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from math import pi
import os
from pathlib import Path
from time import perf_counter_ns

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pyrvo2

from clear_nav import (
    Protocol,
    SMGGeometry,
    Scenario,
    doorway_flow_from_trajectory,
    interference_delay_metrics,
    intersection_flow_from_trajectory,
    make_smg_scenario,
)
from clear_nav.geometry import Circle, Rectangle
from clear_nav.guidance import CostFieldGuidance, GridPlanner
from clear_nav.simulator import (
    _minimum_obstacle_clearance,
    _swept_pair_distance,
)
from clear_nav.unicycle import UnicycleConfig
from run_unicycle import make_unicycle_scenario
from timing_v2_common import aggregate_record, save_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("nh-orca", "orca"),
        default="nh-orca",
        help=(
            "NH-ORCA uses the public forward effective-point mapping; ORCA "
            "tracks the standard center-velocity command with bounded yaw."
        ),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=(
            "free",
            "swap",
            "circ15",
            "rect15",
            "doorway12",
            "intersection24",
        ),
        default=("free", "swap", "circ15", "rect15"),
    )
    parser.add_argument("--robots", nargs="+", type=int, default=(20,))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--lookahead", type=float, default=0.05)
    parser.add_argument("--yaw-rate-limit", type=float, default=pi / 2.0)
    parser.add_argument("--heading-gain", type=float, default=2.5)
    parser.add_argument("--inner-substeps", type=int, default=3)
    parser.add_argument("--neighbor-distance", type=float, default=3.0)
    parser.add_argument("--max-neighbors", type=int, default=128)
    parser.add_argument("--time-horizon", type=float, default=2.0)
    parser.add_argument("--obstacle-time-horizon", type=float, default=2.0)
    parser.add_argument("--obstacle-inflation", type=float, default=0.0)
    parser.add_argument("--pair-inflation", type=float, default=0.0)
    parser.add_argument("--circle-vertices", type=int, default=32)
    parser.add_argument(
        "--isolated-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--timing-hook",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the behavior-equivalent C++ per-agent timing hook.",
    )
    parser.add_argument(
        "--safety-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute swept collision metrics outside the controller timer. "
            "Disable for timing-only runs whose outcomes are audited elsewhere."
        ),
    )
    parser.add_argument("--timing-warmup-steps", type=int, default=40)
    parser.add_argument(
        "--timing-v2-dir",
        type=Path,
        default=Path(
            "baselines/comparison-harness/results/timing_v2/samples"
        ),
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Optional directory for compressed visualization trajectories.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _ccw_rectangle(center: np.ndarray, size: np.ndarray) -> list[np.ndarray]:
    half = 0.5 * np.asarray(size, dtype=float)
    center = np.asarray(center, dtype=float)
    # top-right -> top-left -> bottom-left -> bottom-right is CCW.
    offsets = np.array(
        [
            [half[0], half[1]],
            [-half[0], half[1]],
            [-half[0], -half[1]],
            [half[0], -half[1]],
        ]
    )
    return [np.asarray(value, dtype=float) for value in center + offsets]


def _ccw_circle(obstacle: Circle, count: int) -> list[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * pi, count, endpoint=False)
    vertices = obstacle.center + obstacle.radius * np.stack(
        (np.cos(angles), np.sin(angles)), axis=1
    )
    return [np.asarray(value, dtype=float) for value in vertices]


def _add_obstacles(
    simulator,
    scenario,
    circle_vertices: int,
    inflation: float,
) -> None:
    for obstacle in scenario.arena.obstacles:
        if isinstance(obstacle, Rectangle):
            vertices = _ccw_rectangle(
                obstacle.center,
                obstacle.size + 2.0 * inflation,
            )
        elif isinstance(obstacle, Circle):
            vertices = _ccw_circle(
                Circle(obstacle.center, obstacle.radius + inflation),
                circle_vertices,
            )
        else:
            raise TypeError(f"unsupported obstacle {type(obstacle)!r}")
        simulator.addObstacle(vertices)

    # RVO2 polygon obstacles are one-sided.  Four thin CCW boxes put their
    # collision-facing sides on the workspace boundary.
    half = scenario.protocol.half_width
    span = 2.0 * half + 4.0
    thickness = 2.0
    walls = (
        (
            np.array([-half + inflation - thickness / 2.0, 0.0]),
            np.array([thickness, span]),
        ),
        (
            np.array([half - inflation + thickness / 2.0, 0.0]),
            np.array([thickness, span]),
        ),
        (
            np.array([0.0, -half + inflation - thickness / 2.0]),
            np.array([span, thickness]),
        ),
        (
            np.array([0.0, half - inflation + thickness / 2.0]),
            np.array([span, thickness]),
        ),
    )
    for center, size in walls:
        simulator.addObstacle(_ccw_rectangle(center, size))
    simulator.processObstacle()


def _minimum_pair_distance(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("inf")
    left, right = np.triu_indices(len(positions), k=1)
    return float(
        np.min(np.linalg.norm(positions[left] - positions[right], axis=1))
    )


def _clearance_components(scenario, positions: np.ndarray) -> tuple[float, float]:
    padding = scenario.protocol.robot_clearance
    wall = scenario.protocol.half_width - padding
    boundary = float(
        np.min(wall - np.max(np.abs(positions), axis=1))
    )
    if not scenario.arena.obstacles:
        return boundary, float("inf")
    obstacle = min(
        shape.clearance_normal(point, padding)[0]
        for point in positions
        for shape in scenario.arena.obstacles
    )
    return boundary, float(obstacle)


def _percentile_ms(samples_ns: list[int], percentile: float) -> float:
    if not samples_ns:
        return 0.0
    return float(np.percentile(np.asarray(samples_ns) / 1.0e6, percentile))


def _smg_geometry(family: str) -> SMGGeometry | None:
    if family == "doorway12":
        return SMGGeometry(doorway_width=1.2)
    if family == "intersection24":
        return SMGGeometry(intersection_corridor_width=2.4)
    return None


def _scenario_for(
    args: argparse.Namespace,
    family: str,
    count: int,
    seed: int,
) -> Scenario:
    protocol = Protocol(dt=args.dt, horizon=args.horizon)
    geometry = _smg_geometry(family)
    if geometry is not None:
        smg_family = (
            "doorway" if family.startswith("doorway") else "intersection"
        )
        return make_smg_scenario(
            smg_family, count, seed, protocol, geometry
        )
    unicycle = UnicycleConfig(
        lookahead=args.lookahead,
        yaw_rate_limit=args.yaw_rate_limit,
        inner_substeps=args.inner_substeps,
    )
    return make_unicycle_scenario(
        family, count, seed, protocol, unicycle
    )


def run_case(
    args: argparse.Namespace,
    family: str,
    count: int,
    seed: int,
    *,
    scenario_override: Scenario | None = None,
) -> dict:
    protocol = Protocol(dt=args.dt, horizon=args.horizon)
    scenario = (
        scenario_override
        if scenario_override is not None
        else _scenario_for(args, family, count, seed)
    )
    centers = np.asarray(scenario.starts, dtype=float).copy()
    headings = np.zeros(count)
    heading_vectors = np.stack((np.cos(headings), np.sin(headings)), axis=1)
    virtual = centers + args.lookahead * heading_vectors
    rvo_positions = (
        virtual.copy() if args.mode == "nh-orca" else centers.copy()
    )
    rvo_velocity = np.zeros((count, 2))

    simulator = pyrvo2.RVOSimulator()
    simulator.setTimeStep(args.dt)
    simulator.setAgentDefaults(
        args.neighbor_distance,
        min(args.max_neighbors, max(0, count - 1)),
        args.time_horizon,
        args.obstacle_time_horizon,
        (
            protocol.robot_clearance + args.lookahead
            if args.mode == "nh-orca"
            else protocol.robot_clearance
        )
        + args.pair_inflation,
        protocol.speed_limit,
        np.zeros(2),
    )
    _add_obstacles(
        simulator,
        scenario,
        args.circle_vertices,
        args.obstacle_inflation,
    )
    for position in rvo_positions:
        simulator.addAgent(position)

    planning_start = perf_counter_ns()
    guidance: CostFieldGuidance | None = None
    if scenario.arena.obstacles:
        control_protocol = Protocol(
            **{
                **asdict(protocol),
                "body_radius": (
                    protocol.body_radius + args.lookahead
                    if args.mode == "nh-orca"
                    else protocol.body_radius
                ),
            }
        )
        guidance = GridPlanner(
            scenario.arena, control_protocol
        ).cost_field_plan(scenario.goals)
    planning_ns = perf_counter_ns() - planning_start

    first_arrival = np.full(count, np.inf)
    initial_distance = np.linalg.norm(centers - scenario.goals, axis=1)
    first_arrival[initial_distance <= protocol.arrival_radius] = 0.0
    min_pair = (
        _minimum_pair_distance(centers)
        if args.safety_audit
        else float("inf")
    )
    min_obstacle = (
        _minimum_obstacle_clearance(scenario, centers)
        if args.safety_audit
        else float("inf")
    )
    if args.safety_audit:
        min_boundary, min_static_obstacle = _clearance_components(
            scenario, centers
        )
    else:
        min_boundary = float("inf")
        min_static_obstacle = float("inf")
    maximum_linear = 0.0
    maximum_yaw = 0.0
    path_length = np.zeros(count)
    control_ns: list[int] = []
    timing_batch_ms: list[float] = []
    timing_shared_ms: list[float] = []
    timing_local_ms: list[list[float]] = []
    timing_critical_ms: list[float] = []
    executed_steps = 0
    trajectory = [centers.copy()]
    heading_trajectory = [headings.copy()]
    times = [0.0]

    for step in range(protocol.steps):
        targets = (
            guidance.targets(rvo_positions)
            if guidance is not None
            else scenario.goals
        )
        control_start = perf_counter_ns()
        for robot in range(count):
            simulator.setAgentPosition(robot, rvo_positions[robot])
            simulator.setAgentVelocity(robot, rvo_velocity[robot])

        displacement = targets - rvo_positions
        distance = np.linalg.norm(displacement, axis=1)
        physical_goal_distance = np.linalg.norm(
            centers - scenario.goals, axis=1
        )
        preferred = np.zeros_like(displacement)
        moving = physical_goal_distance > protocol.arrival_radius
        preferred[moving] = (
            displacement[moving]
            / distance[moving, None]
            * np.minimum(
                protocol.speed_limit,
                1.2 * distance[moving],
            )[:, None]
        )
        for robot in range(count):
            simulator.setAgentPrefVelocity(robot, preferred[robot])
        if args.timing_hook:
            native_timing = list(simulator.stepTiming())
            local_ms = [float(value) for value in native_timing[2:]]
        else:
            step_start = perf_counter_ns()
            simulator.step()
            local_ms = [(perf_counter_ns() - step_start) / 1.0e6]
        command = np.stack(
            [simulator.getAgentVelocity(robot) for robot in range(count)]
        )

        heading_vectors = np.stack(
            (np.cos(headings), np.sin(headings)), axis=1
        )
        perpendicular = np.stack(
            (-np.sin(headings), np.cos(headings)), axis=1
        )
        linear = np.einsum("ij,ij->i", heading_vectors, command)
        if args.mode == "nh-orca":
            yaw = (
                np.einsum("ij,ij->i", perpendicular, command)
                / args.lookahead
            )
        else:
            desired_heading = np.arctan2(command[:, 1], command[:, 0])
            heading_error = (
                desired_heading - headings + pi
            ) % (2.0 * pi) - pi
            command_norm = np.linalg.norm(command, axis=1)
            heading_error[command_norm <= 1.0e-12] = 0.0
            yaw = args.heading_gain * heading_error
        linear = np.clip(
            linear, -protocol.speed_limit, protocol.speed_limit
        )
        yaw = np.clip(yaw, -args.yaw_rate_limit, args.yaw_rate_limit)
        batch_ns = perf_counter_ns() - control_start
        control_ns.append(batch_ns)
        batch_ms = batch_ns / 1.0e6
        shared_ms = max(0.0, batch_ms - sum(local_ms))
        timing_batch_ms.append(batch_ms)
        timing_shared_ms.append(shared_ms)
        timing_local_ms.append(local_ms)
        timing_critical_ms.append(
            shared_ms + max(local_ms, default=0.0)
        )

        maximum_linear = max(maximum_linear, float(np.max(np.abs(linear))))
        maximum_yaw = max(maximum_yaw, float(np.max(np.abs(yaw))))
        rvo_before = rvo_positions.copy()
        inner_dt = args.dt / args.inner_substeps
        for _ in range(args.inner_substeps):
            before = centers.copy()
            headings = headings + yaw * inner_dt
            direction = np.stack(
                (np.cos(headings), np.sin(headings)), axis=1
            )
            velocity = linear[:, None] * direction
            centers = centers + inner_dt * velocity
            path_length += np.linalg.norm(centers - before, axis=1)
            if args.safety_audit:
                min_pair = min(
                    min_pair,
                    _swept_pair_distance(before, velocity, inner_dt),
                )
                min_obstacle = min(
                    min_obstacle,
                    _minimum_obstacle_clearance(scenario, centers),
                    _minimum_obstacle_clearance(
                        scenario,
                        0.5 * (before + centers),
                    ),
                )
                for audited_positions in (
                    centers,
                    0.5 * (before + centers),
                ):
                    boundary, static_obstacle = _clearance_components(
                        scenario, audited_positions
                    )
                    min_boundary = min(min_boundary, boundary)
                    min_static_obstacle = min(
                        min_static_obstacle, static_obstacle
                    )
        direction = np.stack(
            (np.cos(headings), np.sin(headings)), axis=1
        )
        virtual = centers + args.lookahead * direction
        rvo_positions = (
            virtual.copy() if args.mode == "nh-orca" else centers.copy()
        )
        rvo_velocity = (rvo_positions - rvo_before) / args.dt

        goal_distance = np.linalg.norm(centers - scenario.goals, axis=1)
        newly_arrived = np.isinf(first_arrival) & (
            goal_distance <= protocol.arrival_radius
        )
        first_arrival[newly_arrived] = (step + 1) * args.dt
        executed_steps = step + 1
        trajectory.append(centers.copy())
        heading_trajectory.append(headings.copy())
        times.append(executed_steps * args.dt)
        if np.all(goal_distance <= protocol.arrival_radius):
            break

    final_distance = np.linalg.norm(centers - scenario.goals, axis=1)
    final_arrived = final_distance <= protocol.arrival_radius
    ever_arrived = np.isfinite(first_arrival)
    safe = (
        bool(
            min_pair >= protocol.pair_clearance - 1.0e-6
            and min_obstacle >= -1.0e-6
        )
        if args.safety_audit
        else None
    )
    warmup = min(
        max(args.timing_warmup_steps, 0),
        max(0, len(timing_batch_ms) - 1),
    )
    retained_batch = timing_batch_ms[warmup:]
    retained_shared = timing_shared_ms[warmup:]
    retained_local = timing_local_ms[warmup:]
    retained_critical = timing_critical_ms[warmup:]
    sample_path = args.timing_v2_dir / (
        f"{args.mode}_{scenario.family}_n{count}_s{seed}.npz"
    )
    sample_sha = save_samples(
        sample_path,
        batch_step_ms=retained_batch,
        shared_coordination_ms=retained_shared,
        local_unit_ms=retained_local,
        critical_path_ms=retained_critical,
    )
    timing_record = aggregate_record(
        batch_step_ms=retained_batch,
        shared_coordination_ms=retained_shared,
        local_unit_ms=retained_local,
        critical_path_ms=retained_critical,
        controller_backend=(
            "public-rvo2-c++-per-agent-hook"
            if args.timing_hook
            else "public-rvo2-c++-legacy-step"
        ),
        worker_count=1,
        warmup_steps=warmup,
        scenario_fingerprint=scenario.fingerprint(),
    )
    state = np.concatenate(
        (
            np.asarray(centers, dtype="<f8").reshape(-1),
            np.asarray(headings, dtype="<f8").reshape(-1),
        )
    )
    record = {
        "algorithm": "NH-ORCA" if args.mode == "nh-orca" else "ORCA",
        "family": scenario.family,
        "n_robots": count,
        "seed": seed,
        "fingerprint": scenario.fingerprint(),
        "dynamics": "bounded-lookahead-unicycle",
        "safe": safe,
        "success": (
            bool(safe and np.all(final_arrived))
            if safe is not None
            else None
        ),
        "safety_audit": args.safety_audit,
        "arrival_success": bool(np.all(final_arrived)),
        "ever_arrival_success": bool(np.all(ever_arrived)),
        "final_arrival_success": bool(np.all(final_arrived)),
        "robot_arrival_rate": float(np.mean(final_arrived)),
        "ever_arrival_rate": float(np.mean(ever_arrived)),
        "makespan_s": (
            float(np.max(first_arrival))
            if np.all(ever_arrived)
            else None
        ),
        "minimum_pair_distance_m": (
            min_pair if args.safety_audit else None
        ),
        "minimum_obstacle_clearance_m": (
            min_obstacle if args.safety_audit else None
        ),
        "minimum_boundary_clearance_m": (
            min_boundary if args.safety_audit else None
        ),
        "minimum_static_obstacle_clearance_m": (
            min_static_obstacle if args.safety_audit else None
        ),
        "maximum_linear_speed_mps": maximum_linear,
        "maximum_yaw_rate_rps": maximum_yaw,
        "mean_path_length_m": float(np.mean(path_length)),
        "planning_time_ms": planning_ns / 1.0e6,
        "termination_time_s": executed_steps * args.dt,
        "steps": executed_steps,
        **timing_record,
        "local_unit_definition": (
            "one RVO2 agent computeNeighbors plus computeNewVelocity"
            if args.timing_hook
            else "complete legacy RVO2 step"
        ),
        "timing_v2_samples_file": sample_path.as_posix(),
        "timing_v2_samples_sha256": sample_sha,
        "final_state_sha256": hashlib.sha256(
            state.tobytes()
        ).hexdigest(),
        "final_centers": np.round(centers, 12).tolist(),
        "final_headings": np.round(headings, 12).tolist(),
        "control_time_mean_ms": _percentile_ms(control_ns, 50.0)
        if not control_ns
        else float(np.mean(control_ns)) / 1.0e6,
        "control_time_p95_ms": _percentile_ms(control_ns, 95.0),
        "control_time_max_ms": _percentile_ms(control_ns, 100.0),
        "final_maximum_goal_distance_m": float(np.max(final_distance)),
        "parameters": {
            "lookahead_m": args.lookahead,
            "mode": args.mode,
            "heading_gain": args.heading_gain,
            "neighbor_distance_m": args.neighbor_distance,
            "max_neighbors": min(args.max_neighbors, max(0, count - 1)),
            "time_horizon_s": args.time_horizon,
            "obstacle_time_horizon_s": args.obstacle_time_horizon,
            "circle_vertices": args.circle_vertices,
            "obstacle_inflation_m": args.obstacle_inflation,
            "pair_inflation_m": args.pair_inflation,
        },
    }
    geometry = _smg_geometry(family)
    if geometry is not None:
        trajectory_array = np.asarray(trajectory)
        times_array = np.asarray(times)
        record["first_arrival_times_s"] = [
            None if not np.isfinite(value) else float(value)
            for value in first_arrival
        ]
        if family.startswith("doorway"):
            record.update(
                doorway_flow_from_trajectory(
                    times_array,
                    trajectory_array,
                    scenario.starts,
                    scenario.goals,
                    body_radius=protocol.body_radius,
                    dt=protocol.dt,
                    doorway_width=geometry.doorway_width,
                    doorway_thickness=geometry.doorway_thickness,
                )
            )
        else:
            record.update(
                intersection_flow_from_trajectory(
                    times_array,
                    trajectory_array,
                    scenario.starts,
                    scenario.goals,
                    body_radius=protocol.body_radius,
                    dt=protocol.dt,
                    corridor_width=geometry.intersection_corridor_width,
                )
            )
    if args.trajectory_dir is not None:
        args.trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory_family = (
            "doorway"
            if family.startswith("doorway")
            else "intersection"
            if family.startswith("intersection")
            else family
        )
        trajectory_path = args.trajectory_dir / (
            f"{args.mode}_{trajectory_family}_n{count}_s{seed}.npz"
        )
        np.savez_compressed(
            trajectory_path,
            times=np.asarray(times, dtype=np.float64),
            positions=np.asarray(trajectory, dtype=np.float64),
            headings=np.asarray(heading_trajectory, dtype=np.float64),
            method=np.asarray(record["algorithm"]),
            scenario_fingerprint=np.asarray(scenario.fingerprint()),
        )
        record["trajectory_file"] = trajectory_path.as_posix()
        record["trajectory_sha256"] = hashlib.sha256(
            trajectory_path.read_bytes()
        ).hexdigest()
    return record


def main() -> None:
    args = parse_args()
    if args.lookahead <= 0.0:
        raise ValueError("lookahead must be positive")
    if args.obstacle_inflation < 0.0:
        raise ValueError("obstacle inflation must be nonnegative")
    if args.pair_inflation < 0.0:
        raise ValueError("pair inflation must be nonnegative")
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("baselines/comparison-harness/results") / (
            f"nhorca_{stamp}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "algorithm": "NH-ORCA" if args.mode == "nh-orca" else "ORCA",
        "source": (
            "public NH-ORCA Python implementation"
            if args.mode == "nh-orca"
            else "public RVO2 ORCA implementation with bounded-unicycle tracking"
        ),
        "physical_protocol": asdict(
            Protocol(dt=args.dt, horizon=args.horizon)
        ),
        "initial_heading_rad": 0.0,
        "shared_static_guidance": "cost-field",
        "records": [],
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("algorithm") != metadata["algorithm"]:
            raise ValueError("existing output algorithm does not match")
        metadata["records"] = existing.get("records", [])
    completed = {
        (
            (
                "doorway"
                if str(record["family"]).startswith("doorway")
                else "intersection"
                if str(record["family"]).startswith("intersection")
                else str(record["family"])
            ),
            int(record["n_robots"]),
            int(record["seed"]),
        )
        for record in metadata["records"]
    }
    total = len(args.families) * len(args.robots) * len(args.seeds)
    for family in args.families:
        for count in args.robots:
            for seed in args.seeds:
                family_key = (
                    "doorway"
                    if family.startswith("doorway")
                    else "intersection"
                    if family.startswith("intersection")
                    else family
                )
                if (family_key, count, seed) in completed:
                    continue
                record = run_case(args, family, count, seed)
                geometry = _smg_geometry(family)
                if args.isolated_reference and geometry is not None:
                    scenario = _scenario_for(
                        args, family, count, seed
                    )
                    solo_times = np.full(count, np.inf)
                    for index in range(count):
                        solo_scenario = Scenario(
                            family=f"{scenario.family}_solo",
                            n_robots=1,
                            seed=seed,
                            starts=scenario.starts[
                                index : index + 1
                            ].copy(),
                            goals=scenario.goals[
                                index : index + 1
                            ].copy(),
                            arena=scenario.arena,
                            protocol=scenario.protocol,
                        )
                        solo_record = run_case(
                            args,
                            family,
                            1,
                            seed,
                            scenario_override=solo_scenario,
                        )
                        solo_makespan = solo_record.get("makespan_s")
                        if solo_makespan is not None:
                            solo_times[index] = solo_makespan
                    multi_times = np.asarray(
                        [
                            np.inf if value is None else value
                            for value in record["first_arrival_times_s"]
                        ],
                        dtype=float,
                    )
                    record.update(
                        interference_delay_metrics(
                            multi_times, solo_times
                        )
                    )
                    record["isolated_ttg_s"] = [
                        None if not np.isfinite(value) else float(value)
                        for value in solo_times
                    ]
                metadata["records"].append(record)
                completed.add((family_key, count, seed))
                output.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(json.dumps(record, sort_keys=True), flush=True)
                print(
                    f"[progress] {len(metadata['records'])}/{total}",
                    flush=True,
                )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
