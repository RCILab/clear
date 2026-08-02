"""Run IMPC-DR on the common bounded-unicycle Social Mini-Games."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, perf_counter_ns

import numpy as np

from clear_nav import (
    Protocol,
    SMGGeometry,
    Scenario,
    interference_delay_metrics,
    make_smg_scenario,
)
from clear_nav.unicycle import UnicycleConfig
from clear_nav.smg_metrics import (
    doorway_flow_from_trajectory,
    intersection_flow_from_trajectory,
)
from run_unicycle import make_unicycle_scenario
from timing_v2_common import aggregate_record, save_samples, stats_ms


METHOD = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "SMGLib"
    / "src"
    / "methods"
    / "Social-IMPC-DR"
)
sys.path.insert(0, str(METHOD))

import SET  # noqa: E402
from others import get_obstacle_list  # noqa: E402
from unicycle_run import run_one_step  # noqa: E402
from unicycle_uav import UnicycleUAV  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("doorway", "intersection", "rect15"),
        default=("doorway", "intersection"),
    )
    parser.add_argument("--robots", nargs="+", type=int, default=(8, 16))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--doorway-width", type=float, default=1.2)
    parser.add_argument("--intersection-width", type=float, default=2.4)
    parser.add_argument("--wall-thickness", type=float, default=0.8)
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--prediction-steps", type=int, default=24)
    parser.add_argument("--scp-iterations", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--timing-warmup-steps", type=int, default=40)
    parser.add_argument(
        "--timing-v2-dir",
        type=Path,
        default=Path("validation/timing_v2/samples"),
    )
    parser.add_argument(
        "--isolated-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Optional directory for compressed visualization trajectories.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def simulate(scenario, geometry, args):
    count = scenario.n_robots
    protocol = scenario.protocol
    pair_planning_guard = 2.0e-4
    obstacle_planning_guard = 1.0e-4
    velocities = [np.zeros(2) for _ in range(count)]
    SET.initialize_set(
        count,
        list(scenario.starts),
        velocities,
        list(scenario.goals),
        protocol.pair_clearance + pair_planning_guard,
        0.1,
        protocol.dt,
        args.prediction_steps,
        protocol.steps,
        0.5,
    )
    agents = [
        UnicycleUAV(
            i,
            scenario.starts[i],
            velocities[i],
            scenario.goals[i],
            ini_heading=0.0,
            ini_K=args.prediction_steps,
            max_speed=protocol.speed_limit,
            max_acceleration=1.2,
            max_yaw_rate=np.pi / 2.0,
        )
        for i in range(count)
    ]
    for agent in agents:
        agent.arena = scenario.arena
        agent.static_padding = (
            protocol.robot_clearance + obstacle_planning_guard
        )

    times = [0.0]
    trajectory = [scenario.starts.copy()]
    headings = [np.zeros(count)]
    first_arrivals = np.full(count, np.inf)
    pair_minimum = np.inf
    obstacle_minimum = np.inf
    maximum_speed = 0.0
    maximum_yaw = 0.0
    accepted = True
    solver_counts = {}
    solver_status_counts = {}
    step_times = []
    timing_batch_ms = []
    timing_shared_ms = []
    timing_local_ms = []
    timing_critical_ms = []
    phase_samples = {}
    arrived = np.linalg.norm(
        scenario.starts - scenario.goals, axis=1
    ) <= protocol.arrival_radius
    first_arrivals[arrived] = 0.0

    for step_index in range(1, protocol.steps + 1):
        shared_started = perf_counter_ns()
        obstacles = get_obstacle_list(agents, count)
        shared_ns = perf_counter_ns() - shared_started
        started = perf_counter()
        agents = run_one_step(
            agents,
            obstacles,
            verbose=False,
            scp_iterations=args.scp_iterations,
        )
        solve_wall_s = perf_counter() - started
        step_times.append(solve_wall_s)
        local_ns = [
            int(agent.last_timing_profile["local_unit_total_ns"])
            for agent in agents
        ]
        shared_ms = shared_ns / 1.0e6
        local_ms = [value / 1.0e6 for value in local_ns]
        timing_batch_ms.append(shared_ms + 1.0e3 * solve_wall_s)
        timing_shared_ms.append(shared_ms)
        timing_local_ms.append(local_ms)
        timing_critical_ms.append(
            shared_ms + max(local_ms, default=0.0)
        )
        for agent in agents:
            for key, value in agent.last_timing_profile.items():
                if key.endswith("_ns") and key != "local_unit_total_ns":
                    phase_samples.setdefault(key, []).append(
                        value / 1.0e6
                    )
        positions = np.asarray([agent.p for agent in agents])
        current_headings = np.asarray([agent.heading for agent in agents])
        speeds = np.asarray([agent.speed for agent in agents])
        yaws = np.asarray([agent.u[1] for agent in agents])
        accepted = accepted and all(
            agent.last_solver_status in ("optimal", "optimal_inaccurate")
            for agent in agents
        )
        for agent in agents:
            solver_counts[agent.last_solver_name] = (
                solver_counts.get(agent.last_solver_name, 0) + 1
            )
            solver_status_counts[agent.last_solver_status] = (
                solver_status_counts.get(agent.last_solver_status, 0) + 1
            )
        if count > 1:
            rows, columns = np.triu_indices(count, 1)
            pair_minimum = min(
                pair_minimum,
                float(np.min(np.linalg.norm(
                    positions[rows] - positions[columns], axis=1
                ))),
            )
        obstacle_minimum = min(
            obstacle_minimum,
            min(
                scenario.arena.minimum_clearance(
                    point, protocol.robot_clearance
                )
                for point in positions
            ),
        )
        maximum_speed = max(maximum_speed, float(np.max(np.abs(speeds))))
        maximum_yaw = max(maximum_yaw, float(np.max(np.abs(yaws))))
        time_s = step_index * protocol.dt
        distances = np.linalg.norm(positions - scenario.goals, axis=1)
        new_arrivals = (~arrived) & (distances <= protocol.arrival_radius)
        first_arrivals[new_arrivals] = time_s
        arrived |= new_arrivals
        times.append(time_s)
        trajectory.append(positions)
        headings.append(current_headings)
        if np.all(arrived):
            break

    times_array = np.asarray(times)
    trajectory_array = np.asarray(trajectory)
    final_distances = np.linalg.norm(
        trajectory_array[-1] - scenario.goals, axis=1
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
        f"impc_{scenario.family}_n{count}_s{scenario.seed}.npz"
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
        controller_backend="cvxpy-clarabel-persistent-cpu",
        worker_count=int(args.cores),
        warmup_steps=warmup,
        scenario_fingerprint=scenario.fingerprint(),
    )
    record = {
        "method": "IMPC-DR",
        "family": scenario.family,
        "n_robots": count,
        "seed": scenario.seed,
        "fingerprint": scenario.fingerprint(),
        "mission_success": bool(
            np.all(arrived)
            and pair_minimum >= protocol.pair_clearance - 1e-6
            and obstacle_minimum >= -1e-6
            and accepted
        ),
        "physical_mission_success": bool(
            np.all(arrived)
            and pair_minimum >= protocol.pair_clearance - 1e-6
            and obstacle_minimum >= -1e-6
        ),
        "robot_arrival_rate": float(np.mean(arrived)),
        "makespan_s": (
            float(np.max(first_arrivals))
            if np.all(np.isfinite(first_arrivals))
            else None
        ),
        "unarrived_robots": int(np.sum(~arrived)),
        "maximum_final_goal_distance_m": float(np.max(final_distances)),
        "minimum_physical_pair_distance_m": float(pair_minimum),
        "minimum_physical_obstacle_clearance_m": float(obstacle_minimum),
        "maximum_linear_speed_mps": maximum_speed,
        "maximum_yaw_rate_rps": maximum_yaw,
        "all_solves_accepted": accepted,
        "final_solve_backend_counts": solver_counts,
        "final_solve_status_counts": solver_status_counts,
        "controller_wall_time_s": float(np.sum(step_times)),
        "mean_control_step_ms": float(1e3 * np.mean(step_times)),
        "p95_control_step_ms": float(
            1e3 * np.quantile(step_times, 0.95)
        ),
        "executed_steps": len(step_times),
        **timing_record,
        "local_unit_definition": (
            "one robot neighbor/static constraint update, two SCP solves, "
            "nonlinear rollout, and post-processing"
        ),
        "timing_v2_samples_file": sample_path.as_posix(),
        "timing_v2_samples_sha256": sample_sha,
        "phase_profile_per_agent_call_ms": {
            key.removesuffix("_ns"): stats_ms(values)
            for key, values in sorted(phase_samples.items())
        },
        "persistent_problem_enabled": (
            os.environ.get("IMPC_PERSISTENT", "1") != "0"
        ),
        "pair_planning_guard_m": pair_planning_guard,
        "obstacle_planning_guard_m": obstacle_planning_guard,
        "first_arrival_times_s": [
            None if not np.isfinite(value) else float(value)
            for value in first_arrivals
        ],
        "final_positions_m": trajectory_array[-1].tolist(),
    }
    if scenario.family.startswith("doorway"):
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
    elif scenario.family.startswith("intersection"):
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
            if scenario.family.startswith("doorway")
            else "intersection"
            if scenario.family.startswith("intersection")
            else scenario.family
        )
        trajectory_path = args.trajectory_dir / (
            f"impc-dr_{trajectory_family}_n{count}_s{scenario.seed}.npz"
        )
        np.savez_compressed(
            trajectory_path,
            times=times_array.astype(np.float64),
            positions=trajectory_array.astype(np.float64),
            headings=np.asarray(headings, dtype=np.float64),
            method=np.asarray("IMPC-DR"),
            scenario_fingerprint=np.asarray(scenario.fingerprint()),
        )
        record["trajectory_file"] = trajectory_path.as_posix()
        record["trajectory_sha256"] = hashlib.sha256(
            trajectory_path.read_bytes()
        ).hexdigest()
    return record


def main():
    args = parse_args()
    os.environ["SMGLIB_CORES"] = str(args.cores)
    protocol = Protocol(horizon=args.horizon, dt=args.dt)
    geometry = SMGGeometry(
        doorway_width=args.doorway_width,
        doorway_thickness=args.wall_thickness,
        intersection_corridor_width=args.intersection_width,
    )
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("results") / f"smg_impc_{stamp}.json"
    payload = {
        "initial_heading_rad": 0.0,
        "protocol": asdict(protocol),
        "geometry": geometry.metadata(protocol, lookahead=0.0),
        "method": "IMPC-DR",
        "implementation": (
            "public IMPC-DR structure adapted to [x,y,v,theta] and "
            "[a,omega] with sequential affine prediction"
        ),
        "prediction_steps": args.prediction_steps,
        "scp_iterations": args.scp_iterations,
        "records": [],
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("method") != "IMPC-DR":
            raise ValueError("existing output is not an IMPC-DR result")
        payload["records"] = existing.get("records", [])
    completed = {
        (
            (
                "doorway"
                if str(record["family"]).startswith("doorway")
                else "intersection"
                if str(record["family"]).startswith("intersection")
                else "rect15"
            ),
            int(record["n_robots"]),
            int(record["seed"]),
        )
        for record in payload["records"]
    }
    jobs = [
        (family, count, seed)
        for family in args.families
        for count in args.robots
        for seed in args.seeds
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for job_index, (family, count, seed) in enumerate(jobs, 1):
        if (family, count, seed) in completed:
            continue
        scenario = (
            make_unicycle_scenario(
                "rect15",
                count,
                seed,
                protocol,
                UnicycleConfig(),
            )
            if family == "rect15"
            else make_smg_scenario(
                family, count, seed, protocol, geometry
            )
        )
        record = simulate(scenario, geometry, args)
        if args.isolated_reference and family != "rect15":
            solo_times = np.full(count, np.inf)
            for robot_index in range(count):
                solo = Scenario(
                    family=f"{scenario.family}_solo",
                    n_robots=1,
                    seed=seed,
                    starts=scenario.starts[
                        robot_index : robot_index + 1
                    ].copy(),
                    goals=scenario.goals[
                        robot_index : robot_index + 1
                    ].copy(),
                    arena=scenario.arena,
                    protocol=scenario.protocol,
                )
                solo_record = simulate(solo, geometry, args)
                solo_time = solo_record.get("makespan_s")
                if solo_time is not None:
                    solo_times[robot_index] = solo_time
            multi_times = np.asarray(
                [
                    np.inf if value is None else float(value)
                    for value in record["first_arrival_times_s"]
                ]
            )
            record.update(
                interference_delay_metrics(multi_times, solo_times)
            )
            record["isolated_ttg_s"] = [
                None if not np.isfinite(value) else float(value)
                for value in solo_times
            ]
        payload["records"].append(record)
        completed.add((family, count, seed))
        args.output.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(
            f"[{job_index}/{len(jobs)}] IMPC-DR {record['family']} "
            f"N={count}: success={record['mission_success']}, "
            f"arrival={record['robot_arrival_rate']:.1%}, "
            f"step={record['mean_control_step_ms']:.1f}ms",
            flush=True,
        )
    print(args.output)


if __name__ == "__main__":
    main()
