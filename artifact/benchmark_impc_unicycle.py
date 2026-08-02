"""Smoke-test and time the common-unicycle IMPC-DR adaptation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


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
    parser.add_argument("--robots", nargs="+", type=int, default=(2, 4, 8, 16))
    parser.add_argument("--horizon-steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--scp-iterations", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def ring_instance(count):
    radius = max(2.0, 0.14 * count)
    angles = 2.0 * np.pi * np.arange(count) / count
    starts = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1
    )
    return starts, -starts


def evaluate(count, args):
    starts, goals = ring_instance(count)
    velocities = [np.zeros(2) for _ in range(count)]
    SET.initialize_set(
        count,
        list(starts),
        velocities,
        list(goals),
        0.4,
        0.1,
        args.dt,
        args.horizon_steps,
        args.steps,
        2.0,
    )
    agents = [
        UnicycleUAV(
            i,
            starts[i],
            velocities[i],
            goals[i],
            ini_heading=0.0,
            ini_K=args.horizon_steps,
        )
        for i in range(count)
    ]
    initial = starts.copy()
    replans = []
    statuses = []
    minimum_distance = np.inf
    for _ in range(args.steps):
        obstacle_list = get_obstacle_list(agents, count)
        started = perf_counter()
        agents = run_one_step(
            agents,
            obstacle_list,
            verbose=False,
            scp_iterations=args.scp_iterations,
        )
        replans.append(perf_counter() - started)
        statuses.extend(agent.last_solver_status for agent in agents)
        positions = np.asarray([agent.p for agent in agents])
        if count > 1:
            rows, columns = np.triu_indices(count, 1)
            minimum_distance = min(
                minimum_distance,
                float(np.min(np.linalg.norm(
                    positions[rows] - positions[columns], axis=1
                ))),
            )
    final = np.asarray([agent.p for agent in agents])
    return {
        "n_robots": count,
        "steps": args.steps,
        "mean_replan_wall_time_s": float(np.mean(replans)),
        "maximum_replan_wall_time_s": float(np.max(replans)),
        "per_robot_replan_ms": float(1e3 * np.mean(replans) / count),
        "moving_robot_count": int(
            np.sum(np.linalg.norm(final - initial, axis=1) > 1e-5)
        ),
        "minimum_pair_distance_m": minimum_distance,
        "all_solves_accepted": all(
            status in ("optimal", "optimal_inaccurate") for status in statuses
        ),
        "maximum_abs_speed_mps": float(
            max(abs(agent.speed) for agent in agents)
        ),
        "maximum_abs_yaw_rate_rps": float(
            max(abs(agent.u[1]) for agent in agents)
        ),
    }


def main():
    args = parse_args()
    os.environ["SMGLIB_CORES"] = str(args.cores)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path("results") / f"impc_unicycle_{stamp}.json"
    payload = {
        "method": "IMPC-DR",
        "implementation": "public algorithm adapted to common unicycle dynamics",
        "state": "[x, y, v, theta]",
        "input": "[a, omega]",
        "dt_s": args.dt,
        "horizon_steps": args.horizon_steps,
        "scp_iterations": args.scp_iterations,
        "records": [],
    }
    for count in args.robots:
        record = evaluate(count, args)
        payload["records"].append(record)
        print(
            f"N={count}: {record['mean_replan_wall_time_s']:.3f}s, "
            f"moving={record['moving_robot_count']}/{count}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

