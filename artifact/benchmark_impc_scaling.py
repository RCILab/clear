"""Audit and time the public SMGLib Social-IMPC-DR core."""

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
from others import collect_data, get_obstacle_list  # noqa: E402
from run import run_one_step  # noqa: E402
from test import initialize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robots",
        nargs="+",
        type=int,
        default=(2, 4, 8, 16, 40, 80),
    )
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def ring_instance(count: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    radius = max(3.0, 0.14 * count)
    angle = 2.0 * np.pi * np.arange(count) / count
    starts = np.stack(
        (radius * np.cos(angle), radius * np.sin(angle)),
        axis=1,
    )
    goals = -starts
    return list(starts), list(goals)


def time_one_step(count: int, horizon_steps: int, dt: float) -> dict:
    starts, goals = ring_instance(count)
    velocities = [np.zeros(2) for _ in range(count)]
    SET.initialize_set(
        count,
        starts,
        velocities,
        goals,
        0.2,
        0.1,
        dt,
        horizon_steps,
        1,
        2.0,
    )
    agents = initialize()
    collect_data(agents)
    obstacles = get_obstacle_list(agents, count)
    before = np.asarray([agent.p.copy() for agent in agents])
    started = perf_counter()
    advanced = run_one_step(agents, obstacles, verbose=False)
    elapsed = perf_counter() - started
    after = np.asarray([agent.p for agent in advanced])
    displacement = np.linalg.norm(after - before, axis=1)
    constraint_rows = [agent.cons_A.shape[0] for agent in advanced]
    return {
        "n_robots": count,
        "one_replan_wall_time_s": elapsed,
        "per_robot_replan_ms": 1.0e3 * elapsed / count,
        "moving_robot_count": int(np.sum(displacement > 1.0e-9)),
        "maximum_displacement_m": float(np.max(displacement)),
        "constraint_rows_mean": float(np.mean(constraint_rows)),
        "constraint_rows_max": int(np.max(constraint_rows)),
    }


def main() -> None:
    args = parse_args()
    os.environ["SMGLIB_CORES"] = str(args.cores)
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results") / f"impc_scaling_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "IMPC-DR",
        "source": "SMGLib public implementation",
        "model_audit": {
            "executed_state": "[x, y, vx, vy]",
            "executed_input": "[ax, ay]",
            "prediction_model": "double integrator",
            "bounded_unicycle_compatible": False,
            "common_unicycle_table_eligible": False,
            "reason": (
                "the public execution path does not use the repository's "
                "separate nonholonomic linearization module"
            ),
        },
        "horizon_steps": args.horizon_steps,
        "dt_s": args.dt,
        "parallel_workers": args.cores,
        "records": [],
    }
    for index, count in enumerate(args.robots, start=1):
        record = time_one_step(count, args.horizon_steps, args.dt)
        payload["records"].append(record)
        output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{index}/{len(args.robots)}] N={count}: "
            f"{record['one_replan_wall_time_s']:.3f}s",
            flush=True,
        )
    print(output)


if __name__ == "__main__":
    main()

