"""Controller-only CLEAR timing across team sizes.

This benchmark advances real closed-loop state streams but deliberately times
only ``CLEARController.command``.  Scenario construction, static
cost-field planning, target lookup, and simulator swept-distance audits are
outside the measured interval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from time import perf_counter, perf_counter_ns

import numpy as np

from clear_nav import (
    CLEARController,
    ControllerConfig,
    Protocol,
)
from clear_nav.guidance import GridPlanner
from clear_nav.unicycle import (
    UnicycleConfig,
    _certified_bridge_progress,
    _project_native_unicycle,
    inflated_unicycle_protocol,
)
from clear_nav.safety import OSQPProjectionWorkspace
from run_unicycle import make_unicycle_scenario


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unreported"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("free", "swap", "circ15", "rect15"),
        default=None,
    )
    parser.add_argument("--robots", nargs="+", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/controller_scaling.json"),
    )
    return parser.parse_args()


def _configuration() -> ControllerConfig:
    return ControllerConfig(
        boundary_progress_aligned=True,
        cluster_escape_gain=0.55,
        cluster_escape_hysteresis=True,
        tangent_band=0.08,
        terminal_capture_hysteresis=True,
    )


def _stats_ms(samples_ns: list[int]) -> dict:
    values = np.asarray(samples_ns, dtype=float) / 1.0e6
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": float(np.max(values)),
    }


def run_case(
    family: str,
    count: int,
    seed: int,
    horizon: float,
    warmup_steps: int,
) -> dict:
    protocol = Protocol(horizon=horizon)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    control_protocol = inflated_unicycle_protocol(
        protocol,
        unicycle.lookahead,
    )
    terminal_capture = True
    config = _configuration()
    scenario = make_unicycle_scenario(
        family,
        count,
        seed,
        protocol,
        unicycle,
    )
    controller = CLEARController(control_protocol, config)
    controller.reset()
    headings = np.zeros(count)
    heading_vectors = np.stack(
        (np.cos(headings), np.sin(headings)),
        axis=1,
    )
    centers = np.asarray(scenario.starts, dtype=float).copy()
    virtual = centers + unicycle.lookahead * heading_vectors
    guidance = (
        GridPlanner(scenario.arena, control_protocol).cost_field_plan(
            scenario.goals
        )
        if scenario.arena.obstacles
        else None
    )
    first_arrival = np.full(count, np.inf)
    command_times: list[int] = []
    clear_command_times: list[int] = []
    native_projection_times: list[int] = []
    native_projection_sweeps: list[int] = []
    tangent_constraints: list[int] = []
    nonconverged = 0
    feasibility_restorations = 0
    projection_iteration_caps = 0
    tangent_fallbacks = 0
    native_dual = np.zeros(0)
    native_progress_dual = np.zeros(0)
    native_workspace = OSQPProjectionWorkspace(max_cached_patterns=16)
    case_start = perf_counter()

    for step in range(protocol.steps):
        control_goals = (
            guidance.targets(virtual)
            if guidance is not None
            else scenario.goals
        )
        start_ns = perf_counter_ns()
        audit = controller.command(
            virtual,
            control_goals,
            scenario.arena,
            project_safety=False,
        )
        bridge_progress = _certified_bridge_progress(
            controller,
            virtual,
            scenario.arena,
            audit,
            protocol,
            unicycle,
        )
        if bridge_progress:
            certified_progress_world_rows = np.stack(
                [proposal.world_row for proposal in bridge_progress]
            )
            certified_progress_lower = np.asarray(
                [proposal.lower_bound for proposal in bridge_progress]
            )
            certified_progress_candidate = np.sum(
                np.stack(
                    [
                        proposal.candidate_velocity
                        for proposal in bridge_progress
                    ]
                ),
                axis=0,
            )
        else:
            certified_progress_world_rows = None
            certified_progress_lower = None
            certified_progress_candidate = None
        clear_elapsed = perf_counter_ns() - start_ns
        clear_command_times.append(clear_elapsed)
        tangent_constraints.append(audit.tangent_constraints)
        tangent_fallbacks += int(not audit.tangent_converged)
        virtual_command = audit.nominal_command
        inner_dt = protocol.dt / unicycle.inner_substeps
        projection_elapsed = 0
        native_step_converged = True
        for _ in range(unicycle.inner_substeps):
            projection_start_ns = perf_counter_ns()
            _, yaw, realized_virtual, native_result = (
                _project_native_unicycle(
                    controller,
                    virtual,
                    headings,
                    scenario.arena,
                    virtual_command,
                    protocol,
                    unicycle,
                    progress_virtual_command=audit.nominal_command,
                    certified_progress_world_rows=(
                        certified_progress_world_rows
                    ),
                    certified_progress_lower=certified_progress_lower,
                    certified_progress_candidate=(
                        certified_progress_candidate
                    ),
                    initial_dual=native_dual,
                    initial_progress_dual=native_progress_dual,
                    workspace=native_workspace,
                )
            )
            elapsed = perf_counter_ns() - projection_start_ns
            projection_elapsed += elapsed
            native_projection_times.append(elapsed)
            native_projection_sweeps.append(
                native_result.sweeps
                + (
                    native_result.progress_result.sweeps
                    if native_result.progress_result is not None
                    else 0
                )
            )
            native_dual = native_result.dual.copy()
            native_progress_dual = native_result.progress_dual.copy()
            feasibility_restorations += int(
                native_result.feasibility_restored
            )
            projection_iteration_caps += int(
                native_result.sweeps
                >= unicycle.projection_max_sweeps
            )
            if native_result.progress_result is not None:
                projection_iteration_caps += int(
                    native_result.progress_result.sweeps
                    >= unicycle.projection_max_sweeps
                )
            native_step_converged &= native_result.converged
            virtual = virtual + inner_dt * realized_virtual
            headings = headings + inner_dt * yaw
            heading_vectors = np.stack(
                (np.cos(headings), np.sin(headings)),
                axis=1,
            )
            centers = virtual - unicycle.lookahead * heading_vectors
        command_times.append(clear_elapsed + projection_elapsed)
        nonconverged += int(not native_step_converged)
        distance = np.linalg.norm(centers - scenario.goals, axis=1)
        new_arrivals = np.isinf(first_arrival) & (
            distance <= protocol.arrival_radius
        )
        first_arrival[new_arrivals] = (step + 1) * protocol.dt

    final_distance = np.linalg.norm(centers - scenario.goals, axis=1)
    warm = min(max(warmup_steps, 0), len(command_times) - 1)
    return {
        "family": family,
        "n_robots": count,
        "seed": seed,
        "fingerprint": scenario.fingerprint(),
        "dynamics": "lookahead-unicycle",
        "terminal_capture": terminal_capture,
        "steps": protocol.steps,
        "command_all": _stats_ms(command_times),
        "command_after_warmup": _stats_ms(command_times[warm:]),
        "clear_command_all": _stats_ms(clear_command_times),
        "native_projection_call_all": _stats_ms(
            native_projection_times
        ),
        "mean_native_projection_sweeps": float(
            np.mean(native_projection_sweeps)
        ),
        "maximum_native_projection_sweeps": int(
            np.max(native_projection_sweeps)
        ),
        "mean_tangent_constraints": float(
            np.mean(tangent_constraints)
        ),
        "maximum_tangent_constraints": int(
            np.max(tangent_constraints)
        ),
        "nonconverged_steps": nonconverged,
        "feasibility_restoration_steps": feasibility_restorations,
        "projection_iteration_cap_calls": projection_iteration_caps,
        "tangent_fallback_steps": tangent_fallbacks,
        "final_arrival_rate": float(
            np.mean(final_distance <= protocol.arrival_radius)
        ),
        "all_robots_ever_arrived": bool(
            np.all(np.isfinite(first_arrival))
        ),
        "controller_loop_wall_s": perf_counter() - case_start,
    }


def main() -> None:
    args = parse_args()
    cases: list[tuple[str, int]] = []
    selected_families = (
        tuple(args.families)
        if args.families is not None
        else ("free", "swap", "circ15", "rect15")
    )
    for family in selected_families:
        default_counts = (
            (20, 40, 60, 80, 120)
            if family in ("free", "swap")
            else (20, 40, 60, 80)
        )
        counts = (
            tuple(args.robots)
            if args.robots is not None
            else default_counts
        )
        cases.extend((family, count) for count in counts)

    records: list[dict] = []
    for family, count in cases:
        for seed in args.seeds:
            record = run_case(
                family,
                count,
                seed,
                args.horizon,
                args.warmup_steps,
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        "family": family,
                        "n_robots": count,
                        "seed": seed,
                        "mean_ms": record["command_after_warmup"][
                            "mean_ms"
                        ],
                        "p95_ms": record["command_after_warmup"][
                            "p95_ms"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
        },
        "measurement_scope": (
            "CLEARController.command plus three bounded native-unicycle CBF "
            "projections per 33.3 Hz step; excludes scenario/planner/target "
            "lookup, state integration, and swept-distance audits"
        ),
        "protocol": asdict(Protocol(horizon=args.horizon)),
        "unicycle": asdict(
            UnicycleConfig(
                lookahead=0.05,
                yaw_rate_limit=np.pi / 2.0,
                inner_substeps=3,
            )
        ),
        "warmup_steps": args.warmup_steps,
        "records": records,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
