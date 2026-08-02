"""Architecture-aware CLEAR and Vanilla CBF-QP timing on live rollouts.

The measured batch is the controller command plus the three native-unicycle
projection calls in one 33.3 Hz control period.  Static planning, target
lookup, state integration, and mission audits remain outside the interval.

For component mode, every final sparse QP row induces an exact robot
connected component.  Component QPs are solved serially for reproducible
batch timing.  The reported deployment critical path treats independent
components within each inner projection as parallel, while preserving the
three sequential inner integration nodes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from clear_nav import (
    CLEARController,
    ControllerConfig,
    Protocol,
    REPORTED_CLEAR_CONTROLLER,
    VanillaCBFController,
)
from clear_nav.guidance import GridPlanner
from clear_nav.safety import OSQPProjectionWorkspace
from clear_nav.unicycle import (
    UnicycleConfig,
    _certified_bridge_progress,
    _project_native_unicycle,
    inflated_unicycle_protocol,
)
from run_unicycle import make_unicycle_scenario
from timing_v2_common import (
    aggregate_record,
    cpu_model,
    gpu_model,
    save_samples,
    write_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("clear", "vanilla-cbf-qp"),
        default=("clear", "vanilla-cbf-qp"),
    )
    parser.add_argument(
        "--projection-mode",
        choices=("component", "global"),
        default="component",
    )
    parser.add_argument("--family", default="rect15")
    parser.add_argument("--families", nargs="+")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=int,
        default=(20, 40, 60, 80),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(0, 7, 13),
    )
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "validation/timing_v2/clear_vanilla_component.json"
        ),
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("validation/timing_v2/samples"),
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _controller_config(variant: str) -> ControllerConfig:
    structured = variant == "clear"
    return replace(
        REPORTED_CLEAR_CONTROLLER,
        boundary_progress_aligned=True,
        cluster_escape_gain=0.55 if structured else 0.0,
        cluster_escape_hysteresis=structured,
        tangent_band=0.08,
        terminal_capture_hysteresis=True,
    )


def _state_digest(centers: np.ndarray, headings: np.ndarray) -> str:
    state = np.concatenate(
        (
            np.asarray(centers, dtype="<f8").reshape(-1),
            np.asarray(headings, dtype="<f8").reshape(-1),
        )
    )
    return hashlib.sha256(state.tobytes()).hexdigest()


def run_case(
    *,
    variant: str,
    projection_mode: str,
    family: str,
    count: int,
    seed: int,
    horizon: float,
    warmup_steps: int,
    samples_dir: Path,
) -> dict:
    physical = Protocol(horizon=horizon, dt=0.03)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    control = inflated_unicycle_protocol(
        physical,
        unicycle.lookahead,
    )
    scenario = make_unicycle_scenario(
        family,
        count,
        seed,
        physical,
        unicycle,
    )
    controller_type = (
        CLEARController
        if variant == "clear"
        else VanillaCBFController
    )
    controller = controller_type(
        control,
        _controller_config(variant),
    )
    controller.reset()
    centers = np.asarray(scenario.starts, dtype=float).copy()
    headings = np.zeros(count)
    heading_vectors = np.stack(
        (np.cos(headings), np.sin(headings)),
        axis=1,
    )
    virtual = centers + unicycle.lookahead * heading_vectors
    guidance = (
        GridPlanner(scenario.arena, control).cost_field_plan(
            scenario.goals
        )
        if scenario.arena.obstacles
        else None
    )
    global_workspace = OSQPProjectionWorkspace(
        max_cached_patterns=16
    )
    component_workspaces: (
        dict[tuple[int, ...], OSQPProjectionWorkspace] | None
    ) = {} if projection_mode == "component" else None
    batch_samples: list[float] = []
    shared_samples: list[float] = []
    local_samples: list[list[float]] = []
    critical_samples: list[float] = []
    component_counts: list[int] = []
    component_sizes: list[int] = []
    nonconverged_steps = 0
    restoration_calls = 0

    for step in range(physical.steps):
        control_goals = (
            guidance.targets(virtual)
            if guidance is not None
            else scenario.goals
        )
        batch_started = perf_counter_ns()
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
            physical,
            unicycle,
        )
        if bridge_progress:
            certified_rows = np.stack(
                [proposal.world_row for proposal in bridge_progress]
            )
            certified_lower = np.asarray(
                [proposal.lower_bound for proposal in bridge_progress]
            )
            certified_candidate = np.sum(
                np.stack(
                    [
                        proposal.candidate_velocity
                        for proposal in bridge_progress
                    ]
                ),
                axis=0,
            )
        else:
            certified_rows = None
            certified_lower = None
            certified_candidate = None

        all_local_ns: list[int] = []
        critical_local_ns = 0
        step_component_counts: list[int] = []
        inner_dt = physical.dt / unicycle.inner_substeps
        step_converged = True
        for _ in range(unicycle.inner_substeps):
            timing: dict = {}
            _, yaw, realized, result = _project_native_unicycle(
                controller,
                virtual,
                headings,
                scenario.arena,
                audit.nominal_command,
                physical,
                unicycle,
                progress_virtual_command=audit.nominal_command,
                certified_progress_world_rows=certified_rows,
                certified_progress_lower=certified_lower,
                certified_progress_candidate=certified_candidate,
                workspace=global_workspace,
                component_workspaces=component_workspaces,
                timing=timing,
            )
            local_ns = list(timing["local_unit_ns"])
            all_local_ns.extend(local_ns)
            critical_local_ns += max(local_ns, default=0)
            step_component_counts.append(len(local_ns))
            component_sizes.extend(timing["component_sizes"])
            step_converged &= result.converged
            restoration_calls += int(result.feasibility_restored)
            virtual = virtual + inner_dt * realized
            headings = headings + inner_dt * yaw
            heading_vectors = np.stack(
                (np.cos(headings), np.sin(headings)),
                axis=1,
            )
            centers = virtual - unicycle.lookahead * heading_vectors
        batch_ns = perf_counter_ns() - batch_started
        shared_ns = max(0, batch_ns - sum(all_local_ns))
        critical_ns = shared_ns + critical_local_ns
        if step >= warmup_steps:
            batch_samples.append(batch_ns / 1.0e6)
            shared_samples.append(shared_ns / 1.0e6)
            # One timing-v2 local unit is the serial chain of the slowest
            # independent QP at each of the three inner integration nodes.
            local_samples.append([critical_local_ns / 1.0e6])
            critical_samples.append(critical_ns / 1.0e6)
            component_counts.extend(step_component_counts)
        nonconverged_steps += int(not step_converged)

    sample_name = (
        f"{variant}_{projection_mode}_{family}_n{count}_s{seed}.npz"
    )
    sample_path = samples_dir / sample_name
    sample_sha = save_samples(
        sample_path,
        batch_step_ms=batch_samples,
        shared_coordination_ms=shared_samples,
        local_unit_ms=local_samples,
        critical_path_ms=critical_samples,
    )
    record = aggregate_record(
        batch_step_ms=batch_samples,
        shared_coordination_ms=shared_samples,
        local_unit_ms=local_samples,
        critical_path_ms=critical_samples,
        controller_backend=(
            f"numpy+scipy+osqp-{projection_mode}"
        ),
        worker_count=1,
        warmup_steps=warmup_steps,
        scenario_fingerprint=scenario.fingerprint(),
        cpu=cpu_model(),
        gpu=gpu_model(),
        algorithm=(
            "CLEAR"
            if variant == "clear"
            else "Vanilla CBF-QP"
        ),
        variant=variant,
        family=family,
        n_robots=count,
        seed=seed,
        horizon_s=horizon,
        dt_s=physical.dt,
        measured_steps=len(batch_samples),
        projection_mode=projection_mode,
        inner_substeps=unicycle.inner_substeps,
        local_unit_definition=(
            "sum across sequential inner nodes of the maximum exact "
            "connected-component QP solve at each node"
        ),
        mean_components_per_projection=(
            float(np.mean(component_counts))
            if component_counts
            else 0.0
        ),
        maximum_component_size=(
            max(component_sizes) if component_sizes else 0
        ),
        nonconverged_steps=nonconverged_steps,
        feasibility_restoration_calls=restoration_calls,
        final_state_sha256=_state_digest(centers, headings),
        final_centers=np.round(centers, 12).tolist(),
        final_headings=np.round(headings, 12).tolist(),
        samples_file=sample_path.as_posix(),
        samples_sha256=sample_sha,
    )
    return record


def _record_key(record: dict) -> tuple[str, str, int, int, str]:
    return (
        record["variant"],
        record["family"],
        int(record["n_robots"]),
        int(record["seed"]),
        record["projection_mode"],
    )


def _load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def main() -> None:
    args = parse_args()
    records = [] if args.no_resume else _load_records(args.output)
    completed = {_record_key(record) for record in records}
    metadata = {
        "measurement_scope": (
            "controller command plus three native-unicycle projections; "
            "excludes static planning, target lookup, integration, and "
            "mission audits"
        ),
        "protocol": asdict(
            Protocol(horizon=args.horizon, dt=0.03)
        ),
        "unicycle": asdict(UnicycleConfig()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "logical_cpu_count": os.cpu_count(),
            "blas_threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
    }
    families = args.families or (args.family,)
    for variant in args.variants:
        for family in families:
            for count in args.robots:
                for seed in args.seeds:
                    key = (
                        variant,
                        family,
                        count,
                        seed,
                        args.projection_mode,
                    )
                    if key in completed:
                        continue
                    record = run_case(
                        variant=variant,
                        projection_mode=args.projection_mode,
                        family=family,
                        count=count,
                        seed=seed,
                        horizon=args.horizon,
                        warmup_steps=args.warmup_steps,
                        samples_dir=args.samples_dir,
                    )
                    records.append(record)
                    records.sort(key=_record_key)
                    write_payload(args.output, records, **metadata)
                    print(
                        json.dumps(
                            {
                                "variant": variant,
                                "family": family,
                                "n_robots": count,
                                "seed": seed,
                                "batch_mean_ms": record[
                                    "batch_step_ms"
                                ]["mean"],
                                "critical_mean_ms": record[
                                    "critical_path_ms"
                                ]["mean"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
