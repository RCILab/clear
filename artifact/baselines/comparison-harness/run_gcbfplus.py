"""Evaluate the public GCBF+ DubinsCar policy on canonical scenarios.

The official checkpoint uses a 4 m workspace and 0.05 m car radius.  The
canonical 16 m geometry is therefore represented at a 1:4 spatial scale,
which preserves every dimensionless clearance and sensing-radius ratio.
Physical speed and yaw limits are imposed after this coordinate conversion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from math import pi
from pathlib import Path
from time import perf_counter_ns

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from gcbfplus.algo import make_algo
from train_gcbfplus_common import CommonBoundedDubinsCar

from clear_nav import (
    Protocol,
    SMGGeometry,
    doorway_flow_from_trajectory,
    interference_delay_metrics,
    intersection_flow_from_trajectory,
    make_smg_scenario,
)
from clear_nav.geometry import Rectangle as ClearRectangle
from clear_nav.guidance import CostFieldGuidance, GridPlanner
from clear_nav.simulator import (
    _minimum_obstacle_clearance,
    _swept_pair_distance,
)
from clear_nav.unicycle import UnicycleConfig
from clear_nav.scenarios import Scenario
from run_unicycle import make_unicycle_scenario
from timing_v2_common import aggregate_record, save_samples


SPATIAL_SCALE = 4.0
EGO_ACTION_PARITY_TOLERANCE = 5.0e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=(
            "free",
            "swap",
            "rect15",
            "doorway08",
            "doorway12",
            "intersection24",
        ),
        default=("free", "swap", "rect15"),
    )
    parser.add_argument("--robots", nargs="+", type=int, default=(20,))
    parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--yaw-rate-limit", type=float, default=pi / 2.0)
    parser.add_argument("--n-rays", type=int)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/opt/gcbfplus/pretrained/DubinsCar/gcbf+"),
    )
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    parser.add_argument(
        "--isolated-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ego-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Time one induced sensing graph per robot and validate its ego "
            "action against the official batched graph."
        ),
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


def _scaled_point(point: np.ndarray, protocol: Protocol) -> np.ndarray:
    return (
        np.asarray(point, dtype=float) + protocol.half_width
    ) / SPATIAL_SCALE


def _obstacle_arrays(scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    widths: list[float] = []
    heights: list[float] = []
    angles: list[float] = []
    for obstacle in scenario.arena.obstacles:
        if not isinstance(obstacle, ClearRectangle):
            raise ValueError(
                "the official GCBF+ DubinsCar environment supports rectangle "
                "obstacles; Circ15 is intentionally not approximated"
            )
        centers.append(_scaled_point(obstacle.center, scenario.protocol))
        widths.append(float(obstacle.size[0] / SPATIAL_SCALE))
        heights.append(float(obstacle.size[1] / SPATIAL_SCALE))
        angles.append(0.0)

    # Make the common closed workspace observable to the learned controller.
    half = scenario.protocol.half_width
    thickness = 2.0
    span = 2.0 * half + 4.0
    walls = (
        (np.array([-half - thickness / 2.0, 0.0]), (thickness, span)),
        (np.array([half + thickness / 2.0, 0.0]), (thickness, span)),
        (np.array([0.0, -half - thickness / 2.0]), (span, thickness)),
        (np.array([0.0, half + thickness / 2.0]), (span, thickness)),
    )
    for center, size in walls:
        centers.append(_scaled_point(center, scenario.protocol))
        widths.append(size[0] / SPATIAL_SCALE)
        heights.append(size[1] / SPATIAL_SCALE)
        angles.append(0.0)
    return (
        np.asarray(centers),
        np.asarray(widths),
        np.asarray(heights),
        np.asarray(angles),
    )


def _load_policy(env: CommonBoundedDubinsCar, checkpoint: Path, step: int):
    with (checkpoint / "config.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.UnsafeLoader)

    def cfg(name: str, default):
        return getattr(config, name, default)

    algo = make_algo(
        algo=cfg("algo", "gcbf+"),
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=cfg("gnn_layers", 1),
        batch_size=cfg("batch_size", 256),
        buffer_size=cfg("buffer_size", 512),
        horizon=cfg("horizon", 32),
        lr_actor=cfg("lr_actor", 3e-5),
        lr_cbf=cfg("lr_cbf", 3e-5),
        alpha=cfg("alpha", 1.0),
        eps=0.02,
        inner_epoch=8,
        loss_action_coef=cfg("loss_action_coef", 1e-4),
        loss_unsafe_coef=cfg("loss_unsafe_coef", 1.0),
        loss_safe_coef=cfg("loss_safe_coef", 1.0),
        loss_h_dot_coef=cfg("loss_h_dot_coef", 0.01),
        max_grad_norm=2.0,
        seed=cfg("seed", 0),
    )
    algo.load(str(checkpoint / "models"), step)
    return jax.jit(algo.act)


def _minimum_pair_distance(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("inf")
    left, right = np.triu_indices(len(positions), k=1)
    return float(
        np.min(np.linalg.norm(positions[left] - positions[right], axis=1))
    )


def _physical_positions(states: np.ndarray, protocol: Protocol) -> np.ndarray:
    return states[:, :2] * SPATIAL_SCALE - protocol.half_width


def _make_controller(
    env: CommonBoundedDubinsCar,
    act_fn,
):
    def controller_fn(
        agent_states,
        target_states,
        obstacle_centers,
        obstacle_widths,
        obstacle_heights,
        obstacle_angles,
    ):
        obstacles = env.create_obstacles(
            obstacle_centers,
            obstacle_widths,
            obstacle_heights,
            obstacle_angles,
        )
        state = env.EnvState(agent_states, target_states, obstacles)
        return act_fn(env.get_graph(state))

    return jax.jit(controller_fn)


def _make_case_scenario(
    family: str,
    count: int,
    seed: int,
    protocol: Protocol,
) -> tuple[Scenario, SMGGeometry | None]:
    if family == "doorway08":
        geometry = SMGGeometry(doorway_width=0.8)
        return (
            make_smg_scenario(
                "doorway", count, seed, protocol, geometry
            ),
            geometry,
        )
    if family == "doorway12":
        geometry = SMGGeometry(doorway_width=1.2)
        return (
            make_smg_scenario(
                "doorway", count, seed, protocol, geometry
            ),
            geometry,
        )
    if family == "intersection24":
        geometry = SMGGeometry(intersection_corridor_width=2.4)
        return (
            make_smg_scenario(
                "intersection", count, seed, protocol, geometry
            ),
            geometry,
        )
    unicycle = UnicycleConfig()
    return (
        make_unicycle_scenario(
            family, count, seed, protocol, unicycle
        ),
        None,
    )


def _geometry_for_family(family: str) -> SMGGeometry | None:
    if family == "doorway08":
        return SMGGeometry(doorway_width=0.8)
    if family == "doorway12":
        return SMGGeometry(doorway_width=1.2)
    if family == "intersection24":
        return SMGGeometry(intersection_corridor_width=2.4)
    return None


def run_case(
    args: argparse.Namespace,
    family: str,
    count: int,
    seed: int,
    n_rays: int,
    policy_provider,
    controller_cache: dict,
    *,
    scenario_override: Scenario | None = None,
) -> dict:
    protocol = Protocol(dt=args.dt, horizon=args.horizon)
    if scenario_override is None:
        scenario, geometry = _make_case_scenario(
            family, count, seed, protocol
        )
    else:
        scenario = scenario_override
        geometry = _geometry_for_family(family)
    centers, widths, heights, angles = _obstacle_arrays(scenario)
    params = {
        **CommonBoundedDubinsCar.PARAMS,
        "n_obs": len(centers),
        "n_rays": n_rays,
    }
    controller_key = ("batch", count, len(centers))
    if controller_key not in controller_cache:
        env = CommonBoundedDubinsCar(
            num_agents=count,
            area_size=protocol.workspace_size / SPATIAL_SCALE,
            max_step=protocol.steps,
            dt=args.dt,
            params=params,
        )
        controller_cache[controller_key] = _make_controller(
            env,
            policy_provider(count),
        )
    controller_fn = controller_cache[controller_key]
    obstacle_data = tuple(
        jax.device_put(jnp.asarray(value))
        for value in (centers, widths, heights, angles)
    )
    starts = np.asarray(
        [_scaled_point(point, protocol) for point in scenario.starts]
    )
    goals = np.asarray(
        [_scaled_point(point, protocol) for point in scenario.goals]
    )
    states = np.column_stack(
        (starts, np.zeros(count), np.zeros(count))
    )
    goal_states = np.column_stack(
        (
            goals,
            np.arctan2(
                goals[:, 1] - starts[:, 1],
                goals[:, 0] - starts[:, 0],
            ),
            np.zeros(count),
        )
    )
    # Compile graph construction and policy once per shape.  Compilation is
    # not a per-cycle cost, matching the upstream jitted rollout.
    jax.block_until_ready(
        controller_fn(
            jnp.asarray(states),
            jnp.asarray(goal_states),
            *obstacle_data,
        )
    )

    planning_start = perf_counter_ns()
    guidance: CostFieldGuidance | None = None
    if scenario.arena.obstacles:
        guidance = GridPlanner(
            scenario.arena, protocol
        ).cost_field_plan(scenario.goals)
    planning_ns = perf_counter_ns() - planning_start

    positions = np.asarray(scenario.starts, dtype=float).copy()
    recorded_times = [0.0]
    recorded_positions = [positions.copy()]
    recorded_headings = [np.asarray(states[:, 2]).copy()]
    first_arrival = np.full(count, np.inf)
    first_arrival[
        np.linalg.norm(positions - scenario.goals, axis=1)
        <= protocol.arrival_radius
    ] = 0.0
    min_pair = (
        _minimum_pair_distance(positions)
        if args.safety_audit
        else float("inf")
    )
    min_obstacle = (
        _minimum_obstacle_clearance(scenario, positions)
        if args.safety_audit
        else float("inf")
    )
    maximum_linear = 0.0
    maximum_yaw = 0.0
    path_length = np.zeros(count)
    control_ns: list[int] = []
    timing_local_ms: list[list[float]] = []
    ego_action_max_error = 0.0
    ego_action_violations = 0
    scaled_speed_limit = protocol.speed_limit / SPATIAL_SCALE
    scaled_arrival = protocol.arrival_radius / SPATIAL_SCALE
    action_yaw_limit = args.yaw_rate_limit / 20.0
    executed_steps = 0

    for step in range(protocol.steps):
        if guidance is not None:
            physical_targets = guidance.targets(positions)
            control_goals = np.asarray(
                [_scaled_point(value, protocol) for value in physical_targets]
            )
        else:
            control_goals = goals
        goal_states = np.column_stack(
            (
                control_goals,
                np.arctan2(
                    control_goals[:, 1] - states[:, 1],
                    control_goals[:, 0] - states[:, 0],
                ),
                np.zeros(count),
            )
        )
        control_start = perf_counter_ns()
        action_device = controller_fn(
            jnp.asarray(states),
            jnp.asarray(goal_states),
            *obstacle_data,
        )
        jax.block_until_ready(action_device)
        action = np.asarray(action_device).copy()
        control_ns.append(perf_counter_ns() - control_start)
        raw_batch_action = action.copy()
        local_step_ms: list[float] = []
        if args.ego_timing:
            comm_radius = float(params["comm_radius"])
            state_distance = np.linalg.norm(
                states[:, None, :2] - states[None, :, :2],
                axis=-1,
            )
            for robot in range(count):
                neighbors = np.flatnonzero(
                    state_distance[robot] < comm_radius
                )
                neighbors = np.concatenate(
                    (
                        np.array([robot], dtype=int),
                        neighbors[neighbors != robot],
                    )
                )
                local_count = len(neighbors)
                local_key = (
                    "ego",
                    local_count,
                    len(centers),
                )
                if local_key not in controller_cache:
                    local_env = CommonBoundedDubinsCar(
                        num_agents=local_count,
                        area_size=(
                            protocol.workspace_size / SPATIAL_SCALE
                        ),
                        max_step=protocol.steps,
                        dt=args.dt,
                        params=params,
                    )
                    local_controller = _make_controller(
                        local_env,
                        policy_provider(local_count),
                    )
                    controller_cache[local_key] = local_controller
                    jax.block_until_ready(
                        local_controller(
                            jnp.asarray(states[neighbors]),
                            jnp.asarray(goal_states[neighbors]),
                            *obstacle_data,
                        )
                    )
                local_controller = controller_cache[local_key]
                local_started = perf_counter_ns()
                local_action_device = local_controller(
                    jnp.asarray(states[neighbors]),
                    jnp.asarray(goal_states[neighbors]),
                    *obstacle_data,
                )
                jax.block_until_ready(local_action_device)
                local_step_ms.append(
                    (perf_counter_ns() - local_started) / 1.0e6
                )
                ego_action = np.asarray(local_action_device)[0]
                error = float(
                    np.max(
                        np.abs(
                            ego_action - raw_batch_action[robot]
                        )
                    )
                )
                ego_action_max_error = max(
                    ego_action_max_error,
                    error,
                )
                # Induced graphs change float32 segment-reduction order even
                # with the same one-hop messages.  The observed command
                # difference is bounded against this explicit tolerance.
                ego_action_violations += int(
                    error > EGO_ACTION_PARITY_TOLERANCE
                )
        else:
            local_step_ms.append(control_ns[-1] / 1.0e6)
        timing_local_ms.append(local_step_ms)
        action[:, 0] = np.clip(
            action[:, 0], -action_yaw_limit, action_yaw_limit
        )
        action[:, 1] = np.clip(action[:, 1], -3.0, 3.0)
        stop = (
            np.linalg.norm(positions - scenario.goals, axis=1)
            <= protocol.arrival_radius
        )
        old_states = states.copy()
        heading = old_states[:, 2]
        physical_velocity = (
            SPATIAL_SCALE
            * old_states[:, 3, None]
            * np.stack((np.cos(heading), np.sin(heading)), axis=1)
        )
        new_states = old_states.copy()
        new_states[:, 0] += (
            np.cos(heading) * old_states[:, 3] * args.dt
        ) * (~stop)
        new_states[:, 1] += (
            np.sin(heading) * old_states[:, 3] * args.dt
        ) * (~stop)
        yaw = np.clip(
            action[:, 0] * 20.0,
            -args.yaw_rate_limit,
            args.yaw_rate_limit,
        )
        new_states[:, 2] += yaw * args.dt * (~stop)
        new_states[:, 3] += action[:, 1] * args.dt * (~stop)
        new_states[:, 3] = np.clip(
            new_states[:, 3], -scaled_speed_limit, scaled_speed_limit
        )
        new_states[stop, 3] = 0.0
        new_positions = _physical_positions(new_states, protocol)

        if args.safety_audit:
            min_pair = min(
                min_pair,
                _swept_pair_distance(
                    positions, physical_velocity, args.dt
                ),
            )
            min_obstacle = min(
                min_obstacle,
                _minimum_obstacle_clearance(scenario, new_positions),
                _minimum_obstacle_clearance(
                    scenario, 0.5 * (positions + new_positions)
                ),
            )
        path_length += np.linalg.norm(new_positions - positions, axis=1)
        positions = new_positions
        states = new_states
        recorded_times.append((step + 1) * args.dt)
        recorded_positions.append(positions.copy())
        recorded_headings.append(np.asarray(states[:, 2]).copy())
        maximum_linear = max(
            maximum_linear,
            float(np.max(np.abs(old_states[:, 3] * SPATIAL_SCALE))),
        )
        maximum_yaw = max(maximum_yaw, float(np.max(np.abs(yaw))))

        goal_distance = np.linalg.norm(positions - scenario.goals, axis=1)
        newly_arrived = np.isinf(first_arrival) & (
            goal_distance <= protocol.arrival_radius
        )
        first_arrival[newly_arrived] = (step + 1) * args.dt
        executed_steps = step + 1
        if np.all(goal_distance <= protocol.arrival_radius):
            break

    final_distance = np.linalg.norm(positions - scenario.goals, axis=1)
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
    samples_ms = np.asarray(control_ns, dtype=float) / 1.0e6
    warmup = min(
        max(args.timing_warmup_steps, 0),
        max(0, len(samples_ms) - 1),
    )
    retained_batch = samples_ms[warmup:].tolist()
    retained_local = timing_local_ms[warmup:]
    retained_shared = [0.0] * len(retained_batch)
    retained_critical = [
        max(values, default=0.0) for values in retained_local
    ]
    sample_path = args.timing_v2_dir / (
        f"gcbfplus_{scenario.family}_n{count}_s{seed}_"
        f"{jax.default_backend()}.npz"
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
            f"public-gcbfplus-jax-{jax.default_backend()}"
        ),
        worker_count=1,
        warmup_steps=warmup,
        scenario_fingerprint=scenario.fingerprint(),
        batch_contains_local_units=False,
        gpu=(
            jax.devices()[0].device_kind
            if jax.default_backend() == "gpu"
            else "none"
        ),
    )
    final_state = np.asarray(states, dtype="<f8")
    record = {
        "algorithm": "GCBF+",
        "family": scenario.family,
        "n_robots": count,
        "seed": seed,
        "fingerprint": scenario.fingerprint(),
        "dynamics": "bounded-unicycle",
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
        "first_arrival_times_s": [
            None if not np.isfinite(value) else float(value)
            for value in first_arrival
        ],
        "minimum_pair_distance_m": (
            min_pair if args.safety_audit else None
        ),
        "minimum_obstacle_clearance_m": (
            min_obstacle if args.safety_audit else None
        ),
        "maximum_linear_speed_mps": maximum_linear,
        "maximum_yaw_rate_rps": maximum_yaw,
        "mean_path_length_m": float(np.mean(path_length)),
        "planning_time_ms": planning_ns / 1.0e6,
        "termination_time_s": executed_steps * args.dt,
        "steps": executed_steps,
        **timing_record,
        "local_unit_definition": (
            "induced one-hop sensing graph construction and synchronized "
            "policy inference for one ego robot"
            if args.ego_timing
            else "official full-graph synchronized batch inference"
        ),
        "timing_mode": (
            "batch-and-induced-ego"
            if args.ego_timing
            else "batch-only"
        ),
        "ego_action_parity_tolerance": (
            EGO_ACTION_PARITY_TOLERANCE
            if args.ego_timing
            else None
        ),
        "ego_action_parity_note": (
            "The induced graph contains the identical one-hop sensing "
            "messages, but its smaller padded shape changes float32 segment "
            "reduction order. The tolerance corresponds to at most "
            "0.001 rad/s after the policy yaw scaling and remains over two "
            "orders of magnitude below the rejected GPU induced-graph "
            "difference."
            if args.ego_timing
            else None
        ),
        "ego_action_max_abs_error": (
            ego_action_max_error if args.ego_timing else None
        ),
        "ego_action_parity_violations": (
            ego_action_violations if args.ego_timing else None
        ),
        "timing_v2_samples_file": sample_path.as_posix(),
        "timing_v2_samples_sha256": sample_sha,
        "final_state_sha256": hashlib.sha256(
            final_state.tobytes()
        ).hexdigest(),
        "control_time_mean_ms": float(np.mean(samples_ms)),
        "control_time_p95_ms": float(np.percentile(samples_ms, 95.0)),
        "control_time_max_ms": float(np.max(samples_ms)),
        "final_maximum_goal_distance_m": float(np.max(final_distance)),
        "parameters": {
            "checkpoint_step": args.checkpoint_step,
            "spatial_scale": SPATIAL_SCALE,
            "scaled_speed_limit": scaled_speed_limit,
            "scaled_arrival_radius": scaled_arrival,
            "scaled_acceleration_limit": 3.0,
            "obstacle_count_including_walls": len(centers),
            "n_rays": n_rays,
        },
    }
    if geometry is not None:
        trajectory = np.asarray(recorded_positions)
        times = np.asarray(recorded_times)
        if family.startswith("doorway"):
            record.update(
                doorway_flow_from_trajectory(
                    times,
                    trajectory,
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
                    times,
                    trajectory,
                    scenario.starts,
                    scenario.goals,
                    body_radius=protocol.body_radius,
                    dt=protocol.dt,
                    corridor_width=(
                        geometry.intersection_corridor_width
                    ),
                )
            )
        record["geometry"] = geometry.metadata(
            protocol,
            lookahead=0.05,
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
            f"gcbfplus_{trajectory_family}_n{count}_s{seed}_"
            f"{jax.default_backend()}.npz"
        )
        np.savez_compressed(
            trajectory_path,
            times=np.asarray(recorded_times, dtype=np.float64),
            positions=np.asarray(recorded_positions, dtype=np.float64),
            headings=np.asarray(recorded_headings, dtype=np.float64),
            method=np.asarray("GCBF+"),
            scenario_fingerprint=np.asarray(scenario.fingerprint()),
        )
        record["trajectory_file"] = trajectory_path.as_posix()
        record["trajectory_sha256"] = hashlib.sha256(
            trajectory_path.read_bytes()
        ).hexdigest()
    return record


def main() -> None:
    args = parse_args()
    with (args.checkpoint / "config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        checkpoint_config = yaml.load(stream, Loader=yaml.UnsafeLoader)
    n_rays = (
        args.n_rays
        if args.n_rays is not None
        else int(getattr(checkpoint_config, "n_rays", 32))
    )
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("baselines/comparison-harness/results") / (
            f"gcbfplus_{stamp}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "GCBF+",
        "source": "public GCBF+ implementation under the common protocol",
        "physical_protocol": asdict(
            Protocol(dt=args.dt, horizon=args.horizon)
        ),
        "initial_heading_rad": 0.0,
        "shared_static_guidance": "cost-field",
        "records": [],
    }
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("algorithm") != payload["algorithm"]:
            raise ValueError("existing output algorithm does not match")
        payload["records"] = existing.get("records", [])
    completed = {
        (
            str(record["family"]),
            int(record["n_robots"]),
            int(record["seed"]),
        )
        for record in payload["records"]
    }
    total = len(args.families) * len(args.robots) * len(args.seeds)
    policy_cache = {}
    controller_cache = {}

    def get_policy(count: int):
        if count not in policy_cache:
            policy_params = {
                **CommonBoundedDubinsCar.PARAMS,
                "n_obs": 0,
                "n_rays": n_rays,
            }
            policy_env = CommonBoundedDubinsCar(
                num_agents=count,
                area_size=Protocol().workspace_size / SPATIAL_SCALE,
                max_step=Protocol().steps,
                dt=args.dt,
                params=policy_params,
            )
            policy_cache[count] = _load_policy(
                policy_env,
                args.checkpoint,
                args.checkpoint_step,
            )
        return policy_cache[count]

    for family in args.families:
        for count in args.robots:
            for seed in args.seeds:
                if (family, count, seed) in completed:
                    continue
                scenario, geometry = _make_case_scenario(
                    family,
                    count,
                    seed,
                    Protocol(dt=args.dt, horizon=args.horizon),
                )
                record = run_case(
                    args,
                    family,
                    count,
                    seed,
                    n_rays,
                    get_policy,
                    controller_cache,
                    scenario_override=scenario,
                )
                if args.isolated_reference and geometry is not None:
                    solo_times = np.full(count, np.inf)
                    for robot in range(count):
                        solo = Scenario(
                            family=f"{scenario.family}_solo",
                            n_robots=1,
                            seed=seed,
                            starts=scenario.starts[
                                robot : robot + 1
                            ].copy(),
                            goals=scenario.goals[
                                robot : robot + 1
                            ].copy(),
                            arena=scenario.arena,
                            protocol=scenario.protocol,
                        )
                        solo_record = run_case(
                            args,
                            family,
                            1,
                            seed,
                            n_rays,
                            get_policy,
                            controller_cache,
                            scenario_override=solo,
                        )
                        value = solo_record[
                            "first_arrival_times_s"
                        ][0]
                        if value is not None:
                            solo_times[robot] = value
                    multi_times = np.asarray(
                        [
                            np.inf if value is None else value
                            for value in record[
                                "first_arrival_times_s"
                            ]
                        ]
                    )
                    record.update(
                        interference_delay_metrics(
                            multi_times,
                            solo_times,
                        )
                    )
                    record["isolated_ttg_s"] = [
                        None
                        if not np.isfinite(value)
                        else float(value)
                        for value in solo_times
                    ]
                payload["records"].append(record)
                completed.add((family, count, seed))
                output.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(json.dumps(record, sort_keys=True), flush=True)
                print(
                    f"[progress] {len(payload['records'])}/{total}",
                    flush=True,
                )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
