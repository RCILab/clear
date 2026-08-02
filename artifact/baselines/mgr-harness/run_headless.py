"""Headless runner for the public Merry-Go-Round implementation.

The controller and simulator are imported from the upstream checkout.  This
file only replaces the Qt timer/render/export layer in visualization.py with a
tight loop and adds evaluator-side safety measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or "unreported"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("MGR_REPO", "/mgr")),
        help="Path to the official MGR checkout (default: /mgr).",
    )
    parser.add_argument(
        "--instance",
        type=Path,
        action="append",
        required=True,
        help="Instance YAML path, relative to --repo or absolute; repeatable.",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument(
        "--arrival-threshold",
        type=float,
        help="Override the official radius/2 arrival threshold.",
    )
    parser.add_argument(
        "--safe-ratio",
        type=float,
        default=1.1,
        help="Upstream SAFE_RATIO used for declared safety margins.",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1.0e-9,
        help="Evaluator tolerance for geometric collision/margin predicates.",
    )
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--env-scale", type=float, default=16.0)
    parser.add_argument("--sim-scale", type=int, default=600)
    parser.add_argument(
        "--smg-resource",
        choices=("none", "doorway", "intersection"),
        default="none",
    )
    parser.add_argument("--smg-resource-width", type=float, default=1.2)
    parser.add_argument("--smg-doorway-thickness", type=float, default=0.8)
    parser.add_argument(
        "--termination-mode",
        choices=("official", "mission", "horizon"),
        default="official",
        help=(
            "official stops on all-arrived or elapsed>timeout; horizon runs "
            "exactly round(timeout/dt) steps and evaluates the final state; "
            "mission uses exact end-of-step labels and stops on all-arrived "
            "or the common finite horizon."
        ),
    )
    parser.add_argument(
        "--audit-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use conservative lower bounds to skip swept obstacle-distance "
            "calculations that cannot improve the running minimum. This "
            "changes evaluator cost only, not the imported MGR controller."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSONL output path. Parent directories are created.",
    )
    parser.add_argument(
        "--trace-npz",
        type=Path,
        help=(
            "Optional compressed state/control trace for one-instance "
            "optimization-equivalence audits."
        ),
    )
    parser.add_argument(
        "--trace-digest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record SHA-256 digests of every state and command sample.",
    )
    parser.add_argument(
        "--timing-warmup-steps",
        type=int,
        default=40,
        help="Initial controller iterations excluded from mean/p95 timing.",
    )
    parser.add_argument(
        "--timing-v2-dir",
        type=Path,
        help="Optional directory for architecture-aware raw NPZ samples.",
    )
    return parser.parse_args()


def configure_imports(repo: Path):
    repo = repo.resolve()
    if not (repo / "main.py").is_file():
        raise FileNotFoundError(f"Not an MGR checkout: {repo}")
    sys.path.insert(0, str(repo))

    from PyQt5.QtCore import QPointF
    from config import detect_env_type, load_config, scale_coordinates
    from main import build_simulator

    return repo, QPointF, detect_env_type, load_config, scale_coordinates, build_simulator


def resolve_instance(repo: Path, instance: Path) -> Path:
    path = instance if instance.is_absolute() else repo / instance
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Instance not found: {path}")
    return path


def pair_metrics(agents, safe_ratio: float) -> tuple[float, float, float]:
    if len(agents) < 2:
        return math.inf, math.inf, math.inf
    positions = np.asarray([[a.position.x(), a.position.y()] for a in agents])
    radii = np.asarray([a.radius for a in agents])
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    upper = np.triu_indices(len(agents), 1)
    pair_distances = distances[upper]
    pair_clearances = pair_distances - (radii[:, None] + radii[None, :])[upper]
    pair_safety_margins = pair_distances - (
        safe_ratio * (radii[:, None] + radii[None, :])
    )[upper]
    return (
        float(pair_distances.min()),
        float(pair_clearances.min()),
        float(pair_safety_margins.min()),
    )


def swept_pair_metrics(
    start: np.ndarray,
    end: np.ndarray,
    radii: np.ndarray,
    safe_ratio: float,
) -> tuple[float, float, float]:
    if len(start) < 2:
        return math.inf, math.inf, math.inf
    relative_start = start[:, None, :] - start[None, :, :]
    displacement = (end - start)
    relative_displacement = (
        displacement[:, None, :] - displacement[None, :, :]
    )
    denominator = np.sum(relative_displacement**2, axis=-1)
    numerator = -np.sum(relative_start * relative_displacement, axis=-1)
    fraction = np.zeros_like(denominator)
    np.divide(
        numerator,
        denominator,
        out=fraction,
        where=denominator > 1.0e-24,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = (
        relative_start + fraction[:, :, None] * relative_displacement
    )
    distances = np.linalg.norm(closest, axis=-1)
    upper = np.triu_indices(len(start), 1)
    pair_distances = distances[upper]
    radius_sums = (radii[:, None] + radii[None, :])[upper]
    return (
        float(pair_distances.min()),
        float((pair_distances - radius_sums).min()),
        float((pair_distances - safe_ratio * radius_sums).min()),
    )


def signed_rect_point_distance(x: float, y: float, rect) -> float:
    left = rect.x()
    right = rect.x() + rect.width()
    top = rect.y()
    bottom = rect.y() + rect.height()
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    if dx > 0.0 or dy > 0.0:
        return math.hypot(dx, dy)
    return -min(x - left, right - x, y - top, bottom - y)


def point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    displacement = end - start
    denominator = float(displacement @ displacement)
    if denominator <= 1.0e-24:
        return float(np.linalg.norm(point - start))
    fraction = float((point - start) @ displacement / denominator)
    closest = start + min(max(fraction, 0.0), 1.0) * displacement
    return float(np.linalg.norm(point - closest))


def segment_intersects_rect(
    start: np.ndarray,
    end: np.ndarray,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> bool:
    """Liang-Barsky segment/AABB intersection test."""
    lower = 0.0
    upper = 1.0
    displacement = end - start
    for origin, delta, minimum, maximum in (
        (start[0], displacement[0], left, right),
        (start[1], displacement[1], top, bottom),
    ):
        if abs(delta) <= 1.0e-15:
            if origin < minimum or origin > maximum:
                return False
            continue
        first = (minimum - origin) / delta
        second = (maximum - origin) / delta
        if first > second:
            first, second = second, first
        lower = max(lower, first)
        upper = min(upper, second)
        if lower > upper:
            return False
    return True


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segments_intersect(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> bool:
    direction_a = a1 - a0
    direction_b = b1 - b0
    denominator = cross2d(direction_a, direction_b)
    offset = b0 - a0
    if abs(denominator) <= 1.0e-15:
        if abs(cross2d(offset, direction_a)) > 1.0e-12:
            return False
        axis = int(np.argmax(np.abs(direction_a)))
        if abs(direction_a[axis]) <= 1.0e-15:
            return float(np.linalg.norm(a0 - b0)) <= 1.0e-12
        a_min, a_max = sorted((a0[axis], a1[axis]))
        b_min, b_max = sorted((b0[axis], b1[axis]))
        return max(a_min, b_min) <= min(a_max, b_max) + 1.0e-12
    fraction_a = cross2d(offset, direction_b) / denominator
    fraction_b = cross2d(offset, direction_a) / denominator
    return (
        -1.0e-12 <= fraction_a <= 1.0 + 1.0e-12
        and -1.0e-12 <= fraction_b <= 1.0 + 1.0e-12
    )


def segment_segment_distance(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> float:
    if segments_intersect(a0, a1, b0, b1):
        return 0.0
    return min(
        point_segment_distance(a0, b0, b1),
        point_segment_distance(a1, b0, b1),
        point_segment_distance(b0, a0, a1),
        point_segment_distance(b1, a0, a1),
    )


def segment_rect_distance(
    start: np.ndarray, end: np.ndarray, rect
) -> float:
    """Exact minimum unsigned distance from a segment to an axis-aligned box.

    The squared point-to-box distance is a convex piecewise quadratic in the
    segment parameter.  Box-face crossing times partition [0, 1] into regions
    with fixed active coordinates; each quadratic is minimized in closed form.
    """
    left = rect.x()
    right = rect.x() + rect.width()
    top = rect.y()
    bottom = rect.y() + rect.height()
    displacement = end - start
    breakpoints = [0.0, 1.0]
    for axis, bounds in enumerate(((left, right), (top, bottom))):
        if abs(displacement[axis]) <= 1.0e-15:
            continue
        for boundary in bounds:
            fraction = (
                boundary - start[axis]
            ) / displacement[axis]
            if 0.0 < fraction < 1.0:
                breakpoints.append(float(fraction))
    breakpoints = sorted(set(breakpoints))

    def squared_distance(fraction: float) -> float:
        point = start + fraction * displacement
        dx = max(left - point[0], 0.0, point[0] - right)
        dy = max(top - point[1], 0.0, point[1] - bottom)
        return float(dx * dx + dy * dy)

    minimum_squared = min(squared_distance(t) for t in breakpoints)
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        midpoint = 0.5 * (lower + upper)
        point = start + midpoint * displacement
        active_slopes = []
        active_offsets = []
        for axis, bounds in enumerate(((left, right), (top, bottom))):
            if point[axis] < bounds[0]:
                active_slopes.append(displacement[axis])
                active_offsets.append(start[axis] - bounds[0])
            elif point[axis] > bounds[1]:
                active_slopes.append(displacement[axis])
                active_offsets.append(start[axis] - bounds[1])
        if not active_slopes:
            return 0.0
        slopes = np.asarray(active_slopes)
        offsets = np.asarray(active_offsets)
        denominator = float(slopes @ slopes)
        if denominator > 1.0e-24:
            optimum = -float(slopes @ offsets) / denominator
            optimum = min(max(optimum, lower), upper)
            minimum_squared = min(
                minimum_squared, squared_distance(optimum)
            )
    return math.sqrt(max(minimum_squared, 0.0))


def obstacle_metrics(
    sim, safe_ratio: float
) -> tuple[float | None, float | None, float, float]:
    min_obstacle_clearance = math.inf
    min_obstacle_safety_margin = math.inf
    min_workspace_clearance = math.inf
    min_workspace_safety_margin = math.inf
    world_width = sim.sim_width / sim.scale
    world_height = sim.sim_height / sim.scale

    for agent in sim.agents:
        x = agent.position.x()
        y = agent.position.y()
        r = agent.radius
        min_workspace_clearance = min(
            min_workspace_clearance,
            x - r,
            world_width - x - r,
            y - r,
            world_height - y - r,
        )
        safe_r = safe_ratio * r
        min_workspace_safety_margin = min(
            min_workspace_safety_margin,
            x - safe_r,
            world_width - x - safe_r,
            y - safe_r,
            world_height - y - safe_r,
        )
        for obstacle in sim.static_obstacles:
            if obstacle.type == "circ":
                center_distance = math.hypot(
                    x - obstacle.center.x(), y - obstacle.center.y()
                )
                clearance = center_distance - obstacle.radius - r
                safety_margin = center_distance - obstacle.radius - safe_r
            else:
                point_distance = signed_rect_point_distance(
                    x, y, obstacle.rect
                )
                clearance = point_distance - r
                safety_margin = point_distance - safe_r
            min_obstacle_clearance = min(min_obstacle_clearance, clearance)
            min_obstacle_safety_margin = min(
                min_obstacle_safety_margin, safety_margin
            )

    obstacle_value = (
        None if math.isinf(min_obstacle_clearance) else min_obstacle_clearance
    )
    obstacle_safety_value = (
        None
        if math.isinf(min_obstacle_safety_margin)
        else min_obstacle_safety_margin
    )
    return (
        obstacle_value,
        obstacle_safety_value,
        min_workspace_clearance,
        min_workspace_safety_margin,
    )


def swept_obstacle_metrics(
    sim,
    start: np.ndarray,
    end: np.ndarray,
    safe_ratio: float,
    incumbent_obstacle_clearance: float | None = None,
    incumbent_obstacle_safety_margin: float | None = None,
    prune: bool = False,
) -> tuple[float | None, float | None, float, float]:
    min_obstacle_clearance = (
        math.inf
        if incumbent_obstacle_clearance is None
        else incumbent_obstacle_clearance
    )
    min_obstacle_safety_margin = (
        math.inf
        if incumbent_obstacle_safety_margin is None
        else incumbent_obstacle_safety_margin
    )
    min_workspace_clearance = math.inf
    min_workspace_safety_margin = math.inf
    world_width = sim.sim_width / sim.scale
    world_height = sim.sim_height / sim.scale

    for index, agent in enumerate(sim.agents):
        r = agent.radius
        safe_r = safe_ratio * r
        for point in (start[index], end[index]):
            x, y = map(float, point)
            min_workspace_clearance = min(
                min_workspace_clearance,
                x - r,
                world_width - x - r,
                y - r,
                world_height - y - r,
            )
            min_workspace_safety_margin = min(
                min_workspace_safety_margin,
                x - safe_r,
                world_width - x - safe_r,
                y - safe_r,
                world_height - y - safe_r,
            )
        for obstacle in sim.static_obstacles:
            if prune:
                segment_length = math.hypot(
                    float(end[index, 0] - start[index, 0]),
                    float(end[index, 1] - start[index, 1]),
                )
                if obstacle.type == "circ":
                    start_boundary_distance = (
                        math.hypot(
                            float(start[index, 0] - obstacle.center.x()),
                            float(start[index, 1] - obstacle.center.y()),
                        )
                        - obstacle.radius
                    )
                    boundary_lower_bound = (
                        start_boundary_distance - segment_length - 1.0e-12
                    )
                else:
                    start_set_distance = max(
                        signed_rect_point_distance(
                            float(start[index, 0]),
                            float(start[index, 1]),
                            obstacle.rect,
                        ),
                        0.0,
                    )
                    boundary_lower_bound = max(
                        start_set_distance - segment_length - 1.0e-12,
                        0.0,
                    )
                cannot_improve_clearance = (
                    not math.isinf(min_obstacle_clearance)
                    and boundary_lower_bound - r
                    >= min_obstacle_clearance
                )
                cannot_improve_safety = (
                    not math.isinf(min_obstacle_safety_margin)
                    and boundary_lower_bound - safe_r
                    >= min_obstacle_safety_margin
                )
                if (
                    cannot_improve_clearance
                    and cannot_improve_safety
                ):
                    continue
            if obstacle.type == "circ":
                center = np.array(
                    [obstacle.center.x(), obstacle.center.y()]
                )
                center_distance = point_segment_distance(
                    center, start[index], end[index]
                )
                boundary_distance = center_distance - obstacle.radius
            else:
                boundary_distance = segment_rect_distance(
                    start[index], end[index], obstacle.rect
                )
            min_obstacle_clearance = min(
                min_obstacle_clearance, boundary_distance - r
            )
            min_obstacle_safety_margin = min(
                min_obstacle_safety_margin, boundary_distance - safe_r
            )

    return (
        None
        if math.isinf(min_obstacle_clearance)
        else min_obstacle_clearance,
        None
        if math.isinf(min_obstacle_safety_margin)
        else min_obstacle_safety_margin,
        min_workspace_clearance,
        min_workspace_safety_margin,
    )


def run_instance(
    instance_path: Path,
    args: argparse.Namespace,
    QPointF,
    detect_env_type,
    load_config,
    scale_coordinates,
    build_simulator,
) -> dict:
    env_type = detect_env_type(str(instance_path))
    config = load_config(str(instance_path))
    config = scale_coordinates(config, env_type, args.env_scale / 40.0)
    sim = build_simulator(
        config,
        env_type,
        args.radius,
        args.tau,
        args.env_scale,
        args.sim_scale,
    )

    agents = sim.agents
    arrival_threshold = (
        args.radius / 2.0
        if args.arrival_threshold is None
        else args.arrival_threshold
    )
    for agent in agents:
        agent.arrival_threshold = arrival_threshold
    arrival_times = [None] * len(agents)
    first_arrival_times = [None] * len(agents)
    arrived_robots = [False] * len(agents)
    total_frame_count = 0
    iteration_times = []
    max_agent_times = []
    timing_v2_batch_ms = []
    timing_v2_shared_ms = []
    timing_v2_local_ms = []
    timing_v2_critical_ms = []
    trace_states = None
    trace_commands = None
    state_digest = hashlib.sha256() if args.trace_digest else None
    command_digest = hashlib.sha256() if args.trace_digest else None
    initial_state = np.asarray(
        [
            [
                agent.position.x(),
                agent.position.y(),
                agent.heading,
            ]
            for agent in agents
        ],
        dtype=np.float64,
    )
    resource_entry_times = [None] * len(agents)
    resource_exit_times = [None] * len(agents)
    resource_axis = None
    resource_direction = None
    if args.smg_resource != "none":
        initial_positions = initial_state[:, :2]
        target_positions = np.asarray(
            [[agent.target.x(), agent.target.y()] for agent in agents],
            dtype=float,
        )
        displacement = target_positions - initial_positions
        if args.smg_resource == "doorway":
            resource_axis = np.zeros(len(agents), dtype=int)
        else:
            resource_axis = np.argmax(np.abs(displacement), axis=1)
        resource_direction = np.sign(
            displacement[np.arange(len(agents)), resource_axis]
        )
        if np.any(resource_direction == 0.0):
            raise ValueError("SMG robots must cross the shared resource")
    if state_digest is not None:
        state_digest.update(initial_state.tobytes(order="C"))
    if args.trace_npz is not None:
        trace_states = [initial_state]
        trace_commands = []
    nonfinite_command_events = 0
    nonfinite_heading_events = 0
    nonfinite_raw_position_events = 0
    workspace_clamp_events = 0
    excessive_step_displacement_events = 0
    maximum_step_displacement = 0.0
    radii = np.asarray([agent.radius for agent in agents])
    (
        min_node_pair_distance,
        min_node_pair_clearance,
        min_node_pair_safety_margin,
    ) = pair_metrics(agents, args.safe_ratio)
    min_swept_pair_distance = min_node_pair_distance
    min_swept_pair_clearance = min_node_pair_clearance
    min_swept_pair_safety_margin = min_node_pair_safety_margin
    (
        min_node_obstacle_clearance,
        min_node_obstacle_safety_margin,
        min_node_workspace_clearance,
        min_node_workspace_safety_margin,
    ) = obstacle_metrics(sim, args.safe_ratio)
    min_swept_obstacle_clearance = min_node_obstacle_clearance
    min_swept_obstacle_safety_margin = min_node_obstacle_safety_margin
    min_swept_workspace_clearance = min_node_workspace_clearance
    min_swept_workspace_safety_margin = (
        min_node_workspace_safety_margin
    )

    wall_start = time.perf_counter()
    while True:
        computation_start = time.perf_counter()
        sim.sim_time = total_frame_count * args.dt
        shared_start = time.perf_counter_ns()
        sim.update()
        shared_ns = time.perf_counter_ns() - shared_start
        start_positions = np.asarray(
            [[agent.position.x(), agent.position.y()] for agent in agents]
        )

        max_agent_time = 0.0
        local_control_ns = []
        step_commands = (
            np.empty((len(agents), 2), dtype=np.float64)
            if trace_commands is not None or command_digest is not None
            else None
        )
        for i, agent in enumerate(agents):
            agent_start = time.perf_counter()
            local_start = time.perf_counter_ns()
            agent.update_escape_status()
            v, omega = agent.decide_velocity()
            local_control_ns.append(
                time.perf_counter_ns() - local_start
            )
            if step_commands is not None:
                step_commands[i] = (v, omega)
            if not np.all(np.isfinite([v, omega])):
                nonfinite_command_events += 1

            # This is deliberately identical to MainWindow.update_simulation:
            # semi-implicit heading update, then sequential position updates.
            agent.heading += omega * args.dt
            if not math.isfinite(agent.heading):
                nonfinite_heading_events += 1
            raw_x = (
                agent.position.x()
                + v * math.cos(agent.heading) * args.dt
            )
            raw_y = (
                agent.position.y()
                + v * math.sin(agent.heading) * args.dt
            )
            if not np.all(np.isfinite([raw_x, raw_y])):
                nonfinite_raw_position_events += 1

            margin = agent.radius
            new_x = max(
                margin,
                min(raw_x, sim.sim_width / sim.scale - margin),
            )
            new_y = max(
                margin,
                min(raw_y, sim.sim_height / sim.scale - margin),
            )
            if (
                not math.isclose(new_x, raw_x, abs_tol=1.0e-15)
                or not math.isclose(new_y, raw_y, abs_tol=1.0e-15)
            ):
                workspace_clamp_events += 1
            agent.position = QPointF(new_x, new_y)
            displacement = math.hypot(
                new_x - start_positions[i, 0],
                new_y - start_positions[i, 1],
            )
            maximum_step_displacement = max(
                maximum_step_displacement, displacement
            )
            if displacement > agent.max_speed * args.dt + 1.0e-9:
                excessive_step_displacement_events += 1

            distance_to_target = math.hypot(
                agent.position.x() - agent.target.x(),
                agent.position.y() - agent.target.y(),
            )
            agent.at_target = distance_to_target <= agent.arrival_threshold
            measurement_time = (
                total_frame_count * args.dt
                if args.termination_mode == "official"
                else (total_frame_count + 1) * args.dt
            )
            if agent.at_target and first_arrival_times[i] is None:
                first_arrival_times[i] = measurement_time
            if agent.at_target and not arrived_robots[i]:
                arrival_times[i] = measurement_time
                arrived_robots[i] = True
            elif not agent.at_target:
                arrived_robots[i] = False

            max_agent_time = max(
                max_agent_time, time.perf_counter() - agent_start
            )

        iteration_times.append(time.perf_counter() - computation_start)
        max_agent_times.append(max_agent_time)
        controller_batch_ns = shared_ns + sum(local_control_ns)
        timing_v2_batch_ms.append(controller_batch_ns / 1.0e6)
        timing_v2_shared_ms.append(shared_ns / 1.0e6)
        timing_v2_local_ms.append(
            np.asarray(local_control_ns, dtype=float) / 1.0e6
        )
        timing_v2_critical_ms.append(
            (shared_ns + max(local_control_ns, default=0)) / 1.0e6
        )
        end_positions = np.asarray(
            [[agent.position.x(), agent.position.y()] for agent in agents]
        )
        if resource_axis is not None:
            measurement_time = (total_frame_count + 1) * args.dt
            centered_coordinate = (
                end_positions[np.arange(len(agents)), resource_axis]
                - 0.5 * args.env_scale
            )
            progress = resource_direction * centered_coordinate
            entry_threshold = (
                -0.5 * args.smg_doorway_thickness
                if args.smg_resource == "doorway"
                else -0.5 * args.smg_resource_width
            )
            exit_threshold = (
                0.5 * args.smg_doorway_thickness + args.radius
                if args.smg_resource == "doorway"
                else 0.5 * args.smg_resource_width + args.radius
            )
            for index in range(len(agents)):
                if (
                    resource_entry_times[index] is None
                    and progress[index] >= entry_threshold
                ):
                    resource_entry_times[index] = measurement_time
                if (
                    resource_exit_times[index] is None
                    and progress[index] >= exit_threshold
                ):
                    resource_exit_times[index] = measurement_time
        if trace_states is not None:
            state_sample = np.column_stack(
                (
                    end_positions,
                    np.asarray(
                        [agent.heading for agent in agents],
                        dtype=np.float64,
                    ),
                )
            )
            trace_states.append(state_sample)
            trace_commands.append(step_commands)
        elif state_digest is not None:
            state_sample = np.column_stack(
                (
                    end_positions,
                    np.asarray(
                        [agent.heading for agent in agents],
                        dtype=np.float64,
                    ),
                )
            )
        if state_digest is not None:
            state_digest.update(state_sample.tobytes(order="C"))
            command_digest.update(step_commands.tobytes(order="C"))

        (
            pair_distance,
            pair_clearance,
            pair_safety_margin,
        ) = pair_metrics(agents, args.safe_ratio)
        min_node_pair_distance = min(
            min_node_pair_distance, pair_distance
        )
        min_node_pair_clearance = min(
            min_node_pair_clearance, pair_clearance
        )
        min_node_pair_safety_margin = min(
            min_node_pair_safety_margin, pair_safety_margin
        )
        (
            swept_pair_distance,
            swept_pair_clearance,
            swept_pair_safety_margin,
        ) = swept_pair_metrics(
            start_positions,
            end_positions,
            radii,
            args.safe_ratio,
        )
        min_swept_pair_distance = min(
            min_swept_pair_distance, swept_pair_distance
        )
        min_swept_pair_clearance = min(
            min_swept_pair_clearance, swept_pair_clearance
        )
        min_swept_pair_safety_margin = min(
            min_swept_pair_safety_margin, swept_pair_safety_margin
        )
        (
            obstacle_clearance,
            obstacle_safety_margin,
            workspace_clearance,
            workspace_safety_margin,
        ) = obstacle_metrics(sim, args.safe_ratio)
        if obstacle_clearance is not None:
            if min_node_obstacle_clearance is None:
                min_node_obstacle_clearance = obstacle_clearance
            else:
                min_node_obstacle_clearance = min(
                    min_node_obstacle_clearance, obstacle_clearance
                )
        if obstacle_safety_margin is not None:
            if min_node_obstacle_safety_margin is None:
                min_node_obstacle_safety_margin = obstacle_safety_margin
            else:
                min_node_obstacle_safety_margin = min(
                    min_node_obstacle_safety_margin,
                    obstacle_safety_margin,
                )
        min_node_workspace_clearance = min(
            min_node_workspace_clearance, workspace_clearance
        )
        min_node_workspace_safety_margin = min(
            min_node_workspace_safety_margin, workspace_safety_margin
        )
        (
            swept_obstacle_clearance,
            swept_obstacle_safety_margin,
            swept_workspace_clearance,
            swept_workspace_safety_margin,
        ) = swept_obstacle_metrics(
            sim,
            start_positions,
            end_positions,
            args.safe_ratio,
            min_swept_obstacle_clearance,
            min_swept_obstacle_safety_margin,
            prune=args.audit_pruning,
        )
        if swept_obstacle_clearance is not None:
            if min_swept_obstacle_clearance is None:
                min_swept_obstacle_clearance = swept_obstacle_clearance
            else:
                min_swept_obstacle_clearance = min(
                    min_swept_obstacle_clearance,
                    swept_obstacle_clearance,
                )
        if swept_obstacle_safety_margin is not None:
            if min_swept_obstacle_safety_margin is None:
                min_swept_obstacle_safety_margin = (
                    swept_obstacle_safety_margin
                )
            else:
                min_swept_obstacle_safety_margin = min(
                    min_swept_obstacle_safety_margin,
                    swept_obstacle_safety_margin,
                )
        min_swept_workspace_clearance = min(
            min_swept_workspace_clearance, swept_workspace_clearance
        )
        min_swept_workspace_safety_margin = min(
            min_swept_workspace_safety_margin,
            swept_workspace_safety_margin,
        )

        if args.termination_mode == "official":
            elapsed_time = total_frame_count * args.dt
            if (
                all(agent.at_target for agent in agents)
                or elapsed_time > args.timeout
            ):
                break
        elif args.termination_mode == "mission":
            elapsed_time = (total_frame_count + 1) * args.dt
            if (
                all(agent.at_target for agent in agents)
                or total_frame_count + 1
                >= int(round(args.timeout / args.dt))
            ):
                break
        else:
            elapsed_time = (total_frame_count + 1) * args.dt
            if total_frame_count + 1 >= int(round(args.timeout / args.dt)):
                break
        total_frame_count += 1

    success = all(arrived_robots)
    arrived_count = sum(arrived_robots)
    observed_arrivals = [t for t in arrival_times if t is not None]
    observed_first_arrivals = [
        t for t in first_arrival_times if t is not None
    ]
    final_goal_distances = [
        math.hypot(
            agent.position.x() - agent.target.x(),
            agent.position.y() - agent.target.y(),
        )
        for agent in agents
    ]
    warmup = min(max(args.timing_warmup_steps, 0), len(iteration_times))
    retained_iteration_times = iteration_times[warmup:]
    if not retained_iteration_times:
        retained_iteration_times = iteration_times
    retained_batch_ms = timing_v2_batch_ms[warmup:]
    retained_shared_ms = timing_v2_shared_ms[warmup:]
    retained_local_ms = timing_v2_local_ms[warmup:]
    retained_critical_ms = timing_v2_critical_ms[warmup:]
    if not retained_batch_ms:
        retained_batch_ms = timing_v2_batch_ms
        retained_shared_ms = timing_v2_shared_ms
        retained_local_ms = timing_v2_local_ms
        retained_critical_ms = timing_v2_critical_ms

    def timing_stats(values) -> dict:
        samples = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(samples)),
            "median": float(np.median(samples)),
            "p95": float(np.percentile(samples, 95)),
            "maximum": float(np.max(samples)),
            "sample_count": int(samples.size),
        }

    retained_local_flat = np.concatenate(retained_local_ms)
    retained_local_max = np.asarray(
        [float(np.max(values)) for values in retained_local_ms]
    )
    qp_solve_count = sum(
        getattr(agent, "qp_solve_count", 0) for agent in agents
    )
    qp_failure_events = sum(
        getattr(agent, "qp_failure_count", 0) for agent in agents
    )
    qp_iteration_sum = sum(
        getattr(agent, "qp_iteration_sum", 0) for agent in agents
    )
    qp_max_iterations = max(
        (getattr(agent, "qp_max_iterations", 0) for agent in agents),
        default=0,
    )
    qp_status_counts = {}
    for agent in agents:
        for status, count in getattr(
            agent, "qp_status_counts", {}
        ).items():
            qp_status_counts[status] = (
                qp_status_counts.get(status, 0) + count
            )
    result = {
        "instance": str(instance_path),
        "environment": env_type,
        "num_agents": len(agents),
        "success": success,
        "arrival_rate": arrived_count / len(agents),
        "arrived_count": arrived_count,
        "termination_time_s": elapsed_time,
        "makespan_s": max(observed_arrivals) if observed_arrivals else None,
        "first_arrival_makespan_s": (
            max(observed_first_arrivals)
            if len(observed_first_arrivals) == len(agents)
            else None
        ),
        "sum_arrival_time_s": sum(observed_arrivals),
        "arrival_times_s": arrival_times,
        "first_arrival_times_s": first_arrival_times,
        "maximum_final_goal_distance_m": max(final_goal_distances),
        "dt_s": args.dt,
        "timeout_s": args.timeout,
        "termination_mode": args.termination_mode,
        "arrival_threshold_m": arrival_threshold,
        "qp_backend": "cvxopt",
        "qp_solve_count": qp_solve_count,
        "qp_failure_events": qp_failure_events,
        "qp_mean_iterations": (
            qp_iteration_sum / qp_solve_count if qp_solve_count else None
        ),
        "qp_max_iterations": qp_max_iterations,
        "qp_status_counts": qp_status_counts,
        "safe_ratio": args.safe_ratio,
        "safety_sampling": "exact linear sweep between control nodes",
        "safety_audit_pruning": bool(args.audit_pruning),
        "min_pair_distance_m": min_swept_pair_distance,
        "min_pair_physical_clearance_m": min_swept_pair_clearance,
        "min_pair_declared_safety_margin_m": (
            min_swept_pair_safety_margin
        ),
        "min_obstacle_physical_clearance_m": (
            min_swept_obstacle_clearance
        ),
        "min_obstacle_declared_safety_margin_m": (
            min_swept_obstacle_safety_margin
        ),
        "min_workspace_physical_clearance_m": (
            min_swept_workspace_clearance
        ),
        "min_workspace_safe_radius_margin_m": (
            min_swept_workspace_safety_margin
        ),
        "min_pair_node_distance_m": min_node_pair_distance,
        "min_pair_node_declared_safety_margin_m": (
            min_node_pair_safety_margin
        ),
        "min_obstacle_node_declared_safety_margin_m": (
            min_node_obstacle_safety_margin
        ),
        "min_workspace_node_safe_radius_margin_m": (
            min_node_workspace_safety_margin
        ),
        "metric_tolerance_m": args.metric_tolerance,
        "physical_collision": (
            min_swept_pair_clearance < -args.metric_tolerance
        )
        or (
            min_swept_obstacle_clearance is not None
            and min_swept_obstacle_clearance < -args.metric_tolerance
        )
        or min_swept_workspace_clearance < -args.metric_tolerance,
        "declared_safety_margin_violation": (
            min_swept_pair_safety_margin < -args.metric_tolerance
            or (
                min_swept_obstacle_safety_margin is not None
                and min_swept_obstacle_safety_margin
                < -args.metric_tolerance
            )
        ),
        "timing_warmup_steps": warmup,
        "batch_step_ms": timing_stats(retained_batch_ms),
        "shared_coordination_ms": timing_stats(retained_shared_ms),
        "local_unit_ms": {
            "per_step_max": timing_stats(retained_local_max),
            "per_call": timing_stats(retained_local_flat),
            "unit_count": timing_stats(
                [len(values) for values in retained_local_ms]
            ),
        },
        "critical_path_ms": timing_stats(retained_critical_ms),
        "controller_backend": "official-mgr-cvxopt",
        "worker_count": 1,
        "cpu_model": cpu_model(),
        "gpu_model": "none",
        "local_unit_definition": (
            "update_escape_status plus decide_velocity for one robot"
        ),
        "critical_path_definition": (
            "sim.update shared coordination plus maximum local robot control"
        ),
        "mean_iteration_wall_s": float(
            np.mean(retained_iteration_times)
        ),
        "p95_iteration_wall_s": float(
            np.quantile(retained_iteration_times, 0.95)
        ),
        "max_iteration_wall_s": max(iteration_times),
        "sum_max_agent_wall_s": sum(max_agent_times),
        "nonfinite_command_events": nonfinite_command_events,
        "nonfinite_heading_events": nonfinite_heading_events,
        "nonfinite_raw_position_events": nonfinite_raw_position_events,
        "workspace_clamp_events": workspace_clamp_events,
        "excessive_step_displacement_events": (
            excessive_step_displacement_events
        ),
        "maximum_step_displacement_m": maximum_step_displacement,
        "wall_time_s": time.perf_counter() - wall_start,
        "steps": len(iteration_times),
    }
    if args.timing_v2_dir is not None:
        args.timing_v2_dir.mkdir(parents=True, exist_ok=True)
        sample_id = hashlib.sha256(instance_path.read_bytes()).hexdigest()[:16]
        sample_path = (
            args.timing_v2_dir
            / f"mgr_{instance_path.stem}_{sample_id}.npz"
        )
        local_matrix = np.stack(retained_local_ms)
        np.savez_compressed(
            sample_path,
            batch_step_ms=np.asarray(
                retained_batch_ms,
                dtype=np.float64,
            ),
            shared_coordination_ms=np.asarray(
                retained_shared_ms,
                dtype=np.float64,
            ),
            local_unit_ms=local_matrix.astype(np.float64),
            critical_path_ms=np.asarray(
                retained_critical_ms,
                dtype=np.float64,
            ),
        )
        result["timing_v2_samples_file"] = str(sample_path)
        result["timing_v2_samples_sha256"] = hashlib.sha256(
            sample_path.read_bytes()
        ).hexdigest()
    if args.trace_npz is not None:
        args.trace_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace_npz,
            states=np.asarray(trace_states, dtype=np.float64),
            commands=np.asarray(trace_commands, dtype=np.float64),
        )
        result["trace_npz"] = str(args.trace_npz)
    if state_digest is not None:
        result["state_trace_sha256"] = state_digest.hexdigest()
        result["command_trace_sha256"] = command_digest.hexdigest()
    if args.smg_resource != "none":
        finite_entries = [
            value for value in resource_entry_times if value is not None
        ]
        finite_exits = [
            value for value in resource_exit_times if value is not None
        ]
        completed = len(finite_exits)
        throughput = 0.0
        flow_rate = 0.0
        if finite_entries and completed == len(agents):
            interval = max(
                max(finite_exits) - min(finite_entries),
                args.dt,
            )
            throughput = len(agents) / interval
            flow_rate = throughput / args.smg_resource_width
        result.update(
            {
                "gap_completed_robots": completed,
                "gap_entered_robots": len(finite_entries),
                "gap_entry_start_s": (
                    min(finite_entries) if finite_entries else None
                ),
                "gap_exit_end_s": (
                    max(finite_exits) if finite_exits else None
                ),
                "parallel_throughput_robots_per_s": throughput,
                "smg_flow_rate_robots_per_m_s": flow_rate,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.trace_npz is not None and len(args.instance) != 1:
        raise ValueError("--trace-npz requires exactly one --instance")
    (
        repo,
        QPointF,
        detect_env_type,
        load_config,
        scale_coordinates,
        build_simulator,
    ) = configure_imports(args.repo)

    output_handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")

    try:
        for raw_instance in args.instance:
            instance_path = resolve_instance(repo, raw_instance)
            result = run_instance(
                instance_path,
                args,
                QPointF,
                detect_env_type,
                load_config,
                scale_coordinates,
                build_simulator,
            )
            record = json.dumps(result, ensure_ascii=False, sort_keys=True)
            print(record, flush=True)
            if output_handle is not None:
                output_handle.write(record + "\n")
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    main()
