"""SMG flow and counterfactual delay metrics."""

from __future__ import annotations

import numpy as np

from .unicycle import UnicycleRollout

Array = np.ndarray


def doorway_flow_from_trajectory(
    times: Array,
    trajectory: Array,
    starts: Array,
    goals: Array,
    *,
    body_radius: float,
    dt: float,
    doorway_width: float,
    doorway_thickness: float,
) -> dict:
    """Compute the SMG entry-to-exit throughput convention.

    ``t_start`` is the first entry into the doorway resource. ``t_end`` is the
    last assigned robot to clear the far face.  The width-normalized flow rate
    follows SMGLib; throughput is also reported in robots/s for readability.
    """
    times = np.asarray(times, dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    count = len(starts)
    direction = np.sign(goals[:, 0] - starts[:, 0])
    if np.any(direction == 0.0):
        raise ValueError("doorway agents must cross the x=0 bottleneck")
    near_face = -direction * 0.5 * doorway_thickness
    far_face = direction * (
        0.5 * doorway_thickness
        + body_radius
    )
    entry_times = np.full(count, np.inf)
    exit_times = np.full(count, np.inf)
    for index in range(count):
        progress = direction[index] * trajectory[:, index, 0]
        entry_threshold = direction[index] * near_face[index]
        exit_threshold = direction[index] * far_face[index]
        entered = np.flatnonzero(progress >= entry_threshold)
        exited = np.flatnonzero(progress >= exit_threshold)
        if len(entered):
            entry_times[index] = times[entered[0]]
        if len(exited):
            exit_times[index] = times[exited[0]]
    completed = np.isfinite(exit_times)
    finite_entries = entry_times[np.isfinite(entry_times)]
    finite_exits = exit_times[completed]
    if not len(finite_entries) or not np.all(completed):
        return {
            "gap_completed_robots": int(np.sum(completed)),
            "gap_entered_robots": int(np.sum(np.isfinite(entry_times))),
            "gap_entry_start_s": (
                float(np.min(finite_entries))
                if len(finite_entries)
                else None
            ),
            "gap_exit_end_s": (
                float(np.max(finite_exits)) if len(finite_exits) else None
            ),
            "parallel_throughput_robots_per_s": 0.0,
            "smg_flow_rate_robots_per_m_s": 0.0,
        }
    start_time = float(np.min(entry_times))
    end_time = float(np.max(exit_times))
    interval = max(
        end_time - start_time,
        dt,
    )
    throughput = count / interval
    return {
        "gap_completed_robots": count,
        "gap_entered_robots": count,
        "gap_entry_start_s": start_time,
        "gap_exit_end_s": end_time,
        "parallel_throughput_robots_per_s": throughput,
        "smg_flow_rate_robots_per_m_s": throughput / doorway_width,
    }


def doorway_flow_metrics(
    rollout: UnicycleRollout,
    *,
    doorway_width: float,
    doorway_thickness: float,
) -> dict:
    return doorway_flow_from_trajectory(
        rollout.times,
        rollout.trajectory,
        rollout.scenario.starts,
        rollout.scenario.goals,
        body_radius=rollout.scenario.protocol.body_radius,
        dt=rollout.scenario.protocol.dt,
        doorway_width=doorway_width,
        doorway_thickness=doorway_thickness,
    )


def intersection_flow_from_trajectory(
    times: Array,
    trajectory: Array,
    starts: Array,
    goals: Array,
    *,
    body_radius: float,
    dt: float,
    corridor_width: float,
) -> dict:
    """Measure entry-to-exit throughput through the central conflict square."""
    times = np.asarray(times, dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    count = len(starts)
    displacement = goals - starts
    axis = np.argmax(np.abs(displacement), axis=1)
    direction = np.sign(displacement[np.arange(len(displacement)), axis])
    resource_half = 0.5 * corridor_width
    entry_times = np.full(count, np.inf)
    exit_times = np.full(count, np.inf)
    for index in range(count):
        coordinate = trajectory[:, index, axis[index]]
        progress = direction[index] * coordinate
        entered = np.flatnonzero(progress >= -resource_half)
        exited = np.flatnonzero(
            progress
            >= resource_half + body_radius
        )
        if len(entered):
            entry_times[index] = times[entered[0]]
        if len(exited):
            exit_times[index] = times[exited[0]]
    completed = np.isfinite(exit_times)
    finite_entries = entry_times[np.isfinite(entry_times)]
    finite_exits = exit_times[completed]
    if not len(finite_entries) or not np.all(completed):
        return {
            "gap_completed_robots": int(np.sum(completed)),
            "gap_entered_robots": int(np.sum(np.isfinite(entry_times))),
            "gap_entry_start_s": (
                float(np.min(finite_entries))
                if len(finite_entries)
                else None
            ),
            "gap_exit_end_s": (
                float(np.max(finite_exits)) if len(finite_exits) else None
            ),
            "parallel_throughput_robots_per_s": 0.0,
            "smg_flow_rate_robots_per_m_s": 0.0,
        }
    start_time = float(np.min(entry_times))
    end_time = float(np.max(exit_times))
    interval = max(
        end_time - start_time,
        dt,
    )
    throughput = count / interval
    return {
        "gap_completed_robots": count,
        "gap_entered_robots": count,
        "gap_entry_start_s": start_time,
        "gap_exit_end_s": end_time,
        "parallel_throughput_robots_per_s": throughput,
        "smg_flow_rate_robots_per_m_s": throughput / corridor_width,
    }


def intersection_flow_metrics(
    rollout: UnicycleRollout,
    *,
    corridor_width: float,
) -> dict:
    return intersection_flow_from_trajectory(
        rollout.times,
        rollout.trajectory,
        rollout.scenario.starts,
        rollout.scenario.goals,
        body_radius=rollout.scenario.protocol.body_radius,
        dt=rollout.scenario.protocol.dt,
        corridor_width=corridor_width,
    )


def interference_delay_metrics(
    multi_agent_times: Array,
    isolated_times: Array,
) -> dict:
    """Compare each TTG against the same method navigating alone."""
    multi = np.asarray(multi_agent_times, dtype=float)
    solo = np.asarray(isolated_times, dtype=float)
    if multi.shape != solo.shape:
        raise ValueError("multi-agent and isolated TTGs must have equal shape")
    valid = np.isfinite(multi) & np.isfinite(solo) & (solo > 0.0)
    if not np.any(valid):
        return {
            "delay_evaluated_robots": 0,
            "average_interference_delay_s": None,
            "p95_interference_delay_s": None,
            "normalized_average_interference_delay": None,
        }
    delay = multi[valid] - solo[valid]
    normalized = delay / solo[valid]
    return {
        "delay_evaluated_robots": int(np.sum(valid)),
        "average_interference_delay_s": float(np.mean(delay)),
        "p95_interference_delay_s": float(np.quantile(delay, 0.95)),
        "normalized_average_interference_delay": float(
            np.mean(normalized)
        ),
    }
