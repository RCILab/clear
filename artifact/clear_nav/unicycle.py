"""Look-ahead-point realization of CLEAR for unicycle robots.

For center position ``x`` and heading ``theta``, the controlled point is

    y = x + ell [cos(theta), sin(theta)].

CLEAR is applied to ``y`` after inflating pair clearance by ``2*ell`` and
static-boundary clearance by ``ell``.  The default realization projects
directly in the bounded unicycle coordinates ``[v, ell*omega]``.  Pair,
boundary, linear-speed, and yaw-rate inequalities therefore enter one convex
projection, and triangle inequalities transfer virtual-point safety to the
physical centers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter_ns

import numpy as np

from .config import ControllerConfig, Protocol
from .controller import CLEARController
from .guidance import CostFieldGuidance, GridPlanner, GuidancePlan
from .scenarios import Scenario
from .safety import (
    OSQPProjectionWorkspace,
    ProjectionResult,
    maximize_halfspace_linear_osqp,
    project_halfspaces,
    project_halfspaces_osqp,
)
from .theorem_audit import audit_straight_bridge_tokens

Array = np.ndarray


@dataclass(frozen=True)
class UnicycleConfig:
    lookahead: float = 0.05
    yaw_rate_limit: float = np.pi / 2.0
    inner_substeps: int = 3
    projection_backend: str = "osqp"
    projection_max_sweeps: int = 4000
    projection_extension_sweeps: int = 0
    projection_tolerance: float = 1.0e-6
    projection_solver_tolerance: float = 1.0e-7
    projection_solver_margin: float = 0.0
    minimum_restoration_retention: float = 0.25
    hierarchical_progress: bool = False
    hqp_progress_retention: float = 0.995
    hqp_regularization: float = 1.0e-6
    hqp_minimum_direction_norm: float = 1.0e-8
    certified_bridge_progress: bool = True
    bridge_progress_retention: float = 0.90
    bridge_geometry_tolerance: float = 1.0e-8
    actuation_mode: str = "native-cbf"

    def __post_init__(self) -> None:
        if self.lookahead <= 0.0:
            raise ValueError("lookahead must be positive")
        if self.yaw_rate_limit <= 0.0:
            raise ValueError("yaw_rate_limit must be positive")
        if self.inner_substeps < 1:
            raise ValueError("inner_substeps must be positive")
        if self.projection_max_sweeps < 1:
            raise ValueError("projection_max_sweeps must be positive")
        if self.projection_tolerance <= 0.0:
            raise ValueError("projection_tolerance must be positive")
        if self.projection_solver_tolerance <= 0.0:
            raise ValueError(
                "projection_solver_tolerance must be positive"
            )
        if self.projection_solver_margin < 0.0:
            raise ValueError(
                "projection_solver_margin must be nonnegative"
            )
        if self.projection_backend not in ("osqp", "hildreth"):
            raise ValueError(
                "projection_backend must be osqp or hildreth"
            )
        if self.projection_extension_sweeps < 0:
            raise ValueError(
                "projection_extension_sweeps must be nonnegative"
            )
        if not 0.0 <= self.minimum_restoration_retention <= 1.0:
            raise ValueError(
                "minimum_restoration_retention must lie in [0, 1]"
            )
        if not 0.0 < self.hqp_progress_retention <= 1.0:
            raise ValueError(
                "hqp_progress_retention must lie in (0, 1]"
            )
        if self.hqp_regularization <= 0.0:
            raise ValueError("hqp_regularization must be positive")
        if self.hqp_minimum_direction_norm < 0.0:
            raise ValueError(
                "hqp_minimum_direction_norm must be nonnegative"
            )
        if not 0.0 < self.bridge_progress_retention < 1.0:
            raise ValueError(
                "bridge_progress_retention must lie in (0, 1)"
            )
        if self.bridge_geometry_tolerance <= 0.0:
            raise ValueError(
                "bridge_geometry_tolerance must be positive"
            )
        if self.actuation_mode not in (
            "native-cbf",
            "common-scale",
            "component-scale",
            "robotarium-clip",
        ):
            raise ValueError(
                "actuation_mode must be native-cbf, common-scale, "
                "component-scale, or robotarium-clip"
            )


@dataclass(frozen=True)
class UnicycleRollout:
    scenario: Scenario
    config: UnicycleConfig
    control_protocol: Protocol
    times: Array
    trajectory: Array
    headings: Array
    first_arrival_times: Array
    final_goal_distances: Array
    minimum_physical_pair_distance: float
    minimum_physical_obstacle_clearance: float
    minimum_virtual_pair_distance: float
    minimum_virtual_obstacle_clearance: float
    maximum_linear_speed: float
    maximum_yaw_rate: float
    minimum_yaw_scale: float
    nonconverged_steps: int
    feasibility_restoration_steps: int
    solver_nonconverged_calls: int
    actuator_contraction_calls: int
    actuator_restoration_calls: int
    witness_restoration_calls: int
    common_contraction_calls: int
    hqp_restoration_calls: int
    projection_iteration_cap_calls: int
    hqp_active_calls: int
    hqp_nonconverged_calls: int
    minimum_hqp_progress_retention: float
    tangent_fallback_steps: int
    tangent_solver_nonconverged_steps: int
    cluster_escape_steps: int
    cluster_escape_component_steps: int
    minimum_cbf_residual: float
    minimum_tangent_residual: float
    static_certificate_candidate_steps: int
    static_certificate_excluded_projection_steps: int
    static_certificate_steps: int
    static_certificate_violations: int
    certified_bridge_steps: int
    certified_bridge_projection_calls: int
    certified_bridge_row_rejections: int
    certified_bridge_progress_violations: int
    minimum_certified_bridge_progress_residual: float

    @property
    def robot_arrival_rate(self) -> float:
        return float(
            np.mean(
                self.final_goal_distances
                <= self.scenario.protocol.arrival_radius
            )
        )

    @property
    def transferred_pair_lower_bound(self) -> float:
        return self.minimum_virtual_pair_distance - 2.0 * self.config.lookahead

    @property
    def mission_success(self) -> bool:
        return bool(
            self.physical_mission_success
            and self.transferred_pair_lower_bound
            >= self.scenario.protocol.pair_clearance - 1.0e-6
            and self.minimum_virtual_obstacle_clearance >= -1.0e-6
            and self.nonconverged_steps == 0
        )

    @property
    def physical_mission_success(self) -> bool:
        return bool(
            self.robot_arrival_rate == 1.0
            and self.minimum_physical_pair_distance
            >= self.scenario.protocol.pair_clearance - 1.0e-6
            and self.minimum_physical_obstacle_clearance >= -1.0e-6
        )

    @property
    def makespan(self) -> float:
        if np.any(~np.isfinite(self.first_arrival_times)):
            return float("nan")
        return float(np.max(self.first_arrival_times))

    def summary(self) -> dict:
        return {
            "family": self.scenario.family,
            "n_robots": self.scenario.n_robots,
            "seed": self.scenario.seed,
            "fingerprint": self.scenario.fingerprint(),
            "mission_success": self.mission_success,
            "physical_mission_success": self.physical_mission_success,
            "robot_arrival_rate": self.robot_arrival_rate,
            "makespan_s": self.makespan,
            "unarrived_robots": int(
                np.sum(
                    self.final_goal_distances
                    > self.scenario.protocol.arrival_radius
                )
            ),
            "maximum_final_goal_distance_m": float(
                np.max(self.final_goal_distances)
            ),
            "minimum_physical_pair_distance_m": (
                self.minimum_physical_pair_distance
            ),
            "minimum_transferred_pair_lower_bound_m": (
                self.transferred_pair_lower_bound
            ),
            "minimum_physical_obstacle_clearance_m": (
                self.minimum_physical_obstacle_clearance
            ),
            "minimum_virtual_obstacle_clearance_m": (
                self.minimum_virtual_obstacle_clearance
            ),
            "maximum_linear_speed_mps": self.maximum_linear_speed,
            "maximum_yaw_rate_rps": self.maximum_yaw_rate,
            "minimum_yaw_scale": (
                None
                if self.config.actuation_mode == "native-cbf"
                else self.minimum_yaw_scale
            ),
            "minimum_yaw_retention": (
                self.minimum_yaw_scale
                if self.config.actuation_mode == "native-cbf"
                else None
            ),
            "theoretical_yaw_scale_floor": (
                None
                if self.config.actuation_mode == "native-cbf"
                else unicycle_yaw_scale_floor(
                    self.scenario.protocol,
                    self.config,
                )
            ),
            "nonconverged_steps": self.nonconverged_steps,
            "command_infeasible_steps": self.nonconverged_steps,
            "feasibility_restoration_steps": (
                self.feasibility_restoration_steps
            ),
            "feasibility_restoration_calls": (
                self.feasibility_restoration_steps
            ),
            "solver_nonconverged_calls": self.solver_nonconverged_calls,
            "actuator_contraction_calls": self.actuator_contraction_calls,
            "restoration_calls_by_type": {
                "actuator_contraction": self.actuator_restoration_calls,
                "certified_witness": self.witness_restoration_calls,
                "common_contraction": self.common_contraction_calls,
                "hqp_fallback": self.hqp_restoration_calls,
            },
            "projection_iteration_cap_calls": (
                self.projection_iteration_cap_calls
            ),
            "hqp_active_calls": self.hqp_active_calls,
            "hqp_nonconverged_calls": self.hqp_nonconverged_calls,
            "minimum_hqp_progress_retention": (
                self.minimum_hqp_progress_retention
                if self.hqp_active_calls
                else None
            ),
            "tangent_fallback_steps": self.tangent_fallback_steps,
            "tangent_solver_nonconverged_steps": (
                self.tangent_solver_nonconverged_steps
            ),
            "cluster_escape_steps": self.cluster_escape_steps,
            "cluster_escape_component_steps": (
                self.cluster_escape_component_steps
            ),
            "minimum_cbf_residual": self.minimum_cbf_residual,
            "minimum_tangent_residual": self.minimum_tangent_residual,
            "static_certificate_candidate_steps": (
                self.static_certificate_candidate_steps
            ),
            "static_certificate_excluded_projection_steps": (
                self.static_certificate_excluded_projection_steps
            ),
            "static_certificate_steps": self.static_certificate_steps,
            "static_certificate_violations": (
                self.static_certificate_violations
            ),
            "certified_bridge_steps": self.certified_bridge_steps,
            "certified_bridge_projection_calls": (
                self.certified_bridge_projection_calls
            ),
            "certified_bridge_row_rejections": (
                self.certified_bridge_row_rejections
            ),
            "certified_bridge_progress_violations": (
                self.certified_bridge_progress_violations
            ),
            "minimum_certified_bridge_progress_residual": (
                self.minimum_certified_bridge_progress_residual
                if self.certified_bridge_projection_calls
                else None
            ),
        }


@dataclass(frozen=True)
class CertifiedBridgeProgress:
    """One native-input progress row backed by a feasible rigid witness."""

    component: tuple[int, ...]
    direction: Array
    rate_floor: float
    world_row: Array
    candidate_velocity: Array

    @property
    def lower_bound(self) -> float:
        return len(self.component) * self.rate_floor


def inflated_unicycle_protocol(
    physical: Protocol,
    lookahead: float,
) -> Protocol:
    """Return the virtual-point protocol giving physical-body safety."""
    return replace(
        physical,
        safety_margin=physical.safety_margin + lookahead,
    )


def unicycle_yaw_scale_floor(
    physical: Protocol,
    config: UnicycleConfig,
) -> float:
    """Controller-independent lower bound on every yaw-rate scale."""
    return min(
        1.0,
        config.lookahead
        * config.yaw_rate_limit
        / physical.speed_limit,
    )


def _minimum_pair_distance(positions: Array) -> float:
    if len(positions) < 2:
        return float("inf")
    left, right = np.triu_indices(len(positions), k=1)
    return float(
        np.min(np.linalg.norm(positions[left] - positions[right], axis=1))
    )


def _minimum_obstacle_clearance(
    scenario: Scenario,
    positions: Array,
    clearance: float,
) -> float:
    return min(
        scenario.arena.minimum_clearance(point, clearance)
        for point in positions
    )


def _component_yaw_scales(
    positions: Array,
    raw_yaw: Array,
    yaw_rate_limit: float,
    pair_row_distance: float,
) -> Array:
    """Return one yaw scale per connected active pair-CBF component."""
    count = len(positions)
    parent = np.arange(count)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    if count > 1:
        left, right = np.triu_indices(count, k=1)
        distance = np.linalg.norm(
            positions[left] - positions[right],
            axis=1,
        )
        for first, second in zip(
            left[distance <= pair_row_distance],
            right[distance <= pair_row_distance],
        ):
            union(int(first), int(second))

    roots = np.array([find(robot) for robot in range(count)])
    scales = np.ones(count)
    for root in np.unique(roots):
        members = roots == root
        peak = float(np.max(np.abs(raw_yaw[members])))
        scales[members] = min(
            1.0,
            yaw_rate_limit / max(peak, 1.0e-12),
        )
    return scales


def _world_velocity_rows_to_unicycle(
    matrix: Array,
    headings: Array,
) -> Array:
    """Map world-velocity rows to ``z_i = [v_i, ell*omega_i]``."""
    matrix = np.asarray(matrix, dtype=float)
    count = len(headings)
    if matrix.shape[1] != 2 * count:
        raise ValueError("world-row width and heading count disagree")
    heading_vectors = np.stack(
        (np.cos(headings), np.sin(headings)),
        axis=1,
    )
    perpendicular = np.stack(
        (-np.sin(headings), np.cos(headings)),
        axis=1,
    )
    blocks = matrix.reshape(len(matrix), count, 2)
    transformed = np.empty_like(blocks)
    transformed[:, :, 0] = np.einsum(
        "mij,ij->mi",
        blocks,
        heading_vectors,
    )
    transformed[:, :, 1] = np.einsum(
        "mij,ij->mi",
        blocks,
        perpendicular,
    )
    return transformed.reshape(len(matrix), 2 * count)


def _sparse_geometric_unicycle_rows(
    controller: CLEARController,
    positions: Array,
    headings: Array,
    arena,
    *,
    speed_limit: float,
):
    """Assemble native-unicycle CBF rows without a dense 2N-wide matrix."""
    from scipy import sparse

    count = len(positions)
    heading_vectors = np.stack(
        (np.cos(headings), np.sin(headings)),
        axis=1,
    )
    perpendicular = np.stack(
        (-np.sin(headings), np.cos(headings)),
        axis=1,
    )
    (
        pair_left,
        pair_right,
        pair_normal,
        pair_bounds,
        boundary_robot,
        boundary_normal,
        boundary_bounds,
    ) = controller._geometric_row_blocks(
        positions,
        arena,
        tangent_only=False,
        speed_limit=speed_limit,
    )
    pair_count = len(pair_left)
    boundary_count = len(boundary_robot)

    pair_rows = np.repeat(np.arange(pair_count), 4)
    pair_columns = np.column_stack(
        (
            2 * pair_left,
            2 * pair_left + 1,
            2 * pair_right,
            2 * pair_right + 1,
        )
    ).reshape(-1)
    pair_data = np.column_stack(
        (
            np.einsum(
                "ij,ij->i",
                pair_normal,
                heading_vectors[pair_left],
            ),
            np.einsum(
                "ij,ij->i",
                pair_normal,
                perpendicular[pair_left],
            ),
            -np.einsum(
                "ij,ij->i",
                pair_normal,
                heading_vectors[pair_right],
            ),
            -np.einsum(
                "ij,ij->i",
                pair_normal,
                perpendicular[pair_right],
            ),
        )
    ).reshape(-1)

    boundary_rows = np.repeat(
        pair_count + np.arange(boundary_count),
        2,
    )
    boundary_columns = np.column_stack(
        (2 * boundary_robot, 2 * boundary_robot + 1)
    ).reshape(-1)
    boundary_data = np.column_stack(
        (
            np.einsum(
                "ij,ij->i",
                boundary_normal,
                heading_vectors[boundary_robot],
            ),
            np.einsum(
                "ij,ij->i",
                boundary_normal,
                perpendicular[boundary_robot],
            ),
        )
    ).reshape(-1)
    matrix = sparse.coo_matrix(
        (
            np.concatenate((pair_data, boundary_data)),
            (
                np.concatenate((pair_rows, boundary_rows)),
                np.concatenate((pair_columns, boundary_columns)),
            ),
        ),
        shape=(pair_count + boundary_count, 2 * count),
    ).tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, np.concatenate((pair_bounds, boundary_bounds))


def _certified_bridge_progress(
    controller: CLEARController,
    positions: Array,
    arena,
    audit,
    physical: Protocol,
    config: UnicycleConfig,
) -> list[CertifiedBridgeProgress]:
    """Construct closed-form native progress rows for exact bridge cells.

    The returned rigid witness is only a proposal.  The native projector
    independently checks it against every current CBF row and actuator box
    before appending any progress constraint.
    """
    if (
        not config.certified_bridge_progress
        or config.actuation_mode != "native-cbf"
    ):
        return []
    certificates = audit_straight_bridge_tokens(
        controller,
        positions,
        arena,
        audit,
        geometry_tolerance=config.bridge_geometry_tolerance,
    )
    proposals: list[CertifiedBridgeProgress] = []
    count = len(positions)
    actuator_floor = min(
        physical.speed_limit,
        config.lookahead * config.yaw_rate_limit,
    )
    for certificate in certificates:
        if not certificate.domain_satisfied:
            continue
        geometric_floor = (
            certificate.guide_progress_floor
            + controller.config.cluster_escape_gain
            * certificate.pair_gate
            * certificate.boundary_gate
        )
        capacity = min(geometric_floor, actuator_floor)
        if capacity <= config.projection_tolerance:
            continue
        # The enforced row retains a fixed fraction of the closed-form
        # capacity.  The midpoint witness leaves positive progress slack
        # without exceeding the capacity used in the actuator proof.
        rate_floor = config.bridge_progress_retention * capacity
        witness_rate = (
            0.5 * (1.0 + config.bridge_progress_retention) * capacity
        )
        indices = np.asarray(certificate.component, dtype=int)
        world_row = np.zeros((count, 2))
        world_row[indices] = certificate.direction
        candidate = np.zeros((count, 2))
        candidate[indices] = witness_rate * certificate.direction
        proposals.append(
            CertifiedBridgeProgress(
                component=certificate.component,
                direction=certificate.direction.copy(),
                rate_floor=rate_floor,
                world_row=world_row,
                candidate_velocity=candidate,
            )
        )
    return proposals


@dataclass(frozen=True)
class NativeProjectionAudit:
    """Final safety result plus optimizer and restoration diagnostics."""

    result: ProjectionResult
    progress_result: ProjectionResult | None
    solver_converged: bool
    solver_status: str | None
    command_feasible: bool
    restoration_type: str | None
    actuator_contracted: bool
    optimum_progress: float
    required_progress: float
    achieved_progress: float
    certified_progress_rows_requested: int
    certified_progress_rows_accepted: int
    minimum_certified_progress_residual: float

    @property
    def converged(self) -> bool:
        """Backward-compatible alias for final command feasibility."""
        return self.command_feasible

    @property
    def sweeps(self) -> int:
        return self.result.sweeps

    @property
    def minimum_residual(self) -> float:
        return self.result.minimum_residual

    @property
    def dual(self) -> Array:
        return self.result.dual

    @property
    def feasibility_restored(self) -> bool:
        return self.restoration_type is not None

    @property
    def hierarchy_active(self) -> bool:
        return self.progress_result is not None

    @property
    def hierarchy_converged(self) -> bool:
        return bool(
            self.progress_result is None
            or self.progress_result.converged
        )

    @property
    def progress_dual(self) -> Array:
        return (
            np.zeros(0)
            if self.progress_result is None
            else self.progress_result.dual
        )

    @property
    def progress_retention(self) -> float:
        if self.optimum_progress <= 1.0e-12:
            return 1.0
        return self.achieved_progress / self.optimum_progress


def _component_partition(matrix, count: int) -> list[tuple[int, ...]]:
    """Return exact robot blocks induced by the final sparse QP rows."""
    from scipy import sparse

    constraint = sparse.csr_matrix(matrix)
    parent = np.arange(count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in range(constraint.shape[0]):
        begin = constraint.indptr[row]
        end = constraint.indptr[row + 1]
        robots = np.unique(constraint.indices[begin:end] // 2)
        if robots.size > 1:
            anchor = int(robots[0])
            for robot in robots[1:]:
                union(anchor, int(robot))
    groups: dict[int, list[int]] = {}
    for robot in range(count):
        groups.setdefault(find(robot), []).append(robot)
    return [
        tuple(robots)
        for _, robots in sorted(groups.items(), key=lambda item: item[1][0])
    ]


def _project_component_blocks(
    nominal: Array,
    matrix,
    lower: Array,
    upper: Array,
    solver_margin: Array,
    count: int,
    workspaces: dict[tuple[int, ...], OSQPProjectionWorkspace],
    config: UnicycleConfig,
) -> tuple[ProjectionResult, list[int], list[tuple[int, ...]]]:
    """Solve an exactly block-diagonal native QP one robot block at a time."""
    from scipy import sparse

    constraint = sparse.csr_matrix(matrix)
    components = _component_partition(constraint, count)
    robot_component = np.empty(count, dtype=int)
    for component_index, component in enumerate(components):
        robot_component[np.asarray(component, dtype=int)] = component_index
    component_rows: list[list[int]] = [[] for _ in components]
    for row in range(constraint.shape[0]):
        begin = constraint.indptr[row]
        end = constraint.indptr[row + 1]
        if begin == end:
            raise RuntimeError("a zero-support projection row cannot be split")
        robots = np.unique(constraint.indices[begin:end] // 2)
        component_indices = np.unique(robot_component[robots])
        if component_indices.size != 1:
            raise RuntimeError("a projection row spans component partitions")
        component_rows[int(component_indices[0])].append(row)
    combined_value = np.zeros(2 * count)
    combined_dual = np.zeros(constraint.shape[0])
    combined_solver_dual = np.zeros(constraint.shape[0])
    component_times: list[int] = []
    component_results: list[ProjectionResult] = []

    for component_index, component in enumerate(components):
        started = perf_counter_ns()
        columns = np.asarray(
            [column for robot in component for column in (2 * robot, 2 * robot + 1)],
            dtype=int,
        )
        rows = np.asarray(component_rows[component_index], dtype=int)
        submatrix = constraint[rows][:, columns].tocsc()
        workspace = workspaces.setdefault(
            component,
            OSQPProjectionWorkspace(max_cached_patterns=16),
        )
        result = workspace.project(
            nominal[columns],
            submatrix,
            lower[rows],
            upper[rows],
            tolerance=config.projection_tolerance,
            solver_tolerance=config.projection_solver_tolerance,
            solver_margin=solver_margin[rows],
            max_iterations=config.projection_max_sweeps,
        )
        component_results.append(result)
        combined_value[columns] = result.value
        combined_dual[rows] = result.dual
        if result.solver_dual is not None:
            combined_solver_dual[rows] = result.solver_dual
        component_times.append(perf_counter_ns() - started)

    if sum(len(rows) for rows in component_rows) != constraint.shape[0]:
        raise RuntimeError("a native projection row was not assigned")
    values = np.asarray(constraint @ combined_value).reshape(-1)
    residuals = [values - lower]
    finite_upper = np.isfinite(upper)
    if np.any(finite_upper):
        residuals.append(upper[finite_upper] - values[finite_upper])
    minimum_residual = float(np.min(np.concatenate(residuals)))
    solver_statuses = {
        result.solver_status
        for result in component_results
        if result.solver_status is not None
    }
    combined = ProjectionResult(
        value=combined_value,
        converged=bool(
            minimum_residual >= -config.projection_tolerance
        ),
        sweeps=max((result.sweeps for result in component_results), default=0),
        minimum_residual=minimum_residual,
        displacement=float(np.linalg.norm(combined_value - nominal)),
        dual=combined_dual,
        feasibility_restored=any(
            result.feasibility_restored for result in component_results
        ),
        solver_dual=combined_solver_dual,
        solver_converged=all(
            result.optimizer_converged for result in component_results
        ),
        solver_status=(
            next(iter(solver_statuses))
            if len(solver_statuses) == 1
            else "+".join(sorted(solver_statuses))
        ),
    )
    return combined, component_times, components


def _project_native_unicycle(
    controller: CLEARController,
    virtual_positions: Array,
    headings: Array,
    arena,
    nominal_virtual_command: Array,
    physical: Protocol,
    config: UnicycleConfig,
    *,
    progress_virtual_command: Array | None = None,
    certified_progress_world_rows: Array | None = None,
    certified_progress_lower: Array | None = None,
    certified_progress_candidate: Array | None = None,
    initial_dual: Array | None = None,
    initial_progress_dual: Array | None = None,
    workspace: OSQPProjectionWorkspace | None = None,
    component_workspaces: (
        dict[tuple[int, ...], OSQPProjectionWorkspace] | None
    ) = None,
    timing: dict | None = None,
) -> tuple[Array, Array, Array, NativeProjectionAudit]:
    """Project one command into safe bounded unicycle input coordinates.

    The optimization variable for robot ``i`` is
    ``z_i = [v_i, ell*omega_i]``.  Its Euclidean objective is exactly the
    squared error of the realized look-ahead-point velocity because
    ``[e_i, J e_i]`` is orthogonal.
    """
    timing_started = perf_counter_ns()
    count = len(virtual_positions)
    heading_vectors = np.stack(
        (np.cos(headings), np.sin(headings)),
        axis=1,
    )
    perpendicular = np.stack(
        (-np.sin(headings), np.cos(headings)),
        axis=1,
    )
    nominal = np.empty((count, 2))
    nominal[:, 0] = np.einsum(
        "ij,ij->i",
        heading_vectors,
        nominal_virtual_command,
    )
    nominal[:, 1] = np.einsum(
        "ij,ij->i",
        perpendicular,
        nominal_virtual_command,
    )
    requested_certified_rows = 0
    certified_world_rows = np.zeros((0, 2 * count))
    certified_lower = np.zeros(0)
    certified_candidate_world = np.zeros((count, 2))
    supplied_certified_data = (
        certified_progress_world_rows is not None
        or certified_progress_lower is not None
        or certified_progress_candidate is not None
    )
    if supplied_certified_data:
        if (
            certified_progress_world_rows is None
            or certified_progress_lower is None
            or certified_progress_candidate is None
        ):
            raise ValueError(
                "certified progress rows, lower bounds, and candidate "
                "must be supplied together"
            )
        certified_world_rows = np.asarray(
            certified_progress_world_rows,
            dtype=float,
        )
        if certified_world_rows.ndim == 3:
            certified_world_rows = certified_world_rows.reshape(
                len(certified_world_rows),
                -1,
            )
        certified_lower = np.asarray(
            certified_progress_lower,
            dtype=float,
        ).reshape(-1)
        certified_candidate_world = np.asarray(
            certified_progress_candidate,
            dtype=float,
        )
        if certified_world_rows.ndim != 2 or (
            certified_world_rows.shape[1] != 2 * count
        ):
            raise ValueError(
                "each certified progress row must contain 2N entries"
            )
        if certified_lower.shape != (len(certified_world_rows),):
            raise ValueError(
                "one lower bound is required per certified progress row"
            )
        if certified_candidate_world.shape != (count, 2):
            raise ValueError(
                "certified progress candidate must contain N planar vectors"
            )
        if np.any(certified_lower <= 0.0):
            raise ValueError(
                "certified progress lower bounds must be positive"
            )
        requested_certified_rows = len(certified_world_rows)

    maximum_virtual_speed = float(
        np.hypot(
            physical.speed_limit,
            config.lookahead * config.yaw_rate_limit,
        )
    )
    optimized_osqp = (
        config.projection_backend == "osqp"
        and not config.hierarchical_progress
    )
    if optimized_osqp:
        from scipy import sparse

        geometric_matrix, geometric_bounds = (
            _sparse_geometric_unicycle_rows(
                controller,
                virtual_positions,
                headings,
                arena,
                speed_limit=maximum_virtual_speed,
            )
        )
    else:
        world_rows, bounds, _, _ = controller._geometric_rows(
            virtual_positions,
            arena,
            tangent_only=False,
            speed_limit=maximum_virtual_speed,
        )
        if world_rows:
            geometric_matrix = _world_velocity_rows_to_unicycle(
                np.asarray(world_rows),
                headings,
            )
            geometric_bounds = np.asarray(bounds)
        else:
            geometric_matrix = np.zeros((0, 2 * count))
            geometric_bounds = np.zeros(0)

    limits = np.tile(
        np.array(
            [
                physical.speed_limit,
                config.lookahead * config.yaw_rate_limit,
            ]
        ),
        count,
    )
    if optimized_osqp:
        identity = sparse.eye(2 * count, format="csc")
        matrix = sparse.vstack(
            (geometric_matrix, identity),
            format="csc",
        )
        lower = np.concatenate((geometric_bounds, -limits))
        upper = np.concatenate(
            (np.full(len(geometric_bounds), np.inf), limits)
        )
    else:
        identity = np.eye(2 * count)
        matrix = np.vstack((geometric_matrix, identity, -identity))
        lower = np.concatenate(
            (geometric_bounds, -limits, -limits)
        )
        upper = np.full(len(lower), np.inf)
    accepted_certified_rows = 0
    certified_candidate_coordinates: Array | None = None
    certified_matrix = np.zeros((0, 2 * count))
    if requested_certified_rows:
        certified_matrix = _world_velocity_rows_to_unicycle(
            certified_world_rows,
            headings,
        )
        certified_candidate_coordinates = (
            _world_velocity_rows_to_unicycle(
                certified_candidate_world.reshape(1, -1),
                headings,
            ).reshape(-1)
        )
        base_values = np.asarray(
            matrix @ certified_candidate_coordinates
        ).reshape(-1)
        base_feasible = bool(
            np.all(base_values >= lower - config.projection_tolerance)
            and np.all(
                base_values[np.isfinite(upper)]
                <= upper[np.isfinite(upper)]
                + config.projection_tolerance
            )
        )
        certified_values = (
            certified_matrix @ certified_candidate_coordinates
        )
        progress_feasible = bool(
            np.all(
                certified_values
                >= certified_lower - config.projection_tolerance
            )
        )
        if base_feasible and progress_feasible:
            if optimized_osqp:
                from scipy import sparse

                matrix = sparse.vstack(
                    (matrix, sparse.csc_matrix(certified_matrix)),
                    format="csc",
                )
            else:
                matrix = np.vstack((matrix, certified_matrix))
            lower = np.concatenate((lower, certified_lower))
            upper = np.concatenate(
                (upper, np.full(requested_certified_rows, np.inf))
            )
            accepted_certified_rows = requested_certified_rows
    projection_solver = (
        project_halfspaces_osqp
        if config.projection_backend == "osqp"
        else project_halfspaces
    )
    solver_limit_name = (
        "max_iterations"
        if config.projection_backend == "osqp"
        else "max_sweeps"
    )
    final_matrix = matrix
    final_lower = lower
    final_upper = upper
    progress_result: ProjectionResult | None = None
    progress_direction = np.zeros(2 * count)
    optimum_progress = 0.0
    required_progress = 0.0
    if (
        config.hierarchical_progress
        and config.projection_backend == "osqp"
        and progress_virtual_command is not None
    ):
        progress_virtual_command = np.asarray(
            progress_virtual_command,
            dtype=float,
        )
        if progress_virtual_command.shape != (count, 2):
            raise ValueError(
                "progress command must contain one planar vector per robot"
            )
        progress_coordinates = np.empty((count, 2))
        progress_coordinates[:, 0] = np.einsum(
            "ij,ij->i",
            heading_vectors,
            progress_virtual_command,
        )
        progress_coordinates[:, 1] = np.einsum(
            "ij,ij->i",
            perpendicular,
            progress_virtual_command,
        )
        progress_direction = progress_coordinates.reshape(-1)
        progress_norm = float(np.linalg.norm(progress_direction))
        if progress_norm > config.hqp_minimum_direction_norm:
            progress_direction = progress_direction / progress_norm
            progress_result = maximize_halfspace_linear_osqp(
                progress_direction,
                matrix,
                lower,
                tolerance=config.projection_tolerance,
                max_iterations=config.projection_max_sweeps,
                regularization=config.hqp_regularization,
                initial_dual=initial_progress_dual,
            )
            if progress_result.converged:
                optimum_progress = max(
                    0.0,
                    float(progress_direction @ progress_result.value),
                )
                required_progress = max(
                    0.0,
                    config.hqp_progress_retention * optimum_progress
                    - 10.0 * config.projection_tolerance,
                )
                if required_progress > config.projection_tolerance:
                    final_matrix = np.vstack(
                        (matrix, progress_direction[None, :])
                    )
                    final_lower = np.concatenate(
                        (lower, [required_progress])
                    )
                    final_upper = np.concatenate((upper, [np.inf]))

    local_solve_times: list[int] = []
    projection_components: list[tuple[int, ...]] = [
        tuple(range(count))
    ]
    if optimized_osqp:
        active_workspace = (
            workspace
            if workspace is not None
            else OSQPProjectionWorkspace(max_cached_patterns=1)
        )
        solver_margin = np.concatenate(
            (
                np.full(
                    len(geometric_bounds),
                    config.projection_solver_margin,
                ),
                np.zeros(len(limits)),
                np.zeros(accepted_certified_rows),
            )
        )
        if component_workspaces is None:
            solve_started = perf_counter_ns()
            result = active_workspace.project(
                nominal.reshape(-1),
                final_matrix,
                final_lower,
                final_upper,
                tolerance=config.projection_tolerance,
                solver_tolerance=config.projection_solver_tolerance,
                solver_margin=solver_margin,
                max_iterations=config.projection_max_sweeps,
            )
            local_solve_times.append(perf_counter_ns() - solve_started)
        else:
            result, local_solve_times, projection_components = (
                _project_component_blocks(
                    nominal.reshape(-1),
                    final_matrix,
                    final_lower,
                    final_upper,
                    solver_margin,
                    count,
                    component_workspaces,
                    config,
                )
            )
    else:
        solve_started = perf_counter_ns()
        result = projection_solver(
            nominal.reshape(-1),
            final_matrix,
            final_lower,
            tolerance=config.projection_tolerance,
            initial_dual=initial_dual,
            **{solver_limit_name: config.projection_max_sweeps},
        )
        local_solve_times.append(perf_counter_ns() - solve_started)
    solver_converged = result.optimizer_converged
    solver_status = result.solver_status
    restoration_type: str | None = None
    actuator_contracted = False

    def row_values(value: Array) -> Array:
        return np.asarray(final_matrix @ value).reshape(-1)

    def feasibility_minimum(value: Array) -> float:
        values = row_values(value)
        residuals = [values - final_lower]
        finite_upper = np.isfinite(final_upper)
        if np.any(finite_upper):
            residuals.append(
                final_upper[finite_upper] - values[finite_upper]
            )
        return float(np.min(np.concatenate(residuals)))

    def required_contraction(candidate: ProjectionResult) -> float:
        values = row_values(candidate.value)
        ratios: list[Array] = []
        violated_lower = (
            values < final_lower - config.projection_tolerance
        )
        if np.any(violated_lower):
            ratios.append(
                np.divide(
                    final_lower[violated_lower],
                    values[violated_lower],
                    out=np.zeros(np.count_nonzero(violated_lower)),
                    where=values[violated_lower] < 0.0,
                )
            )
        violated_upper = (
            values > final_upper + config.projection_tolerance
        )
        if np.any(violated_upper):
            ratios.append(
                np.divide(
                    final_upper[violated_upper],
                    values[violated_upper],
                    out=np.zeros(np.count_nonzero(violated_upper)),
                    where=values[violated_upper] > 0.0,
                )
            )
        if not ratios:
            return 1.0
        return max(
            0.0,
            min(
                1.0,
                float(np.min(np.concatenate(ratios)))
                * (1.0 - 1.0e-10),
            ),
        )

    safe_right_hand_side = (
        float(np.max(final_lower)) <= config.projection_tolerance
        and float(np.min(final_upper)) >= -config.projection_tolerance
    )
    zero_is_exactly_feasible = bool(
        np.all(final_lower <= 0.0)
        and np.all(final_upper >= 0.0)
    )
    if (
        not result.converged
        and accepted_certified_rows
        and certified_candidate_coordinates is not None
    ):
        # The rigid witness was checked against the complete native feasible
        # set before the progress rows were appended.  It is therefore a safe
        # deterministic fallback if the numerical projection reaches its
        # iteration budget.
        candidate_residual = feasibility_minimum(
            certified_candidate_coordinates
        )
        if candidate_residual >= -config.projection_tolerance:
            restoration_type = "certified-witness"
            result = ProjectionResult(
                value=certified_candidate_coordinates.copy(),
                converged=True,
                sweeps=result.sweeps,
                minimum_residual=candidate_residual,
                displacement=float(
                    np.linalg.norm(
                        certified_candidate_coordinates
                        - nominal.reshape(-1)
                    )
                ),
                dual=np.zeros(len(final_lower)),
                feasibility_restored=True,
                solver_converged=solver_converged,
                solver_status=solver_status,
            )
    if optimized_osqp and zero_is_exactly_feasible:
        # OSQP's termination tolerance may leave a tiny actuator-box
        # overshoot.  A common contraction enforces the physical limits
        # exactly and preserves every CBF row because zero is feasible.
        actuator_scale = min(
            1.0,
            float(
                np.min(
                    np.divide(
                        limits,
                        np.abs(result.value),
                        out=np.full_like(limits, np.inf),
                        where=np.abs(result.value) > limits,
                    )
                )
            ),
        )
        if actuator_scale < 1.0:
            feasible_before_actuator = result.converged
            actuator_contracted = True
            actuator_scale = float(
                np.nextafter(actuator_scale, 0.0)
            )
            bounded_value = actuator_scale * result.value
            bounded_residual = feasibility_minimum(bounded_value)
            bounded_feasible = bool(
                bounded_residual >= -config.projection_tolerance
            )
            if (
                not feasible_before_actuator
                and bounded_feasible
                and restoration_type is None
            ):
                restoration_type = "actuator-contraction"
            result = ProjectionResult(
                value=bounded_value,
                converged=bounded_feasible,
                sweeps=result.sweeps,
                minimum_residual=bounded_residual,
                displacement=float(
                    np.linalg.norm(
                        bounded_value - nominal.reshape(-1)
                    )
                ),
                dual=result.dual.copy(),
                feasibility_restored=(
                    result.feasibility_restored
                    or restoration_type == "actuator-contraction"
                ),
                solver_dual=result.solver_dual,
                solver_converged=solver_converged,
                solver_status=solver_status,
            )
    contraction = (
        required_contraction(result)
        if not result.converged and safe_right_hand_side
        else 1.0
    )
    if (
        not result.converged
        and safe_right_hand_side
        and contraction < config.minimum_restoration_retention
        and config.projection_extension_sweeps > 0
        and config.projection_backend == "hildreth"
    ):
        result = project_halfspaces(
            nominal.reshape(-1),
            matrix,
            lower,
            tolerance=config.projection_tolerance,
            max_sweeps=config.projection_extension_sweeps,
            initial_dual=result.dual,
        )
        solver_converged = result.optimizer_converged
        solver_status = result.solver_status
        contraction = (
            required_contraction(result)
            if not result.converged
            else 1.0
        )
    if not result.converged and safe_right_hand_side:
        # At a safe state all geometric right-hand sides are nonpositive, so
        # zero unicycle input is feasible.  A common contraction toward zero
        # can therefore restore feasibility without invalidating any row.
        restored_value = contraction * result.value
        restored_residual = feasibility_minimum(restored_value)
        if restored_residual >= -config.projection_tolerance:
            restoration_type = "common-contraction"
            result = ProjectionResult(
                value=restored_value,
                converged=True,
                sweeps=result.sweeps,
                minimum_residual=restored_residual,
                displacement=float(
                    np.linalg.norm(restored_value - nominal.reshape(-1))
                ),
                dual=result.dual.copy(),
                feasibility_restored=True,
                solver_dual=result.solver_dual,
                solver_converged=solver_converged,
                solver_status=solver_status,
            )
    elif (
        not result.converged
        and progress_result is not None
        and progress_result.converged
        and required_progress > config.projection_tolerance
    ):
        # The first-level solution is a known feasible point for the safety
        # rows and attains the retained progress row by construction.
        restored_value = progress_result.value.copy()
        restored_residual = feasibility_minimum(restored_value)
        if restored_residual >= -config.projection_tolerance:
            restoration_type = "hqp-fallback"
            result = ProjectionResult(
                value=restored_value,
                converged=True,
                sweeps=result.sweeps,
                minimum_residual=restored_residual,
                displacement=float(
                    np.linalg.norm(restored_value - nominal.reshape(-1))
                ),
                dual=np.concatenate(
                    (progress_result.dual.copy(), [0.0])
                ),
                feasibility_restored=True,
                solver_converged=solver_converged,
                solver_status=solver_status,
            )
    realized = result.value.reshape(count, 2)
    linear = realized[:, 0]
    yaw = realized[:, 1] / config.lookahead
    virtual_velocity = (
        linear[:, None] * heading_vectors
        + realized[:, 1, None] * perpendicular
    )
    achieved_progress = (
        float(progress_direction @ result.value)
        if progress_result is not None
        else 0.0
    )
    minimum_certified_progress_residual = (
        float(
            np.min(
                certified_matrix @ result.value - certified_lower
            )
        )
        if accepted_certified_rows
        else float("inf")
    )
    if timing is not None:
        batch_ns = perf_counter_ns() - timing_started
        timing.update(
            {
                "batch_ns": batch_ns,
                "shared_ns": max(0, batch_ns - sum(local_solve_times)),
                "local_unit_ns": local_solve_times,
                "critical_path_ns": (
                    max(0, batch_ns - sum(local_solve_times))
                    + (max(local_solve_times) if local_solve_times else 0)
                ),
                "component_sizes": [
                    len(component) for component in projection_components
                ],
            }
        )
    return (
        linear,
        yaw,
        virtual_velocity,
        NativeProjectionAudit(
            result=result,
            progress_result=progress_result,
            solver_converged=solver_converged,
            solver_status=solver_status,
            command_feasible=bool(
                result.minimum_residual >= -config.projection_tolerance
            ),
            restoration_type=restoration_type,
            actuator_contracted=actuator_contracted,
            optimum_progress=optimum_progress,
            required_progress=required_progress,
            achieved_progress=achieved_progress,
            certified_progress_rows_requested=requested_certified_rows,
            certified_progress_rows_accepted=accepted_certified_rows,
            minimum_certified_progress_residual=(
                minimum_certified_progress_residual
            ),
        ),
    )


def simulate_unicycle(
    scenario: Scenario,
    controller_config: ControllerConfig,
    unicycle_config: UnicycleConfig = UnicycleConfig(),
    *,
    initial_headings: Array | None = None,
    record_stride: int = 4,
    guidance_mode: str = "direct",
    controller: CLEARController | None = None,
    on_record: Callable[[float, Array, Array, Array], None] | None = None,
) -> UnicycleRollout:
    """Simulate CLEAR through a look-ahead unicycle realization.

    ``native-cbf`` projects the nominal CLEAR command directly in bounded
    ``[v, ell*omega]`` coordinates while enforcing the virtual CBF rows.
    ``common-scale`` preserves the virtual-point command up to one positive
    team-wide scale.  ``component-scale`` uses one positive scale per
    connected active pair-CBF component; every pair row still sees one common
    scale, while boundary rows remain safe under their robot's positive scale.
    ``robotarium-clip`` clips each robot's angular velocity independently and
    therefore does not inherit the virtual CBF proof.
    """
    physical = scenario.protocol
    control_protocol = inflated_unicycle_protocol(
        physical,
        unicycle_config.lookahead,
    )
    if controller is None:
        controller = CLEARController(control_protocol, controller_config)
    elif controller.protocol != control_protocol:
        raise ValueError(
            "the supplied controller must use the inflated unicycle protocol"
        )
    if guidance_mode not in ("direct", "waypoint", "cost"):
        raise ValueError("guidance_mode must be direct, waypoint, or cost")
    centers = np.asarray(scenario.starts, dtype=float).copy()
    if initial_headings is None:
        direction = np.asarray(scenario.goals) - centers
        headings = np.arctan2(direction[:, 1], direction[:, 0])
    else:
        headings = np.asarray(initial_headings, dtype=float).copy()
        if headings.shape != (scenario.n_robots,):
            raise ValueError("one initial heading is required per robot")
    heading_vectors = np.stack(
        (np.cos(headings), np.sin(headings)),
        axis=1,
    )
    virtual = centers + unicycle_config.lookahead * heading_vectors
    controller.reset()
    guidance: GuidancePlan | CostFieldGuidance | None = None
    if guidance_mode != "direct" and scenario.arena.obstacles:
        planner = GridPlanner(scenario.arena, control_protocol)
        guidance = (
            planner.plan(virtual, scenario.goals)
            if guidance_mode == "waypoint"
            else planner.cost_field_plan(scenario.goals)
        )

    if (
        _minimum_pair_distance(virtual)
        < control_protocol.pair_clearance - 1.0e-9
    ):
        raise ValueError("inflated virtual pair clearance is initially unsafe")
    if (
        _minimum_obstacle_clearance(
            scenario,
            virtual,
            control_protocol.robot_clearance,
        )
        < -1.0e-9
    ):
        raise ValueError(
            "inflated virtual obstacle clearance is initially unsafe"
        )

    first_arrival = np.full(scenario.n_robots, np.inf)
    initial_distance = np.linalg.norm(centers - scenario.goals, axis=1)
    first_arrival[initial_distance <= physical.arrival_radius] = 0.0
    recorded_times = [0.0]
    recorded_centers = [centers.copy()]
    recorded_headings = [headings.copy()]

    min_physical_pair = _minimum_pair_distance(centers)
    min_virtual_pair = _minimum_pair_distance(virtual)
    min_physical_obstacle = _minimum_obstacle_clearance(
        scenario,
        centers,
        physical.robot_clearance,
    )
    min_virtual_obstacle = _minimum_obstacle_clearance(
        scenario,
        virtual,
        control_protocol.robot_clearance,
    )
    max_linear = 0.0
    max_yaw = 0.0
    min_yaw_scale = 1.0
    nonconverged = 0
    feasibility_restorations = 0
    solver_nonconverged_calls = 0
    actuator_contractions = 0
    actuator_restorations = 0
    witness_restorations = 0
    common_contractions = 0
    hqp_restorations = 0
    projection_iteration_caps = 0
    hqp_active_calls = 0
    hqp_nonconverged_calls = 0
    minimum_hqp_progress_retention = 1.0
    tangent_fallbacks = 0
    tangent_solver_nonconverged = 0
    cluster_escape_steps = 0
    cluster_escape_component_steps = 0
    min_cbf_residual = float("inf")
    min_tangent_residual = float("inf")
    static_candidate_steps = 0
    static_excluded_projection_steps = 0
    static_steps = 0
    static_violations = 0
    certified_bridge_steps = 0
    certified_bridge_projection_calls = 0
    certified_bridge_row_rejections = 0
    certified_bridge_progress_violations = 0
    minimum_certified_bridge_progress_residual = float("inf")
    inner_dt = physical.dt / unicycle_config.inner_substeps
    yaw_scale_floor = unicycle_yaw_scale_floor(
        physical,
        unicycle_config,
    )
    pair_row_distance = (
        control_protocol.pair_clearance
        + 2.0 * control_protocol.speed_limit / controller_config.cbf_rate
    )
    native_dual = np.zeros(0)
    native_progress_dual = np.zeros(0)
    native_workspace = (
        OSQPProjectionWorkspace(max_cached_patterns=16)
        if (
            unicycle_config.actuation_mode == "native-cbf"
            and unicycle_config.projection_backend == "osqp"
            and not unicycle_config.hierarchical_progress
        )
        else None
    )

    for step in range(physical.steps):
        if isinstance(guidance, GuidancePlan):
            control_goals = guidance.targets(
                virtual,
                scenario.arena,
                control_protocol.robot_clearance + 0.01,
            )
        elif isinstance(guidance, CostFieldGuidance):
            control_goals = guidance.targets(virtual)
        else:
            control_goals = scenario.goals
        audit = controller.command(
            virtual,
            control_goals,
            scenario.arena,
            project_safety=(
                unicycle_config.actuation_mode != "native-cbf"
            ),
        )
        virtual_command = (
            audit.nominal_command
            if unicycle_config.actuation_mode == "native-cbf"
            else audit.executed_command
        )
        bridge_progress = _certified_bridge_progress(
            controller,
            virtual,
            scenario.arena,
            audit,
            physical,
            unicycle_config,
        )
        if bridge_progress:
            certified_bridge_steps += 1
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
        if unicycle_config.actuation_mode != "native-cbf":
            nonconverged += int(not audit.cbf_converged)
            min_cbf_residual = min(
                min_cbf_residual,
                audit.cbf_minimum_residual,
            )
        tangent_fallbacks += int(not audit.tangent_converged)
        tangent_solver_nonconverged += int(
            not audit.tangent_solver_converged
        )
        cluster_escape_steps += int(audit.active_cluster_escapes > 0)
        cluster_escape_component_steps += audit.active_cluster_escapes
        min_tangent_residual = min(
            min_tangent_residual,
            audit.tangent_minimum_residual,
        )
        static_candidate = (
            audit.tangent_converged
            and audit.cbf_converged
            and audit.tangent_norm > 1.0e-10
            and audit.global_tangent_margin > 1.0e-9
        )
        if (
            unicycle_config.actuation_mode != "native-cbf"
            and static_candidate
        ):
            static_candidate_steps += 1
            if (
                audit.tangent_solver_converged
                and audit.cbf_solver_converged
            ):
                static_steps += 1
                static_violations += int(
                    np.linalg.norm(virtual_command) <= 1.0e-10
                )
            else:
                static_excluded_projection_steps += 1

        native_step_converged = True
        native_step_residual = float("inf")
        native_first_command_norm: float | None = None
        native_first_solver_converged: bool | None = None
        virtual_before_step = virtual.copy()
        accepted_bridge_rows_this_step = 0
        for _ in range(unicycle_config.inner_substeps):
            heading_vectors = np.stack(
                (np.cos(headings), np.sin(headings)),
                axis=1,
            )
            perpendicular = np.stack(
                (-np.sin(headings), np.cos(headings)),
                axis=1,
            )
            raw_linear = np.einsum(
                "ij,ij->i",
                heading_vectors,
                virtual_command,
            )
            raw_yaw = (
                np.einsum("ij,ij->i", perpendicular, virtual_command)
                / unicycle_config.lookahead
            )
            if unicycle_config.actuation_mode == "native-cbf":
                linear, yaw, realized_virtual, native_result = (
                    _project_native_unicycle(
                        controller,
                        virtual,
                        headings,
                        scenario.arena,
                        virtual_command,
                        physical,
                        unicycle_config,
                        # The hierarchy preserves progress along the complete
                        # CLEAR task (goal plus transverse escape).  Using the
                        # transverse witness alone would reward perpetual
                        # circulation after it has served its purpose.
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
                native_dual = native_result.dual.copy()
                native_progress_dual = (
                    native_result.progress_dual.copy()
                )
                feasibility_restorations += int(
                    native_result.feasibility_restored
                )
                solver_nonconverged_calls += int(
                    not native_result.solver_converged
                )
                actuator_contractions += int(
                    native_result.actuator_contracted
                )
                actuator_restorations += int(
                    native_result.restoration_type
                    == "actuator-contraction"
                )
                witness_restorations += int(
                    native_result.restoration_type == "certified-witness"
                )
                common_contractions += int(
                    native_result.restoration_type == "common-contraction"
                )
                hqp_restorations += int(
                    native_result.restoration_type == "hqp-fallback"
                )
                projection_iteration_caps += int(
                    native_result.sweeps
                    >= unicycle_config.projection_max_sweeps
                )
                if native_result.progress_result is not None:
                    hqp_active_calls += 1
                    hqp_nonconverged_calls += int(
                        not native_result.progress_result.converged
                    )
                    projection_iteration_caps += int(
                        native_result.progress_result.sweeps
                        >= unicycle_config.projection_max_sweeps
                    )
                    minimum_hqp_progress_retention = min(
                        minimum_hqp_progress_retention,
                        native_result.progress_retention,
                    )
                native_step_converged &= native_result.converged
                native_step_residual = min(
                    native_step_residual,
                    native_result.minimum_residual,
                )
                certified_bridge_projection_calls += (
                    native_result.certified_progress_rows_accepted
                )
                accepted_bridge_rows_this_step += (
                    native_result.certified_progress_rows_accepted
                )
                certified_bridge_row_rejections += (
                    native_result.certified_progress_rows_requested
                    - native_result.certified_progress_rows_accepted
                )
                minimum_certified_bridge_progress_residual = min(
                    minimum_certified_bridge_progress_residual,
                    native_result.minimum_certified_progress_residual,
                )
                if native_first_command_norm is None:
                    native_first_command_norm = float(
                        np.linalg.norm(realized_virtual)
                    )
                    native_first_solver_converged = (
                        native_result.solver_converged
                    )
                active_yaw = np.abs(raw_yaw) > 1.0e-12
                local_min_scale = (
                    float(
                        np.min(
                            np.abs(yaw[active_yaw])
                            / np.abs(raw_yaw[active_yaw])
                        )
                    )
                    if np.any(active_yaw)
                    else 1.0
                )
            elif unicycle_config.actuation_mode in (
                "common-scale",
                "component-scale",
            ):
                peak_yaw = float(np.max(np.abs(raw_yaw)))
                if unicycle_config.actuation_mode == "common-scale":
                    yaw_scales = np.full(
                        scenario.n_robots,
                        min(
                            1.0,
                            unicycle_config.yaw_rate_limit
                            / max(peak_yaw, 1.0e-12),
                        ),
                    )
                else:
                    yaw_scales = _component_yaw_scales(
                        virtual,
                        raw_yaw,
                        unicycle_config.yaw_rate_limit,
                        pair_row_distance,
                    )
                local_min_scale = float(np.min(yaw_scales))
                if local_min_scale < yaw_scale_floor - 1.0e-12:
                    raise RuntimeError(
                        "yaw scale violated its analytic lower bound"
                    )
                linear = yaw_scales * raw_linear
                yaw = yaw_scales * raw_yaw
            else:
                # Robotarium's public create_si_to_uni_mapping applies
                # [v, omega]^T = diag(1, 1/ell) R(theta)^T u.  MGR states an
                # angular-speed limit, so impose it per robot after mapping.
                # The held command is already norm-limited, hence |v| <=
                # physical.speed_limit without another saturation.
                linear = raw_linear
                yaw = np.clip(
                    raw_yaw,
                    -unicycle_config.yaw_rate_limit,
                    unicycle_config.yaw_rate_limit,
                )
                active_yaw = np.abs(raw_yaw) > 1.0e-12
                local_min_scale = (
                    float(
                        np.min(
                            np.abs(yaw[active_yaw])
                            / np.abs(raw_yaw[active_yaw])
                        )
                    )
                    if np.any(active_yaw)
                    else 1.0
                )
            max_linear = max(max_linear, float(np.max(np.abs(linear))))
            max_yaw = max(max_yaw, float(np.max(np.abs(yaw))))
            min_yaw_scale = min(min_yaw_scale, local_min_scale)

            if unicycle_config.actuation_mode == "native-cbf":
                virtual = virtual + inner_dt * realized_virtual
                headings = headings + inner_dt * yaw
                heading_vectors = np.stack(
                    (np.cos(headings), np.sin(headings)),
                    axis=1,
                )
                centers = (
                    virtual
                    - unicycle_config.lookahead * heading_vectors
                )
            elif unicycle_config.actuation_mode in (
                "common-scale",
                "component-scale",
            ):
                # Each virtual point follows a positive scaling of CLEAR's
                # held command. Pair-CBF connected robots use the same scale.
                virtual = (
                    virtual
                    + inner_dt * yaw_scales[:, None] * virtual_command
                )
                headings = headings + inner_dt * yaw
                heading_vectors = np.stack(
                    (np.cos(headings), np.sin(headings)),
                    axis=1,
                )
                centers = (
                    virtual
                    - unicycle_config.lookahead * heading_vectors
                )
            else:
                # Standard explicit integration of first-order unicycle
                # kinematics after independent angular-velocity clipping.
                centers = (
                    centers
                    + inner_dt * linear[:, None] * heading_vectors
                )
                headings = headings + inner_dt * yaw
                heading_vectors = np.stack(
                    (np.cos(headings), np.sin(headings)),
                    axis=1,
                )
                virtual = (
                    centers
                    + unicycle_config.lookahead * heading_vectors
                )

            min_physical_pair = min(
                min_physical_pair,
                _minimum_pair_distance(centers),
            )
            min_virtual_pair = min(
                min_virtual_pair,
                _minimum_pair_distance(virtual),
            )

        if unicycle_config.actuation_mode == "native-cbf":
            nonconverged += int(not native_step_converged)
            min_cbf_residual = min(
                min_cbf_residual,
                native_step_residual,
            )
            static_candidate = (
                audit.tangent_converged
                and audit.tangent_norm > 1.0e-10
                and audit.global_tangent_margin > 1.0e-9
            )
            if static_candidate:
                static_candidate_steps += 1
            if (
                static_candidate
                and audit.tangent_solver_converged
                and native_first_solver_converged
            ):
                static_steps += 1
                static_violations += int(
                    native_first_command_norm is None
                    or native_first_command_norm <= 1.0e-10
                )
            elif static_candidate:
                static_excluded_projection_steps += 1
        expected_bridge_rows = (
            len(bridge_progress) * unicycle_config.inner_substeps
        )
        if (
            bridge_progress
            and accepted_bridge_rows_this_step == expected_bridge_rows
        ):
            virtual_increment = virtual - virtual_before_step
            held_velocity_lipschitz = (
                unicycle_config.yaw_rate_limit
                * np.hypot(
                    physical.speed_limit,
                    unicycle_config.lookahead
                    * unicycle_config.yaw_rate_limit,
                )
            )
            for proposal in bridge_progress:
                indices = np.asarray(proposal.component, dtype=int)
                actual_progress = float(
                    np.mean(
                        virtual_increment[indices]
                        @ proposal.direction
                    )
                )
                certified_increment = physical.dt * (
                    proposal.rate_floor
                    - 0.5 * held_velocity_lipschitz * inner_dt
                )
                certified_bridge_progress_violations += int(
                    actual_progress
                    < certified_increment
                    - 10.0 * unicycle_config.projection_tolerance
                )

        # Pair distances are cheap and are sampled at the inner integration
        # nodes above.  Static geometry is audited once per 33.3 Hz control
        # period; continuous physical safety is supplied by the inflated
        # virtual-point certificate rather than by these samples.
        min_physical_obstacle = min(
            min_physical_obstacle,
            _minimum_obstacle_clearance(
                scenario,
                centers,
                physical.robot_clearance,
            ),
        )
        min_virtual_obstacle = min(
            min_virtual_obstacle,
            _minimum_obstacle_clearance(
                scenario,
                virtual,
                control_protocol.robot_clearance,
            ),
        )

        time = (step + 1) * physical.dt
        goal_distance = np.linalg.norm(centers - scenario.goals, axis=1)
        new_arrivals = (
            np.isinf(first_arrival)
            & (goal_distance <= physical.arrival_radius)
        )
        first_arrival[new_arrivals] = time
        if (step + 1) % record_stride == 0 or step + 1 == physical.steps:
            recorded_times.append(time)
            recorded_centers.append(centers.copy())
            recorded_headings.append(headings.copy())
            if on_record is not None:
                on_record(
                    time,
                    centers.copy(),
                    headings.copy(),
                    first_arrival.copy(),
                )

    final_distance = np.linalg.norm(centers - scenario.goals, axis=1)
    return UnicycleRollout(
        scenario=scenario,
        config=unicycle_config,
        control_protocol=control_protocol,
        times=np.asarray(recorded_times),
        trajectory=np.asarray(recorded_centers),
        headings=np.asarray(recorded_headings),
        first_arrival_times=first_arrival,
        final_goal_distances=final_distance,
        minimum_physical_pair_distance=min_physical_pair,
        minimum_physical_obstacle_clearance=min_physical_obstacle,
        minimum_virtual_pair_distance=min_virtual_pair,
        minimum_virtual_obstacle_clearance=min_virtual_obstacle,
        maximum_linear_speed=max_linear,
        maximum_yaw_rate=max_yaw,
        minimum_yaw_scale=min_yaw_scale,
        nonconverged_steps=nonconverged,
        feasibility_restoration_steps=feasibility_restorations,
        solver_nonconverged_calls=solver_nonconverged_calls,
        actuator_contraction_calls=actuator_contractions,
        actuator_restoration_calls=actuator_restorations,
        witness_restoration_calls=witness_restorations,
        common_contraction_calls=common_contractions,
        hqp_restoration_calls=hqp_restorations,
        projection_iteration_cap_calls=projection_iteration_caps,
        hqp_active_calls=hqp_active_calls,
        hqp_nonconverged_calls=hqp_nonconverged_calls,
        minimum_hqp_progress_retention=(
            minimum_hqp_progress_retention
        ),
        tangent_fallback_steps=tangent_fallbacks,
        tangent_solver_nonconverged_steps=(
            tangent_solver_nonconverged
        ),
        cluster_escape_steps=cluster_escape_steps,
        cluster_escape_component_steps=cluster_escape_component_steps,
        minimum_cbf_residual=min_cbf_residual,
        minimum_tangent_residual=min_tangent_residual,
        static_certificate_candidate_steps=static_candidate_steps,
        static_certificate_excluded_projection_steps=(
            static_excluded_projection_steps
        ),
        static_certificate_steps=static_steps,
        static_certificate_violations=static_violations,
        certified_bridge_steps=certified_bridge_steps,
        certified_bridge_projection_calls=(
            certified_bridge_projection_calls
        ),
        certified_bridge_row_rejections=(
            certified_bridge_row_rejections
        ),
        certified_bridge_progress_violations=(
            certified_bridge_progress_violations
        ),
        minimum_certified_bridge_progress_residual=(
            minimum_certified_bridge_progress_residual
        ),
    )
