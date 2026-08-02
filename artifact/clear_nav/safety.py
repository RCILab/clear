"""Small deterministic projection solver used by CLEAR.

The problem is

    minimize 0.5 ||u - u_nom||^2  subject to A u >= b.

The native bounded-unicycle path uses sparse OSQP, while the structural
tangent projection retains Hildreth's dual coordinate method with projected
successive over-relaxation.  Both solve the same half-space projection.  The
returned primal residual is always audited; a caller must not treat an
infeasible result as certified.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def maximize_halfspace_linear_osqp(
    direction: Array,
    matrix: Array,
    lower: Array,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 4000,
    regularization: float = 1.0e-6,
    initial_dual: Array | None = None,
) -> ProjectionResult:
    """Maximize a linear task over half-spaces with tiny norm regularization.

    This is the first level of the native-unicycle hierarchy.  The actuator
    box makes the linear task bounded; the positive quadratic term only
    selects a unique point on an optimal face and improves OSQP conditioning.
    """
    import osqp
    from scipy import sparse

    direction = np.asarray(direction, dtype=float).reshape(-1)
    matrix = np.asarray(matrix, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != direction.size:
        raise ValueError("matrix shape does not match direction")
    if matrix.shape[0] != lower.size:
        raise ValueError("one lower bound is required per row")
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if not len(lower):
        raise ValueError("a bounded feasible set is required")

    solver = osqp.OSQP()
    solver.setup(
        P=regularization
        * sparse.eye(direction.size, format="csc"),
        q=-direction,
        A=sparse.csc_matrix(matrix),
        l=lower,
        u=np.full(len(lower), np.inf),
        eps_abs=0.1 * tolerance,
        eps_rel=0.1 * tolerance,
        max_iter=max_iterations,
        polishing=False,
        adaptive_rho=True,
        check_termination=10,
        verbose=False,
    )
    if initial_dual is not None:
        candidate = np.asarray(initial_dual, dtype=float).reshape(-1)
        if len(candidate) == len(lower) and np.all(np.isfinite(candidate)):
            solver.warm_start(y=-np.maximum(candidate, 0.0))
    solved = solver.solve(raise_error=False)
    value = np.asarray(solved.x, dtype=float).reshape(-1)
    if value.size != direction.size or not np.all(np.isfinite(value)):
        value = np.zeros_like(direction)
    residual = matrix @ value - lower
    minimum = float(np.min(residual))
    converged = bool(minimum >= -tolerance)
    raw_dual = np.asarray(solved.y, dtype=float).reshape(-1)
    dual = (
        np.maximum(-raw_dual, 0.0)
        if len(raw_dual) == len(lower) and np.all(np.isfinite(raw_dual))
        else np.zeros(len(lower))
    )
    return ProjectionResult(
        value=value,
        converged=converged,
        sweeps=int(solved.info.iter),
        minimum_residual=minimum,
        displacement=float(np.linalg.norm(value)),
        dual=dual,
        solver_converged=int(solved.info.status_val) in (1, 2),
        solver_status=str(solved.info.status),
    )


def project_halfspaces_osqp(
    nominal: Array,
    matrix: Array,
    lower: Array,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 4000,
    initial_dual: Array | None = None,
) -> ProjectionResult:
    """Solve the Euclidean half-space projection with sparse OSQP."""
    import osqp
    from scipy import sparse

    nominal = np.asarray(nominal, dtype=float).reshape(-1)
    matrix = np.asarray(matrix, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != nominal.size:
        raise ValueError("matrix shape does not match nominal")
    if matrix.shape[0] != lower.size:
        raise ValueError("one lower bound is required per row")
    if not len(lower):
        return ProjectionResult(
            nominal.copy(),
            True,
            0,
            float("inf"),
            0.0,
            np.zeros(0),
        )

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.eye(nominal.size, format="csc"),
        q=-nominal,
        A=sparse.csc_matrix(matrix),
        l=lower,
        u=np.full(len(lower), np.inf),
        eps_abs=0.1 * tolerance,
        eps_rel=0.1 * tolerance,
        max_iter=max_iterations,
        polishing=False,
        adaptive_rho=True,
        check_termination=10,
        verbose=False,
    )
    if initial_dual is not None:
        candidate = np.asarray(initial_dual, dtype=float).reshape(-1)
        if len(candidate) == len(lower) and np.all(np.isfinite(candidate)):
            solver.warm_start(y=-np.maximum(candidate, 0.0))
    solved = solver.solve(raise_error=False)
    value = np.asarray(solved.x, dtype=float).reshape(-1)
    if value.size != nominal.size or not np.all(np.isfinite(value)):
        value = np.zeros_like(nominal)
    residual = matrix @ value - lower
    minimum = float(np.min(residual))
    # Certification depends on primal CBF/input feasibility.  OSQP may reach
    # its dual-optimality iteration cap at a nearly degenerate contact while
    # already satisfying every safety row to the declared tolerance.
    converged = bool(minimum >= -tolerance)
    raw_dual = np.asarray(solved.y, dtype=float).reshape(-1)
    dual = (
        np.maximum(-raw_dual, 0.0)
        if len(raw_dual) == len(lower) and np.all(np.isfinite(raw_dual))
        else np.zeros(len(lower))
    )
    return ProjectionResult(
        value=value,
        converged=converged,
        sweeps=int(solved.info.iter),
        minimum_residual=minimum,
        displacement=float(np.linalg.norm(value - nominal)),
        dual=dual,
        solver_converged=int(solved.info.status_val) in (1, 2),
        solver_status=str(solved.info.status),
    )


def _orthogonal_row_groups(matrix: Array) -> list[Array]:
    """Greedily color rows whose nonzero supports do not overlap.

    Updates within one returned group commute exactly, so a Hildreth sweep can
    evaluate that group with matrix operations rather than Python row loops.
    """
    occupied: list[set[int]] = []
    groups: list[list[int]] = []
    for row_index, row in enumerate(matrix):
        support = set(np.flatnonzero(row))
        for group_index, used in enumerate(occupied):
            if support.isdisjoint(used):
                groups[group_index].append(row_index)
                used.update(support)
                break
        else:
            groups.append([row_index])
            occupied.append(set(support))
    return [np.asarray(group, dtype=int) for group in groups]


@dataclass(frozen=True)
class ProjectionResult:
    """Projection output with feasibility and solver termination kept separate.

    ``converged`` is the historical primal-feasibility flag used by controller
    safety checks.  ``solver_converged`` records whether the optimizer itself
    reached a solved termination status.  The distinction matters when an
    iteration-limited iterate is already feasible or is subsequently repaired.
    """

    value: Array
    converged: bool
    sweeps: int
    minimum_residual: float
    displacement: float
    dual: Array
    feasibility_restored: bool = False
    solver_dual: Array | None = None
    solver_converged: bool | None = None
    solver_status: str | None = None

    @property
    def command_feasible(self) -> bool:
        return self.converged

    @property
    def optimizer_converged(self) -> bool:
        return (
            self.converged
            if self.solver_converged is None
            else self.solver_converged
        )


class OSQPProjectionWorkspace:
    """Persistent sparse OSQP workspaces keyed by constraint sparsity.

    Reusing a matching solver avoids symbolic setup and preserves both the
    primal and signed OSQP dual iterate.  A small LRU cache tolerates the few
    active-neighbour patterns that recur as robots move.
    """

    def __init__(self, max_cached_patterns: int = 8) -> None:
        if max_cached_patterns < 1:
            raise ValueError("max_cached_patterns must be positive")
        self.max_cached_patterns = int(max_cached_patterns)
        self._entries: OrderedDict[
            tuple[int, int, bytes, bytes],
            tuple[object, Array, Array],
        ] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def project(
        self,
        nominal: Array,
        matrix,
        lower: Array,
        upper: Array,
        *,
        tolerance: float = 1.0e-8,
        solver_tolerance: float | None = None,
        solver_margin: float | Array = 0.0,
        max_iterations: int = 4000,
    ) -> ProjectionResult:
        """Project onto sparse two-sided rows ``lower <= A z <= upper``."""
        import osqp
        from scipy import sparse

        nominal = np.asarray(nominal, dtype=float).reshape(-1)
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        constraint = sparse.csc_matrix(matrix, dtype=float)
        constraint.sum_duplicates()
        constraint.sort_indices()
        if constraint.shape[1] != nominal.size:
            raise ValueError("matrix shape does not match nominal")
        if constraint.shape[0] != lower.size or lower.size != upper.size:
            raise ValueError("one lower and upper bound is required per row")
        if np.any(lower > upper):
            raise ValueError("lower bound exceeds upper bound")
        effective_solver_tolerance = (
            tolerance
            if solver_tolerance is None
            else float(solver_tolerance)
        )
        if effective_solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be positive")
        margin = np.asarray(solver_margin, dtype=float)
        if margin.ndim == 0:
            margin = np.full(len(lower), float(margin))
        else:
            margin = margin.reshape(-1)
        if margin.size != lower.size:
            raise ValueError("solver_margin must be scalar or one per row")
        if np.any(margin < 0.0):
            raise ValueError("solver_margin must be nonnegative")
        solver_lower = lower.copy()
        solver_upper = upper.copy()
        finite_lower = np.isfinite(solver_lower)
        finite_upper = np.isfinite(solver_upper)
        solver_lower[finite_lower] += margin[finite_lower]
        solver_upper[finite_upper] -= margin[finite_upper]
        if np.any(solver_lower > solver_upper):
            raise ValueError("solver_margin makes a row infeasible")
        if not len(lower):
            return ProjectionResult(
                nominal.copy(),
                True,
                0,
                float("inf"),
                0.0,
                np.zeros(0),
                solver_dual=np.zeros(0),
            )

        key = (
            constraint.shape[0],
            constraint.shape[1],
            constraint.indptr.tobytes(),
            constraint.indices.tobytes(),
        )
        entry = self._entries.pop(key, None)
        if entry is None:
            solver = osqp.OSQP()
            solver.setup(
                P=sparse.eye(nominal.size, format="csc"),
                q=-nominal,
                A=constraint,
                l=solver_lower,
                u=solver_upper,
                eps_abs=effective_solver_tolerance,
                eps_rel=effective_solver_tolerance,
                max_iter=max_iterations,
                polishing=False,
                adaptive_rho=True,
                check_termination=10,
                warm_starting=True,
                verbose=False,
            )
            previous_value = np.zeros(0)
            previous_dual = np.zeros(0)
        else:
            solver, previous_value, previous_dual = entry
            solver.update(
                q=-nominal,
                l=solver_lower,
                u=solver_upper,
                Ax=constraint.data,
            )
            solver.update_settings(
                eps_abs=effective_solver_tolerance,
                eps_rel=effective_solver_tolerance,
                max_iter=max_iterations,
            )
            if (
                previous_value.size == nominal.size
                and previous_dual.size == lower.size
                and np.all(np.isfinite(previous_value))
                and np.all(np.isfinite(previous_dual))
            ):
                solver.warm_start(x=previous_value, y=previous_dual)

        solved = solver.solve(raise_error=False)
        value = (
            np.asarray(solved.x, dtype=float).reshape(-1)
            if solved.x is not None
            else np.zeros(0)
        )
        if value.size != nominal.size or not np.all(np.isfinite(value)):
            value = np.zeros_like(nominal)
        raw_dual = (
            np.asarray(solved.y, dtype=float).reshape(-1)
            if solved.y is not None
            else np.zeros(0)
        )
        if raw_dual.size != lower.size or not np.all(np.isfinite(raw_dual)):
            raw_dual = np.zeros(len(lower))

        row_value = np.asarray(constraint @ value).reshape(-1)
        residual_parts = []
        finite_lower = np.isfinite(lower)
        if np.any(finite_lower):
            residual_parts.append(row_value[finite_lower] - lower[finite_lower])
        finite_upper = np.isfinite(upper)
        if np.any(finite_upper):
            residual_parts.append(upper[finite_upper] - row_value[finite_upper])
        minimum = float(
            np.min(np.concatenate(residual_parts))
            if residual_parts
            else np.inf
        )
        converged = bool(minimum >= -tolerance)
        self._entries[key] = (solver, value.copy(), raw_dual.copy())
        while len(self._entries) > self.max_cached_patterns:
            self._entries.popitem(last=False)

        return ProjectionResult(
            value=value,
            converged=converged,
            sweeps=int(solved.info.iter),
            minimum_residual=minimum,
            displacement=float(np.linalg.norm(value - nominal)),
            dual=np.maximum(-raw_dual, 0.0),
            solver_dual=raw_dual,
            solver_converged=int(solved.info.status_val) in (1, 2),
            solver_status=str(solved.info.status),
        )


def project_halfspaces(
    nominal: Array,
    matrix: Array,
    lower: Array,
    *,
    tolerance: float = 1.0e-9,
    max_sweeps: int = 500,
    relaxation: float = 1.5,
    acceleration_after_sweeps: int = 16,
    batching_after_sweeps: int = 4,
    batching_minimum_rows: int = 16,
    warm_start_minimum_rows: int = 16,
    initial_dual: Array | None = None,
) -> ProjectionResult:
    nominal = np.asarray(nominal, dtype=float).reshape(-1)
    matrix = np.asarray(matrix, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != nominal.size:
        raise ValueError("matrix shape does not match nominal")
    if matrix.shape[0] != lower.size:
        raise ValueError("one lower bound is required per row")
    if not 0.0 < relaxation < 2.0:
        raise ValueError("relaxation must lie strictly between zero and two")
    if acceleration_after_sweeps < 0:
        raise ValueError("acceleration_after_sweeps must be nonnegative")
    if batching_after_sweeps < 0:
        raise ValueError("batching_after_sweeps must be nonnegative")
    if batching_minimum_rows < 1:
        raise ValueError("batching_minimum_rows must be positive")
    if warm_start_minimum_rows < 1:
        raise ValueError("warm_start_minimum_rows must be positive")
    if not len(lower):
        return ProjectionResult(
            nominal.copy(),
            True,
            0,
            float("inf"),
            0.0,
            np.zeros(0),
        )

    row_norm_sq = np.einsum("ij,ij->i", matrix, matrix)
    if np.any(row_norm_sq <= 1.0e-18):
        raise ValueError("zero constraint row")

    warm_started = False
    if initial_dual is None or len(lower) < warm_start_minimum_rows:
        dual = np.zeros(len(lower))
    else:
        candidate = np.asarray(initial_dual, dtype=float).reshape(-1)
        if len(candidate) == len(lower) and np.all(np.isfinite(candidate)):
            dual = np.maximum(candidate, 0.0).copy()
            warm_started = True
        else:
            dual = np.zeros(len(lower))
    value = nominal + matrix.T @ dual if warm_started else nominal.copy()
    sweeps = 0
    optimizer_converged = False
    orthogonal_groups: list[Array] | None = None
    for sweeps in range(1, max_sweeps + 1):
        largest_change = 0.0
        sweep_relaxation = (
            1.0 if sweeps <= acceleration_after_sweeps else relaxation
        )
        use_batches = (
            len(lower) >= batching_minimum_rows
            and sweeps > batching_after_sweeps
        )
        if not use_batches:
            for k in range(len(lower)):
                old = dual[k]
                violation = lower[k] - float(matrix[k] @ value)
                new = max(
                    0.0,
                    old + sweep_relaxation * violation / row_norm_sq[k],
                )
                change = new - old
                if change:
                    value += change * matrix[k]
                    dual[k] = new
                    largest_change = max(largest_change, abs(change))
        else:
            if orthogonal_groups is None:
                orthogonal_groups = _orthogonal_row_groups(matrix)
            for group in orthogonal_groups:
                old = dual[group]
                violation = lower[group] - matrix[group] @ value
                new = np.maximum(
                    0.0,
                    old + sweep_relaxation * violation / row_norm_sq[group],
                )
                change = new - old
                if np.any(change):
                    value += change @ matrix[group]
                    dual[group] = new
                    largest_change = max(
                        largest_change,
                        float(np.max(np.abs(change))),
                    )
        residual = matrix @ value - lower
        if float(np.min(residual)) >= -tolerance and largest_change <= tolerance:
            optimizer_converged = True
            break

    residual = matrix @ value - lower
    minimum = float(np.min(residual))
    converged = minimum >= -tolerance
    return ProjectionResult(
        value=value,
        converged=converged,
        sweeps=sweeps,
        minimum_residual=minimum,
        displacement=float(np.linalg.norm(value - nominal)),
        dual=dual,
        solver_converged=optimizer_converged,
        solver_status=(
            "solved" if optimizer_converged else "maximum sweeps reached"
        ),
    )


def project_halfspaces_and_balls(
    nominal: Array,
    matrix: Array,
    lower: Array,
    *,
    n_robots: int,
    speed_limit: float,
    tolerance: float = 1.0e-9,
    max_sweeps: int = 500,
) -> ProjectionResult:
    """Dykstra projection onto safety half-spaces and exact velocity balls.

    Unlike a polygonal actuator approximation, the Euclidean balls preserve
    continuous rotation equivariance.
    """
    nominal = np.asarray(nominal, dtype=float).reshape(-1)
    matrix = np.asarray(matrix, dtype=float)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    if nominal.size != 2 * n_robots:
        raise ValueError("nominal size and n_robots disagree")
    if matrix.shape != (len(lower), nominal.size):
        raise ValueError("constraint shape mismatch")

    row_norm_sq = np.einsum("ij,ij->i", matrix, matrix)
    if len(matrix) and np.any(row_norm_sq <= 1.0e-18):
        raise ValueError("zero constraint row")

    value = nominal.copy()
    half_corrections = np.zeros_like(matrix)
    ball_corrections = np.zeros((n_robots, 2))
    sweeps = 0
    optimizer_converged = False
    for sweeps in range(1, max_sweeps + 1):
        largest_move = 0.0
        for k in range(len(lower)):
            shifted = value + half_corrections[k]
            violation = lower[k] - float(matrix[k] @ shifted)
            if violation > 0.0:
                updated = shifted + (violation / row_norm_sq[k]) * matrix[k]
            else:
                updated = shifted
            half_corrections[k] = shifted - updated
            largest_move = max(largest_move, float(np.linalg.norm(updated - value)))
            value = updated

        reshaped = value.reshape(n_robots, 2)
        for robot in range(n_robots):
            shifted = reshaped[robot] + ball_corrections[robot]
            magnitude = float(np.linalg.norm(shifted))
            if magnitude > speed_limit:
                updated = speed_limit * shifted / magnitude
            else:
                updated = shifted
            ball_corrections[robot] = shifted - updated
            largest_move = max(
                largest_move, float(np.linalg.norm(updated - reshaped[robot]))
            )
            reshaped[robot] = updated
        value = reshaped.reshape(-1)

        half_residual = matrix @ value - lower if len(lower) else np.array([np.inf])
        ball_residual = speed_limit - np.linalg.norm(
            value.reshape(n_robots, 2), axis=1
        )
        minimum = min(float(np.min(half_residual)), float(np.min(ball_residual)))
        if minimum >= -tolerance and largest_move <= tolerance:
            optimizer_converged = True
            break

    half_residual = matrix @ value - lower if len(lower) else np.array([np.inf])
    ball_residual = speed_limit - np.linalg.norm(value.reshape(n_robots, 2), axis=1)
    minimum = min(float(np.min(half_residual)), float(np.min(ball_residual)))
    return ProjectionResult(
        value=value,
        converged=minimum >= -tolerance,
        sweeps=sweeps,
        minimum_residual=minimum,
        displacement=float(np.linalg.norm(value - nominal)),
        dual=np.zeros(0),
        solver_converged=optimizer_converged,
        solver_status=(
            "solved" if optimizer_converged else "maximum sweeps reached"
        ),
    )
