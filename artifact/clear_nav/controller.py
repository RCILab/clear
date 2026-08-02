"""Constraint-Tangent Circulation controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    DEFAULT_CONTROLLER,
    DEFAULT_PROTOCOL,
    ControllerConfig,
    Protocol,
)
from .geometry import Arena, Circle, Rectangle, rotate90
from .safety import project_halfspaces

Array = np.ndarray


def _smooth_gate_batch(clearance: Array, radius: float) -> Array:
    """Vectorized ``smooth_gate`` with identical branch semantics."""
    clearance = np.asarray(clearance, dtype=float)
    if radius <= 0.0:
        return np.zeros_like(clearance)
    z = 1.0 - clearance / radius
    polynomial = z * z * (3.0 - 2.0 * z)
    return np.where(
        clearance >= radius,
        0.0,
        np.where(clearance <= 0.0, 1.0, polynomial),
    )


def _boundary_batch(
    arena: Arena,
    points: Array,
    padding: float,
) -> tuple[Array, Array]:
    """Evaluate ``Arena.boundary_queries`` in its exact feature order."""
    points = np.asarray(points, dtype=float)
    count = len(points)
    wall = arena.half_width - padding
    x, y = points[:, 0], points[:, 1]
    clearance_columns = [x + wall, wall - x, y + wall, wall - y]
    wall_normals = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    normal_columns = [
        np.broadcast_to(normal, (count, 2)) for normal in wall_normals
    ]

    for obstacle in arena.obstacles:
        if isinstance(obstacle, Circle):
            delta = points - obstacle.center
            distance = np.linalg.norm(delta, axis=1)
            degenerate = distance < 1.0e-12
            safe_distance = np.where(degenerate, 1.0, distance)
            normal = delta / safe_distance[:, None]
            normal[degenerate] = (1.0, 0.0)
            clearance = np.where(
                degenerate,
                -(obstacle.radius + padding),
                distance - obstacle.radius - padding,
            )
        elif isinstance(obstacle, Rectangle):
            half = 0.5 * obstacle.size
            local = points - obstacle.center
            outside = np.abs(local) - half
            positive = np.maximum(outside, 0.0)
            external_distance = np.linalg.norm(positive, axis=1)
            external = external_distance > 1.0e-12

            closest = np.clip(local, -half, half)
            delta = local - closest
            delta_norm = np.linalg.norm(delta, axis=1)
            external_normal = delta / np.where(
                delta_norm > 0.0, delta_norm, 1.0
            )[:, None]

            face_gap = half - np.abs(local)
            axis = np.argmin(face_gap, axis=1)
            rows = np.arange(count)
            sign = np.where(local[rows, axis] >= 0.0, 1.0, -1.0)
            internal_normal = np.zeros((count, 2))
            internal_normal[rows, axis] = sign
            internal_clearance = -face_gap[rows, axis] - padding

            clearance = np.where(
                external,
                external_distance - padding,
                internal_clearance,
            )
            normal = np.where(
                external[:, None], external_normal, internal_normal
            )
        else:  # pragma: no cover - Arena currently admits only these types.
            raise TypeError(type(obstacle))

        clearance_columns.append(clearance)
        normal_columns.append(normal)

    return np.stack(clearance_columns, axis=1), np.stack(normal_columns, axis=1)


@dataclass(frozen=True)
class ControllerAudit:
    goal_command: Array
    raw_circulation: Array
    cluster_circulation: Array
    cone_circulation: Array
    nominal_command: Array
    projected_command: Array
    executed_command: Array
    common_scale: float
    circulation_gains: Array
    active_pair_edges: int
    active_boundary_edges: int
    active_cluster_escapes: int
    tangent_constraints: int
    tangent_minimum_residual: float
    tangent_converged: bool
    tangent_solver_converged: bool
    cbf_minimum_residual: float
    cbf_converged: bool
    cbf_solver_converged: bool
    cbf_sweeps: int
    tangent_margin: float
    global_tangent_margin: float
    tangent_norm: float


class CLEARController:
    def __init__(
        self,
        protocol: Protocol = DEFAULT_PROTOCOL,
        config: ControllerConfig = DEFAULT_CONTROLLER,
    ) -> None:
        self.protocol = protocol
        self.config = config
        self._cluster_tokens: dict[tuple[int, ...], Array] = {}
        self._terminal_captured = np.zeros(0, dtype=bool)
        self._geometry_positions: Array | None = None
        self._geometry_arena: Arena | None = None
        self._geometry_cache: (
            tuple[Array, Array, Array, Array, Array, Array] | None
        ) = None
        self._tangent_dual = np.zeros(0)
        self._cbf_dual = np.zeros(0)

    def reset(self) -> None:
        """Clear event memory before a logically independent rollout."""
        self._cluster_tokens.clear()
        self._terminal_captured = np.zeros(0, dtype=bool)
        self._geometry_positions = None
        self._geometry_arena = None
        self._geometry_cache = None
        self._tangent_dual = np.zeros(0)
        self._cbf_dual = np.zeros(0)

    @property
    def cluster_tokens(self) -> dict[tuple[int, ...], Array]:
        """Return a defensive snapshot of the event-held passage tokens."""
        return {
            key: direction.copy()
            for key, direction in self._cluster_tokens.items()
        }

    def _prepare_geometry(self, positions: Array, arena: Arena) -> None:
        """Cache pair and boundary geometry once for the current command."""
        positions = np.asarray(positions, dtype=float)
        if (
            self._geometry_positions is positions
            and self._geometry_arena is arena
            and self._geometry_cache is not None
        ):
            return
        count = len(positions)
        left, right = np.triu_indices(count, k=1)
        delta = positions[left] - positions[right]
        distance = np.linalg.norm(delta, axis=1)
        degenerate = distance < 1.0e-12
        safe_distance = np.where(degenerate, 1.0, distance)
        normal = delta / safe_distance[:, None]
        normal[degenerate] = (1.0, 0.0)
        boundary_clearance, boundary_normal = _boundary_batch(
            arena,
            positions,
            self.protocol.robot_clearance,
        )
        self._geometry_positions = positions
        self._geometry_arena = arena
        self._geometry_cache = (
            left,
            right,
            distance,
            normal,
            boundary_clearance,
            boundary_normal,
        )

    def _geometry_data(
        self,
        positions: Array,
        arena: Arena,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        self._prepare_geometry(positions, arena)
        assert self._geometry_cache is not None
        return self._geometry_cache

    @staticmethod
    def _union_components(
        count: int,
        left: Array,
        right: Array,
        mask: Array,
    ) -> list[list[int]]:
        """Preserve the loop formulation's deterministic union-find order."""
        parent = list(range(count))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for first, second in zip(left[mask], right[mask]):
            root_first = find(int(first))
            root_second = find(int(second))
            if root_first != root_second:
                parent[root_second] = root_first

        groups: dict[int, list[int]] = {}
        for index in range(count):
            groups.setdefault(find(index), []).append(index)
        return list(groups.values())

    def _update_terminal_capture(
        self,
        positions: Array,
        goals: Array,
        arena: Arena,
    ) -> None:
        """Latch near goals without suppressing clutter-passage circulation.

        In open scenes a wider latch removes the residual pair circulation
        that otherwise balances goal attraction near dense exchange goals.
        Static-obstacle scenes retain the arrival-scale latch because their
        near-goal circulation may still be required to clear a boundary.
        """
        count = len(positions)
        if len(self._terminal_captured) != count:
            self._terminal_captured = np.zeros(count, dtype=bool)
        if not self.config.terminal_capture_hysteresis:
            self._terminal_captured.fill(False)
            return
        distance = np.linalg.norm(np.asarray(goals) - np.asarray(positions), axis=1)
        capture_radius = (
            self.config.terminal_open_capture_radius
            if not arena.obstacles
            else self.config.terminal_capture_radius
        )
        self._terminal_captured[
            distance > self.config.terminal_release_radius
        ] = False
        self._terminal_captured[
            distance <= capture_radius
        ] = True

    def goal_field(self, positions: Array, goals: Array) -> Array:
        error = np.asarray(goals) - np.asarray(positions)
        distance = np.linalg.norm(error, axis=1)
        direction = error / np.maximum(distance[:, None], 1.0e-12)
        magnitude = self.protocol.speed_limit * np.tanh(
            self.config.goal_gain * distance / self.config.goal_slow_radius
        )
        command = magnitude[:, None] * direction
        command[distance <= 1.0e-12] = 0.0
        return command

    def _geometric_row_blocks(
        self,
        positions: Array,
        arena: Arena,
        *,
        tangent_only: bool,
        speed_limit: float | None = None,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        """Return compact pair/boundary data in canonical row order."""
        left, right, distance, normal, boundary_h, boundary_n = (
            self._geometry_data(positions, arena)
        )
        effective_speed_limit = (
            self.protocol.speed_limit
            if speed_limit is None
            else float(speed_limit)
        )
        pair_limit = (
            self.config.tangent_band
            if tangent_only
            else 2.0 * effective_speed_limit / self.config.cbf_rate
        )
        boundary_limit = (
            self.config.tangent_band
            if tangent_only
            else effective_speed_limit / self.config.cbf_rate
        )
        pair_clearance = distance - self.protocol.pair_clearance
        pair_mask = pair_clearance <= pair_limit
        pair_left = left[pair_mask]
        pair_right = right[pair_mask]
        pair_normal = normal[pair_mask]
        kept_clearance = pair_clearance[pair_mask]
        pair_count = len(pair_left)
        pair_bounds = (
            np.zeros(pair_count)
            if tangent_only
            else -self.config.cbf_rate * kept_clearance
        )

        boundary_mask = boundary_h <= boundary_limit
        robot_index, feature_index = np.nonzero(boundary_mask)
        boundary_count = len(robot_index)
        kept_normals = boundary_n[robot_index, feature_index]
        boundary_bounds = (
            np.zeros(boundary_count)
            if tangent_only
            else -self.config.cbf_rate
            * boundary_h[robot_index, feature_index]
        )

        return (
            pair_left,
            pair_right,
            pair_normal,
            pair_bounds,
            robot_index,
            kept_normals,
            boundary_bounds,
        )

    def _geometric_rows(
        self,
        positions: Array,
        arena: Arena,
        *,
        tangent_only: bool,
        speed_limit: float | None = None,
    ) -> tuple[list[Array], list[float], int, int]:
        """Build rows in loop-identical pair-first, robot-major order."""
        count = len(positions)
        (
            pair_left,
            pair_right,
            pair_normal,
            pair_bounds,
            robot_index,
            kept_normals,
            boundary_bounds,
        ) = self._geometric_row_blocks(
            positions,
            arena,
            tangent_only=tangent_only,
            speed_limit=speed_limit,
        )
        pair_count = len(pair_left)
        pair_rows = np.zeros((pair_count, 2 * count))
        pair_index = np.arange(pair_count)
        pair_rows[pair_index, 2 * pair_left] = pair_normal[:, 0]
        pair_rows[pair_index, 2 * pair_left + 1] = pair_normal[:, 1]
        pair_rows[pair_index, 2 * pair_right] = -pair_normal[:, 0]
        pair_rows[pair_index, 2 * pair_right + 1] = -pair_normal[:, 1]

        boundary_count = len(robot_index)
        boundary_rows = np.zeros((boundary_count, 2 * count))
        boundary_index = np.arange(boundary_count)
        boundary_rows[boundary_index, 2 * robot_index] = kept_normals[:, 0]
        boundary_rows[boundary_index, 2 * robot_index + 1] = kept_normals[:, 1]

        rows = list(pair_rows) + list(boundary_rows)
        bounds = list(pair_bounds) + list(boundary_bounds)
        return rows, bounds, pair_count, boundary_count

    def circulation_field(
        self,
        positions: Array,
        goals: Array,
        arena: Arena,
        goal_command: Array,
    ) -> tuple[Array, int, int]:
        """Vectorized loop-equivalent pair and boundary circulation."""
        positions = np.asarray(positions)
        count = len(positions)
        left, right, distance, normal, boundary_h, boundary_n = (
            self._geometry_data(positions, arena)
        )
        circulation = np.zeros_like(positions, dtype=float)
        chi = float(self.config.handedness)

        pair_clearance = distance - self.protocol.pair_clearance
        pair_gate = _smooth_gate_batch(
            pair_clearance,
            self.config.pair_sense_radius - self.protocol.pair_clearance,
        )
        relative_goal = goal_command[left] - goal_command[right]
        closing = np.maximum(
            0.0,
            -np.einsum("ij,ij->i", normal, relative_goal),
        )
        conflict = np.minimum(
            1.0,
            (closing + self.config.closing_bias)
            / (self.protocol.speed_limit + self.config.closing_bias),
        )
        if len(self._terminal_captured) == count:
            captured_pair = (
                self._terminal_captured[left]
                & self._terminal_captured[right]
            )
        else:
            captured_pair = np.zeros(len(left), dtype=bool)
        weight = pair_gate * conflict * np.where(captured_pair, 0.0, 1.0)
        contribution = weight[:, None] * (chi * rotate90(normal))
        np.add.at(circulation, left, contribution)
        np.add.at(circulation, right, -contribution)
        active_pairs = int(np.count_nonzero(weight != 0.0))

        boundary_gate = _smooth_gate_batch(
            boundary_h,
            self.config.obstacle_sense_radius,
        )
        inward = np.maximum(
            0.0,
            -np.einsum("nmj,nj->nm", boundary_n, goal_command),
        )
        boundary_conflict = np.minimum(
            1.0,
            (inward + self.config.closing_bias)
            / (self.protocol.speed_limit + self.config.closing_bias),
        )
        positive_tangent = rotate90(boundary_n)
        if self.config.boundary_progress_aligned:
            alignment = np.einsum(
                "nmj,nj->nm",
                positive_tangent,
                goal_command,
            )
            boundary_sense = np.where(
                alignment > 1.0e-12,
                1.0,
                np.where(alignment < -1.0e-12, -1.0, chi),
            )
        else:
            boundary_sense = chi
        coefficient = (
            self.config.obstacle_circulation_gain
            / self.config.circulation_gain
        )
        circulation += coefficient * np.einsum(
            "nm,nmj->nj",
            boundary_gate * boundary_conflict * boundary_sense,
            positive_tangent,
        )
        active_boundaries = int(np.count_nonzero(boundary_gate != 0.0))
        return circulation, active_pairs, active_boundaries

    def cluster_escape_field(
        self,
        positions: Array,
        goal_command: Array,
        arena: Arena,
        tangent_matrix: Array,
    ) -> tuple[Array, int]:
        """Construct loop-equivalent rigid translations at boundary bridges."""
        positions = np.asarray(positions)
        count = len(positions)
        cluster = np.zeros_like(positions, dtype=float)
        active = 0
        if self.config.cluster_escape_gain <= 0.0:
            return cluster, active

        left, right, distance, _, boundary_h, boundary_n = (
            self._geometry_data(positions, arena)
        )
        pair_clearance = distance - self.protocol.pair_clearance
        pair_gates = _smooth_gate_batch(
            pair_clearance,
            self.config.cluster_pair_band,
        )
        components = self._union_components(
            count,
            left,
            right,
            pair_clearance < self.config.tangent_band,
        )
        boundary_close = boundary_h < self.config.tangent_band
        boundary_gates = _smooth_gate_batch(
            boundary_h,
            self.config.cluster_boundary_band,
        )

        active_keys: set[tuple[int, ...]] = set()
        for component in components:
            if len(component) < 2:
                continue
            component_array = np.asarray(component, dtype=int)
            in_component = np.zeros(count, dtype=bool)
            in_component[component_array] = True
            pair_selection = in_component[left] & in_component[right]
            pair_gate = (
                float(np.max(pair_gates[pair_selection]))
                if pair_selection.any()
                else 0.0
            )
            if pair_gate == 0.0:
                continue

            local_mask = boundary_close[component_array]
            robot_offset, feature_index = np.nonzero(local_mask)
            normals = [
                boundary_n[component_array[offset], feature]
                for offset, feature in zip(robot_offset, feature_index)
            ]
            boundary_gate = float(
                np.max(boundary_gates[component_array])
            )
            if boundary_gate == 0.0 or not normals:
                continue

            token_key = tuple(component)
            active_keys.add(token_key)
            direction = self._cluster_tokens.get(token_key)
            if direction is not None and not np.all(
                np.asarray(normals) @ direction >= -1.0e-10
            ):
                direction = None
            if direction is None:
                direction = self._common_tangent_direction(
                    normals,
                    np.sum(goal_command[component_array], axis=0),
                )
                if (
                    direction is not None
                    and self.config.cluster_escape_hysteresis
                ):
                    self._cluster_tokens[token_key] = direction.copy()
            if direction is None:
                continue

            weight = (
                self.config.cluster_escape_gain
                / self.config.circulation_gain
                * pair_gate
                * boundary_gate
            )
            contribution = np.zeros_like(cluster)
            contribution[component_array] = weight * direction
            if len(tangent_matrix) and np.min(
                tangent_matrix @ contribution.reshape(-1)
            ) < -self.config.projection_tolerance:
                continue
            cluster += contribution
            active += 1

        if self.config.cluster_escape_hysteresis:
            for key in list(self._cluster_tokens):
                if key not in active_keys:
                    del self._cluster_tokens[key]
        return cluster, active

    @staticmethod
    def _common_tangent_direction(
        normals: list[Array],
        aggregate_goal: Array,
    ) -> Array | None:
        """Return a unit rigid velocity in all supplied boundary half-spaces."""
        aggregate_goal = np.asarray(aggregate_goal, dtype=float)
        candidates: list[Array] = []
        goal_norm = float(np.linalg.norm(aggregate_goal))
        if goal_norm > 1.0e-12:
            candidates.append(aggregate_goal / goal_norm)
        for normal in normals:
            normal = np.asarray(normal, dtype=float)
            candidates.extend((normal, rotate90(normal), -rotate90(normal)))
        summed_normal = np.sum(normals, axis=0)
        summed_norm = float(np.linalg.norm(summed_normal))
        if summed_norm > 1.0e-12:
            candidates.append(summed_normal / summed_norm)

        normal_matrix = np.asarray(normals)
        feasible: list[Array] = []
        for candidate in candidates:
            magnitude = float(np.linalg.norm(candidate))
            if magnitude <= 1.0e-12:
                continue
            direction = candidate / magnitude
            if np.all(normal_matrix @ direction >= -1.0e-10):
                feasible.append(direction)
        if not feasible:
            return None
        scores = np.asarray([aggregate_goal @ direction for direction in feasible])
        best = int(np.argmax(scores))
        if scores[best] <= 1.0e-8:
            return None
        return feasible[best]

    def _circulation_components(self, positions: Array) -> list[list[int]]:
        """Connected components with loop-identical strict edge semantics."""
        positions = np.asarray(positions)
        count = len(positions)
        if (
            self._geometry_positions is positions
            and self._geometry_cache is not None
        ):
            left, right, distance = self._geometry_cache[:3]
        else:
            left, right = np.triu_indices(count, k=1)
            distance = np.linalg.norm(
                positions[left] - positions[right],
                axis=1,
            )
        return self._union_components(
            count,
            left,
            right,
            distance < self.config.pair_sense_radius,
        )

    def command(
        self,
        positions: Array,
        goals: Array,
        arena: Arena,
        *,
        project_safety: bool = True,
    ) -> ControllerAudit:
        positions = np.asarray(positions, dtype=float)
        goals = np.asarray(goals, dtype=float)
        n_robots = len(positions)
        # The cache is command-local: callers may reuse and mutate the same
        # positions array between invocations.
        self._geometry_positions = None
        self._geometry_arena = None
        self._geometry_cache = None
        self._prepare_geometry(positions, arena)
        self._update_terminal_capture(positions, goals, arena)
        goal = self.goal_field(positions, goals)
        raw, active_pairs, active_boundaries = self.circulation_field(
            positions, goals, arena, goal
        )

        tangent_rows, _, _, _ = self._geometric_rows(
            positions, arena, tangent_only=True
        )
        if tangent_rows:
            tangent_matrix = np.asarray(tangent_rows)
            tangent_result = project_halfspaces(
                raw.reshape(-1),
                tangent_matrix,
                np.zeros(len(tangent_rows)),
                tolerance=self.config.projection_tolerance,
                max_sweeps=self.config.tangent_projection_max_sweeps,
                initial_dual=self._tangent_dual,
            )
            self._tangent_dual = tangent_result.dual.copy()
            tangent_converged = tangent_result.converged
            tangent_solver_converged = tangent_result.optimizer_converged
            if tangent_converged:
                cone = tangent_result.value.reshape(n_robots, 2)
                tangent_residual = tangent_result.minimum_residual
            else:
                # The zero vector belongs to every tangent cone.  Falling back
                # to it preserves the structural claim instead of silently
                # using an uncertified approximate projection.
                cone = np.zeros_like(raw)
                tangent_residual = 0.0
        else:
            self._tangent_dual = np.zeros(0)
            tangent_matrix = np.zeros((0, 2 * n_robots))
            cone = raw.copy()
            tangent_residual = float("inf")
            tangent_converged = True
            tangent_solver_converged = True

        cluster, active_cluster_escapes = self.cluster_escape_field(
            positions, goal, arena, tangent_matrix
        )
        cone = cone + cluster
        if len(tangent_matrix):
            tangent_residual = float(
                np.min(tangent_matrix @ cone.reshape(-1))
            )

        # Select one gain per circulation-connected component.  Whenever that
        # component has a nonzero tangent witness, its nominal dot witness is
        # made strictly positive.  This removes exact fixed-gain cancellation.
        gains = np.full(n_robots, self.config.circulation_gain)
        component_margins: list[float] = []
        for component in self._circulation_components(positions):
            indices = np.asarray(component, dtype=int)
            z_component = cone[indices].reshape(-1)
            g_component = goal[indices].reshape(-1)
            norm_sq = float(z_component @ z_component)
            if norm_sq <= 1.0e-14:
                continue
            if self.config.adaptive_certificate_gain:
                required = (
                    -float(g_component @ z_component) / norm_sq
                    + self.config.certificate_margin_gain
                )
                gain = max(self.config.circulation_gain, required)
            else:
                gain = self.config.circulation_gain
            gains[indices] = gain
            component_margins.append(
                float(g_component @ z_component) + gain * norm_sq
            )
        nominal = goal + gains[:, None] * cone
        if project_safety:
            safety_rows, safety_bounds, _, _ = self._geometric_rows(
                positions, arena, tangent_only=False
            )
            if safety_rows:
                matrix = np.asarray(safety_rows)
                lower = np.asarray(safety_bounds)
            else:
                matrix = np.zeros((0, 2 * n_robots))
                lower = np.zeros(0)
            result = project_halfspaces(
                nominal.reshape(-1),
                matrix,
                lower,
                tolerance=self.config.projection_tolerance,
                max_sweeps=self.config.projection_max_sweeps,
                initial_dual=self._cbf_dual,
            )
            self._cbf_dual = result.dual.copy()
            filtered = result.value.reshape(n_robots, 2)
            peak_speed = float(np.max(np.linalg.norm(filtered, axis=1)))
            common_scale = min(
                1.0,
                self.protocol.speed_limit / max(peak_speed, 1.0e-12),
            )
            executed = common_scale * filtered
            if len(lower):
                scaled_residual = float(
                    np.min(matrix @ executed.reshape(-1) - lower)
                )
            else:
                scaled_residual = float("inf")
            cbf_residual = min(
                scaled_residual,
                self.protocol.speed_limit
                - float(np.max(np.linalg.norm(executed, axis=1))),
            )
            cbf_converged = (
                result.converged
                and cbf_residual >= -self.config.projection_tolerance
            )
            cbf_solver_converged = result.optimizer_converged
            cbf_sweeps = result.sweeps
        else:
            # Native-unicycle callers apply the geometric rows and actuator
            # bounds together in their final projection.
            self._cbf_dual = np.zeros(0)
            filtered = nominal.copy()
            executed = nominal.copy()
            common_scale = 1.0
            scaled_residual = float("inf")
            cbf_residual = float("inf")
            cbf_converged = True
            cbf_solver_converged = True
            cbf_sweeps = 0

        # Feasible-direction witness.  At a safe state, every sufficiently
        # small positive multiple of `cone` satisfies the geometric CBF
        # inequalities.  The projection variational inequality
        # therefore proves u* != 0 for each audited component.
        tangent_margin = min(component_margins) if component_margins else 0.0
        global_tangent_margin = float(
            nominal.reshape(-1) @ cone.reshape(-1)
        )
        tangent_norm = float(np.linalg.norm(cone))

        return ControllerAudit(
            goal_command=goal,
            raw_circulation=raw,
            cluster_circulation=cluster,
            cone_circulation=cone,
            nominal_command=nominal,
            projected_command=filtered,
            executed_command=executed,
            common_scale=common_scale,
            circulation_gains=gains,
            active_pair_edges=active_pairs,
            active_boundary_edges=active_boundaries,
            active_cluster_escapes=active_cluster_escapes,
            tangent_constraints=len(tangent_rows),
            tangent_minimum_residual=tangent_residual,
            tangent_converged=tangent_converged,
            tangent_solver_converged=tangent_solver_converged,
            cbf_minimum_residual=cbf_residual,
            cbf_converged=cbf_converged,
            cbf_solver_converged=cbf_solver_converged,
            cbf_sweeps=cbf_sweeps,
            tangent_margin=tangent_margin,
            global_tangent_margin=global_tangent_margin,
            tangent_norm=tangent_norm,
        )
