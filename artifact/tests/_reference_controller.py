"""Pre-vectorization loop bodies used as an independent parity oracle."""

from __future__ import annotations

import numpy as np

from clear_nav.controller import CLEARController
from clear_nav.geometry import Arena, rotate90, smooth_gate

Array = np.ndarray


class ReferenceCLEARController(CLEARController):
    """CLEAR with the four original Python-loop hot paths restored."""

    def _geometric_rows(
        self,
        positions: Array,
        arena: Arena,
        *,
        tangent_only: bool,
    ) -> tuple[list[Array], list[float], int, int]:
        n_robots = len(positions)
        pair_limit = (
            self.config.tangent_band
            if tangent_only
            else 2.0 * self.protocol.speed_limit / self.config.cbf_rate
        )
        boundary_limit = (
            self.config.tangent_band
            if tangent_only
            else self.protocol.speed_limit / self.config.cbf_rate
        )
        rows: list[Array] = []
        bounds: list[float] = []
        pair_count = 0
        boundary_count = 0

        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                delta = positions[i] - positions[j]
                distance = float(np.linalg.norm(delta))
                if distance < 1.0e-12:
                    normal = np.array([1.0, 0.0])
                else:
                    normal = delta / distance
                clearance = distance - self.protocol.pair_clearance
                if clearance > pair_limit:
                    continue
                row = np.zeros(2 * n_robots)
                row[2 * i : 2 * i + 2] = normal
                row[2 * j : 2 * j + 2] = -normal
                rows.append(row)
                bounds.append(
                    0.0
                    if tangent_only
                    else -self.config.cbf_rate * clearance
                )
                pair_count += 1

        for i, point in enumerate(positions):
            for clearance, normal, _ in arena.boundary_queries(
                point,
                self.protocol.robot_clearance,
            ):
                if clearance > boundary_limit:
                    continue
                row = np.zeros(2 * n_robots)
                row[2 * i : 2 * i + 2] = normal
                rows.append(row)
                bounds.append(
                    0.0
                    if tangent_only
                    else -self.config.cbf_rate * clearance
                )
                boundary_count += 1
        return rows, bounds, pair_count, boundary_count

    def circulation_field(
        self,
        positions: Array,
        goals: Array,
        arena: Arena,
        goal_command: Array,
    ) -> tuple[Array, int, int]:
        positions = np.asarray(positions)
        n_robots = len(positions)
        circulation = np.zeros_like(positions, dtype=float)
        active_pairs = 0
        active_boundaries = 0
        chi = float(self.config.handedness)

        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                delta = positions[i] - positions[j]
                distance = float(np.linalg.norm(delta))
                if distance < 1.0e-12:
                    normal = np.array([1.0, 0.0])
                else:
                    normal = delta / distance
                clearance = distance - self.protocol.pair_clearance
                gate = smooth_gate(
                    clearance,
                    self.config.pair_sense_radius
                    - self.protocol.pair_clearance,
                )
                if gate == 0.0:
                    continue
                relative_goal = goal_command[i] - goal_command[j]
                closing = max(0.0, -float(normal @ relative_goal))
                conflict = min(
                    1.0,
                    (closing + self.config.closing_bias)
                    / (
                        self.protocol.speed_limit
                        + self.config.closing_bias
                    ),
                )
                captured_pair = (
                    len(self._terminal_captured) == n_robots
                    and self._terminal_captured[i]
                    and self._terminal_captured[j]
                )
                weight = gate * conflict * float(not captured_pair)
                if weight == 0.0:
                    continue
                tangent = chi * rotate90(normal)
                circulation[i] += weight * tangent
                circulation[j] -= weight * tangent
                active_pairs += 1

        for i, point in enumerate(positions):
            goal_direction = goal_command[i]
            for clearance, normal, _ in arena.boundary_queries(
                point,
                self.protocol.robot_clearance,
            ):
                gate = smooth_gate(
                    clearance,
                    self.config.obstacle_sense_radius,
                )
                if gate == 0.0:
                    continue
                inward = max(0.0, -float(normal @ goal_direction))
                conflict = min(
                    1.0,
                    (inward + self.config.closing_bias)
                    / (
                        self.protocol.speed_limit
                        + self.config.closing_bias
                    ),
                )
                boundary_sense = chi
                if self.config.boundary_progress_aligned:
                    positive_tangent = rotate90(normal)
                    alignment = float(positive_tangent @ goal_direction)
                    if alignment > 1.0e-12:
                        boundary_sense = 1.0
                    elif alignment < -1.0e-12:
                        boundary_sense = -1.0
                circulation[i] += (
                    self.config.obstacle_circulation_gain
                    / self.config.circulation_gain
                    * gate
                    * conflict
                    * boundary_sense
                    * rotate90(normal)
                )
                active_boundaries += 1
        return circulation, active_pairs, active_boundaries

    def cluster_escape_field(
        self,
        positions: Array,
        goal_command: Array,
        arena: Arena,
        tangent_matrix: Array,
    ) -> tuple[Array, int]:
        positions = np.asarray(positions)
        n_robots = len(positions)
        cluster = np.zeros_like(positions, dtype=float)
        active = 0
        if self.config.cluster_escape_gain <= 0.0:
            return cluster, active

        parent = list(range(n_robots))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        pair_gates: dict[tuple[int, int], float] = {}
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                pair_clearance = (
                    float(np.linalg.norm(positions[i] - positions[j]))
                    - self.protocol.pair_clearance
                )
                if pair_clearance < self.config.tangent_band:
                    union(i, j)
                pair_gates[(i, j)] = smooth_gate(
                    pair_clearance,
                    self.config.cluster_pair_band,
                )

        groups: dict[int, list[int]] = {}
        for robot in range(n_robots):
            groups.setdefault(find(robot), []).append(robot)

        active_keys: set[tuple[int, ...]] = set()
        for component in groups.values():
            if len(component) < 2:
                continue
            component_set = set(component)
            pair_gate = max(
                (
                    gate
                    for (left, right), gate in pair_gates.items()
                    if left in component_set and right in component_set
                ),
                default=0.0,
            )
            if pair_gate == 0.0:
                continue

            normals: list[Array] = []
            boundary_gate = 0.0
            for robot in component:
                for clearance, normal, _ in arena.boundary_queries(
                    positions[robot],
                    self.protocol.robot_clearance,
                ):
                    if clearance < self.config.tangent_band:
                        normals.append(normal)
                    boundary_gate = max(
                        boundary_gate,
                        smooth_gate(
                            clearance,
                            self.config.cluster_boundary_band,
                        ),
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
                    np.sum(goal_command[component], axis=0),
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
            contribution[component] = weight * direction
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

    def _circulation_components(
        self,
        positions: Array,
    ) -> list[list[int]]:
        count = len(positions)
        parent = list(range(count))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for i in range(count):
            for j in range(i + 1, count):
                if (
                    np.linalg.norm(positions[i] - positions[j])
                    < self.config.pair_sense_radius
                ):
                    union(i, j)
        groups: dict[int, list[int]] = {}
        for index in range(count):
            groups.setdefault(find(index), []).append(index)
        return list(groups.values())
