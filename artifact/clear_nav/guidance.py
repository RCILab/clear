"""Deterministic static-obstacle geodesic guidance.

The planner handles only the topology of the fixed free space.  Other robots
are deliberately absent from its occupancy grid; their conflicts remain the
job of CLEAR's circulation and CBF layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import ceil, sqrt

import numpy as np

from .config import Protocol
from .geometry import Arena

Array = np.ndarray

_MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, sqrt(2.0)),
    (-1, 1, sqrt(2.0)),
    (1, -1, sqrt(2.0)),
    (1, 1, sqrt(2.0)),
)


@dataclass
class GuidancePlan:
    paths: tuple[Array, ...]
    resolution: float
    lookahead_distance: float = 0.90

    def targets(
        self,
        positions: Array,
        arena: Arena,
        clearance: float,
    ) -> Array:
        targets = np.empty_like(positions)
        for robot, (position, path) in enumerate(zip(positions, self.paths)):
            targets[robot] = _polyline_lookahead(
                position, path, self.lookahead_distance
            )
        return targets


@dataclass
class CostFieldGuidance:
    """Memoryless discrete geodesic descent fields, one per robot goal."""

    axis: Array
    free: Array
    costs: tuple[Array, ...]
    goals: Array
    lookahead_steps: int = 5

    def _nearest_free(self, point: Array) -> tuple[int, int] | None:
        column = int(np.argmin(np.abs(self.axis - point[0])))
        row = int(np.argmin(np.abs(self.axis - point[1])))
        if self.free[row, column]:
            return row, column
        rows, columns = self.free.shape
        for radius in range(1, 7):
            best: tuple[float, tuple[int, int]] | None = None
            for rr in range(max(0, row - radius), min(rows, row + radius + 1)):
                for cc in range(
                    max(0, column - radius), min(columns, column + radius + 1)
                ):
                    if not self.free[rr, cc]:
                        continue
                    distance = float(
                        np.linalg.norm(
                            point - np.array([self.axis[cc], self.axis[rr]])
                        )
                    )
                    candidate = (distance, (rr, cc))
                    if best is None or candidate < best:
                        best = candidate
            if best is not None:
                return best[1]
        return None

    def targets(self, positions: Array) -> Array:
        targets = np.empty_like(positions)
        rows, columns = self.free.shape
        for robot, (position, cost, goal) in enumerate(
            zip(positions, self.costs, self.goals)
        ):
            node = self._nearest_free(position)
            if node is None or not np.isfinite(cost[node]):
                targets[robot] = goal
                continue
            current = node
            for _ in range(self.lookahead_steps):
                row, column = current
                best = current
                best_cost = float(cost[current])
                for dr, dc, _ in _MOVES:
                    nr, nc = row + dr, column + dc
                    if not (0 <= nr < rows and 0 <= nc < columns):
                        continue
                    if not self.free[nr, nc]:
                        continue
                    if dr and dc and not (
                        self.free[row + dr, column]
                        and self.free[row, column + dc]
                    ):
                        continue
                    candidate_cost = float(cost[nr, nc])
                    if candidate_cost < best_cost - 1.0e-12:
                        best_cost = candidate_cost
                        best = (nr, nc)
                if best == current:
                    break
                current = best
            if float(cost[current]) <= self.lookahead_steps * (
                self.axis[1] - self.axis[0]
            ):
                targets[robot] = goal
            else:
                targets[robot] = np.array(
                    [self.axis[current[1]], self.axis[current[0]]]
                )
        return targets


def segment_is_free(
    start: Array,
    end: Array,
    arena: Arena,
    clearance: float,
    *,
    sample_step: float,
) -> bool:
    distance = float(np.linalg.norm(end - start))
    samples = max(1, int(ceil(distance / sample_step)))
    for fraction in np.linspace(0.0, 1.0, samples + 1):
        point = (1.0 - fraction) * start + fraction * end
        if not arena.free(point, clearance):
            return False
    return True


class GridPlanner:
    def __init__(
        self,
        arena: Arena,
        protocol: Protocol,
        *,
        resolution: float = 0.25,
        extra_clearance: float = 0.025,
    ) -> None:
        self.arena = arena
        self.protocol = protocol
        self.resolution = resolution
        self.clearance = protocol.robot_clearance + extra_clearance
        limit = protocol.half_width - self.clearance
        count = int(np.floor(2.0 * limit / resolution)) + 1
        self.axis = np.linspace(-limit, limit, count)
        self.free = np.zeros((count, count), dtype=bool)
        for row, y in enumerate(self.axis):
            for column, x in enumerate(self.axis):
                self.free[row, column] = arena.free(
                    np.array([x, y]), self.clearance
                )

    def _nearest_free(self, point: Array) -> tuple[int, int]:
        column = int(np.argmin(np.abs(self.axis - point[0])))
        row = int(np.argmin(np.abs(self.axis - point[1])))
        if self.free[row, column]:
            return row, column
        candidates = np.argwhere(self.free)
        world = np.stack(
            (self.axis[candidates[:, 1]], self.axis[candidates[:, 0]]), axis=1
        )
        index = int(np.argmin(np.linalg.norm(world - point, axis=1)))
        return tuple(map(int, candidates[index]))

    def _world(self, node: tuple[int, int]) -> Array:
        row, column = node
        return np.array([self.axis[column], self.axis[row]])

    def path(self, start: Array, goal: Array) -> Array:
        source = self._nearest_free(start)
        target = self._nearest_free(goal)
        queue: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(queue, (0.0, 0.0, source))
        cost = {source: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        rows, columns = self.free.shape

        while queue:
            _, current_cost, current = heapq.heappop(queue)
            if current == target:
                break
            if current_cost != cost.get(current):
                continue
            row, column = current
            for dr, dc, step_cost in _MOVES:
                nr, nc = row + dr, column + dc
                if not (0 <= nr < rows and 0 <= nc < columns):
                    continue
                if not self.free[nr, nc]:
                    continue
                if dr and dc:
                    if not (self.free[row + dr, column] and self.free[row, column + dc]):
                        continue
                neighbour = (nr, nc)
                candidate_cost = current_cost + step_cost
                if candidate_cost >= cost.get(neighbour, float("inf")):
                    continue
                cost[neighbour] = candidate_cost
                parent[neighbour] = current
                heuristic = float(np.hypot(nr - target[0], nc - target[1]))
                heapq.heappush(
                    queue,
                    (candidate_cost + heuristic, candidate_cost, neighbour),
                )
        else:
            raise RuntimeError("no static-obstacle route exists")

        nodes = [target]
        while nodes[-1] != source:
            nodes.append(parent[nodes[-1]])
        nodes.reverse()
        waypoints = [np.asarray(start, dtype=float)]
        waypoints.extend(self._world(node) for node in nodes[1:-1])
        waypoints.append(np.asarray(goal, dtype=float))
        return np.asarray(waypoints)

    def plan(self, starts: Array, goals: Array) -> GuidancePlan:
        paths = tuple(
            self._shortcut(self.path(start, goal))
            for start, goal in zip(starts, goals)
        )
        return GuidancePlan(paths, self.resolution)

    def cost_field_plan(self, goals: Array) -> CostFieldGuidance:
        return CostFieldGuidance(
            self.axis.copy(),
            self.free.copy(),
            tuple(self._cost_to_goal(goal) for goal in goals),
            np.asarray(goals, dtype=float).copy(),
        )

    def _cost_to_goal(self, goal: Array) -> Array:
        target = self._nearest_free(goal)
        costs = np.full(self.free.shape, np.inf)
        costs[target] = 0.0
        queue: list[tuple[float, tuple[int, int]]] = [(0.0, target)]
        rows, columns = self.free.shape
        while queue:
            current_cost, (row, column) = heapq.heappop(queue)
            if current_cost != costs[row, column]:
                continue
            for dr, dc, step_cost in _MOVES:
                nr, nc = row + dr, column + dc
                if not (0 <= nr < rows and 0 <= nc < columns):
                    continue
                if not self.free[nr, nc]:
                    continue
                if dr and dc and not (
                    self.free[row + dr, column]
                    and self.free[row, column + dc]
                ):
                    continue
                candidate = current_cost + self.resolution * step_cost
                if candidate >= costs[nr, nc]:
                    continue
                costs[nr, nc] = candidate
                heapq.heappush(queue, (candidate, (nr, nc)))
        return costs

    def _shortcut(self, path: Array) -> Array:
        """Greedily remove grid staircasing once, before the rollout."""
        if len(path) <= 2:
            return path
        result = [path[0]]
        current = 0
        while current < len(path) - 1:
            chosen = current + 1
            for candidate in range(len(path) - 1, current, -1):
                if segment_is_free(
                    path[current],
                    path[candidate],
                    self.arena,
                    self.clearance,
                    sample_step=0.20 * self.resolution,
                ):
                    chosen = candidate
                    break
            result.append(path[chosen])
            current = chosen
        return np.asarray(result)


def _polyline_lookahead(position: Array, path: Array, lookahead: float) -> Array:
    """Project onto a route polyline and return a point `lookahead` metres ahead."""
    if len(path) == 1:
        return path[0]
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_distance = float("inf")
    best_arc = 0.0
    for index, length in enumerate(lengths):
        if length <= 1.0e-12:
            continue
        segment = path[index + 1] - path[index]
        fraction = float(
            np.clip(
                (position - path[index]) @ segment / (length * length),
                0.0,
                1.0,
            )
        )
        projection = path[index] + fraction * segment
        distance = float(np.linalg.norm(position - projection))
        if distance < best_distance:
            best_distance = distance
            best_arc = cumulative[index] + fraction * length

    target_arc = min(cumulative[-1], best_arc + lookahead)
    index = min(int(np.searchsorted(cumulative, target_arc, side="right") - 1), len(lengths) - 1)
    if lengths[index] <= 1.0e-12:
        return path[index + 1]
    fraction = (target_arc - cumulative[index]) / lengths[index]
    return (1.0 - fraction) * path[index] + fraction * path[index + 1]
