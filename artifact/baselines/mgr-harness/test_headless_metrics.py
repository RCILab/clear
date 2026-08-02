"""Evaluator-only tests; no upstream MGR source is imported."""

from __future__ import annotations

import unittest

import numpy as np

from run_headless import (
    segment_rect_distance,
    swept_obstacle_metrics,
    swept_pair_metrics,
)


class FakeRect:
    def __init__(self, x: float, y: float, width: float, height: float):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y

    def width(self) -> float:
        return self._width

    def height(self) -> float:
        return self._height


class FakePoint:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y


class FakeAgent:
    def __init__(self, radius: float = 0.2):
        self.radius = radius


class FakeObstacle:
    def __init__(
        self,
        obstacle_type: str,
        *,
        center: tuple[float, float] | None = None,
        radius: float = 0.0,
        rect: FakeRect | None = None,
    ):
        self.type = obstacle_type
        self.center = None if center is None else FakePoint(*center)
        self.radius = radius
        self.rect = rect


class FakeSimulator:
    def __init__(self):
        self.agents = [FakeAgent(), FakeAgent()]
        self.static_obstacles = [
            FakeObstacle("circ", center=(7.0, 7.0), radius=0.5),
            FakeObstacle(
                "rect",
                rect=FakeRect(2.0, 2.0, 1.0, 1.0),
            ),
        ]
        self.sim_width = 16.0
        self.sim_height = 16.0
        self.scale = 1.0


class SweptMetricTests(unittest.TestCase):
    def test_pair_crossing_is_detected_between_safe_endpoints(self) -> None:
        start = np.array([[-1.0, 0.0], [1.0, 0.0]])
        end = np.array([[1.0, 0.0], [-1.0, 0.0]])
        distance, physical, declared = swept_pair_metrics(
            start,
            end,
            np.array([0.2, 0.2]),
            1.1,
        )
        self.assertAlmostEqual(distance, 0.0)
        self.assertAlmostEqual(physical, -0.4)
        self.assertAlmostEqual(declared, -0.44)

    def test_parallel_pair_uses_interior_minimum(self) -> None:
        start = np.array([[0.0, 0.0], [0.8, 1.0]])
        end = np.array([[1.0, 0.0], [0.8, -1.0]])
        distance, _, _ = swept_pair_metrics(
            start,
            end,
            np.array([0.2, 0.2]),
            1.1,
        )
        self.assertAlmostEqual(distance, np.hypot(0.24, 0.12))

    def test_segment_rectangle_distance(self) -> None:
        rect = FakeRect(-0.5, -0.5, 1.0, 1.0)
        crossing = segment_rect_distance(
            np.array([-2.0, 0.0]),
            np.array([2.0, 0.0]),
            rect,
        )
        parallel = segment_rect_distance(
            np.array([-2.0, 1.0]),
            np.array([2.0, 1.0]),
            rect,
        )
        diagonal = segment_rect_distance(
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            rect,
        )
        self.assertAlmostEqual(crossing, 0.0)
        self.assertAlmostEqual(parallel, 0.5)
        self.assertAlmostEqual(diagonal, np.sqrt(0.5))

    def test_pruned_obstacle_audit_matches_unpruned_minimum(self) -> None:
        sim = FakeSimulator()
        start = np.array([[0.5, 0.5], [5.0, 5.0]])
        end = np.array([[0.52, 0.51], [5.02, 5.01]])
        reference = swept_obstacle_metrics(
            sim,
            start,
            end,
            1.1,
        )
        pruned = swept_obstacle_metrics(
            sim,
            start,
            end,
            1.1,
            incumbent_obstacle_clearance=10.0,
            incumbent_obstacle_safety_margin=10.0,
            prune=True,
        )
        np.testing.assert_allclose(pruned, reference, rtol=0.0, atol=0.0)

    def test_pruned_obstacle_audit_preserves_better_incumbent(self) -> None:
        sim = FakeSimulator()
        start = np.array([[0.5, 0.5], [5.0, 5.0]])
        end = np.array([[0.52, 0.51], [5.02, 5.01]])
        reference = swept_obstacle_metrics(
            sim,
            start,
            end,
            1.1,
        )
        incumbent_clearance = reference[0] - 0.1
        incumbent_safety = reference[1] - 0.1
        pruned = swept_obstacle_metrics(
            sim,
            start,
            end,
            1.1,
            incumbent_obstacle_clearance=incumbent_clearance,
            incumbent_obstacle_safety_margin=incumbent_safety,
            prune=True,
        )
        self.assertEqual(pruned[0], incumbent_clearance)
        self.assertEqual(pruned[1], incumbent_safety)
        self.assertEqual(pruned[2:], reference[2:])


if __name__ == "__main__":
    unittest.main()
