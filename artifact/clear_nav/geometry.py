"""Signed-clearance geometry with outward normals.

Every query returns a clearance `h` that is non-negative in free space and an
outward unit normal `n` satisfying, away from medial-axis ties,

    d h / d t = n.T velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol as TypingProtocol

import numpy as np

Array = np.ndarray


class Obstacle(TypingProtocol):
    def clearance_normal(self, point: Array, padding: float) -> tuple[float, Array]:
        ...


@dataclass(frozen=True)
class Circle:
    center: Array
    radius: float

    def clearance_normal(self, point: Array, padding: float) -> tuple[float, Array]:
        delta = np.asarray(point, dtype=float) - self.center
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-12:
            return -(self.radius + padding), np.array([1.0, 0.0])
        return distance - self.radius - padding, delta / distance


@dataclass(frozen=True)
class Rectangle:
    center: Array
    size: Array

    def clearance_normal(self, point: Array, padding: float) -> tuple[float, Array]:
        point = np.asarray(point, dtype=float)
        half = 0.5 * self.size
        local = point - self.center
        outside = np.abs(local) - half
        positive = np.maximum(outside, 0.0)
        external_distance = float(np.linalg.norm(positive))

        if external_distance > 1.0e-12:
            closest = np.clip(local, -half, half)
            delta = local - closest
            normal = delta / np.linalg.norm(delta)
            return external_distance - padding, normal

        # Inside/on the box: choose the nearest face.  Ties are resolved by the
        # first axis, deterministically.
        face_gap = half - np.abs(local)
        axis = int(np.argmin(face_gap))
        sign = 1.0 if local[axis] >= 0.0 else -1.0
        normal = np.zeros(2)
        normal[axis] = sign
        return -float(face_gap[axis]) - padding, normal


@dataclass(frozen=True)
class Arena:
    half_width: float
    obstacles: tuple[Obstacle, ...] = ()

    def boundary_queries(self, point: Array, padding: float) -> list[tuple[float, Array, str]]:
        x, y = map(float, point)
        wall = self.half_width - padding
        queries: list[tuple[float, Array, str]] = [
            (x + wall, np.array([1.0, 0.0]), "wall-left"),
            (wall - x, np.array([-1.0, 0.0]), "wall-right"),
            (y + wall, np.array([0.0, 1.0]), "wall-bottom"),
            (wall - y, np.array([0.0, -1.0]), "wall-top"),
        ]
        for index, obstacle in enumerate(self.obstacles):
            h, normal = obstacle.clearance_normal(point, padding)
            queries.append((h, normal, f"obstacle-{index}"))
        return queries

    def minimum_clearance(self, point: Array, padding: float) -> float:
        return min(h for h, _, _ in self.boundary_queries(point, padding))

    def free(self, point: Array, padding: float) -> bool:
        return self.minimum_clearance(point, padding) >= 0.0


def rotate90(vector: Array) -> Array:
    vector = np.asarray(vector, dtype=float)
    return np.stack((-vector[..., 1], vector[..., 0]), axis=-1)


def smooth_gate(clearance: float, radius: float) -> float:
    """C1 gate equal to one at contact and zero outside `radius`."""
    if radius <= 0.0 or clearance >= radius:
        return 0.0
    if clearance <= 0.0:
        return 1.0
    z = 1.0 - clearance / radius
    return z * z * (3.0 - 2.0 * z)


def obstacles_from_arrays(
    circles: Iterable[tuple[Iterable[float], float]] = (),
    rectangles: Iterable[tuple[Iterable[float], Iterable[float]]] = (),
) -> tuple[Obstacle, ...]:
    result: list[Obstacle] = []
    result.extend(Circle(np.asarray(center, dtype=float), float(radius)) for center, radius in circles)
    result.extend(
        Rectangle(np.asarray(center, dtype=float), np.asarray(size, dtype=float))
        for center, size in rectangles
    )
    return tuple(result)

