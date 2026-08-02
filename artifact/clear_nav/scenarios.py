"""Deterministic clean-room benchmark scenario generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import pi

import numpy as np

from .config import DEFAULT_PROTOCOL, Protocol
from .geometry import Arena, Circle, Rectangle

Array = np.ndarray
FAMILIES = ("free", "circ15", "rect15", "swap")


@dataclass(frozen=True)
class Scenario:
    family: str
    n_robots: int
    seed: int
    starts: Array
    goals: Array
    arena: Arena
    protocol: Protocol

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.family.encode("ascii"))
        digest.update(np.asarray([self.n_robots, self.seed], dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(self.starts, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(self.goals, dtype=np.float64).tobytes())
        for obstacle in self.arena.obstacles:
            if isinstance(obstacle, Circle):
                digest.update(b"C")
                digest.update(np.asarray([*obstacle.center, obstacle.radius], dtype=np.float64).tobytes())
            elif isinstance(obstacle, Rectangle):
                digest.update(b"R")
                digest.update(np.asarray([*obstacle.center, *obstacle.size], dtype=np.float64).tobytes())
        return digest.hexdigest()[:16]


def _obstacle_gap(a: object, b: object) -> float:
    if isinstance(a, Circle) and isinstance(b, Circle):
        return float(np.linalg.norm(a.center - b.center) - a.radius - b.radius)
    if isinstance(a, Rectangle) and isinstance(b, Rectangle):
        delta = np.abs(a.center - b.center) - 0.5 * (a.size + b.size)
        if np.all(delta <= 0.0):
            return float(np.max(delta))
        return float(np.linalg.norm(np.maximum(delta, 0.0)))
    raise TypeError("mixed obstacle families are not generated")


def _make_circles(rng: np.random.Generator, p: Protocol) -> tuple[Circle, ...]:
    factors = rng.uniform(0.72, 1.28, p.circle_count)
    target_area = p.obstacle_fraction * p.workspace_size**2
    radii = factors * np.sqrt(target_area / (pi * np.sum(factors**2)))
    obstacles: list[Circle] = []
    for radius in sorted(radii, reverse=True):
        limit = p.half_width - radius - p.robot_clearance
        for _ in range(30_000):
            candidate = Circle(rng.uniform(-limit, limit, size=2), float(radius))
            if all(_obstacle_gap(candidate, old) >= 0.25 for old in obstacles):
                obstacles.append(candidate)
                break
        else:
            raise RuntimeError("circle field generation failed")
    return tuple(obstacles)


def _make_rectangles(rng: np.random.Generator, p: Protocol) -> tuple[Rectangle, ...]:
    aspect = rng.uniform(0.65, 1.55, size=(p.rectangle_count, 2))
    target_area = p.obstacle_fraction * p.workspace_size**2
    scale = np.sqrt(target_area / np.sum(aspect[:, 0] * aspect[:, 1]))
    sizes = aspect * scale
    obstacles: list[Rectangle] = []
    for size in sorted(sizes, key=lambda value: -float(np.prod(value))):
        limit = p.half_width - 0.5 * size - p.robot_clearance
        for _ in range(30_000):
            candidate = Rectangle(rng.uniform(-limit, limit), np.asarray(size, dtype=float))
            if all(_obstacle_gap(candidate, old) >= 0.25 for old in obstacles):
                obstacles.append(candidate)
                break
        else:
            raise RuntimeError("rectangle field generation failed")
    return tuple(obstacles)


def _sample_sites(
    rng: np.random.Generator,
    count: int,
    arena: Arena,
    p: Protocol,
    separation: float,
) -> Array:
    limit = p.half_width - p.robot_clearance
    accepted: list[Array] = []
    for _ in range(200_000):
        point = rng.uniform(-limit, limit, size=2)
        if not arena.free(point, p.robot_clearance + 0.03):
            continue
        if any(np.linalg.norm(point - old) < separation for old in accepted):
            continue
        accepted.append(point)
        if len(accepted) == count:
            return np.asarray(accepted)
    raise RuntimeError(f"could not place {count} sites")


def _make_swap_sites(
    rng: np.random.Generator,
    count: int,
    protocol: Protocol,
) -> Array:
    """Place an antipodal Swap task on the fewest safe staggered rings."""
    phase = float(rng.uniform(0.0, 2.0 * pi))
    required = protocol.pair_clearance + 0.02
    radial_step = required + 0.10
    for ring_count in range(1, 9):
        radii = protocol.swap_radius - radial_step * np.arange(ring_count)
        if radii[-1] <= protocol.robot_clearance:
            break
        base, remainder = divmod(count, ring_count)
        allocations = [
            base + int(index < remainder) for index in range(ring_count)
        ]
        rings: list[Array] = []
        for index, (radius, allocation) in enumerate(
            zip(radii, allocations)
        ):
            if allocation == 0:
                continue
            offset = index * pi / max(allocation, 1)
            angle = (
                phase
                + offset
                + 2.0 * pi * np.arange(allocation) / allocation
            )
            rings.append(
                radius * np.stack((np.cos(angle), np.sin(angle)), axis=1)
            )
        starts = np.concatenate(rings, axis=0)
        if len(starts) < 2:
            return starts
        left, right = np.triu_indices(len(starts), k=1)
        minimum = float(
            np.min(np.linalg.norm(starts[left] - starts[right], axis=1))
        )
        if minimum >= required:
            return starts
    raise RuntimeError(
        f"could not place {count} safe Swap sites on staggered rings"
    )


def make_scenario(
    family: str,
    n_robots: int,
    seed: int,
    protocol: Protocol = DEFAULT_PROTOCOL,
) -> Scenario:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if n_robots < 1:
        raise ValueError("n_robots must be positive")
    rng = np.random.default_rng(seed)

    if family == "swap":
        starts = _make_swap_sites(rng, n_robots, protocol)
        arena = Arena(protocol.half_width)
        return Scenario(family, n_robots, seed, starts, -starts, arena, protocol)

    if family == "free":
        obstacles = ()
    elif family == "circ15":
        obstacles = _make_circles(rng, protocol)
    else:
        obstacles = _make_rectangles(rng, protocol)
    arena = Arena(protocol.half_width, obstacles)
    separation = protocol.pair_clearance + 0.10
    starts = _sample_sites(rng, n_robots, arena, protocol, separation)
    goals = _sample_sites(rng, n_robots, arena, protocol, separation)
    return Scenario(family, n_robots, seed, starts, goals, arena, protocol)
