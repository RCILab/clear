"""Compare geometric difficulty proxies for official and paired MGR YAML sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


GRID_SIZE = 40.0
METERS_PER_GRID = 0.4
ROBOT_CLEARANCE_GRID = 0.22 / METERS_PER_GRID
PAIR_CLEARANCE_M = 0.44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--paired-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def obstacle_kind(obstacle: dict) -> str:
    return "circle" if "radius" in obstacle else "rectangle"


def obstacle_area(obstacle: dict) -> float:
    if obstacle_kind(obstacle) == "circle":
        return float(np.pi * obstacle["radius"] ** 2)
    return float(obstacle["width"] * obstacle["height"])


def signed_clearance(points: np.ndarray, obstacle: dict) -> np.ndarray:
    center = np.asarray(obstacle["center"], dtype=float)
    delta = points - center
    if obstacle_kind(obstacle) == "circle":
        return np.linalg.norm(delta, axis=1) - float(obstacle["radius"])
    half = 0.5 * np.asarray(
        [obstacle["width"], obstacle["height"]], dtype=float
    )
    q = np.abs(delta) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outside + inside


def obstacle_gap(first: dict, second: dict) -> float:
    c1 = np.asarray(first["center"], dtype=float)
    c2 = np.asarray(second["center"], dtype=float)
    if obstacle_kind(first) == obstacle_kind(second) == "circle":
        return float(
            np.linalg.norm(c1 - c2) - first["radius"] - second["radius"]
        )
    if obstacle_kind(first) == obstacle_kind(second) == "rectangle":
        h1 = 0.5 * np.asarray([first["width"], first["height"]])
        h2 = 0.5 * np.asarray([second["width"], second["height"]])
        q = np.abs(c1 - c2) - h1 - h2
        if np.all(q <= 0.0):
            return float(np.max(q))
        return float(np.linalg.norm(np.maximum(q, 0.0)))
    raise ValueError("mixed obstacle types are unsupported")


def segment_hits_obstacle(start: np.ndarray, goal: np.ndarray, obstacle: dict) -> bool:
    center = np.asarray(obstacle["center"], dtype=float)
    direction = goal - start
    if obstacle_kind(obstacle) == "circle":
        denominator = float(direction @ direction)
        fraction = (
            0.0
            if denominator == 0.0
            else float(np.clip((center - start) @ direction / denominator, 0.0, 1.0))
        )
        closest = start + fraction * direction
        radius = float(obstacle["radius"]) + ROBOT_CLEARANCE_GRID
        return bool(np.linalg.norm(closest - center) <= radius)

    half = (
        0.5 * np.asarray([obstacle["width"], obstacle["height"]], dtype=float)
        + ROBOT_CLEARANCE_GRID
    )
    lower = center - half
    upper = center + half
    t_low, t_high = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) < 1.0e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - start[axis]) / direction[axis]
        second = (upper[axis] - start[axis]) / direction[axis]
        t_low = max(t_low, min(first, second))
        t_high = min(t_high, max(first, second))
        if t_low > t_high:
            return False
    return True


def union_fraction(obstacles: list[dict], resolution: int = 400) -> float:
    axis = (np.arange(resolution, dtype=float) + 0.5) * GRID_SIZE / resolution
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    occupied = np.zeros(len(points), dtype=bool)
    for obstacle in obstacles:
        occupied |= signed_clearance(points, obstacle) <= 0.0
    return float(np.mean(occupied))


def minimum_pair_distance(points: np.ndarray) -> float:
    left, right = np.triu_indices(len(points), k=1)
    return float(np.min(np.linalg.norm(points[left] - points[right], axis=1)))


def synchronized_pair_conflicts(starts: np.ndarray, goals: np.ndarray) -> int:
    conflicts = 0
    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            relative_start = starts[first] - starts[second]
            relative_delta = (
                goals[first] - starts[first] - goals[second] + starts[second]
            )
            denominator = float(relative_delta @ relative_delta)
            fraction = (
                0.0
                if denominator == 0.0
                else float(
                    np.clip(
                        -(relative_start @ relative_delta) / denominator,
                        0.0,
                        1.0,
                    )
                )
            )
            if np.linalg.norm(relative_start + fraction * relative_delta) <= PAIR_CLEARANCE_M:
                conflicts += 1
    return conflicts


def scenario_metrics(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    starts_grid = np.asarray(data["startPoints"], dtype=float)
    goals_grid = np.asarray(data["goalPoints"], dtype=float)
    starts = METERS_PER_GRID * starts_grid
    goals = METERS_PER_GRID * goals_grid
    obstacles = data["obstacles"]
    gaps = [
        obstacle_gap(obstacles[first], obstacles[second])
        for first in range(len(obstacles))
        for second in range(first + 1, len(obstacles))
    ]
    start_clearance = np.min(
        np.column_stack(
            [signed_clearance(starts_grid, obstacle) for obstacle in obstacles]
        ),
        axis=1,
    )
    goal_clearance = np.min(
        np.column_stack(
            [signed_clearance(goals_grid, obstacle) for obstacle in obstacles]
        ),
        axis=1,
    )
    blocked = sum(
        any(segment_hits_obstacle(start, goal, obstacle) for obstacle in obstacles)
        for start, goal in zip(starts_grid, goals_grid)
    )
    possible_pairs = len(starts) * (len(starts) - 1) // 2
    return {
        "obstacle_count": float(len(obstacles)),
        "nominal_obstacle_fraction": float(
            sum(obstacle_area(obstacle) for obstacle in obstacles) / GRID_SIZE**2
        ),
        "union_obstacle_fraction": union_fraction(obstacles),
        "overlapping_obstacle_pairs": float(sum(gap < 0.0 for gap in gaps)),
        "minimum_obstacle_gap_m": METERS_PER_GRID * min(gaps),
        "direct_paths_blocked_fraction": blocked / len(starts),
        "mean_start_goal_distance_m": float(
            np.mean(np.linalg.norm(goals - starts, axis=1))
        ),
        "synchronized_pair_conflict_fraction": (
            synchronized_pair_conflicts(starts, goals) / possible_pairs
        ),
        "minimum_start_pair_distance_m": minimum_pair_distance(starts),
        "minimum_goal_pair_distance_m": minimum_pair_distance(goals),
        "minimum_start_obstacle_clearance_m": float(
            METERS_PER_GRID * np.min(start_clearance)
        ),
        "minimum_goal_obstacle_clearance_m": float(
            METERS_PER_GRID * np.min(goal_clearance)
        ),
    }


def family_directory(root: Path, family: str) -> Path:
    if family == "circ15":
        return root / "circle_maps" / "CircleEnv15" / "agents20"
    return root / "rect_maps" / "RectEnv15" / "agents20"


def aggregate(paths: list[Path]) -> dict:
    records = [scenario_metrics(path) for path in paths]
    keys = records[0]
    summary = {}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=float)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {"instances": len(records), "summary": summary}


def main() -> None:
    args = parse_args()
    result = {"units": "physical meters unless stated otherwise", "groups": {}}
    for source, root in (
        ("official_mgr", args.official_root),
        ("canonical_clear", args.paired_root),
    ):
        result["groups"][source] = {}
        for family in ("circ15", "rect15"):
            paths = sorted(family_directory(root, family).glob("*.yaml"))
            if len(paths) != 20:
                raise RuntimeError(
                    f"expected 20 {source} {family} instances, found {len(paths)}"
                )
            result["groups"][source][family] = aggregate(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
