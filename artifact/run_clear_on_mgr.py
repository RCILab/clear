"""Run CLEAR on the official Merry-Go-Round YAML benchmark instances.

Coordinates are scaled from the upstream 40-by-40 map to 16 m and translated
from [0, 16]^2 to CLEAR's equivalent centered arena [-8, 8]^2.  No rotation,
resampling, or obstacle modification is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
import yaml

from clear_nav import ControllerConfig, Protocol, Scenario
from clear_nav.geometry import Arena, Circle, Rectangle
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        type=Path,
        action="append",
        required=True,
        help="Official MGR YAML path; repeatable.",
    )
    parser.add_argument("--horizon", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--arrival-radius", type=float, default=0.22)
    parser.add_argument("--lookahead", type=float, default=0.05)
    parser.add_argument("--yaw-rate-limit", type=float, default=np.pi / 2.0)
    parser.add_argument("--inner-substeps", type=int, default=3)
    parser.add_argument("--record-stride", type=int, default=20)
    parser.add_argument(
        "--guidance",
        choices=("direct", "waypoint", "cost"),
        default="cost",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON output path.",
    )
    return parser.parse_args()


def family_from_path(path: Path) -> str:
    normalized = str(path).lower()
    if "swap" in normalized:
        return "swap"
    if "circle" in normalized:
        return "circ15"
    if "rect" in normalized:
        return "rect15"
    return "free"


def seed_from_path(path: Path) -> int:
    match = re.search(r"_(\d+)\.ya?ml$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def scaled_point(value, scale: float, offset: float) -> np.ndarray:
    return np.asarray(value, dtype=float) * scale + offset


def load_mgr_scenario(path: Path, protocol: Protocol) -> Scenario:
    path = path.resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    family = family_from_path(path)
    scale = protocol.workspace_size / 40.0
    offset = -protocol.half_width

    starts = np.asarray(
        [scaled_point(point, scale, offset) for point in data["startPoints"]]
    )
    goals = np.asarray(
        [scaled_point(point, scale, offset) for point in data["goalPoints"]]
    )
    count = int(data["agentNum"])
    if starts.shape != (count, 2) or goals.shape != (count, 2):
        raise ValueError(f"inconsistent agent arrays in {path}")

    obstacles = []
    for obstacle in data.get("obstacles", []):
        center = scaled_point(obstacle["center"], scale, offset)
        if "radius" in obstacle:
            obstacles.append(Circle(center, float(obstacle["radius"]) * scale))
        elif "width" in obstacle and "height" in obstacle:
            obstacles.append(
                Rectangle(
                    center,
                    scale
                    * np.asarray(
                        [obstacle["width"], obstacle["height"]],
                        dtype=float,
                    ),
                )
            )
        else:
            raise ValueError(f"unknown obstacle record in {path}: {obstacle}")

    return Scenario(
        family=family,
        n_robots=count,
        seed=seed_from_path(path),
        starts=starts,
        goals=goals,
        arena=Arena(protocol.half_width, tuple(obstacles)),
        protocol=protocol,
    )


def clear_controller_config() -> ControllerConfig:
    """The manuscript's retained CLEAR configuration."""
    return ControllerConfig(
        handedness=1,
        boundary_progress_aligned=True,
        cluster_escape_gain=0.55,
        cluster_escape_hysteresis=True,
        tangent_band=0.08,
        terminal_capture_hysteresis=True,
        terminal_capture_radius=0.22,
        terminal_open_capture_radius=0.60,
        terminal_release_radius=0.80,
    )


def main() -> None:
    args = parse_args()
    protocol = Protocol(
        arrival_radius=args.arrival_radius,
        dt=args.dt,
        horizon=args.horizon,
    )
    unicycle = UnicycleConfig(
        lookahead=args.lookahead,
        yaw_rate_limit=args.yaw_rate_limit,
        inner_substeps=args.inner_substeps,
        projection_backend="osqp",
        projection_tolerance=1.0e-6,
        projection_max_sweeps=4000,
        certified_bridge_progress=True,
        bridge_progress_retention=0.90,
        actuation_mode="native-cbf",
    )
    controller = clear_controller_config()
    records = []

    for path in args.instance:
        scenario = load_mgr_scenario(path, protocol)
        rollout = simulate_unicycle(
            scenario,
            controller,
            unicycle,
            initial_headings=np.zeros(scenario.n_robots),
            record_stride=args.record_stride,
            guidance_mode=args.guidance,
        )
        record = {
            "source_instance": str(path.resolve()),
            **rollout.summary(),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    result = {
        "algorithm": "CLEAR",
        "scenario_source": "official MGR YAML",
        "coordinate_transform": "p_clear = 0.4 * p_mgr - [8, 8]",
        "physical_protocol": asdict(protocol),
        "control_protocol": asdict(
            inflated_unicycle_protocol(protocol, unicycle.lookahead)
        ),
        "unicycle": asdict(unicycle),
        "controller": asdict(controller),
        "guidance": args.guidance,
        "initial_headings_rad": 0.0,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
