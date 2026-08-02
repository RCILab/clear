"""Export the canonical 320 CLEAR scenarios as official-MGR-compatible YAML.

The generated files contain the exact physical-center starts, goals, and
obstacles used by the retained CLEAR raw records.  Export aborts if any
regenerated scenario fingerprint differs from the canonical record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from clear_nav import Protocol
from clear_nav.geometry import Circle, Rectangle
from clear_nav.unicycle import UnicycleConfig
from run_clear_on_mgr import load_mgr_scenario
from run_unicycle import make_unicycle_scenario


FAMILY_DIRS = {
    "free": Path("Free_maps"),
    "circ15": Path("circle_maps") / "CircleEnv15",
    "rect15": Path("rect_maps") / "RectEnv15",
    "swap": Path("swap_maps"),
}
FAMILY_LABELS = {
    "free": "Free",
    "circ15": "CircleEnv_15",
    "rect15": "RectEnv_15",
    "swap": "Free_Swap_18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("results/unicycle_clear_all_sizes_raw.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mgr_paired_instances"),
    )
    return parser.parse_args()


def mgr_point(point: np.ndarray, workspace_size: float) -> list[float]:
    scale = workspace_size / 40.0
    shifted = (np.asarray(point, dtype=float) + workspace_size / 2.0) / scale
    return [float(shifted[0]), float(shifted[1])]


def scenario_yaml(scenario) -> dict:
    scale = scenario.protocol.workspace_size / 40.0
    obstacles = []
    for obstacle in scenario.arena.obstacles:
        if isinstance(obstacle, Circle):
            obstacles.append(
                {
                    "center": mgr_point(
                        obstacle.center, scenario.protocol.workspace_size
                    ),
                    "radius": float(obstacle.radius / scale),
                }
            )
        elif isinstance(obstacle, Rectangle):
            obstacles.append(
                {
                    "center": mgr_point(
                        obstacle.center, scenario.protocol.workspace_size
                    ),
                    "width": float(obstacle.size[0] / scale),
                    "height": float(obstacle.size[1] / scale),
                }
            )
        else:
            raise TypeError(f"unsupported obstacle: {type(obstacle)!r}")
    return {
        "agentNum": scenario.n_robots,
        "agent_size": 1.0,
        "goalPoints": [
            mgr_point(point, scenario.protocol.workspace_size)
            for point in scenario.goals
        ],
        "obstacles": obstacles,
        "startPoints": [
            mgr_point(point, scenario.protocol.workspace_size)
            for point in scenario.starts
        ],
    }


def roundtrip_error(original, loaded) -> float:
    if (
        original.family != loaded.family
        or original.n_robots != loaded.n_robots
        or original.seed != loaded.seed
        or len(original.arena.obstacles) != len(loaded.arena.obstacles)
    ):
        raise RuntimeError("YAML round-trip changed scenario metadata")
    errors = [
        float(np.max(np.abs(original.starts - loaded.starts))),
        float(np.max(np.abs(original.goals - loaded.goals))),
    ]
    for expected, actual in zip(
        original.arena.obstacles, loaded.arena.obstacles
    ):
        if type(expected) is not type(actual):
            raise RuntimeError("YAML round-trip changed obstacle type")
        errors.append(
            float(np.max(np.abs(expected.center - actual.center)))
        )
        if isinstance(expected, Circle):
            errors.append(abs(expected.radius - actual.radius))
        else:
            errors.append(float(np.max(np.abs(expected.size - actual.size))))
    return max(errors, default=0.0)


def main() -> None:
    args = parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    physical = Protocol(**raw["physical_protocol"])
    unicycle = UnicycleConfig(**raw["unicycle"])
    records = raw["records"]
    if len(records) != 320:
        raise RuntimeError(f"expected 320 canonical records, found {len(records)}")

    manifest_records = []
    seen = set()
    for record in records:
        key = (record["family"], record["n_robots"], record["seed"])
        if key in seen:
            raise RuntimeError(f"duplicate canonical scenario: {key}")
        seen.add(key)
        scenario = make_unicycle_scenario(
            record["family"],
            record["n_robots"],
            record["seed"],
            physical,
            unicycle,
        )
        if scenario.fingerprint() != record["fingerprint"]:
            raise RuntimeError(
                f"fingerprint mismatch for {key}: "
                f"{scenario.fingerprint()} != {record['fingerprint']}"
            )

        relative = (
            FAMILY_DIRS[record["family"]]
            / f"agents{record['n_robots']}"
            / (
                f"{FAMILY_LABELS[record['family']]}_"
                f"{record['n_robots']}_{record['seed']}.yaml"
            )
        )
        output = args.output_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(
                scenario_yaml(scenario),
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        roundtrip = load_mgr_scenario(output, physical)
        maximum_roundtrip_error = roundtrip_error(scenario, roundtrip)
        if maximum_roundtrip_error > 1.0e-12:
            raise RuntimeError(
                f"YAML round-trip error for {key}: "
                f"{maximum_roundtrip_error:.3e} m"
            )
        yaml_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest_records.append(
            {
                "family": record["family"],
                "n_robots": record["n_robots"],
                "seed": record["seed"],
                "fingerprint": record["fingerprint"],
                "relative_yaml": relative.as_posix(),
                "roundtrip_max_abs_error_m": maximum_roundtrip_error,
                "yaml_sha256": yaml_sha256,
            }
        )

    manifest = {
        "description": (
            "Canonical CLEAR physical scenarios exported for paired execution "
            "by the official MGR implementation"
        ),
        "source_raw": str(args.raw),
        "coordinate_transform": "p_mgr = (p_clear + [8, 8]) / 0.4",
        "physical_protocol": raw["physical_protocol"],
        "unicycle_initial_heading_rad": 0.0,
        "record_count": len(manifest_records),
        "records": manifest_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"verified and exported {len(manifest_records)} scenarios")
    print(manifest_path)


if __name__ == "__main__":
    main()
