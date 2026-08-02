"""Export paired SMG multi-agent and solo instances for the MGR runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from clear_nav import Protocol, SMGGeometry, Scenario, make_smg_scenario
from export_mgr_pairing_instances import scenario_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smg_mgr_instances"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=range(20))
    parser.add_argument("--robots", nargs="+", type=int, default=(8, 16))
    return parser.parse_args()


def write_instance(path: Path, scenario: Scenario) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            scenario_yaml(scenario),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    protocol = Protocol()
    specifications = (
        ("doorway", "doorway12", SMGGeometry(doorway_width=1.2)),
        (
            "intersection",
            "intersection24",
            SMGGeometry(intersection_corridor_width=2.4),
        ),
    )
    multi_records = []
    solo_records = []
    for family, label, geometry in specifications:
        for count in args.robots:
            for seed in args.seeds:
                scenario = make_smg_scenario(
                    family, count, seed, protocol, geometry
                )
                relative = (
                    Path("RectSMG")
                    / label
                    / f"agents{count}"
                    / f"{label}_{count}_{seed}.yaml"
                )
                sha = write_instance(args.output_dir / relative, scenario)
                multi_records.append(
                    {
                        "record_kind": "multi",
                        "family": family,
                        "n_robots": count,
                        "seed": seed,
                        "fingerprint": scenario.fingerprint(),
                        "relative_yaml": relative.as_posix(),
                        "yaml_sha256": sha,
                    }
                )
                for robot_index in range(count):
                    solo = Scenario(
                        family=f"{scenario.family}_solo",
                        n_robots=1,
                        seed=seed,
                        starts=scenario.starts[
                            robot_index : robot_index + 1
                        ].copy(),
                        goals=scenario.goals[
                            robot_index : robot_index + 1
                        ].copy(),
                        arena=scenario.arena,
                        protocol=scenario.protocol,
                    )
                    solo_relative = (
                        Path("RectSMG")
                        / label
                        / f"solo_from_agents{count}"
                        / f"{label}_{count}_{seed}_robot{robot_index}.yaml"
                    )
                    solo_sha = write_instance(
                        args.output_dir / solo_relative, solo
                    )
                    solo_records.append(
                        {
                            "record_kind": "solo",
                            "family": family,
                            "n_robots": 1,
                            "parent_n_robots": count,
                            "robot_index": robot_index,
                            "seed": seed,
                            "fingerprint": solo.fingerprint(),
                            "parent_fingerprint": scenario.fingerprint(),
                            "relative_yaml": solo_relative.as_posix(),
                            "yaml_sha256": solo_sha,
                        }
                    )

    common = {
        "description": "Paired common-unicycle Social Mini-Game instances",
        "physical_protocol": {
            "dt_s": protocol.dt,
            "horizon_s": protocol.horizon,
            "body_radius_m": protocol.body_radius,
            "pair_clearance_m": protocol.pair_clearance,
        },
    }
    for name, records in (
        ("multi_manifest.json", multi_records),
        ("solo_manifest.json", solo_records),
    ):
        payload = {**common, "record_count": len(records), "records": records}
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"exported {len(multi_records)} multi-agent instances")
    print(f"exported {len(solo_records)} solo instances")


if __name__ == "__main__":
    main()
