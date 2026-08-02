"""Aggregate one or more CLEAR JSON result files without mixing configurations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _metric(record: dict, *keys: str) -> float:
    for key in keys:
        if key in record:
            return float(record[key])
    raise KeyError(f"record contains none of the metric keys: {keys}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    fingerprints: set[str] = set()
    configurations: set[tuple] = set()
    for path in args.files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        configurations.add(
            (
                payload.get("algorithm", "CLEAR"),
                payload.get("dynamics", "single-integrator"),
                payload.get("variant", "clear"),
                payload.get("guidance", "cost"),
                payload.get("boundary_mode", "progress"),
                json.dumps(
                    payload.get("physical_protocol", {}),
                    sort_keys=True,
                ),
                json.dumps(payload.get("controller", {}), sort_keys=True),
                json.dumps(payload.get("unicycle", {}), sort_keys=True),
                json.dumps(
                    payload.get("native_projection", {}),
                    sort_keys=True,
                ),
            )
        )
        for record in payload["records"]:
            fingerprint = record["fingerprint"]
            if fingerprint in fingerprints:
                raise ValueError(f"duplicate scenario fingerprint: {fingerprint}")
            fingerprints.add(fingerprint)
            groups[(record["family"], record["n_robots"])].append(record)
    if len(configurations) != 1:
        raise ValueError("input files contain mixed controller configurations")

    summaries: list[dict] = []
    for (family, count), records in sorted(groups.items()):
        all_arrived_makespans = [
            record["makespan_s"]
            for record in records
            if record["robot_arrival_rate"] == 1.0
            and record["makespan_s"] is not None
            and np.isfinite(record["makespan_s"])
        ]
        success_makespans = [
            record["makespan_s"]
            for record in records
            if record["mission_success"]
            and record["makespan_s"] is not None
            and np.isfinite(record["makespan_s"])
        ]
        ever_all_arrived = [
            record
            for record in records
            if record["makespan_s"] is not None
            and np.isfinite(record["makespan_s"])
        ]
        summary = {
            "family": family,
            "n_robots": count,
            "instances": len(records),
            "mission_successes": sum(record["mission_success"] for record in records),
            "mission_success_rate": float(
                np.mean([record["mission_success"] for record in records])
            ),
            "all_arrived_successes": len(all_arrived_makespans),
            "ever_all_arrived_successes": len(ever_all_arrived),
            "all_arrived_success_rate": float(
                len(all_arrived_makespans) / len(records)
            ),
            "robot_arrival_rate": float(
                np.mean([record["robot_arrival_rate"] for record in records])
            ),
            "all_arrived_makespan_mean_s": (
                float(np.mean(all_arrived_makespans))
                if all_arrived_makespans
                else None
            ),
            "all_arrived_makespan_std_s": (
                float(np.std(all_arrived_makespans, ddof=1))
                if len(all_arrived_makespans) > 1
                else None
            ),
            "successful_makespan_mean_s": (
                float(np.mean(success_makespans)) if success_makespans else None
            ),
            "successful_makespan_std_s": (
                float(np.std(success_makespans, ddof=1))
                if len(success_makespans) > 1
                else None
            ),
            "minimum_pair_distance_m": min(
                _metric(
                    record,
                    "minimum_pair_distance_m",
                    "minimum_physical_pair_distance_m",
                )
                for record in records
            ),
            "minimum_obstacle_clearance_m": min(
                _metric(
                    record,
                    "minimum_obstacle_clearance_m",
                    "minimum_physical_obstacle_clearance_m",
                )
                for record in records
            ),
            "nonconverged_steps": sum(
                record["nonconverged_steps"] for record in records
            ),
            "feasibility_restoration_steps": sum(
                record.get("feasibility_restoration_steps", 0)
                for record in records
            ),
            "feasibility_restoration_calls": sum(
                record.get(
                    "feasibility_restoration_calls",
                    record.get("feasibility_restoration_steps", 0),
                )
                for record in records
            ),
            "command_infeasible_steps": sum(
                record.get(
                    "command_infeasible_steps",
                    record.get("nonconverged_steps", 0),
                )
                for record in records
            ),
            "solver_nonconverged_calls": sum(
                record.get("solver_nonconverged_calls", 0)
                for record in records
            ),
            "actuator_contraction_calls": sum(
                record.get("actuator_contraction_calls", 0)
                for record in records
            ),
            "restoration_calls_by_type": {
                key: sum(
                    record.get("restoration_calls_by_type", {}).get(key, 0)
                    for record in records
                )
                for key in (
                    "actuator_contraction",
                    "certified_witness",
                    "common_contraction",
                    "hqp_fallback",
                )
            },
            "projection_iteration_cap_calls": sum(
                record.get("projection_iteration_cap_calls", 0)
                for record in records
            ),
            "hqp_active_calls": sum(
                record.get("hqp_active_calls", 0)
                for record in records
            ),
            "hqp_nonconverged_calls": sum(
                record.get("hqp_nonconverged_calls", 0)
                for record in records
            ),
            "minimum_hqp_progress_retention": min(
                (
                    record["minimum_hqp_progress_retention"]
                    for record in records
                    if record.get("minimum_hqp_progress_retention")
                    is not None
                ),
                default=None,
            ),
            "tangent_fallback_steps": sum(
                record["tangent_fallback_steps"] for record in records
            ),
            "tangent_solver_nonconverged_steps": sum(
                record.get("tangent_solver_nonconverged_steps", 0)
                for record in records
            ),
            "cluster_escape_steps": sum(
                record.get("cluster_escape_steps", 0) for record in records
            ),
            "cluster_escape_component_steps": sum(
                record.get("cluster_escape_component_steps", 0)
                for record in records
            ),
            "static_certificate_violations": sum(
                record.get("static_certificate_violations", 0)
                for record in records
            ),
            "static_certificate_candidate_steps": sum(
                record.get("static_certificate_candidate_steps", 0)
                for record in records
            ),
            "static_certificate_excluded_projection_steps": sum(
                record.get(
                    "static_certificate_excluded_projection_steps",
                    0,
                )
                for record in records
            ),
            "static_certificate_steps": sum(
                record.get("static_certificate_steps", 0)
                for record in records
            ),
            "static_certificate_applicable_steps": sum(
                record.get("static_certificate_steps", 0)
                for record in records
            ),
            "seeds": sorted(record["seed"] for record in records),
        }
        if "physical_mission_success" in records[0]:
            summary.update(
                {
                    "physical_mission_successes": sum(
                        record["physical_mission_success"]
                        for record in records
                    ),
                    "physical_mission_success_rate": float(
                        np.mean(
                            [
                                record["physical_mission_success"]
                                for record in records
                            ]
                        )
                    ),
                }
            )
        if "minimum_transferred_pair_lower_bound_m" in records[0]:
            yaw_scales = [
                record["minimum_yaw_scale"]
                for record in records
                if record.get("minimum_yaw_scale") is not None
            ]
            yaw_retentions = [
                record["minimum_yaw_retention"]
                for record in records
                if record.get("minimum_yaw_retention") is not None
            ]
            summary.update(
                {
                    "minimum_physical_pair_distance_m": min(
                        record["minimum_physical_pair_distance_m"]
                        for record in records
                    ),
                    "minimum_transferred_pair_lower_bound_m": min(
                        record["minimum_transferred_pair_lower_bound_m"]
                        for record in records
                    ),
                    "minimum_physical_obstacle_clearance_m": min(
                        record["minimum_physical_obstacle_clearance_m"]
                        for record in records
                    ),
                    "minimum_virtual_obstacle_clearance_m": min(
                        record["minimum_virtual_obstacle_clearance_m"]
                        for record in records
                    ),
                    "maximum_yaw_rate_rps": max(
                        record["maximum_yaw_rate_rps"]
                        for record in records
                    ),
                    "maximum_linear_speed_mps": max(
                        record["maximum_linear_speed_mps"]
                        for record in records
                    ),
                    "minimum_yaw_scale": (
                        min(yaw_scales) if yaw_scales else None
                    ),
                    "minimum_yaw_retention": (
                        min(yaw_retentions)
                        if yaw_retentions
                        else None
                    ),
                }
            )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "configuration": list(configurations)[0],
                    "summaries": summaries,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
