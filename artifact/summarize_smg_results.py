"""Create the student-facing SMG comparison report from canonical records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear-vanilla",
        type=Path,
        default=Path("results/smg_clear_vanilla_yaw0_seed0.json"),
    )
    parser.add_argument(
        "--stress",
        type=Path,
        default=Path("validation/smg_yaw0_probe.json"),
    )
    parser.add_argument(
        "--gcbfplus",
        type=Path,
        default=Path(
            "baselines/comparison-harness/results/"
            "gcbfplus_smg_yaw0_seed0.json"
        ),
    )
    parser.add_argument(
        "--impc-small",
        type=Path,
        default=Path("validation/impc_scaling_probe.json"),
    )
    parser.add_argument(
        "--impc-large",
        type=Path,
        default=Path("validation/impc_scaling_n80_serial.json"),
    )
    parser.add_argument(
        "--impc-parallel",
        type=Path,
        default=Path("validation/impc_scaling_n80_parallel4.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/smg_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/SMG_REPORT.md"),
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index(records: list[dict], method_key: str) -> dict:
    return {
        (
            str(record[method_key]),
            str(record["family"]),
            int(record["n_robots"]),
        ): record
        for record in records
    }


def reduction(clear_value: float, baseline_value: float) -> float:
    return 100.0 * (1.0 - clear_value / baseline_value)


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    common = load(args.clear_vanilla)
    stress = load(args.stress)
    gcbf = load(args.gcbfplus)
    impc_small = load(args.impc_small)
    impc_large = load(args.impc_large)
    impc_parallel = load(args.impc_parallel)
    common_index = index(common["records"], "method")
    gcbf_index = index(gcbf["records"], "algorithm")
    primary = []
    for family in ("doorway_w1.20", "intersection_w2.40"):
        for count in (8, 16):
            clear = common_index[("CLEAR", family, count)]
            vanilla = common_index[("Vanilla CBF-QP", family, count)]
            learned = gcbf_index[("GCBF+", family, count)]
            comparison = {
                "family": family,
                "n_robots": count,
                "CLEAR": clear,
                "Vanilla CBF-QP": vanilla,
                "GCBF+": learned,
                "clear_vs_vanilla": {
                    "makespan_reduction_pct": reduction(
                        clear["makespan_s"],
                        vanilla["makespan_s"],
                    ),
                    "throughput_ratio": (
                        clear["parallel_throughput_robots_per_s"]
                        / vanilla[
                            "parallel_throughput_robots_per_s"
                        ]
                    ),
                    "interference_delay_reduction_pct": reduction(
                        clear["average_interference_delay_s"],
                        vanilla["average_interference_delay_s"],
                    ),
                },
            }
            primary.append(comparison)

    stress_rows = [
        record
        for record in stress["records"]
        if record["family"] == "doorway_w0.80"
    ]
    serial_scaling = [
        *impc_small["records"],
        *impc_large["records"],
    ]
    n80_serial = next(
        row for row in serial_scaling if row["n_robots"] == 80
    )
    n80_parallel = next(
        row
        for row in impc_parallel["records"]
        if row["n_robots"] == 80
    )
    pair_clearance = 2.0 * (
        common["protocol"]["body_radius"]
        + common["protocol"]["safety_margin"]
    )
    summary = {
        "scope": (
            "canonical seed-0 SMG instances; common bounded unicycle, "
            "yaw(0)=0, dt=0.03 s, horizon=60 s"
        ),
        "primary": primary,
        "doorway_w0.80_stress": stress_rows,
        "theorem_audit": {
            "certified_bridge_steps": max(
                int(record["certified_bridge_steps"])
                for record in common["records"]
                if record["method"] == "CLEAR"
            ),
            "claim": (
                "empirical only; no local Theorem 3 antecedent was "
                "certified in the canonical SMG runs"
            ),
        },
        "impc_dr": {
            "model_audit": impc_small["model_audit"],
            "serial_scaling": serial_scaling,
            "n80_serial_period_ratio_at_public_dt": (
                n80_serial["one_replan_wall_time_s"]
                / impc_small["dt_s"]
            ),
            "n80_serial_period_ratio_at_common_dt": (
                n80_serial["one_replan_wall_time_s"]
                / common["protocol"]["dt"]
            ),
            "n80_parallel4_wall_time_s": n80_parallel[
                "one_replan_wall_time_s"
            ],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Social Mini-Game canonical report",
        "",
        "All primary runs use bounded unicycle kinematics, initial yaw 0, "
        "a 0.03 s control period, and a 60 s horizon. These are canonical "
        "seed-0 instances, not a multi-seed statistical aggregate.",
        "",
        "## Primary comparison",
        "",
        "| Scenario | N | Method | Success | Makespan (s) | Throughput "
        "(robots/s) | Avg. delay (s) | Arrival | Safe |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        for method in ("CLEAR", "Vanilla CBF-QP", "GCBF+"):
            record = row[method]
            success = (
                record["mission_success"]
                if method != "GCBF+"
                else record["success"]
            )
            safe = (
                record["mission_success"]
                or (
                    record["minimum_physical_pair_distance_m"]
                    >= pair_clearance
                    and record["minimum_physical_obstacle_clearance_m"]
                    >= -1.0e-6
                )
                if method != "GCBF+"
                else record["safe"]
            )
            arrival = record["robot_arrival_rate"]
            delay_text = fmt(
                record.get("average_interference_delay_s")
            )
            if (
                method == "GCBF+"
                and record.get("delay_evaluated_robots", 0)
                < row["n_robots"]
                and delay_text != "--"
            ):
                delay_text += (
                    f" ({record['delay_evaluated_robots']}/"
                    f"{row['n_robots']})"
                )
            lines.append(
                f"| {row['family']} | {row['n_robots']} | {method} | "
                f"{int(bool(success))} | {fmt(record.get('makespan_s'))} | "
                f"{fmt(record.get('parallel_throughput_robots_per_s'), 3)} | "
                f"{delay_text} | "
                f"{100.0 * arrival:.1f}% | {int(bool(safe))} |"
            )
    lines.extend(
        [
            "",
            "CLEAR relative to Vanilla CBF-QP:",
            "",
            "| Scenario | N | Makespan reduction | Throughput ratio | "
            "Delay reduction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in primary:
        derived = row["clear_vs_vanilla"]
        lines.append(
            f"| {row['family']} | {row['n_robots']} | "
            f"{derived['makespan_reduction_pct']:.1f}% | "
            f"{derived['throughput_ratio']:.2f}x | "
            f"{derived['interference_delay_reduction_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Tight-doorway stress condition",
            "",
            "The 0.80 m doorway is a one-lane limit case and carries the "
            "baseline-width fairness note defined in `SMG_PROTOCOL.md`.",
            "",
            "| N | Method | Success | Arrival | Pair safe | Obstacle safe |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for record in stress_rows:
        lines.append(
            f"| {record['n_robots']} | {record['method']} | "
            f"{int(bool(record['mission_success']))} | "
            f"{100.0 * record['robot_arrival_rate']:.1f}% | "
            f"{int(record['minimum_physical_pair_distance_m'] >= pair_clearance - 1e-6)} | "
            f"{int(record['minimum_physical_obstacle_clearance_m'] >= -1e-6)} |"
        )
    lines.extend(
        [
            "",
            "CLEAR does not solve this stress condition completely. No "
            "canonical doorway/intersection run activated a certified "
            "straight-bridge progress row, so these scenarios remain "
            "empirical and are not evidence for local Theorem 3 coverage.",
            "",
            "## IMPC-DR audit",
            "",
            "The public SMGLib execution path is a double-integrator model "
            "with state `[x,y,vx,vy]` and input `[ax,ay]`; it is therefore "
            "not eligible for the common-unicycle outcome table without a "
            "separately validated execution adapter.",
            "",
            "| N | Serial replan time (s) | Per robot (ms) |",
            "|---:|---:|---:|",
        ]
    )
    for record in serial_scaling:
        lines.append(
            f"| {record['n_robots']} | "
            f"{record['one_replan_wall_time_s']:.3f} | "
            f"{record['per_robot_replan_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"At N=80 the serial replan takes "
            f"{n80_serial['one_replan_wall_time_s']:.3f} s: "
            f"{summary['impc_dr']['n80_serial_period_ratio_at_public_dt']:.1f}x "
            "the public 0.2 s period and "
            f"{summary['impc_dr']['n80_serial_period_ratio_at_common_dt']:.1f}x "
            "the common 0.03 s period. Four workers increase it to "
            f"{n80_parallel['one_replan_wall_time_s']:.3f} s, so the serial "
            "vectorized path is retained.",
            "",
            "## GCBF+ interpretation",
            "",
            "GCBF+ is fast (about 1.9--3.7 ms mean policy time here), but "
            "does not complete any primary SMG instance. Doorway N=8 remains "
            "safe with 12.5% arrival; doorway N=16 and both intersection "
            "sizes violate at least one common clearance audit. Several solo "
            "counterfactuals also fail to arrive, so delay is undefined "
            "unless both the multi-agent and matching solo run arrive.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
