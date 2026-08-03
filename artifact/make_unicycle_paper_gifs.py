"""Generate the official all-unicycle GIF collection for the CLEAR paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_theorem_extensions import clear_config, make_straight_bridge
from clear_nav import Protocol, SMGGeometry, make_smg_scenario
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)
from make_paper_gifs import Trace, _trim_trace, render_pair, render_single
from run_unicycle import make_unicycle_scenario

FAMILIES = ("free", "swap", "circ15", "rect15")
PAIRED_CASES = (
    ("circ15", 40, 12, "recovery"),
    ("rect15", 20, 16, "recovery"),
    ("rect15", 40, 3, "recovery"),
    ("rect15", 40, 12, "recovery"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("main", "paired", "smg", "bridge"),
        default=("main", "paired", "smg", "bridge"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_gifs"),
    )
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--smg-robots", nargs="+", type=int, default=(8, 16))
    return parser.parse_args()


def _load_trace(path: Path) -> Trace:
    with np.load(path) as payload:
        return Trace(
            np.asarray(payload["times"]),
            np.asarray(payload["positions"]),
            np.asarray(payload["headings"]),
        )


def _settings() -> tuple[Protocol, UnicycleConfig]:
    return (
        Protocol(horizon=60.0, dt=0.03),
        UnicycleConfig(
            lookahead=0.05,
            yaw_rate_limit=np.pi / 2.0,
            inner_substeps=3,
        ),
    )


def make_main(args: argparse.Namespace, records: list[dict]) -> None:
    physical, unicycle = _settings()
    for family in FAMILIES:
        scenario = make_unicycle_scenario(
            family,
            20,
            0,
            physical,
            unicycle,
        )
        rollout = simulate_unicycle(
            scenario,
            clear_config(),
            unicycle,
            initial_headings=np.zeros(20),
            record_stride=4,
            guidance_mode="cost",
        )
        trace = _trim_trace(
            rollout.times,
            rollout.trajectory,
            rollout.makespan,
            rollout.headings,
        )
        filename = f"main_unicycle_{family}_n20_seed0.gif"
        render_single(
            scenario,
            trace,
            args.output_dir / filename,
            f"Main unicycle evaluation — {family.upper()} (N=20, seed=0)",
            max_frames=args.max_frames,
            fps=args.fps,
            dpi=args.dpi,
            heading_lines=True,
        )
        records.append(
            {
                "file": filename,
                "group": "main-unicycle",
                **rollout.summary(),
            }
        )


def make_paired(args: argparse.Namespace, records: list[dict]) -> None:
    physical, unicycle = _settings()
    for family, count, seed, outcome in PAIRED_CASES:
        stem = f"{family}_n{count}_seed{seed}"
        scenario = make_unicycle_scenario(
            family,
            count,
            seed,
            physical,
            unicycle,
        )
        base = _load_trace(
            Path("results/unicycle_diagnostics_component_free")
            / f"{stem}_trace.npz"
        )
        clear = _load_trace(
            Path("results/unicycle_diagnostics_clear")
            / f"{stem}_trace.npz"
        )
        filename = f"paired_unicycle_{stem}_{outcome}.gif"
        render_pair(
            scenario,
            base,
            clear,
            args.output_dir / filename,
            f"Unicycle paired {outcome} — {family.upper()}, "
            f"N={count}, seed={seed}",
            max_frames=args.max_frames,
            fps=args.fps,
            dpi=args.dpi,
        )
        records.append(
            {
                "file": filename,
                "group": "paired-unicycle",
                "family": family,
                "n_robots": count,
                "seed": seed,
                "outcome": outcome,
            }
        )


def make_smg(args: argparse.Namespace, records: list[dict]) -> None:
    physical, unicycle = _settings()
    geometry = SMGGeometry(
        doorway_width=1.2,
        doorway_thickness=0.8,
        intersection_corridor_width=2.4,
    )
    for family in ("doorway", "intersection"):
        for count in args.smg_robots:
            scenario = make_smg_scenario(
                family,
                count,
                0,
                physical,
                geometry,
            )
            rollout = simulate_unicycle(
                scenario,
                clear_config(),
                unicycle,
                initial_headings=np.zeros(count),
                record_stride=4,
                guidance_mode="cost",
            )
            trace = _trim_trace(
                rollout.times,
                rollout.trajectory,
                rollout.makespan,
                rollout.headings,
            )
            filename = f"smg_unicycle_{family}_n{count}_seed0.gif"
            render_single(
                scenario,
                trace,
                args.output_dir / filename,
                f"Social mini-game -- {family.title()} "
                f"(N={count}, seed=0)",
                max_frames=args.max_frames,
                fps=args.fps,
                dpi=args.dpi,
                heading_lines=True,
            )
            records.append(
                {
                    "file": filename,
                    "group": "smg-unicycle",
                    **rollout.summary(),
                }
            )


def make_bridge(args: argparse.Namespace, records: list[dict]) -> None:
    physical = Protocol(horizon=10.0, dt=0.03)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    design = inflated_unicycle_protocol(physical, unicycle.lookahead)
    scenario = make_straight_bridge(
        physical,
        4,
        0,
        design_protocol=design,
    )
    rollout = simulate_unicycle(
        scenario,
        clear_config(),
        unicycle,
        initial_headings=np.zeros(4),
        record_stride=2,
    )
    trace = _trim_trace(
        rollout.times,
        rollout.trajectory,
        rollout.makespan,
        rollout.headings,
    )
    filename = "straight_bridge_unicycle_n4_seed0.gif"
    render_single(
        scenario,
        trace,
        args.output_dir / filename,
        "Unicycle StraightBridge audit (N=4, seed=0)",
        max_frames=args.max_frames,
        fps=args.fps,
        dpi=args.dpi,
        heading_lines=True,
        limits=(-4.0, 2.3, -1.15, 1.15),
    )
    records.append(
        {
            "file": filename,
            "group": "bridge-unicycle",
            **rollout.summary(),
        }
    )


def write_index(args: argparse.Namespace, records: list[dict]) -> None:
    for record in records:
        path = args.output_dir / record["file"]
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            record["frame_count"] = int(getattr(image, "n_frames", 1))
            record["infinite_loop"] = image.info.get("loop") == 0
    payload = {
        "dynamics": "bounded-lookahead-unicycle",
        "file_count": len(records),
        "lookahead_m": 0.05,
        "yaw_rate_limit_rps": float(np.pi / 2.0),
        "inner_substeps": 3,
        "main_horizon_s": 60.0,
        "terminal_capture_clutter_m": 0.22,
        "terminal_capture_open_m": 0.60,
        "terminal_release_m": 0.80,
        "hierarchical_progress": False,
        "actuation_mode": "native-cbf",
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# CLEAR paper GIFs - all-unicycle evaluation",
        "",
        "Every animation in this directory uses the bounded-input look-ahead",
        "unicycle realization with pair, boundary, linear-speed, and yaw-rate",
        "constraints enforced in one native CBF projection.",
        "",
        f"Files: {len(records)}. Per-file frame, loop, and SHA-256 values are",
        "recorded below and in `manifest.json`.",
        "",
    ]
    for record in records:
        lines.append(
            f"- [{record['file']}]({record['file']}) — "
            f"{record['frame_count']} frames, "
            f"infinite loop: {str(record['infinite_loop']).lower()}, "
            f"SHA-256 `{record['sha256']}`"
        )
    lines.append("")
    (args.output_dir / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if "main" in args.groups:
        make_main(args, records)
    if "paired" in args.groups:
        make_paired(args, records)
    if "smg" in args.groups:
        make_smg(args, records)
    if "bridge" in args.groups:
        make_bridge(args, records)
    write_index(args, records)
    print(f"wrote {len(records)} unicycle GIFs to {args.output_dir}")


if __name__ == "__main__":
    main()
