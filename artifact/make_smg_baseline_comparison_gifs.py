"""Generate synchronized N=20 Doorway/Intersection baseline GIFs.

The visualization advances simulated time, not controller wall time.  External
baseline runners write standardized trajectory NPZ files; this script creates
the native CLEAR/Vanilla traces, converts the MGR trace to centered physical
coordinates, and renders both individual and seven-panel comparison GIFs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmark_smg import clear_config, make_controller
from clear_nav import Protocol, SMGGeometry, make_smg_scenario
from clear_nav.unicycle import UnicycleConfig, simulate_unicycle
from make_paper_gifs import (
    Trace,
    _add_dynamic_artists,
    _draw_static_scene,
    _robot_colors,
    _save_animation,
    _update_dynamic,
    render_single,
)


METHODS = (
    ("CLEAR", "clear"),
    ("Vanilla CBF-QP", "vanilla-cbf-qp"),
    ("MGR", "mgr"),
    ("ORCA", "orca"),
    ("NH-ORCA", "nh-orca"),
    ("GCBF+", "gcbfplus"),
    ("IMPC-DR", "impc-dr"),
)
FAMILIES = ("doorway", "intersection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smg_comparison_gifs_n20"),
    )
    parser.add_argument("--horizon", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--robots", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument(
        "--prepare-native",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="reuse existing individual GIFs and rerender only the grids",
    )
    return parser.parse_args()


def geometry() -> SMGGeometry:
    return SMGGeometry(
        doorway_width=1.2,
        doorway_thickness=0.8,
        intersection_corridor_width=2.4,
    )


def save_trace(
    path: Path,
    *,
    times: np.ndarray,
    positions: np.ndarray,
    headings: np.ndarray,
    method: str,
    scenario_fingerprint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=np.asarray(times, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.float64),
        headings=np.asarray(headings, dtype=np.float64),
        method=np.asarray(method),
        scenario_fingerprint=np.asarray(scenario_fingerprint),
    )


def prepare_native(args: argparse.Namespace) -> None:
    protocol = Protocol(horizon=args.horizon, dt=args.dt)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    config = clear_config()
    trace_dir = args.output_dir / "trajectories"
    for family in FAMILIES:
        scenario = make_smg_scenario(
            family,
            args.robots,
            args.seed,
            protocol,
            geometry(),
        )
        for method_key, display in (
            ("clear", "CLEAR"),
            ("vanilla-cbf-qp", "Vanilla CBF-QP"),
        ):
            controller = make_controller(
                method_key, protocol, unicycle, config
            )
            rollout = simulate_unicycle(
                scenario,
                config,
                unicycle,
                initial_headings=np.zeros(args.robots),
                record_stride=1,
                guidance_mode="cost",
                controller=controller,
            )
            save_trace(
                trace_dir
                / f"{method_key}_{family}_n{args.robots}_s{args.seed}.npz",
                times=rollout.times,
                positions=rollout.trajectory,
                headings=rollout.headings,
                method=display,
                scenario_fingerprint=scenario.fingerprint(),
            )


def standardize_mgr(args: argparse.Namespace) -> None:
    trace_dir = args.output_dir / "trajectories"
    protocol = Protocol(horizon=args.horizon, dt=args.dt)
    for family in FAMILIES:
        source = trace_dir / f"mgr_raw_{family}_n{args.robots}_s{args.seed}.npz"
        target = trace_dir / f"mgr_{family}_n{args.robots}_s{args.seed}.npz"
        if target.exists():
            continue
        if not source.exists():
            raise FileNotFoundError(
                f"missing MGR raw trace: {source}; run run_headless.py first"
            )
        with np.load(source) as payload:
            states = np.asarray(payload["states"], dtype=np.float64)
        scenario = make_smg_scenario(
            family,
            args.robots,
            args.seed,
            protocol,
            geometry(),
        )
        positions = states[:, :, :2] - protocol.half_width
        if not np.allclose(positions[0], scenario.starts, atol=1.0e-10):
            raise RuntimeError(f"MGR start state mismatch for {family}")
        save_trace(
            target,
            times=np.arange(len(states), dtype=float) * args.dt,
            positions=positions,
            headings=states[:, :, 2],
            method="MGR",
            scenario_fingerprint=scenario.fingerprint(),
        )


def trace_path(
    trace_dir: Path,
    method_key: str,
    family: str,
    count: int,
    seed: int,
) -> Path:
    suffix = "_cpu" if method_key == "gcbfplus" else ""
    return trace_dir / (
        f"{method_key}_{family}_n{count}_s{seed}{suffix}.npz"
    )


def load_trace(path: Path, expected_count: int) -> tuple[Trace, str, str]:
    with np.load(path) as payload:
        trace = Trace(
            times=np.asarray(payload["times"], dtype=float),
            positions=np.asarray(payload["positions"], dtype=float),
            headings=np.asarray(payload["headings"], dtype=float),
        )
        method = str(np.asarray(payload["method"]).item())
        fingerprint = str(
            np.asarray(payload["scenario_fingerprint"]).item()
        )
    if trace.positions.ndim != 3 or trace.positions.shape[1:] != (
        expected_count,
        2,
    ):
        raise RuntimeError(f"invalid trajectory shape in {path}")
    if len(trace.times) != len(trace.positions):
        raise RuntimeError(f"time/state length mismatch in {path}")
    return trace, method, fingerprint


def padded(trace: Trace, horizon: float) -> Trace:
    if trace.times[-1] >= horizon - 1.0e-12:
        return trace
    return Trace(
        times=np.append(trace.times, horizon),
        positions=np.concatenate(
            (trace.positions, trace.positions[-1:]), axis=0
        ),
        headings=(
            None
            if trace.headings is None
            else np.concatenate(
                (trace.headings, trace.headings[-1:]), axis=0
            )
        ),
    )


def family_label(family: str) -> str:
    return "Doorway" if family.startswith("doorway") else "Intersection"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_grid(
    scenario,
    traces: list[tuple[str, Trace]],
    path: Path,
    *,
    horizon: float,
    max_frames: int,
    fps: int,
    dpi: int,
) -> None:
    colors = _robot_colors(scenario.n_robots)
    figure, axes = plt.subplots(
        4,
        2,
        figsize=(10.5, 16.8),
        sharex=True,
        sharey=True,
    )
    dynamic = []
    for ax, (method, trace) in zip(axes.flat, traces):
        _draw_static_scene(
            ax,
            scenario,
            method,
            colors=colors,
        )
        dynamic.append(
            (
                trace,
                _add_dynamic_artists(
                    ax,
                    scenario.n_robots,
                    colors,
                    headings=False,
                ),
            )
        )
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.text(
        0.04,
        0.96,
        "Synchronized visualization\n\n"
        f"N = {scenario.n_robots}, seed = {scenario.seed}\n"
        f"simulated horizon = {horizon:g} s\n"
        f"dt = {scenario.protocol.dt:g} s\n\n"
        "Playback uses simulated time.\n"
        "Controller wall time is not encoded.\n\n"
        "×  start     ○  goal",
        ha="left",
        va="top",
        fontsize=12,
        transform=legend_ax.transAxes,
    )
    figure.suptitle(
        f"{family_label(scenario.family)} baseline comparison",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    frame_times = np.linspace(0.0, horizon, max_frames)

    def update(time_s: float):
        artists = []
        for (method, trace), (_, handles) in zip(traces, dynamic):
            index = int(
                np.clip(
                    np.searchsorted(trace.times, time_s, side="right") - 1,
                    0,
                    len(trace.times) - 1,
                )
            )
            artists.extend(
                _update_dynamic(
                    handles,
                    trace,
                    index,
                    scenario,
                    label=method,
                )
            )
        return tuple(artists)

    _save_animation(
        figure,
        update,
        frame_times,
        path,
        fps=fps,
        dpi=dpi,
    )


def render_all(args: argparse.Namespace) -> None:
    protocol = Protocol(horizon=args.horizon, dt=args.dt)
    trace_dir = args.output_dir / "trajectories"
    gif_dir = args.output_dir / "gifs"
    records = []
    for family in FAMILIES:
        scenario = make_smg_scenario(
            family,
            args.robots,
            args.seed,
            protocol,
            geometry(),
        )
        method_traces: list[tuple[str, Trace]] = []
        for display, method_key in METHODS:
            path = trace_path(
                trace_dir,
                method_key,
                family,
                args.robots,
                args.seed,
            )
            trace, embedded_method, fingerprint = load_trace(
                path, args.robots
            )
            if embedded_method != display:
                raise RuntimeError(
                    f"method label mismatch in {path}: {embedded_method}"
                )
            if fingerprint != scenario.fingerprint():
                raise RuntimeError(
                    f"scenario fingerprint mismatch in {path}"
                )
            if not np.allclose(
                trace.positions[0], scenario.starts, atol=1.0e-8
            ):
                raise RuntimeError(f"start positions mismatch in {path}")
            trace = padded(trace, args.horizon)
            method_traces.append((display, trace))
            gif_name = (
                f"{method_key}_{family}_n{args.robots}_s{args.seed}.gif"
            )
            gif_path = gif_dir / gif_name
            if not args.comparison_only:
                render_single(
                    scenario,
                    trace,
                    gif_path,
                    f"{display} -- {family_label(family)} "
                    f"(N={args.robots}, seed={args.seed})",
                    max_frames=args.max_frames,
                    fps=args.fps,
                    dpi=args.dpi,
                    heading_lines=False,
                )
            elif not gif_path.exists():
                raise FileNotFoundError(
                    f"missing individual GIF for --comparison-only: {gif_path}"
                )
            records.append(
                {
                    "file": f"gifs/{gif_name}",
                    "kind": "individual",
                    "method": display,
                    "family": family,
                    "trajectory": path.relative_to(
                        args.output_dir
                    ).as_posix(),
                    "trajectory_sha256": sha256(path),
                    "scenario_fingerprint": fingerprint,
                    "final_arrived": int(
                        np.sum(
                            np.linalg.norm(
                                trace.positions[-1] - scenario.goals,
                                axis=1,
                            )
                            <= protocol.arrival_radius
                        )
                    ),
                }
            )
        comparison_name = (
            f"comparison_{family}_n{args.robots}_s{args.seed}.gif"
        )
        render_grid(
            scenario,
            method_traces,
            gif_dir / comparison_name,
            horizon=args.horizon,
            max_frames=args.max_frames,
            fps=args.fps,
            dpi=args.dpi,
        )
        records.append(
            {
                "file": f"gifs/{comparison_name}",
                "kind": "comparison-grid",
                "method": "all",
                "family": family,
            }
        )

    for record in records:
        gif_path = args.output_dir / record["file"]
        record["bytes"] = gif_path.stat().st_size
        record["sha256"] = sha256(gif_path)
        with Image.open(gif_path) as image:
            record["frame_count"] = int(image.n_frames)
            record["infinite_loop"] = image.info.get("loop") == 0
        if record["frame_count"] < 2 or not record["infinite_loop"]:
            raise RuntimeError(f"invalid GIF: {gif_path}")

    gcbf_payload = json.loads(
        (args.output_dir / "gcbfplus.json").read_text(encoding="utf-8")
    )
    checkpoint_steps = sorted(
        {
            int(record["parameters"]["checkpoint_step"])
            for record in gcbf_payload["records"]
        }
    )
    impc_details = {}
    for family in FAMILIES:
        impc_payload = json.loads(
            (args.output_dir / f"impc_{family}.json").read_text(
                encoding="utf-8"
            )
        )
        record = impc_payload["records"][0]
        impc_details[family] = {
            "all_solves_accepted": bool(record["all_solves_accepted"]),
            "final_solve_status_counts": record[
                "final_solve_status_counts"
            ],
            "mean_control_step_ms": record["mean_control_step_ms"],
        }
    payload = {
        "description": (
            "N=20 synchronized algorithm visualization; not an official "
            "Table VIII outcome rerun"
        ),
        "n_robots": args.robots,
        "seed": args.seed,
        "simulated_horizon_s": args.horizon,
        "dt_s": args.dt,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "wall_clock_encoded": False,
        "methods": [display for display, _ in METHODS],
        "method_notes": {
            "GCBF+": {
                "checkpoint_steps": checkpoint_steps,
                "note": (
                    "Uses the checkpoint available in the public "
                    "reproducibility image. The checkpoint 1600 referenced "
                    "by the prior Table VIII record is absent from that image."
                ),
            },
            "IMPC-DR": {
                "note": (
                    "Each solve was allowed to finish without a 30 ms "
                    "deadline. The finite trajectories are visualizable, "
                    "but these N=20 runs did not accept every solve."
                ),
                "scenario_details": impc_details,
            },
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# N=20 Doorway/Intersection comparison GIFs",
        "",
        "These animations synchronize simulated time at dt=0.03 s.",
        "Controller wall time is deliberately not encoded. They are visual",
        "comparisons, not an extension of the official N=8/16 Table VIII",
        "outcome matrix.",
        "",
        "GCBF+ uses public checkpoint 1000 because checkpoint 1600 is not",
        "present in the reproducibility image. IMPC-DR solves were allowed",
        "to finish without a 30 ms deadline; both N=20 traces are finite,",
        "but not every solve was accepted. See `manifest.json` for status",
        "counts and per-scenario details.",
        "",
    ]
    for record in records:
        lines.append(
            f"- [{record['file']}]({record['file']}) — "
            f"{record['frame_count']} frames, "
            f"{record['bytes']} bytes, SHA-256 `{record['sha256']}`"
        )
    lines.append("")
    (args.output_dir / "README.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.prepare_native:
        prepare_native(args)
        standardize_mgr(args)
    if args.render:
        render_all(args)
    print(args.output_dir)


if __name__ == "__main__":
    main()
