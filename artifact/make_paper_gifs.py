"""Generate trajectory GIFs for the motion experiments in the CLEAR paper.

The manuscript also contains aggregate, safety, and timing experiments that do
not define a unique trajectory.  This script covers every experiment with a
motion trace:

* the four main task families;
* the three paired component-free/CLEAR tail-recovery cases;
* the exact StraightBridge audit for component sizes 2, 4, and 8;
* the four randomized unicycle tasks and the four-robot unicycle bridge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import Rectangle as RectanglePatch
import matplotlib.pyplot as plt
import numpy as np

from benchmark_theorem_extensions import (
    clear_config,
    make_straight_bridge,
)
from clear_nav import (
    CLEARController,
    ControllerConfig,
    Protocol,
    Scenario,
    make_scenario,
    simulate,
)
from clear_nav.geometry import Circle, Rectangle
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)

Array = np.ndarray
MAIN_FAMILIES = ("free", "swap", "circ15", "rect15")
RECOVERY_CASES = (
    ("circ15", 20, 19),
    ("rect15", 40, 15),
    ("rect15", 40, 18),
)


@dataclass(frozen=True)
class Trace:
    times: Array
    positions: Array
    headings: Array | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("main", "recovery", "bridge", "unicycle"),
        default=("main", "recovery", "bridge", "unicycle"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_gifs"),
    )
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=90)
    return parser.parse_args()


def _robot_colors(count: int) -> Array:
    return plt.get_cmap("turbo")(np.linspace(0.03, 0.97, count))


def _frame_indices(length: int, maximum: int) -> Array:
    if maximum < 2:
        raise ValueError("max-frames must be at least 2")
    return np.unique(
        np.round(np.linspace(0, length - 1, min(length, maximum))).astype(int)
    )


def _trim_trace(
    times: Array,
    positions: Array,
    makespan: float,
    headings: Array | None = None,
) -> Trace:
    if np.isfinite(makespan):
        end_time = min(float(times[-1]), makespan + 1.5)
        end = min(len(times), int(np.searchsorted(times, end_time)) + 1)
    else:
        end = len(times)
    return Trace(
        np.asarray(times[:end]),
        np.asarray(positions[:end]),
        None if headings is None else np.asarray(headings[:end]),
    )


def _draw_static_scene(
    ax,
    scenario: Scenario,
    title: str,
    *,
    colors: Array,
    limits: tuple[float, float, float, float] | None = None,
) -> None:
    if limits is None:
        half = scenario.protocol.half_width
        limits = (-half, half, -half, half)
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8fafc")
    ax.grid(color="#d7dee8", linewidth=0.45, alpha=0.7)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)

    half = scenario.protocol.half_width
    ax.add_patch(
        RectanglePatch(
            (-half, -half),
            2.0 * half,
            2.0 * half,
            fill=False,
            edgecolor="#111827",
            linewidth=1.4,
            zorder=2,
        )
    )
    for obstacle in scenario.arena.obstacles:
        if isinstance(obstacle, Circle):
            patch = CirclePatch(
                obstacle.center,
                obstacle.radius,
                facecolor="#64748b",
                edgecolor="#334155",
                linewidth=0.8,
                alpha=0.85,
                zorder=2,
            )
        elif isinstance(obstacle, Rectangle):
            lower = obstacle.center - 0.5 * obstacle.size
            patch = RectanglePatch(
                lower,
                obstacle.size[0],
                obstacle.size[1],
                facecolor="#64748b",
                edgecolor="#334155",
                linewidth=0.8,
                alpha=0.85,
                zorder=2,
            )
        else:
            continue
        ax.add_patch(patch)

    ax.scatter(
        scenario.goals[:, 0],
        scenario.goals[:, 1],
        s=44,
        facecolors="none",
        edgecolors=colors,
        linewidths=1.25,
        marker="o",
        zorder=3,
    )
    ax.scatter(
        scenario.starts[:, 0],
        scenario.starts[:, 1],
        s=18,
        c=colors,
        linewidths=0.7,
        marker="x",
        zorder=3,
    )


def _add_dynamic_artists(ax, count: int, colors: Array, *, headings: bool):
    trails = LineCollection(
        [],
        colors=colors,
        linewidths=0.75,
        alpha=0.38,
        zorder=3,
    )
    ax.add_collection(trails)
    robots = ax.scatter(
        np.zeros(count),
        np.zeros(count),
        s=42,
        c=colors,
        edgecolors="#111827",
        linewidths=0.55,
        zorder=5,
    )
    heading_lines = None
    if headings:
        heading_lines = LineCollection(
            [],
            colors="#111827",
            linewidths=1.0,
            zorder=6,
        )
        ax.add_collection(heading_lines)
    status = ax.text(
        0.015,
        0.985,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.9,
        },
        zorder=8,
    )
    return trails, robots, heading_lines, status


def _update_dynamic(
    artists,
    trace: Trace,
    index: int,
    scenario: Scenario,
    *,
    label: str = "",
) -> tuple:
    trails, robots, heading_lines, status = artists
    positions = trace.positions[index]
    trails.set_segments(
        [trace.positions[: index + 1, robot] for robot in range(len(positions))]
    )
    robots.set_offsets(positions)
    if heading_lines is not None and trace.headings is not None:
        directions = np.stack(
            (
                np.cos(trace.headings[index]),
                np.sin(trace.headings[index]),
            ),
            axis=1,
        )
        heading_lines.set_segments(
            np.stack((positions, positions + 0.34 * directions), axis=1)
        )
    arrived = int(
        np.sum(
            np.linalg.norm(positions - scenario.goals, axis=1)
            <= scenario.protocol.arrival_radius
        )
    )
    prefix = f"{label}\n" if label else ""
    status.set_text(
        f"{prefix}t = {trace.times[index]:5.1f} s\n"
        f"arrived = {arrived}/{scenario.n_robots}"
    )
    result = [trails, robots, status]
    if heading_lines is not None:
        result.append(heading_lines)
    return tuple(result)


def _save_animation(
    figure: Figure,
    update,
    frames: Array,
    path: Path,
    *,
    fps: int,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    movie.save(
        path,
        writer=animation.PillowWriter(fps=fps),
        dpi=dpi,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(figure)
    print(f"wrote {path}", flush=True)


def render_single(
    scenario: Scenario,
    trace: Trace,
    path: Path,
    title: str,
    *,
    max_frames: int,
    fps: int,
    dpi: int,
    heading_lines: bool = False,
    limits: tuple[float, float, float, float] | None = None,
) -> None:
    colors = _robot_colors(scenario.n_robots)
    figure, ax = plt.subplots(figsize=(5.25, 5.25))
    _draw_static_scene(ax, scenario, title, colors=colors, limits=limits)
    artists = _add_dynamic_artists(
        ax,
        scenario.n_robots,
        colors,
        headings=heading_lines,
    )
    figure.tight_layout()
    frames = _frame_indices(len(trace.times), max_frames)

    def update(index: int):
        return _update_dynamic(artists, trace, index, scenario)

    _save_animation(figure, update, frames, path, fps=fps, dpi=dpi)


def render_pair(
    scenario: Scenario,
    left: Trace,
    right: Trace,
    path: Path,
    title: str,
    *,
    max_frames: int,
    fps: int,
    dpi: int,
    left_status_label: str = "component-free",
    right_status_label: str = "CLEAR",
) -> None:
    colors = _robot_colors(scenario.n_robots)
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.75), sharex=True, sharey=True)
    _draw_static_scene(
        axes[0],
        scenario,
        "Component-free",
        colors=colors,
    )
    _draw_static_scene(
        axes[1],
        scenario,
        "CLEAR",
        colors=colors,
    )
    left_artists = _add_dynamic_artists(
        axes[0], scenario.n_robots, colors, headings=False
    )
    right_artists = _add_dynamic_artists(
        axes[1], scenario.n_robots, colors, headings=False
    )
    figure.suptitle(title, fontsize=11, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    common_times = np.linspace(
        0.0,
        min(float(left.times[-1]), float(right.times[-1])),
        min(max_frames, max(len(left.times), len(right.times))),
    )

    def update(time: float):
        left_index = min(len(left.times) - 1, int(np.searchsorted(left.times, time)))
        right_index = min(
            len(right.times) - 1,
            int(np.searchsorted(right.times, time)),
        )
        return (
            *_update_dynamic(
                left_artists,
                left,
                left_index,
                scenario,
                label=left_status_label,
            ),
            *_update_dynamic(
                right_artists,
                right,
                right_index,
                scenario,
                label=right_status_label,
            ),
        )

    _save_animation(
        figure,
        update,
        common_times,
        path,
        fps=fps,
        dpi=dpi,
    )


def make_main_gifs(args: argparse.Namespace, records: list[dict]) -> None:
    protocol = Protocol(horizon=60.0, dt=0.03)
    for family in MAIN_FAMILIES:
        scenario = make_scenario(family, 20, 0, protocol)
        rollout = simulate(
            scenario,
            CLEARController(protocol, clear_config()),
            record_stride=4,
            guidance_mode="cost",
        )
        trace = _trim_trace(
            rollout.times,
            rollout.trajectory,
            rollout.makespan,
        )
        filename = f"main_{family}_n20_seed0.gif"
        render_single(
            scenario,
            trace,
            args.output_dir / filename,
            f"Main evaluation — {family.upper()} (N=20, seed=0)",
            max_frames=args.max_frames,
            fps=args.fps,
            dpi=args.dpi,
        )
        records.append(
            {
                "file": filename,
                "group": "main",
                "family": family,
                "n_robots": 20,
                "seed": 0,
                **rollout.summary(),
            }
        )


def _load_recovery_trace(path: Path, dt: float) -> Trace:
    with np.load(path) as payload:
        positions = np.asarray(payload["positions"])
    times = np.arange(len(positions), dtype=float) * dt
    return Trace(times, positions)


def make_recovery_gifs(args: argparse.Namespace, records: list[dict]) -> None:
    protocol = Protocol(horizon=60.0, dt=0.03)
    for family, count, seed in RECOVERY_CASES:
        stem = f"{family}_n{count}_seed{seed}"
        left_path = (
            Path("results/diagnostics_component_free")
            / f"{stem}_trace.npz"
        )
        right_path = Path("results/diagnostics_clear") / f"{stem}_trace.npz"
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(
                f"retained diagnostic traces are missing for {stem}"
            )
        scenario = make_scenario(family, count, seed, protocol)
        left = _load_recovery_trace(left_path, protocol.dt)
        right = _load_recovery_trace(right_path, protocol.dt)
        filename = f"recovery_{stem}.gif"
        render_pair(
            scenario,
            left,
            right,
            args.output_dir / filename,
            f"Paired tail recovery — {family.upper()}, N={count}, seed={seed}",
            max_frames=args.max_frames,
            fps=args.fps,
            dpi=args.dpi,
        )
        records.append(
            {
                "file": filename,
                "group": "recovery",
                "family": family,
                "n_robots": count,
                "seed": seed,
                "left": "component-free",
                "right": "CLEAR",
            }
        )


def _straight_bridge_trace(scenario: Scenario, passage_length: float = 1.5) -> Trace:
    controller = CLEARController(scenario.protocol, clear_config())
    controller.reset()
    positions = np.asarray(scenario.starts, dtype=float).copy()
    initial_progress = float(np.mean(positions[:, 0]))
    times = [0.0]
    trajectory = [positions.copy()]
    for step in range(scenario.protocol.steps):
        velocity = controller.command(
            positions,
            scenario.goals,
            scenario.arena,
        ).executed_command
        positions = positions + scenario.protocol.dt * velocity
        times.append((step + 1) * scenario.protocol.dt)
        trajectory.append(positions.copy())
        if float(np.mean(positions[:, 0])) >= initial_progress + passage_length:
            break
    return Trace(np.asarray(times), np.asarray(trajectory))


def make_bridge_gif(args: argparse.Namespace, records: list[dict]) -> None:
    protocol = Protocol(horizon=8.0, dt=0.03)
    scenarios = [
        make_straight_bridge(protocol, count, 0) for count in (2, 4, 8)
    ]
    traces = [_straight_bridge_trace(scenario) for scenario in scenarios]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.75), sharex=True, sharey=True)
    dynamic = []
    for ax, scenario, trace in zip(axes, scenarios, traces):
        colors = _robot_colors(scenario.n_robots)
        xmin = min(-3.9, float(np.min(scenario.starts[:, 0])) - 0.6)
        limits = (xmin, 0.3, -1.15, 1.15)
        _draw_static_scene(
            ax,
            scenario,
            f"component size m={scenario.n_robots}",
            colors=colors,
            limits=limits,
        )
        dynamic.append(
            _add_dynamic_artists(
                ax,
                scenario.n_robots,
                colors,
                headings=False,
            )
        )
    figure.suptitle(
        "Exact StraightBridge runtime audit (seed=0)",
        fontsize=11,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    end_time = max(float(trace.times[-1]) for trace in traces)
    common_times = np.linspace(
        0.0,
        end_time,
        min(args.max_frames, max(len(trace.times) for trace in traces)),
    )

    def update(time: float):
        changed = []
        for scenario, trace, artists in zip(scenarios, traces, dynamic):
            index = min(
                len(trace.times) - 1,
                int(np.searchsorted(trace.times, time)),
            )
            changed.extend(_update_dynamic(artists, trace, index, scenario))
        return tuple(changed)

    filename = "straight_bridge_m2_m4_m8_seed0.gif"
    _save_animation(
        figure,
        update,
        common_times,
        args.output_dir / filename,
        fps=args.fps,
        dpi=args.dpi,
    )
    records.append(
        {
            "file": filename,
            "group": "bridge",
            "component_sizes": [2, 4, 8],
            "seed": 0,
            "exit_times_s": {
                str(scenario.n_robots): float(trace.times[-1])
                for scenario, trace in zip(scenarios, traces)
            },
        }
    )


def _unicycle_scenario(
    family: str,
    count: int,
    seed: int,
    physical: Protocol,
    config: UnicycleConfig,
) -> Scenario:
    design = inflated_unicycle_protocol(physical, config.lookahead)
    generated = make_scenario(family, count, seed, design)
    return Scenario(
        family=family,
        n_robots=count,
        seed=seed,
        starts=(
            generated.starts
            - config.lookahead * np.array([1.0, 0.0])
        ),
        goals=generated.goals,
        arena=generated.arena,
        protocol=physical,
    )


def make_unicycle_gifs(args: argparse.Namespace, records: list[dict]) -> None:
    config = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    physical = Protocol(horizon=60.0, dt=0.03)
    for family in MAIN_FAMILIES:
        scenario = _unicycle_scenario(family, 20, 0, physical, config)
        rollout = simulate_unicycle(
            scenario,
            clear_config(),
            config,
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
        filename = f"unicycle_{family}_n20_seed0.gif"
        render_single(
            scenario,
            trace,
            args.output_dir / filename,
            f"Look-ahead unicycle — {family.upper()} (N=20, seed=0)",
            max_frames=args.max_frames,
            fps=args.fps,
            dpi=args.dpi,
            heading_lines=True,
        )
        records.append(
            {
                "file": filename,
                "group": "unicycle",
                "family": family,
                "n_robots": 20,
                "seed": 0,
                **rollout.summary(),
            }
        )

    bridge_physical = Protocol(horizon=10.0, dt=0.03)
    bridge_design = inflated_unicycle_protocol(
        bridge_physical,
        config.lookahead,
    )
    bridge = make_straight_bridge(
        bridge_physical,
        4,
        0,
        design_protocol=bridge_design,
    )
    bridge_rollout = simulate_unicycle(
        bridge,
        clear_config(),
        config,
        initial_headings=np.zeros(4),
        record_stride=2,
    )
    bridge_trace = _trim_trace(
        bridge_rollout.times,
        bridge_rollout.trajectory,
        bridge_rollout.makespan,
        bridge_rollout.headings,
    )
    filename = "unicycle_straight_bridge_n4_seed0.gif"
    render_single(
        bridge,
        bridge_trace,
        args.output_dir / filename,
        "Look-ahead unicycle — StraightBridge (N=4, seed=0)",
        max_frames=args.max_frames,
        fps=args.fps,
        dpi=args.dpi,
        heading_lines=True,
        limits=(-4.0, 2.3, -1.15, 1.15),
    )
    records.append(
        {
            "file": filename,
            "group": "unicycle",
            "family": "straight_bridge",
            "n_robots": 4,
            "seed": 0,
            **bridge_rollout.summary(),
        }
    )


def _write_index(output: Path, records: list[dict], args: argparse.Namespace) -> None:
    payload = {
        "source_manuscript": "CLEAR manuscript source",
        "groups": list(args.groups),
        "max_frames": args.max_frames,
        "fps": args.fps,
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["group"], []).append(record)
    labels = {
        "main": "Main evaluation",
        "recovery": "Paired tail recovery",
        "bridge": "StraightBridge theorem audit",
        "unicycle": "Unicycle transfer",
    }
    lines = [
        "# CLEAR paper experiment GIFs",
        "",
        "Generated from the official manuscript settings in `paper/main2.tex`.",
        "Aggregate safety/timing tables are not trajectory experiments and are",
        "not converted into artificial animations. Their source records remain",
        "in the parent `results/` directory and `validation/`.",
        "",
    ]
    for group in args.groups:
        if group not in grouped:
            continue
        lines.extend((f"## {labels[group]}", ""))
        for record in grouped[group]:
            lines.append(f"- [{record['file']}]({record['file']})")
        lines.append("")
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2")
    if args.fps < 1 or args.dpi < 1:
        raise ValueError("--fps and --dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if "main" in args.groups:
        make_main_gifs(args, records)
    if "recovery" in args.groups:
        make_recovery_gifs(args, records)
    if "bridge" in args.groups:
        make_bridge_gif(args, records)
    if "unicycle" in args.groups:
        make_unicycle_gifs(args, records)
    _write_index(args.output_dir, records, args)
    print(
        f"wrote {len(records)} GIF entries to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
