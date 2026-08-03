"""Generate short website GIFs for the internal CLEAR comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

from benchmark_theorem_extensions import clear_config
from clear_nav import Protocol, VanillaCBFController
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)
from make_paper_figures import _draw_arena
from run_unicycle import make_unicycle_scenario


@dataclass(frozen=True)
class Trace:
    times: np.ndarray
    positions: np.ndarray
    goals: np.ndarray


TRACE_PATHS = {
    "clear": Path("results/internal_30ms_diagnostics_clear")
    / "rect15_n20_seed16_trace.npz",
    "vanilla-cbf-qp": Path("results/internal_30ms_diagnostics_vanilla")
    / "rect15_n20_seed16_trace.npz",
    "component-free": Path("results/internal_30ms_diagnostics_component_free")
    / "rect15_n20_seed16_trace.npz",
}
HIGHLIGHTED = (4, 6, 11)
ROBOT_COLORS = {
    4: "#0072B2",
    6: "#D55E00",
    11: "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/internal_comparison_gifs"),
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=14)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--window", nargs=2, type=float, default=(10.0, 20.0))
    return parser.parse_args()


def ensure_trace(
    variant: str,
    protocol: Protocol,
    unicycle: UnicycleConfig,
) -> None:
    trace_path = TRACE_PATHS[variant]
    if trace_path.exists():
        return
    scenario = make_unicycle_scenario("rect15", 20, 16, protocol, unicycle)
    config = clear_config()
    controller = None
    if variant != "clear":
        config = replace(
            config,
            cluster_escape_gain=0.0,
            cluster_escape_hysteresis=False,
        )
    if variant == "vanilla-cbf-qp":
        controller = VanillaCBFController(
            inflated_unicycle_protocol(protocol, unicycle.lookahead),
            config,
        )
    rollout = simulate_unicycle(
        scenario,
        config,
        unicycle,
        initial_headings=np.zeros(20),
        record_stride=1,
        guidance_mode="cost",
        controller=controller,
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        times=rollout.times,
        positions=rollout.trajectory,
        headings=rollout.headings,
        goals=scenario.goals,
    )


def load_trace(path: Path) -> Trace:
    with np.load(path) as payload:
        return Trace(
            times=np.asarray(payload["times"]),
            positions=np.asarray(payload["positions"]),
            goals=np.asarray(payload["goals"]),
        )


def panel_colors(variant: str, robot_colored: bool) -> list[str]:
    colors = ["#AAB4C0"] * 20
    for robot in HIGHLIGHTED:
        if robot_colored or variant == "clear":
            colors[robot] = ROBOT_COLORS[robot]
        else:
            colors[robot] = "#D94747"
    return colors


def render_comparison(
    scenario,
    left_variant: str,
    right_variant: str,
    left_title: str,
    right_title: str,
    output: Path,
    *,
    robot_colored: bool,
    window: tuple[float, float],
    duration: float,
    fps: int,
    dpi: int,
) -> None:
    traces = [
        load_trace(TRACE_PATHS[left_variant]),
        load_trace(TRACE_PATHS[right_variant]),
    ]
    variants = (left_variant, right_variant)
    titles = (left_title, right_title)
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    dynamic = []

    for ax, trace, variant, title in zip(axes, traces, variants, titles):
        _draw_arena(ax, scenario)
        colors = panel_colors(variant, robot_colored)
        rgba = [matplotlib.colors.to_rgba(color, 0.90) for color in colors]
        trail_rgba = [
            matplotlib.colors.to_rgba(color, 0.85 if i in HIGHLIGHTED else 0.30)
            for i, color in enumerate(colors)
        ]
        trails = LineCollection(
            [],
            colors=trail_rgba,
            linewidths=[1.25 if i in HIGHLIGHTED else 0.55 for i in range(20)],
            zorder=3,
        )
        ax.add_collection(trails)
        robots = ax.scatter(
            np.zeros(20),
            np.zeros(20),
            s=[52 if i in HIGHLIGHTED else 30 for i in range(20)],
            c=rgba,
            edgecolors="white",
            linewidths=0.45,
            zorder=5,
        )
        for robot in HIGHLIGHTED:
            ax.scatter(
                trace.goals[robot, 0],
                trace.goals[robot, 1],
                s=18,
                marker=".",
                color=colors[robot],
                linewidth=0,
                zorder=6,
            )
        clock = ax.text(
            0.03,
            0.96,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#454A50",
            zorder=7,
        )
        ax.set_title(title, fontsize=9, weight="bold", pad=4)
        ax.set_xlim(2.0, 8.0)
        ax.set_ylim(-6.0, 0.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        dynamic.append((trails, robots, clock))

    frame_count = max(2, int(round(duration * fps)))
    frame_times = np.linspace(window[0], window[1], frame_count)
    start_indices = [int(np.searchsorted(trace.times, window[0])) for trace in traces]

    def update(time: float):
        artists = []
        for trace, start, (trails, robots, clock) in zip(
            traces, start_indices, dynamic
        ):
            index = min(len(trace.times) - 1, int(np.searchsorted(trace.times, time)))
            trails.set_segments(
                [trace.positions[start : index + 1 : 2, robot] for robot in range(20)]
            )
            robots.set_offsets(trace.positions[index])
            clock.set_text(f"t = {time - window[0]:.1f} s")
            artists.extend((trails, robots, clock))
        return tuple(artists)

    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.90, wspace=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frame_times,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    movie.save(
        output,
        writer=animation.PillowWriter(fps=fps),
        dpi=dpi,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(figure)
    print(f"wrote {output}")


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise ValueError("duration and fps must be positive")
    protocol = Protocol(dt=0.03)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    for variant in TRACE_PATHS:
        ensure_trace(variant, protocol, unicycle)
    scenario = make_unicycle_scenario("rect15", 20, 16, protocol, unicycle)
    window = (float(args.window[0]), float(args.window[1]))
    render_comparison(
        scenario,
        "clear",
        "vanilla-cbf-qp",
        "CLEAR",
        "Plain CBF-QP",
        args.output_dir / "clear_reciprocal_qp_closeup.gif",
        robot_colored=False,
        window=window,
        duration=args.duration,
        fps=args.fps,
        dpi=args.dpi,
    )
    render_comparison(
        scenario,
        "clear",
        "component-free",
        "CLEAR",
        "Without component translation",
        args.output_dir / "clear_component_closeup.gif",
        robot_colored=True,
        window=window,
        duration=args.duration,
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
