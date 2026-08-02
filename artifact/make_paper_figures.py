"""Generate the figures used by the CLEAR T-RO manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
from matplotlib.patches import FancyArrowPatch, Rectangle as RectanglePatch
import numpy as np

from benchmark_theorem_extensions import clear_config, make_straight_bridge
from clear_nav import (
    Protocol,
    REPORTED_CLEAR_CONTROLLER,
    SMGGeometry,
    make_scenario,
    make_smg_scenario,
)
from clear_nav.geometry import Circle, Rectangle
from clear_nav.unicycle import (
    UnicycleConfig,
    inflated_unicycle_protocol,
    simulate_unicycle,
)
from run_unicycle import make_unicycle_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
    )
    parser.add_argument("--smg-task-panels-only", action="store_true")
    parser.add_argument("--internal-closeups-only", action="store_true")
    parser.add_argument("--bridge-panel-only", action="store_true")
    return parser.parse_args()


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {path}")


def make_overview(output: Path) -> None:
    blue = "#2463A5"
    orange = "#D97706"
    green = "#16803C"
    gray = "#B9BEC6"
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.15))

    ax = axes[0]
    for center in ((-0.48, 0.0), (0.48, 0.0)):
        ax.add_patch(CirclePatch(center, 0.31, color=blue, alpha=0.85))
    ax.plot([-0.17, 0.17], [0, 0], color="#A61B1B", lw=3)
    ax.add_patch(FancyArrowPatch((-0.48, 0.35), (-0.48, 1.0),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=3, color=orange))
    ax.add_patch(FancyArrowPatch((0.48, -0.35), (0.48, -1.0),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=3, color=orange))
    ax.text(0, -1.25, "relative circulation", ha="center", fontsize=10)
    ax.set_title("(a) Pair contact", fontweight="bold")

    ax = axes[1]
    ax.add_patch(CirclePatch((0.0, 0.0), 0.8, color=gray))
    ax.add_patch(CirclePatch((1.1, 0.0), 0.28, color=blue))
    ax.add_patch(FancyArrowPatch((1.1, -0.05), (1.1, 0.92),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=3, color=orange))
    ax.add_patch(FancyArrowPatch((1.15, 0.0), (0.72, 0.0),
                                 arrowstyle="-|>", mutation_scale=15,
                                 lw=2, color="#A61B1B", alpha=0.8))
    ax.text(0.5, -1.25, "progress-aligned tangent", ha="center", fontsize=10)
    ax.set_title("(b) Obstacle contact", fontweight="bold")

    ax = axes[2]
    ax.add_patch(RectanglePatch((-1.45, 0.5), 2.9, 0.5, color=gray))
    ax.add_patch(RectanglePatch((-1.45, -1.0), 2.9, 0.5, color=gray))
    centers = [(-0.45, 0.18), (0.0, -0.18), (0.48, 0.18)]
    for center in centers:
        ax.add_patch(CirclePatch(center, 0.27, color=blue, alpha=0.85))
        ax.add_patch(FancyArrowPatch(center, (center[0] + 0.85, center[1]),
                                     arrowstyle="-|>", mutation_scale=16,
                                     lw=2.6, color=green))
    ax.text(0, -1.25, "event-held component motion", ha="center", fontsize=10)
    ax.set_title("(c) Boundary bridge", fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.45, 1.35)
        ax.axis("off")
    figure.tight_layout(w_pad=1.0)
    _save(figure, output / "clear_overview.png")


def _draw_arena(ax: plt.Axes, scenario) -> None:
    limit = scenario.protocol.half_width
    ax.add_patch(
        RectanglePatch(
            (-limit, -limit),
            2 * limit,
            2 * limit,
            facecolor="white",
            edgecolor="#454A50",
            lw=0.8,
        )
    )
    for obstacle in scenario.arena.obstacles:
        if isinstance(obstacle, Circle):
            ax.add_patch(
                CirclePatch(
                    obstacle.center,
                    obstacle.radius,
                    facecolor="#B9BEC6",
                    edgecolor="none",
                    alpha=0.8,
                )
            )
        elif isinstance(obstacle, Rectangle):
            lower = obstacle.center - 0.5 * obstacle.size
            ax.add_patch(
                RectanglePatch(
                    lower,
                    obstacle.size[0],
                    obstacle.size[1],
                    facecolor="#B9BEC6",
                    edgecolor="none",
                    alpha=0.8,
                )
            )


def make_smg_task_panels(output: Path) -> None:
    physical = Protocol(dt=0.03, horizon=60.0)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    geometry = SMGGeometry(
        doorway_width=1.2,
        doorway_thickness=0.8,
        intersection_corridor_width=2.4,
    )
    for family in ("doorway", "intersection"):
        scenario = make_smg_scenario(
            family,
            16,
            0,
            physical,
            geometry,
        )
        rollout = simulate_unicycle(
            scenario,
            REPORTED_CLEAR_CONTROLLER,
            unicycle,
            initial_headings=np.zeros(16),
            record_stride=10,
            guidance_mode="cost",
        )
        figure, ax = plt.subplots(figsize=(3.05, 3.05))
        _draw_arena(ax, scenario)
        colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, 16))
        for robot, color in enumerate(colors):
            path = rollout.trajectory[:, robot]
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                linewidth=0.85,
                alpha=0.72,
                zorder=2,
            )
            ax.scatter(
                path[0, 0],
                path[0, 1],
                marker="s",
                s=10,
                color=color,
                edgecolors="none",
                zorder=3,
            )
            ax.scatter(
                scenario.goals[robot, 0],
                scenario.goals[robot, 1],
                marker="x",
                s=18,
                color=color,
                linewidths=0.9,
                zorder=4,
            )
            ax.add_patch(
                CirclePatch(
                    path[-1],
                    physical.body_radius,
                    facecolor=color,
                    edgecolor="#20262c",
                    linewidth=0.35,
                    zorder=5,
                )
            )
        limit = physical.half_width
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        tick_values = (-8, -4, 0, 4, 8)
        tick_labels = ("-8", "-4", "0", "4", "8")
        ax.set_xticks(tick_values, labels=tick_labels)
        ax.set_yticks(tick_values, labels=tick_labels)
        ax.tick_params(
            labelsize=7,
            length=2.5,
            width=0.6,
            labelbottom=True,
            labelleft=True,
            colors="#111827",
        )
        ax.grid(color="#d8dde3", linewidth=0.35, alpha=0.7, zorder=-1)
        ax.set_xlabel("x (m)", fontsize=8, labelpad=1)
        ax.set_ylabel("y (m)", fontsize=8, labelpad=1)
        figure.tight_layout(pad=0.25)
        target = output / f"clear_n16_{family}_seed0.png"
        figure.savefig(
            target,
            dpi=320,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)
        print(
            f"wrote {target} success={rollout.mission_success} "
            f"makespan={rollout.makespan:.3f}s"
        )


def make_straight_bridge_panel(output: Path) -> None:
    physical = Protocol(dt=0.03, horizon=10.0)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    scenario = make_straight_bridge(
        physical,
        8,
        0,
        design_protocol=inflated_unicycle_protocol(
            physical,
            unicycle.lookahead,
        ),
    )
    rollout = simulate_unicycle(
        scenario,
        clear_config(),
        unicycle,
        initial_headings=np.zeros(8),
        record_stride=2,
    )
    figure, ax = plt.subplots(figsize=(6.4, 2.15))
    _draw_arena(ax, scenario)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, 8))
    for robot, color in enumerate(colors):
        path = rollout.trajectory[:, robot]
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=1.0,
            alpha=0.78,
            zorder=2,
        )
        ax.scatter(
            path[0, 0],
            path[0, 1],
            marker="s",
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )
        ax.scatter(
            scenario.goals[robot, 0],
            scenario.goals[robot, 1],
            marker="x",
            s=28,
            color=color,
            linewidths=1.0,
            zorder=7,
        )
        ax.add_patch(
            CirclePatch(
                path[-1],
                physical.body_radius,
                facecolor=color,
                edgecolor="#20262c",
                linewidth=0.4,
                zorder=6,
            )
        )
    xmin = float(np.min(scenario.starts[:, 0])) - 0.55
    xmax = float(np.max(scenario.goals[:, 0])) + 0.55
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.95, 0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=8, labelpad=1)
    ax.set_ylabel("y (m)", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7, length=2.5, width=0.6)
    ax.grid(color="#d8dde3", linewidth=0.35, alpha=0.7, zorder=-1)
    figure.tight_layout(pad=0.25)
    target = output / "clear_straight_bridge_n8_seed0.png"
    figure.savefig(
        target,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(
        f"wrote {target} success={rollout.mission_success} "
        f"makespan={rollout.makespan:.3f}s"
    )


def make_failure_recovery(output: Path) -> None:
    cases = [
        ("circ15", 40, 12, "recovery"),
        ("rect15", 20, 16, "recovery"),
        ("rect15", 40, 3, "recovery"),
        ("rect15", 40, 12, "recovery"),
    ]
    protocol = Protocol()
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    figure, axes = plt.subplots(2, 4, figsize=(13.2, 6.6))
    for column, (family, count, seed, outcome) in enumerate(cases):
        scenario = make_unicycle_scenario(
            family,
            count,
            seed,
            protocol,
            unicycle,
        )
        stem = f"{family}_n{count}_seed{seed}"
        paths = [
            Path("results/unicycle_diagnostics_component_free")
            / f"{stem}_trace.npz",
            Path("results/unicycle_diagnostics_clear")
            / f"{stem}_trace.npz",
        ]
        report_paths = [
            Path("results/unicycle_diagnostics_component_free")
            / f"{stem}_report.json",
            Path("results/unicycle_diagnostics_clear")
            / f"{stem}_report.json",
        ]
        for row, (trace_path, report_path) in enumerate(
            zip(paths, report_paths)
        ):
            ax = axes[row, column]
            _draw_arena(ax, scenario)
            with np.load(trace_path) as trace:
                positions = trace["positions"]
                final_distance = trace["final_goal_distances"]
            unfinished = set(
                np.flatnonzero(
                    final_distance > protocol.arrival_radius
                ).tolist()
            )
            for robot in range(count):
                if robot in unfinished:
                    color, width, alpha = "#C72C2C", 1.8, 1.0
                else:
                    color, width, alpha = "#2463A5", 0.65, 0.38
                ax.plot(
                    positions[::8, robot, 0],
                    positions[::8, robot, 1],
                    color=color,
                    lw=width,
                    alpha=alpha,
                )
            ax.scatter(
                scenario.starts[:, 0],
                scenario.starts[:, 1],
                s=7,
                color="#111827",
                alpha=0.55,
                zorder=4,
            )
            ax.scatter(
                scenario.goals[:, 0],
                scenario.goals[:, 1],
                s=18,
                marker="*",
                color="#16803C",
                alpha=0.75,
                zorder=5,
            )
            ax.set_xlim(-protocol.half_width, protocol.half_width)
            ax.set_ylim(-protocol.half_width, protocol.half_width)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f"{family.capitalize()}, $N={count}$, seed {seed}\n"
                    f"({outcome})",
                    fontweight="bold",
                    fontsize=9,
                )
            if column == 0:
                ax.set_ylabel(
                    "Component-free" if row == 0 else "CLEAR",
                    fontsize=11,
                    fontweight="bold",
                )
            arrived = count - len(unfinished)
            ax.text(
                0.03,
                0.03,
                f"{arrived}/{count} arrived",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
            )
    figure.tight_layout(h_pad=0.55, w_pad=0.45)
    _save(figure, output / "clear_failure_recovery.png")


def make_internal_closeups(output: Path) -> None:
    protocol = Protocol(dt=0.03)
    unicycle = UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
    )
    common = {
        "family": "rect15",
        "count": 20,
        "seed": 16,
        "window": (10.0, 20.0),
        "arrow_time": 14.0,
        "xlim": (2.0, 8.0),
        "ylim": (-6.0, 0.0),
        "highlight": (4, 6, 11),
    }
    panels = [
        {
            **common,
            "trace": Path("results/internal_30ms_diagnostics_clear")
            / "rect15_n20_seed16_trace.npz",
            "color": "#1F6FB2",
        },
        {
            **common,
            "trace": Path("results/internal_30ms_diagnostics_vanilla")
            / "rect15_n20_seed16_trace.npz",
            "color": "#C83A3A",
        },
        {
            **common,
            "trace": Path("results/internal_30ms_diagnostics_component_free")
            / "rect15_n20_seed16_trace.npz",
            "color": "#C83A3A",
        },
    ]

    def draw_panels(selected_panels: list[dict], filename: str) -> None:
        figure, axes = plt.subplots(
            1,
            len(selected_panels),
            figsize=(3.2 * len(selected_panels), 3.2),
            squeeze=False,
        )
        for ax, panel in zip(axes[0], selected_panels):
            scenario = make_unicycle_scenario(
                panel["family"],
                panel["count"],
                panel["seed"],
                protocol,
                unicycle,
            )
            _draw_arena(ax, scenario)
            with np.load(panel["trace"]) as trace:
                times = trace["times"]
                positions = trace["positions"]
                goals = trace["goals"]
            begin = int(np.searchsorted(times, panel["window"][0]))
            end = min(
                int(np.searchsorted(times, panel["window"][1])),
                len(times) - 1,
            )
            highlighted = set(panel["highlight"])
            for robot in range(panel["count"]):
                selected = robot in highlighted
                ax.plot(
                    positions[begin : end + 1 : 2, robot, 0],
                    positions[begin : end + 1 : 2, robot, 1],
                    color=panel["color"] if selected else "#AAB4C0",
                    lw=1.25 if selected else 0.55,
                    alpha=0.90 if selected else 0.38,
                    zorder=3 if selected else 2,
                )
            arrow_index = int(np.argmin(np.abs(times - panel["arrow_time"])))
            future_index = min(arrow_index + 7, len(times) - 1)
            for robot in sorted(highlighted):
                direction = positions[future_index, robot] - positions[arrow_index, robot]
                norm = float(np.linalg.norm(direction))
                if norm <= 1.0e-5:
                    continue
                direction *= 0.28 / norm
                ax.add_patch(
                    FancyArrowPatch(
                        positions[arrow_index, robot],
                        positions[arrow_index, robot] + direction,
                        arrowstyle="-|>",
                        mutation_scale=7.5,
                        lw=0.9,
                        color=panel["color"],
                        zorder=5,
                    )
                )
            for robot in range(panel["count"]):
                selected = robot in highlighted
                ax.add_patch(
                    CirclePatch(
                        positions[end, robot],
                        protocol.body_radius,
                        facecolor=panel["color"] if selected else "#9AA4B0",
                        edgecolor="white",
                        lw=0.45,
                        alpha=0.92 if selected else 0.55,
                        zorder=6 if selected else 4,
                    )
                )
            selected = np.array(sorted(highlighted), dtype=int)
            ax.scatter(
                goals[selected, 0],
                goals[selected, 1],
                s=24,
                marker="*",
                color="#16803C",
                edgecolor="white",
                linewidth=0.35,
                zorder=7,
            )
            ax.set_xlim(*panel["xlim"])
            ax.set_ylim(*panel["ylim"])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        figure.tight_layout(pad=0.05, w_pad=0.30)
        _save(figure, output / filename)

    draw_panels([panels[0]], "clear_internal_full_closeup.png")
    draw_panels([panels[1]], "clear_internal_vanilla_closeup.png")
    draw_panels([panels[2]], "clear_internal_component_free_closeup.png")


def make_results(output: Path) -> None:
    labels = ["Circ20", "Rect20", "Circ40", "Rect40"]
    keys = [
        ("circ15", 20),
        ("rect15", 20),
        ("circ15", 40),
        ("rect15", 40),
    ]

    def load_records(paths: list[Path]) -> dict[tuple[str, int, int], dict]:
        result: dict[tuple[str, int, int], dict] = {}
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload["records"]:
                result[
                    (
                        record["family"],
                        record["n_robots"],
                        record["seed"],
                    )
                ] = record
        return result

    clear = load_records(
        [
            Path("results/unicycle_clear_all_sizes_raw.json"),
        ]
    )
    base = load_records(
        [
            Path("results/unicycle_component_free_n20_n40_raw.json"),
        ]
    )
    base_success = np.array(
        [
            100
            * sum(
                base[(family, count, seed)]["robot_arrival_rate"] == 1.0
                for seed in range(20)
            )
            / 20
            for family, count in keys
        ]
    )
    full_success = np.array(
        [
            100
            * sum(
                clear[(family, count, seed)]["robot_arrival_rate"] == 1.0
                for seed in range(20)
            )
            / 20
            for family, count in keys
        ]
    )
    base_fallback = np.array(
        [
            sum(
                base[(family, count, seed)]["tangent_fallback_steps"]
                for seed in range(20)
            )
            for family, count in keys
        ]
    )
    full_fallback = np.array(
        [
            sum(
                clear[(family, count, seed)]["tangent_fallback_steps"]
                for seed in range(20)
            )
            for family, count in keys
        ]
    )
    time_change = []
    for family, count in keys:
        paired_delta = []
        for seed in range(20):
            base_record = base[(family, count, seed)]
            clear_record = clear[(family, count, seed)]
            if (
                base_record["robot_arrival_rate"] == 1.0
                and clear_record["robot_arrival_rate"] == 1.0
            ):
                paired_delta.append(
                    clear_record["makespan_s"]
                    - base_record["makespan_s"]
                )
        time_change.append(float(np.mean(paired_delta)))
    time_change = np.asarray(time_change)
    x = np.arange(len(labels))
    width = 0.36

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.35))
    axes[0].bar(x - width / 2, base_success, width, label="Component-free",
                color="#9CA3AF")
    axes[0].bar(x + width / 2, full_success, width, label="CLEAR",
                color="#2463A5")
    axes[0].set_ylim(
        max(0.0, float(min(np.min(base_success), np.min(full_success))) - 5.0),
        102,
    )
    axes[0].set_ylabel("Mission success (%)")
    axes[0].set_xticks(x, labels, rotation=25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("(a) Tail success", fontweight="bold")

    axes[1].bar(x - width / 2, base_fallback, width, color="#9CA3AF")
    axes[1].bar(x + width / 2, full_fallback, width, color="#2463A5")
    axes[1].set_ylabel("Tangent fallback steps")
    axes[1].set_xticks(x, labels, rotation=25)
    axes[1].set_title("(b) Numerical audit", fontweight="bold")

    axes[2].bar(x, time_change, 0.55, color="#16803C")
    axes[2].axhline(0, color="#333333", lw=0.8)
    padding = max(0.2, 0.15 * float(np.ptp(time_change)))
    axes[2].set_ylim(
        min(0.0, float(np.min(time_change)) - padding),
        max(0.0, float(np.max(time_change)) + padding),
    )
    axes[2].set_ylabel("Mean makespan change (s)")
    axes[2].set_xticks(x, labels, rotation=25)
    axes[2].set_title("(c) Paired makespan effect", fontweight="bold")
    for index, value in enumerate(time_change):
        offset = 0.04 if value >= 0.0 else -0.08
        axes[2].text(
            index,
            value + offset,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=8,
        )

    for ax in axes:
        ax.grid(axis="y", alpha=0.2, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(w_pad=1.2)
    _save(figure, output / "clear_results.png")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smg_task_panels_only:
        make_smg_task_panels(args.output_dir)
        return
    if args.internal_closeups_only:
        make_internal_closeups(args.output_dir)
        return
    if args.bridge_panel_only:
        make_straight_bridge_panel(args.output_dir)
        return
    make_overview(args.output_dir)
    make_failure_recovery(args.output_dir)
    make_internal_closeups(args.output_dir)
    make_straight_bridge_panel(args.output_dir)
    make_results(args.output_dir)


if __name__ == "__main__":
    main()
