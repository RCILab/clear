"""Interactive visualization server for CLEAR unicycle experiments.

Runs the actual ``clear_nav`` controller under the canonical evaluation
protocol (30 ms passage updates and three native projection substeps) and
serves a browser viewer that replays the resulting trajectory in real time
for teams of up to 60 robots.

The evaluation-only dense held-arc clearance audit is kept off the live
execution path.  Display frames independently track sampled physical pair
distance; neither change alters the controller or exact held-input dynamics,
so the replayed motion is identical to the paper pipeline.
This tool is for inspection only -- reported numbers still come from
``run_unicycle.py``.

Usage:
    python visualization/server.py [--port 8765]
    (or inside Docker: docker run --rm -p 8765:8765 -v "${PWD}:/work" \
        -w /work clear-nav python visualization/server.py)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import traceback
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import numpy as np  # noqa: E402

from clear_nav import Protocol, Scenario  # noqa: E402
from clear_nav.geometry import Circle, Rectangle  # noqa: E402
from clear_nav.smg_metrics import (  # noqa: E402
    doorway_flow_from_trajectory,
    interference_delay_metrics,
    intersection_flow_from_trajectory,
)
from clear_nav.smg_scenarios import SMGGeometry, make_smg_scenario  # noqa: E402
from clear_nav.unicycle import UnicycleConfig, simulate_unicycle  # noqa: E402
from run_unicycle import (  # noqa: E402
    controller_config,
    make_unicycle_scenario,
)

FAMILIES = ("free", "swap", "circ15", "rect15")
SMG_FAMILIES = ("doorway", "intersection")
# Paper protocol: 1.2 m wide, 0.8 m thick doorway; 2.4 m intersection corridor.
SMG_GEOMETRY = SMGGeometry(doorway_width=1.2, doorway_thickness=0.8)
MAX_ROBOTS = 60
RECORD_STRIDE = 1  # one display frame per 30 ms passage update
POST_ARRIVAL_TAIL = 2.0  # seconds simulated past team completion
_SIM_LOCK = threading.Lock()


class _StopStream(Exception):
    """Raised inside the record observer to end a live run early."""


def _canonical_unicycle() -> UnicycleConfig:
    """Mirror the run_unicycle.py argparse defaults (canonical protocol).

    The three native substeps match the reported bounded-unicycle interface.
    Evaluation-only dense clearance auditing remains outside this live path.
    """
    return UnicycleConfig(
        lookahead=0.05,
        yaw_rate_limit=np.pi / 2.0,
        inner_substeps=3,
        projection_backend="osqp",
        projection_tolerance=1.0e-6,
        projection_max_sweeps=4000,
        projection_extension_sweeps=0,
        minimum_restoration_retention=0.25,
        hierarchical_progress=False,
        hqp_progress_retention=0.995,
        hqp_regularization=1.0e-6,
        certified_bridge_progress=True,
        bridge_progress_retention=0.90,
        actuation_mode="native-cbf",
    )


def _canonical_controller_args() -> Namespace:
    return Namespace(
        variant="clear",
        handedness=1,
        boundary_mode="progress",
        terminal_capture_radius=0.22,
        terminal_open_capture_radius=0.60,
        terminal_release_radius=0.80,
        cbf_rate=4.0,
    )


def _serialize_obstacles(obstacles) -> list[dict]:
    serialized = []
    for obstacle in obstacles:
        if isinstance(obstacle, Circle):
            serialized.append(
                {
                    "type": "circle",
                    "center": [float(obstacle.center[0]), float(obstacle.center[1])],
                    "radius": float(obstacle.radius),
                }
            )
        elif isinstance(obstacle, Rectangle):
            serialized.append(
                {
                    "type": "rect",
                    "center": [float(obstacle.center[0]), float(obstacle.center[1])],
                    "size": [float(obstacle.size[0]), float(obstacle.size[1])],
                }
            )
    return serialized


def generate_run(family: str, n_robots: int, seed: int) -> dict:
    """Compute one complete run as a single JSON-ready dict.

    Delegates to :func:`stream_run` so the batch path shares the streaming
    path's early termination (stop a little past team completion) and frame
    format exactly.
    """
    meta: dict = {}
    times: list = []
    traj: list = []
    headings: list = []
    summary: dict = {}
    arrivals: list = []

    def emit(obj: dict) -> None:
        nonlocal arrivals
        if obj["type"] == "meta":
            meta.update(obj)
        elif obj["type"] == "frame":
            times.append(obj["t"])
            traj.append(obj["p"])
            headings.append(obj["h"])
            arrivals = obj["arr"]
        elif obj["type"] == "done":
            summary.update(obj)

    stream_run(family, n_robots, seed, emit)
    return {
        "family": family,
        "n": n_robots,
        "seed": seed,
        "smg": bool(meta.get("smg", False)),
        "workspace": meta["workspace"],
        "body_radius": meta["body_radius"],
        "arrival_radius": meta["arrival_radius"],
        "dt": meta["dt"],
        "horizon": meta["horizon"],
        "obstacles": meta["obstacles"],
        "goals": meta["goals"],
        "times": times,
        "traj": traj,
        "headings": headings,
        "arrivals": arrivals,
        "completed": summary["completed"],
        "makespan": summary["makespan"],
        "min_pair": summary["min_pair"],
        "flow": summary.get("flow"),
        "delay": summary.get("delay"),
    }


def _min_pair_distance(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return float("inf")
    deltas = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    upper = distances[np.triu_indices(len(centers), k=1)]
    return float(np.min(upper))


def _solo_arrival_time(
    scenario: Scenario,
    robot_index: int,
    config,
    unicycle: UnicycleConfig,
) -> float:
    """Time-to-goal of one robot navigating the same map alone."""
    solo = Scenario(
        family=f"{scenario.family}_solo",
        n_robots=1,
        seed=scenario.seed,
        starts=np.asarray(scenario.starts)[robot_index : robot_index + 1].copy(),
        goals=np.asarray(scenario.goals)[robot_index : robot_index + 1].copy(),
        arena=scenario.arena,
        protocol=scenario.protocol,
    )
    state = {"arrival": float("inf")}

    def on_record(t, centers, headings, first_arrival) -> None:
        if math.isfinite(first_arrival[0]):
            state["arrival"] = float(first_arrival[0])
            raise _StopStream

    try:
        simulate_unicycle(
            solo,
            config,
            unicycle,
            initial_headings=np.zeros(1),
            record_stride=RECORD_STRIDE,
            guidance_mode="cost",
            on_record=on_record,
        )
    except _StopStream:
        pass
    return state["arrival"]


def stream_run(family: str, n_robots: int, seed: int, emit) -> None:
    """Run one mission live, emitting NDJSON-ready dicts per recorded tick.

    ``emit`` is called with a metadata dict, then one frame dict per recorded
    control tick as soon as it is computed, then a final ``done`` dict.  If
    ``emit`` raises (client disconnected), the simulation stops immediately.
    For the shared-resource families (doorway, intersection) the done dict
    additionally reports the trial's normalized flow and, after per-robot
    isolated reference runs, the pooled interference delay.
    """
    is_smg = family in SMG_FAMILIES
    if not is_smg and family not in FAMILIES:
        raise ValueError(
            f"family must be one of {FAMILIES + SMG_FAMILIES}"
        )
    if not 2 <= n_robots <= MAX_ROBOTS:
        raise ValueError(f"n must be between 2 and {MAX_ROBOTS}")

    physical = Protocol(horizon=60.0, dt=0.03)
    unicycle = _canonical_unicycle()
    config = controller_config(_canonical_controller_args())
    if is_smg:
        # Mirrors benchmark_smg.py: sites are physical centers, no shift.
        scenario = make_smg_scenario(
            family, n_robots, seed, physical, SMG_GEOMETRY
        )
    else:
        scenario = make_unicycle_scenario(
            family, n_robots, seed, physical, unicycle
        )
    emit(
        {
            "type": "meta",
            "family": family,
            "n": n_robots,
            "seed": seed,
            "smg": is_smg,
            "workspace": float(scenario.protocol.workspace_size),
            "body_radius": float(scenario.protocol.body_radius),
            "arrival_radius": float(scenario.protocol.arrival_radius),
            "dt": float(physical.dt) * RECORD_STRIDE,
            "horizon": float(physical.horizon),
            "obstacles": _serialize_obstacles(scenario.arena.obstacles),
            "goals": np.round(np.asarray(scenario.goals, float), 4).tolist(),
        }
    )

    state = {
        "min_pair": float("inf"),
        "stop_at": None,
        "last_t": 0.0,
        "times": [],
        "centers": [],
        "arrivals": None,
    }

    def on_record(t, centers, headings, first_arrival) -> None:
        state["min_pair"] = min(state["min_pair"], _min_pair_distance(centers))
        state["last_t"] = float(t)
        state["times"].append(float(t))
        state["centers"].append(np.array(centers, dtype=float))
        state["arrivals"] = np.array(first_arrival, dtype=float)
        arrivals = [
            (float(a) if math.isfinite(a) else None) for a in first_arrival
        ]
        emit(
            {
                "type": "frame",
                "t": round(float(t), 4),
                "p": np.round(centers, 4).tolist(),
                "h": np.round(headings, 4).tolist(),
                "arr": arrivals,
            }
        )
        if state["stop_at"] is None and all(
            a is not None for a in arrivals
        ):
            state["stop_at"] = max(arrivals) + POST_ARRIVAL_TAIL
        if state["stop_at"] is not None and t >= state["stop_at"]:
            raise _StopStream

    try:
        simulate_unicycle(
            scenario,
            config,
            unicycle,
            initial_headings=np.zeros(n_robots),
            record_stride=RECORD_STRIDE,
            guidance_mode="cost",
            on_record=on_record,
        )
    except _StopStream:
        pass
    team_arrivals = (
        state["arrivals"]
        if state["arrivals"] is not None
        else np.full(n_robots, np.inf)
    )
    completed = bool(np.all(np.isfinite(team_arrivals)))
    makespan = float(np.max(team_arrivals)) if completed else None

    flow = None
    delay = None
    if is_smg and state["times"]:
        times = np.asarray(state["times"], dtype=float)
        trajectory = np.stack(state["centers"], axis=0)
        if family == "doorway":
            flow_record = doorway_flow_from_trajectory(
                times,
                trajectory,
                np.asarray(scenario.starts, dtype=float),
                np.asarray(scenario.goals, dtype=float),
                body_radius=scenario.protocol.body_radius,
                dt=float(physical.dt) * RECORD_STRIDE,
                doorway_width=SMG_GEOMETRY.doorway_width,
                doorway_thickness=SMG_GEOMETRY.doorway_thickness,
            )
        else:
            flow_record = intersection_flow_from_trajectory(
                times,
                trajectory,
                np.asarray(scenario.starts, dtype=float),
                np.asarray(scenario.goals, dtype=float),
                body_radius=scenario.protocol.body_radius,
                dt=float(physical.dt) * RECORD_STRIDE,
                corridor_width=SMG_GEOMETRY.intersection_corridor_width,
            )
        flow = float(flow_record["smg_flow_rate_robots_per_m_s"])

        solo_times = np.full(n_robots, np.inf)
        for index in range(n_robots):
            emit(
                {
                    "type": "phase",
                    "msg": (
                        "computing isolated reference runs "
                        f"({index + 1}/{n_robots}) for the delay metric"
                    ),
                }
            )
            solo_times[index] = _solo_arrival_time(
                scenario, index, config, unicycle
            )
        delay_record = interference_delay_metrics(team_arrivals, solo_times)
        delay = delay_record["average_interference_delay_s"]

    emit(
        {
            "type": "done",
            "completed": completed,
            "makespan": makespan,
            "min_pair": state["min_pair"],
            "flow": flow,
            "delay": delay,
        }
    )


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/stream":
            params = parse_qs(parsed.query)
            try:
                family = params.get("family", ["free"])[0]
                n_robots = int(params.get("n", ["20"])[0])
                seed = int(params.get("seed", ["0"])[0])
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def emit(obj: dict) -> None:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()

            try:
                with _SIM_LOCK:
                    stream_run(family, n_robots, seed, emit)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ):
                pass  # client aborted; the simulation stopped with it
            except ValueError as error:
                try:
                    emit({"type": "error", "error": str(error)})
                except OSError:
                    pass
            except Exception:  # noqa: BLE001
                try:
                    emit(
                        {
                            "type": "error",
                            "error": traceback.format_exc(limit=6),
                        }
                    )
                except OSError:
                    pass
            return
        if parsed.path == "/api/generate":
            params = parse_qs(parsed.query)
            try:
                family = params.get("family", ["free"])[0]
                n_robots = int(params.get("n", ["20"])[0])
                seed = int(params.get("seed", ["0"])[0])
                with _SIM_LOCK:
                    payload = generate_run(family, n_robots, seed)
                self._send_json(payload)
            except ValueError as error:
                self._send_json({"error": str(error)}, status=400)
            except Exception:  # noqa: BLE001 (surface to the browser)
                self._send_json(
                    {"error": traceback.format_exc(limit=6)}, status=500
                )
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("[server] " + fmt % args + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"CLEAR visualization: http://localhost:{args.port}")
    print("(live simulation supports N=20, 40, and 60)")
    server.serve_forever()


if __name__ == "__main__":
    main()
