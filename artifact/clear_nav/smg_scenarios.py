"""Deterministic Social Mini-Game geometry for common-controller evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from .config import DEFAULT_PROTOCOL, Protocol
from .geometry import Arena, Rectangle
from .scenarios import Scenario

Array = np.ndarray
SMG_FAMILIES = ("doorway", "intersection")


@dataclass(frozen=True)
class SMGGeometry:
    """Physical geometry parameters, expressed in metres."""

    doorway_width: float = 0.80
    doorway_thickness: float = 0.80
    intersection_corridor_width: float = 2.40
    staging_distance: float = 3.20
    queue_spacing: float = 0.60
    staging_row_spacing: float = 0.56
    reported_cost_field_passage_floor: float = 0.59
    reported_baseline_local_model_width: float = 0.74

    def __post_init__(self) -> None:
        positive = (
            self.doorway_width,
            self.doorway_thickness,
            self.intersection_corridor_width,
            self.staging_distance,
            self.queue_spacing,
            self.staging_row_spacing,
        )
        if min(positive) <= 0.0:
            raise ValueError("all SMG geometry dimensions must be positive")

    def metadata(
        self,
        protocol: Protocol,
        *,
        lookahead: float,
    ) -> dict:
        virtual_diameter = 2.0 * (
            protocol.robot_clearance + lookahead
        )
        return {
            **asdict(self),
            "robot_diameter_m": 2.0 * protocol.body_radius,
            "pair_clearance_m": protocol.pair_clearance,
            "lookahead_virtual_diameter_m": virtual_diameter,
            "doorway_nominal_safety_lanes": int(
                self.doorway_width // protocol.pair_clearance
            ),
            "doorway_nominal_virtual_lanes": int(
                self.doorway_width // virtual_diameter
            ),
            "doorway_cost_field_margin_m": (
                self.doorway_width
                - self.reported_cost_field_passage_floor
            ),
            "baseline_local_model_margin_m": (
                self.doorway_width
                - self.reported_baseline_local_model_width
            ),
            "fairness_note_required": bool(
                self.doorway_width
                <= self.reported_baseline_local_model_width + 0.10
            ),
        }


def _doorway_obstacles(
    protocol: Protocol,
    geometry: SMGGeometry,
) -> tuple[Rectangle, Rectangle]:
    half = protocol.half_width
    gap_half = 0.5 * geometry.doorway_width
    wall_height = half - gap_half
    center_offset = gap_half + 0.5 * wall_height
    size = np.array([geometry.doorway_thickness, wall_height])
    return (
        Rectangle(np.array([0.0, center_offset]), size.copy()),
        Rectangle(np.array([0.0, -center_offset]), size.copy()),
    )


def _doorway_sites(
    count: int,
    protocol: Protocol,
    geometry: SMGGeometry,
    seed: int = 0,
) -> tuple[Array, Array]:
    if count % 2:
        raise ValueError("doorway requires an even number of robots")
    per_side = count // 2
    # The starting queue reflects the usable bottleneck capacity: 0.8 m is a
    # single-file task under the common safety envelope, while 1.2 m admits
    # two tightly spaced lanes.  This prevents a geometry artefact in which
    # agents are initialized on approach lines that cannot enter the door.
    safe_lane_pitch = max(
        geometry.staging_row_spacing,
        protocol.pair_clearance + 0.10,
    )
    rows = min(
        per_side,
        max(1, int(geometry.doorway_width // safe_lane_pitch)),
    )
    columns = int(ceil(per_side / rows))
    y_values = (
        np.arange(rows, dtype=float) - 0.5 * (rows - 1)
    ) * safe_lane_pitch
    left: list[Array] = []
    for column in range(columns):
        x = -geometry.staging_distance - column * geometry.queue_spacing
        for y in y_values:
            if len(left) == per_side:
                break
            left.append(np.array([x, y]))
    left_array = np.asarray(left)
    if seed:
        rng = np.random.default_rng(seed)
        column = np.rint(
            (-left_array[:, 0] - geometry.staging_distance)
            / geometry.queue_spacing
        ).astype(int)
        column_count = int(np.max(column)) + 1
        gaps = geometry.queue_spacing + rng.uniform(
            -0.02, 0.04, size=max(column_count - 1, 0)
        )
        offsets = np.concatenate(([0.0], np.cumsum(gaps)))
        staging_jitter = rng.uniform(-0.08, 0.08)
        left_array[:, 0] = (
            -geometry.staging_distance
            - staging_jitter
            - offsets[column]
        )
        # A common lateral bias and non-contracting pitch variation preserve
        # the initial pair margin while changing approach asymmetry.
        lateral_bias = rng.uniform(-0.04, 0.04)
        lateral_scale = rng.uniform(1.0, 1.07)
        left_array[:, 1] = (
            lateral_scale * left_array[:, 1] + lateral_bias
        )
    right_array = left_array.copy()
    right_array[:, 0] *= -1.0
    starts = np.concatenate((left_array, right_array), axis=0)
    # Goals lie in a post-resource dispersal region.  Parking agents at the
    # mirrored approach queue would make the leading arrival block every
    # follower and would measure exit parking rather than doorway contention.
    goal_rows = min(4, per_side)
    goal_columns = int(ceil(per_side / goal_rows))
    goal_y = (
        np.arange(goal_rows, dtype=float) - 0.5 * (goal_rows - 1)
    ) * 1.20
    right_goals: list[Array] = []
    for column in range(goal_columns):
        x = 6.80 - column * geometry.queue_spacing
        for y in goal_y:
            if len(right_goals) == per_side:
                break
            right_goals.append(np.array([x, y]))
    right_goal_array = np.asarray(right_goals)
    left_goal_array = -right_goal_array
    if seed:
        permutation = rng.permutation(per_side)
        right_goal_array = right_goal_array[permutation]
        left_goal_array = left_goal_array[permutation]
    goals = np.concatenate(
        (right_goal_array, left_goal_array),
        axis=0,
    )
    if np.max(np.abs(starts)) >= (
        protocol.half_width - protocol.robot_clearance
    ):
        raise ValueError("doorway staging sites do not fit in the workspace")
    return starts, goals


def _intersection_obstacles(
    protocol: Protocol,
    geometry: SMGGeometry,
) -> tuple[Rectangle, ...]:
    half = protocol.half_width
    corridor_half = 0.5 * geometry.intersection_corridor_width
    side = half - corridor_half
    offset = corridor_half + 0.5 * side
    size = np.array([side, side])
    return tuple(
        Rectangle(np.array([sx * offset, sy * offset]), size.copy())
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    )


def _intersection_sites(
    count: int,
    protocol: Protocol,
    geometry: SMGGeometry,
    seed: int = 0,
) -> tuple[Array, Array]:
    if count % 4:
        raise ValueError("intersection requires a multiple of four robots")
    per_arm = count // 4
    lateral = min(
        0.20,
        0.20 * geometry.intersection_corridor_width,
    )
    starts: list[Array] = []
    goals: list[Array] = []
    rng = np.random.default_rng(seed) if seed else None
    for arm in range(4):
        if rng is None:
            staging_jitter = 0.0
            queue_gaps = np.full(max(per_arm - 1, 0), geometry.queue_spacing)
            arm_lateral = lateral
        else:
            staging_jitter = rng.uniform(-0.08, 0.08)
            queue_gaps = geometry.queue_spacing + rng.uniform(
                -0.02, 0.04, size=max(per_arm - 1, 0)
            )
            arm_lateral = lateral * rng.uniform(0.85, 1.10)
        queue_offsets = np.concatenate(([0.0], np.cumsum(queue_gaps)))
        for queue_index in range(per_arm):
            start_distance = (
                geometry.staging_distance
                + staging_jitter
                + queue_offsets[queue_index]
            )
            # Preserve queue order after the shared resource: the front robot
            # receives the farthest goal so that arrivals do not park in front
            # of robots that are still exiting.
            goal_distance = 6.80 - queue_index * geometry.queue_spacing
            offset = arm_lateral * (-1.0 if queue_index % 2 else 1.0)
            if arm == 0:  # west -> east
                start = np.array([-start_distance, offset])
                goal = np.array([goal_distance, offset])
            elif arm == 1:  # east -> west
                start = np.array([start_distance, -offset])
                goal = np.array([-goal_distance, -offset])
            elif arm == 2:  # south -> north
                start = np.array([-offset, -start_distance])
                goal = np.array([-offset, goal_distance])
            else:  # north -> south
                start = np.array([offset, start_distance])
                goal = np.array([offset, -goal_distance])
            starts.append(start)
            goals.append(goal)
    result_starts = np.asarray(starts)
    if np.max(np.abs(result_starts)) >= (
        protocol.half_width - protocol.robot_clearance
    ):
        raise ValueError(
            "intersection staging sites do not fit in the workspace"
        )
    return result_starts, np.asarray(goals)


def make_smg_scenario(
    family: str,
    n_robots: int,
    seed: int,
    protocol: Protocol = DEFAULT_PROTOCOL,
    geometry: SMGGeometry = SMGGeometry(),
) -> Scenario:
    """Build a paired, deterministic doorway or four-arm intersection."""
    if family not in SMG_FAMILIES:
        raise ValueError(f"family must be one of {SMG_FAMILIES}")
    if n_robots < 1:
        raise ValueError("n_robots must be positive")
    # Seeds are retained in fingerprints and result pairing.  The canonical
    # layouts themselves are deterministic, matching the SMG protocol.
    if family == "doorway":
        starts, goals = _doorway_sites(
            n_robots, protocol, geometry, seed
        )
        obstacles = _doorway_obstacles(protocol, geometry)
        label = f"doorway_w{geometry.doorway_width:.2f}"
    else:
        starts, goals = _intersection_sites(
            n_robots,
            protocol,
            geometry,
            seed,
        )
        obstacles = _intersection_obstacles(protocol, geometry)
        label = (
            "intersection_w"
            f"{geometry.intersection_corridor_width:.2f}"
        )
    return Scenario(
        family=label,
        n_robots=n_robots,
        seed=seed,
        starts=starts,
        goals=goals,
        arena=Arena(protocol.half_width, obstacles),
        protocol=protocol,
    )
