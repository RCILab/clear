"""Numerical and task constants for the clean-room CLEAR implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class Protocol:
    workspace_size: float = 16.0
    body_radius: float = 0.20
    safety_margin: float = 0.02
    speed_limit: float = 0.80
    arrival_radius: float = 0.22
    dt: float = 0.03
    horizon: float = 60.0
    obstacle_fraction: float = 0.15
    circle_count: int = 14
    rectangle_count: int = 12
    swap_radius: float = 7.5

    @property
    def robot_clearance(self) -> float:
        return self.body_radius + self.safety_margin

    @property
    def pair_clearance(self) -> float:
        return 2.0 * self.robot_clearance

    @property
    def half_width(self) -> float:
        return 0.5 * self.workspace_size

    @property
    def steps(self) -> int:
        return int(round(self.horizon / self.dt))


@dataclass(frozen=True)
class ControllerConfig:
    # Goal field
    goal_gain: float = 1.2
    goal_slow_radius: float = 0.80

    # Conflict circulation
    pair_sense_radius: float = 2.2
    obstacle_sense_radius: float = 1.2
    circulation_gain: float = 0.72
    obstacle_circulation_gain: float = 0.62
    boundary_progress_aligned: bool = False
    cluster_escape_gain: float = 0.0
    cluster_pair_band: float = 0.08
    cluster_boundary_band: float = 0.08
    cluster_escape_hysteresis: bool = False
    terminal_capture_hysteresis: bool = False
    terminal_capture_radius: float = 0.22
    terminal_open_capture_radius: float = 0.60
    terminal_release_radius: float = 0.80
    certificate_margin_gain: float = 0.08
    adaptive_certificate_gain: bool = False
    handedness: int = 1
    closing_bias: float = 0.12

    # Active tangent bundle
    tangent_band: float = 0.35
    tangent_projection_max_sweeps: int = 1800

    # Safety projection
    cbf_rate: float = 4.0
    projection_tolerance: float = 1.0e-8
    projection_max_sweeps: int = 1800

    def __post_init__(self) -> None:
        if self.handedness not in (-1, 1):
            raise ValueError("handedness must be -1 or +1")
        if self.terminal_capture_radius < 0.0:
            raise ValueError("terminal_capture_radius must be nonnegative")
        if self.terminal_open_capture_radius < 0.0:
            raise ValueError(
                "terminal_open_capture_radius must be nonnegative"
            )
        if self.terminal_release_radius < max(
            self.terminal_capture_radius,
            self.terminal_open_capture_radius,
        ):
            raise ValueError(
                "terminal_release_radius must not be smaller than either "
                "capture radius"
            )


DEFAULT_PROTOCOL = Protocol()
DEFAULT_CONTROLLER = ControllerConfig()
REPORTED_CLEAR_CONTROLLER = ControllerConfig(
    boundary_progress_aligned=True,
    cluster_escape_gain=0.55,
    cluster_escape_hysteresis=True,
    terminal_capture_hysteresis=True,
    tangent_band=0.08,
)
