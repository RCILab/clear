"""Synchronous rollout and certificate-oriented metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import CLEARController
from .guidance import CostFieldGuidance, GridPlanner, GuidancePlan
from .scenarios import Scenario
from .theorem_audit import audit_straight_bridge_tokens

Array = np.ndarray


@dataclass(frozen=True)
class Rollout:
    scenario: Scenario
    times: Array
    trajectory: Array
    first_arrival_times: Array
    final_goal_distances: Array
    minimum_pair_distance: float
    minimum_obstacle_clearance: float
    minimum_cbf_residual: float
    minimum_tangent_residual: float
    maximum_speed: float
    nonconverged_steps: int
    tangent_fallback_steps: int
    cluster_escape_steps: int
    cluster_escape_component_steps: int
    positive_tangent_margin_fraction: float
    positive_global_tangent_margin_fraction: float
    global_tangent_audited_steps: int
    static_certificate_steps: int
    static_certificate_violations: int
    minimum_static_certified_speed: float
    straight_bridge_events: int
    straight_bridge_domain_steps: int
    straight_bridge_bound_violations: int
    straight_bridge_token_flips: int
    maximum_bridge_projection_axis_error: float

    @property
    def robot_arrival_rate(self) -> float:
        return float(np.mean(self.final_goal_distances <= self.scenario.protocol.arrival_radius))

    @property
    def mission_success(self) -> bool:
        safe_pair = self.minimum_pair_distance >= self.scenario.protocol.pair_clearance - 1.0e-6
        safe_obstacle = self.minimum_obstacle_clearance >= -1.0e-6
        return bool(
            self.robot_arrival_rate == 1.0
            and safe_pair
            and safe_obstacle
            and self.nonconverged_steps == 0
        )

    @property
    def makespan(self) -> float:
        if np.any(~np.isfinite(self.first_arrival_times)):
            return float("nan")
        return float(np.max(self.first_arrival_times))

    def summary(self) -> dict:
        return {
            "family": self.scenario.family,
            "n_robots": self.scenario.n_robots,
            "seed": self.scenario.seed,
            "fingerprint": self.scenario.fingerprint(),
            "mission_success": self.mission_success,
            "robot_arrival_rate": self.robot_arrival_rate,
            "makespan_s": self.makespan,
            "minimum_pair_distance_m": self.minimum_pair_distance,
            "minimum_obstacle_clearance_m": self.minimum_obstacle_clearance,
            "minimum_cbf_residual": self.minimum_cbf_residual,
            "minimum_tangent_residual": self.minimum_tangent_residual,
            "maximum_speed_mps": self.maximum_speed,
            "nonconverged_steps": self.nonconverged_steps,
            "tangent_fallback_steps": self.tangent_fallback_steps,
            "cluster_escape_steps": self.cluster_escape_steps,
            "cluster_escape_component_steps": self.cluster_escape_component_steps,
            "positive_tangent_margin_fraction": self.positive_tangent_margin_fraction,
            "positive_global_tangent_margin_fraction": (
                self.positive_global_tangent_margin_fraction
            ),
            "global_tangent_audited_steps": (
                self.global_tangent_audited_steps
            ),
            "static_certificate_steps": self.static_certificate_steps,
            "static_certificate_violations": self.static_certificate_violations,
            "minimum_static_certified_speed_mps": (
                self.minimum_static_certified_speed
            ),
            "straight_bridge_events": self.straight_bridge_events,
            "straight_bridge_domain_steps": (
                self.straight_bridge_domain_steps
            ),
            "straight_bridge_bound_violations": (
                self.straight_bridge_bound_violations
            ),
            "straight_bridge_token_flips": self.straight_bridge_token_flips,
            "maximum_bridge_projection_axis_error": (
                self.maximum_bridge_projection_axis_error
            ),
        }


def _minimum_pair_distance(positions: Array) -> float:
    if len(positions) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(positions)):
        distances = np.linalg.norm(positions[i + 1 :] - positions[i], axis=1)
        if len(distances):
            best = min(best, float(np.min(distances)))
    return best


def _swept_pair_distance(start: Array, velocity: Array, dt: float) -> float:
    best = float("inf")
    for i in range(len(start)):
        for j in range(i + 1, len(start)):
            relative_position = start[i] - start[j]
            relative_velocity = velocity[i] - velocity[j]
            speed_sq = float(relative_velocity @ relative_velocity)
            if speed_sq <= 1.0e-18:
                tau = 0.0
            else:
                tau = float(
                    np.clip(
                        -(relative_position @ relative_velocity) / speed_sq,
                        0.0,
                        dt,
                    )
                )
            best = min(
                best,
                float(np.linalg.norm(relative_position + tau * relative_velocity)),
            )
    return best


def _minimum_obstacle_clearance(scenario: Scenario, positions: Array) -> float:
    return min(
        scenario.arena.minimum_clearance(point, scenario.protocol.robot_clearance)
        for point in positions
    )


def simulate(
    scenario: Scenario,
    controller: CLEARController | None = None,
    *,
    record_stride: int = 4,
    guidance_mode: str = "direct",
) -> Rollout:
    if controller is None:
        controller = CLEARController(scenario.protocol)
    controller.reset()
    p = scenario.protocol
    positions = np.asarray(scenario.starts, dtype=float).copy()
    if guidance_mode not in ("direct", "waypoint", "cost"):
        raise ValueError("guidance_mode must be direct, waypoint, or cost")
    guidance: GuidancePlan | CostFieldGuidance | None = None
    if guidance_mode != "direct" and scenario.arena.obstacles:
        planner = GridPlanner(scenario.arena, p)
        guidance = (
            planner.plan(scenario.starts, scenario.goals)
            if guidance_mode == "waypoint"
            else planner.cost_field_plan(scenario.goals)
        )
    first_arrival = np.full(scenario.n_robots, np.inf)

    recorded_times = [0.0]
    recorded_positions = [positions.copy()]
    min_pair = _minimum_pair_distance(positions)
    min_obstacle = _minimum_obstacle_clearance(scenario, positions)
    min_cbf = float("inf")
    min_tangent = float("inf")
    max_speed = 0.0
    nonconverged = 0
    tangent_fallbacks = 0
    cluster_escape_steps = 0
    cluster_escape_component_steps = 0
    tangent_positive = 0
    tangent_audited = 0
    global_tangent_positive = 0
    global_tangent_audited = 0
    static_certificate_steps = 0
    static_certificate_violations = 0
    minimum_static_certified_speed = float("inf")
    active_bridge_tokens: dict[tuple[int, ...], Array] = {}
    straight_bridge_events = 0
    straight_bridge_domain_steps = 0
    straight_bridge_bound_violations = 0
    straight_bridge_token_flips = 0
    maximum_bridge_projection_axis_error = 0.0

    initial_distance = np.linalg.norm(positions - scenario.goals, axis=1)
    first_arrival[initial_distance <= p.arrival_radius] = 0.0

    for step in range(p.steps):
        if isinstance(guidance, GuidancePlan):
            control_goals = guidance.targets(
                positions,
                scenario.arena,
                p.robot_clearance + 0.01,
            )
        elif isinstance(guidance, CostFieldGuidance):
            control_goals = guidance.targets(positions)
        else:
            control_goals = scenario.goals
        tokens_before = controller.cluster_tokens
        audit = controller.command(positions, control_goals, scenario.arena)
        tokens_after = controller.cluster_tokens
        bridge_certificates = audit_straight_bridge_tokens(
            controller,
            positions,
            scenario.arena,
            audit,
        )
        valid_bridge_steps = []
        for certificate in bridge_certificates:
            key = certificate.component
            if certificate.domain_satisfied and key not in tokens_before:
                active_bridge_tokens[key] = certificate.direction.copy()
                straight_bridge_events += 1
            if key in active_bridge_tokens and certificate.domain_satisfied:
                previous = active_bridge_tokens[key]
                if float(previous @ certificate.direction) < 1.0 - 1.0e-8:
                    straight_bridge_token_flips += 1
                active_bridge_tokens[key] = certificate.direction.copy()
                valid_bridge_steps.append(certificate)
                maximum_bridge_projection_axis_error = max(
                    maximum_bridge_projection_axis_error,
                    certificate.projection_axis_error,
                )
            elif key in active_bridge_tokens:
                del active_bridge_tokens[key]
        for key in list(active_bridge_tokens):
            if key not in tokens_after:
                del active_bridge_tokens[key]
        velocity = audit.executed_command
        min_cbf = min(min_cbf, audit.cbf_minimum_residual)
        min_tangent = min(min_tangent, audit.tangent_minimum_residual)
        max_speed = max(max_speed, float(np.max(np.linalg.norm(velocity, axis=1))))
        nonconverged += int(not audit.cbf_converged)
        tangent_fallbacks += int(not audit.tangent_converged)
        cluster_escape_steps += int(audit.active_cluster_escapes > 0)
        cluster_escape_component_steps += audit.active_cluster_escapes
        if audit.tangent_constraints:
            tangent_audited += 1
            tangent_positive += int(audit.tangent_margin > 0.0)
        if (
            audit.tangent_norm > 1.0e-10
            and audit.tangent_converged
            and audit.cbf_converged
        ):
            global_tangent_audited += 1
            if audit.global_tangent_margin > 1.0e-9:
                global_tangent_positive += 1
                static_certificate_steps += 1
                executed_norm = float(np.linalg.norm(velocity))
                minimum_static_certified_speed = min(
                    minimum_static_certified_speed,
                    executed_norm,
                )
                static_certificate_violations += int(
                    executed_norm <= 1.0e-10
                )

        min_pair = min(min_pair, _swept_pair_distance(positions, velocity, p.dt))
        # Three samples audit obstacle clearance over the zero-order-hold step.
        min_obstacle = min(
            min_obstacle,
            *(
                _minimum_obstacle_clearance(
                    scenario, positions + fraction * p.dt * velocity
                )
                for fraction in (0.5, 1.0)
            ),
        )
        next_positions = positions + p.dt * velocity
        for certificate in valid_bridge_steps:
            indices = np.asarray(certificate.component, dtype=int)
            progress = float(
                np.mean(
                    (next_positions[indices] - positions[indices])
                    @ certificate.direction
                )
            )
            straight_bridge_domain_steps += 1
            straight_bridge_bound_violations += int(
                progress + 1.0e-10
                < p.dt * certificate.progress_rate_bound
            )
        positions = next_positions
        time = (step + 1) * p.dt

        distance = np.linalg.norm(positions - scenario.goals, axis=1)
        new_arrivals = np.isinf(first_arrival) & (distance <= p.arrival_radius)
        first_arrival[new_arrivals] = time
        if (step + 1) % record_stride == 0 or step + 1 == p.steps:
            recorded_times.append(time)
            recorded_positions.append(positions.copy())

    final_distance = np.linalg.norm(positions - scenario.goals, axis=1)
    positive_fraction = (
        tangent_positive / tangent_audited if tangent_audited else float("nan")
    )
    positive_global_fraction = (
        global_tangent_positive / global_tangent_audited
        if global_tangent_audited
        else float("nan")
    )
    if not np.isfinite(minimum_static_certified_speed):
        minimum_static_certified_speed = float("nan")
    return Rollout(
        scenario=scenario,
        times=np.asarray(recorded_times),
        trajectory=np.asarray(recorded_positions),
        first_arrival_times=first_arrival,
        final_goal_distances=final_distance,
        minimum_pair_distance=min_pair,
        minimum_obstacle_clearance=min_obstacle,
        minimum_cbf_residual=min_cbf,
        minimum_tangent_residual=min_tangent,
        maximum_speed=max_speed,
        nonconverged_steps=nonconverged,
        tangent_fallback_steps=tangent_fallbacks,
        cluster_escape_steps=cluster_escape_steps,
        cluster_escape_component_steps=cluster_escape_component_steps,
        positive_tangent_margin_fraction=positive_fraction,
        positive_global_tangent_margin_fraction=positive_global_fraction,
        global_tangent_audited_steps=global_tangent_audited,
        static_certificate_steps=static_certificate_steps,
        static_certificate_violations=static_certificate_violations,
        minimum_static_certified_speed=minimum_static_certified_speed,
        straight_bridge_events=straight_bridge_events,
        straight_bridge_domain_steps=straight_bridge_domain_steps,
        straight_bridge_bound_violations=straight_bridge_bound_violations,
        straight_bridge_token_flips=straight_bridge_token_flips,
        maximum_bridge_projection_axis_error=(
            maximum_bridge_projection_axis_error
        ),
    )
