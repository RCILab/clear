"""Runtime audits for the controller-specific CLEAR theorems.

The functions in this module do not alter the controller.  They reconstruct
the antecedents and one-step conclusions of the state-wise static-equilibrium
certificate and the straight two-sided bridge theorem from one command audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .controller import (
    CLEARController,
    ControllerAudit,
    _smooth_gate_batch,
)
from .geometry import Arena, rotate90

Array = np.ndarray


@dataclass(frozen=True)
class BridgeStepCertificate:
    component: tuple[int, ...]
    direction: Array
    domain_satisfied: bool
    failed_conditions: tuple[str, ...]
    guide_progress_floor: float
    pair_gate: float
    boundary_gate: float
    progress_rate_bound: float
    projection_axis_error: float


def static_certificate_holds(
    audit: ControllerAudit,
    *,
    margin_tolerance: float = 1.0e-9,
    velocity_tolerance: float = 1.0e-10,
) -> tuple[bool, bool]:
    """Return ``(antecedent, conclusion)`` for the no-static-deadlock theorem."""
    antecedent = bool(
        audit.tangent_solver_converged
        and audit.cbf_solver_converged
        and audit.tangent_norm > velocity_tolerance
        and audit.global_tangent_margin > margin_tolerance
    )
    conclusion = bool(
        np.linalg.norm(audit.executed_command) > velocity_tolerance
    )
    return antecedent, conclusion


def audit_straight_bridge_tokens(
    controller: CLEARController,
    positions: Array,
    arena: Arena,
    audit: ControllerAudit,
    *,
    geometry_tolerance: float = 1.0e-8,
) -> list[BridgeStepCertificate]:
    """Audit every held token against the straight-bridge theorem domain."""
    positions = np.asarray(positions, dtype=float)
    count = len(positions)
    (
        left,
        right,
        distance,
        _,
        boundary_h,
        boundary_n,
    ) = controller._geometry_data(positions, arena)
    pair_h = distance - controller.protocol.pair_clearance
    pair_sensed = distance < controller.config.pair_sense_radius
    pair_predictive = (
        pair_h <= 2.0 * controller.protocol.speed_limit
        / controller.config.cbf_rate
    )

    certificates: list[BridgeStepCertificate] = []
    for component, token in controller.cluster_tokens.items():
        indices = np.asarray(component, dtype=int)
        if len(indices) < 2:
            continue
        direction = np.asarray(token, dtype=float)
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= 1.0e-12:
            continue
        direction = direction / magnitude
        normal_axis = -rotate90(direction)
        in_component = np.zeros(count, dtype=bool)
        in_component[indices] = True

        failures: list[str] = []
        cross_pair = in_component[left] ^ in_component[right]
        if np.any(cross_pair & (pair_sensed | pair_predictive)):
            failures.append("cross_pair")

        component_h = boundary_h[indices]
        component_n = boundary_n[indices]
        relevant_boundary = (
            (component_h < controller.config.obstacle_sense_radius)
            | (
                component_h
                <= controller.protocol.speed_limit
                / controller.config.cbf_rate
            )
        )
        relevant_normals = component_n[relevant_boundary]
        structural_boundary = component_h < controller.config.tangent_band
        structural_normals = component_n[structural_boundary]
        if not len(relevant_normals):
            failures.append("no_boundary")
        else:
            axial = relevant_normals @ normal_axis
            transverse = relevant_normals @ direction
            if (
                np.max(np.abs(transverse)) > geometry_tolerance
                or np.max(np.abs(np.abs(axial) - 1.0)) > geometry_tolerance
            ):
                failures.append("nonparallel_boundary")
        if not len(structural_normals):
            failures.append("no_structural_boundary")
        else:
            structural_sign = structural_normals @ normal_axis
            if not (
                np.any(structural_sign > 1.0 - geometry_tolerance)
                and np.any(structural_sign < -1.0 + geometry_tolerance)
            ):
                failures.append("not_two_sided")

        guide = audit.goal_command[indices]
        guide_progress = guide @ direction
        guide_transverse = guide @ normal_axis
        guide_progress_floor = float(np.min(guide_progress))
        if guide_progress_floor <= geometry_tolerance:
            failures.append("nonpositive_guide")
        if float(np.max(np.abs(guide_transverse))) > geometry_tolerance:
            failures.append("nonaxial_guide")

        internal_pair = in_component[left] & in_component[right]
        pair_gate_values = _smooth_gate_batch(
            pair_h[internal_pair],
            controller.config.cluster_pair_band,
        )
        pair_gate = (
            float(np.max(pair_gate_values))
            if len(pair_gate_values)
            else 0.0
        )
        if pair_gate <= 0.0:
            failures.append("no_pair_core")

        boundary_gate_values = _smooth_gate_batch(
            component_h,
            controller.config.cluster_boundary_band,
        )
        boundary_gate = (
            float(np.max(boundary_gate_values))
            if boundary_gate_values.size
            else 0.0
        )
        if boundary_gate <= 0.0:
            failures.append("no_boundary_core")

        rigid_axis = np.zeros_like(positions)
        rigid_axis[indices] = direction
        projection_axis_error = abs(
            float(
                rigid_axis.reshape(-1)
                @ (
                    audit.projected_command.reshape(-1)
                    - audit.nominal_command.reshape(-1)
                )
            )
        ) / len(indices)
        bound_norm = (
            sqrt(count) * controller.protocol.speed_limit
            + sqrt(2.0)
            * controller.config.circulation_gain
            * audit.active_pair_edges
            + controller.config.obstacle_circulation_gain
            * audit.active_boundary_edges
            + sqrt(count) * controller.config.cluster_escape_gain
        )
        scale_floor = min(
            1.0,
            controller.protocol.speed_limit / max(bound_norm, 1.0e-12),
        )
        progress_rate_bound = scale_floor * (
            guide_progress_floor
            + controller.config.cluster_escape_gain
            * pair_gate
            * boundary_gate
        )

        certificates.append(
            BridgeStepCertificate(
                component=component,
                direction=direction,
                domain_satisfied=not failures,
                failed_conditions=tuple(failures),
                guide_progress_floor=guide_progress_floor,
                pair_gate=pair_gate,
                boundary_gate=boundary_gate,
                progress_rate_bound=progress_rate_bound,
                projection_axis_error=projection_axis_error,
            )
        )
    return certificates
