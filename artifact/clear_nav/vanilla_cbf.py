"""Vanilla goal-tracking CBF-QP baseline.

This baseline deliberately contains no circulation, component token, or
deadlock-resolution term.  It shares CLEAR's static guidance, geometric CBF
rows, bounded native-unicycle projection, and numerical tolerances so that the
comparison isolates the liveness mechanism rather than implementation details.
"""

from __future__ import annotations

import numpy as np

from .controller import CLEARController
from .geometry import Arena

Array = np.ndarray


class VanillaCBFController(CLEARController):
    """Goal field followed by the common CBF-QP safety filter."""

    def circulation_field(
        self,
        positions: Array,
        goals: Array,
        arena: Arena,
        goal_command: Array | None = None,
    ) -> tuple[Array, int, int]:
        del goals, goal_command
        positions = np.asarray(positions, dtype=float)
        return np.zeros_like(positions), 0, 0

    def cluster_escape_field(
        self,
        positions: Array,
        goal_command: Array,
        arena: Arena,
        tangent_matrix: Array,
    ) -> tuple[Array, int]:
        del goal_command, arena, tangent_matrix
        return np.zeros_like(np.asarray(positions, dtype=float)), 0

