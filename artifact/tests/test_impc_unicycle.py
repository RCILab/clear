"""Numerical checks for the common-unicycle IMPC-DR adaptation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


METHOD = (
    Path(__file__).resolve().parents[2]
    / "baselines"
    / "SMGLib"
    / "src"
    / "methods"
    / "Social-IMPC-DR"
)
sys.path.insert(0, str(METHOD))

import SET  # noqa: E402
from others import get_obstacle_list  # noqa: E402
from unicycle_run import run_one_step  # noqa: E402
from avoid import GET_cons, MBVC_WB  # noqa: E402
from unicycle_uav import (  # noqa: E402
    UnicycleUAV,
    condense_affine_dynamics,
    linearize_unicycle,
)


class UnicycleDynamicsTests(unittest.TestCase):
    def test_affine_model_is_exact_at_linearization_point(self):
        state = np.array([0.4, -0.2, 0.55, 1.1])
        control = np.array([-0.3, 0.7])
        step = 0.03
        a_matrix, b_matrix, offset = linearize_unicycle(
            state, control, step
        )
        affine = a_matrix @ state + b_matrix @ control + offset
        expected = np.array(
            [
                state[0] + step * state[2] * np.cos(state[3]),
                state[1] + step * state[2] * np.sin(state[3]),
                state[2] + step * control[0],
                state[3] + step * control[1],
            ]
        )
        np.testing.assert_allclose(affine, expected, atol=1e-12)

    def test_condensed_model_matches_recursive_affine_rollout(self):
        rng = np.random.default_rng(4)
        horizon = 5
        step = 0.03
        nominal_states = rng.normal(size=(horizon, 4))
        nominal_controls = rng.normal(size=(horizon, 2))
        models = [
            linearize_unicycle(
                nominal_states[k], nominal_controls[k], step
            )
            for k in range(horizon)
        ]
        va, vb, vc = condense_affine_dynamics(
            [item[0] for item in models],
            [item[1] for item in models],
            [item[2] for item in models],
        )
        initial = rng.normal(size=4)
        controls = rng.normal(size=2 * horizon)
        condensed = (va @ initial + vb @ controls + vc).reshape(horizon, 4)

        recursive = []
        state = initial.copy()
        for k, (a_matrix, b_matrix, offset) in enumerate(models):
            control = controls[2 * k : 2 * (k + 1)]
            state = a_matrix @ state + b_matrix @ control + offset
            recursive.append(state)
        np.testing.assert_allclose(condensed, recursive, atol=1e-12)

    def test_deadlock_warning_band_is_active_only_in_adapter_mode(self):
        own = np.array([0.0, 0.0])
        other = np.array([1.0, 0.0])
        target = np.array([0.0, 1.0])
        public = MBVC_WB(
            own, other, target, 0.4, 1.0, True, False
        )
        adapted = MBVC_WB(
            own, other, target, 0.4, 1.0, True, True
        )
        self.assertEqual(public[2], 0.0)
        self.assertGreater(adapted[2], 0.0)


class UnicycleSolverTests(unittest.TestCase):
    def setUp(self):
        starts = [np.array([-1.0, -0.8]), np.array([-1.0, 0.8])]
        goals = [np.array([1.0, 0.8]), np.array([1.0, -0.8])]
        SET.initialize_set(
            2,
            starts,
            [np.zeros(2), np.zeros(2)],
            goals,
            0.4,
            0.1,
            0.10,
            12,
            4,
            2.0,
        )
        self.agents = [
            UnicycleUAV(
                i,
                starts[i],
                np.zeros(2),
                goals[i],
                ini_heading=0.0,
                ini_K=12,
            )
            for i in range(2)
        ]

    def test_stopped_robot_is_not_classified_as_a_wall(self):
        obstacles = get_obstacle_list(self.agents, len(self.agents))
        public = GET_cons(
            self.agents[0],
            obstacles,
            wall_collision_multiplier=0.5,
        )
        adapted = GET_cons(
            self.agents[0],
            obstacles,
            wall_collision_multiplier=0.5,
            dynamic_agent_count=len(self.agents),
        )
        self.assertGreater(
            float(np.max(adapted[1] - public[1])),
            0.09,
        )

    def test_solver_respects_unicycle_input_and_speed_bounds(self):
        for _ in range(3):
            obstacles = get_obstacle_list(self.agents, len(self.agents))
            self.agents = run_one_step(
                self.agents,
                obstacles,
                verbose=False,
                scp_iterations=2,
            )
        for agent in self.agents:
            self.assertIn(agent.last_solver_status, ("optimal", "optimal_inaccurate"))
            self.assertLessEqual(abs(agent.u[0]), agent.Umax + 2e-4)
            self.assertLessEqual(abs(agent.u[1]), agent.Wmax + 2e-4)
            self.assertLessEqual(abs(agent.speed), agent.Vmax + 2e-4)
            predicted = np.asarray(agent.cache[0]).reshape(agent.K, 4)
            self.assertLessEqual(abs(predicted[-1, 2]), 2e-4)

        positions = np.asarray([agent.p for agent in self.agents])
        self.assertGreaterEqual(
            np.linalg.norm(positions[0] - positions[1]),
            SET.r_min - 2e-3,
        )
        self.assertTrue(
            any(np.linalg.norm(agent.position[-1] - agent.ini_p) > 1e-4
                for agent in self.agents)
        )


if __name__ == "__main__":
    unittest.main()
