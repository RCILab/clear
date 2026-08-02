from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from clear_nav.config import ControllerConfig, Protocol
from clear_nav.controller import CLEARController
from clear_nav.geometry import Arena, Circle, Rectangle, rotate90
from clear_nav.guidance import GridPlanner, segment_is_free
from clear_nav.safety import (
    OSQPProjectionWorkspace,
    project_halfspaces,
    project_halfspaces_and_balls,
    project_halfspaces_osqp,
)
from clear_nav.scenarios import Scenario, make_scenario
from clear_nav.simulator import simulate
from clear_nav.unicycle import (
    UnicycleConfig,
    _component_partition,
    _component_yaw_scales,
    _project_native_unicycle,
    inflated_unicycle_protocol,
    unicycle_yaw_scale_floor,
)
from run_unicycle import controller_config


class GeometryTests(unittest.TestCase):
    def test_rotate90_is_positive_quarter_turn(self) -> None:
        np.testing.assert_allclose(rotate90(np.array([1.0, 0.0])), [0.0, 1.0])

    def test_circle_clearance_and_normal(self) -> None:
        circle = Circle(np.zeros(2), 1.0)
        h, n = circle.clearance_normal(np.array([2.0, 0.0]), 0.2)
        self.assertAlmostEqual(h, 0.8)
        np.testing.assert_allclose(n, [1.0, 0.0])

    def test_rectangle_corner_normal(self) -> None:
        box = Rectangle(np.zeros(2), np.array([2.0, 2.0]))
        h, n = box.clearance_normal(np.array([2.0, 2.0]), 0.0)
        self.assertAlmostEqual(h, np.sqrt(2.0))
        np.testing.assert_allclose(n, np.ones(2) / np.sqrt(2.0))


class ProjectionTests(unittest.TestCase):
    def test_halfspace_projection(self) -> None:
        result = project_halfspaces(
            np.array([-2.0, 3.0]),
            np.array([[1.0, 0.0]]),
            np.array([0.0]),
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.value, [0.0, 3.0], atol=1.0e-8)

    def test_persistent_two_sided_osqp_matches_explicit_box_rows(
        self,
    ) -> None:
        from scipy import sparse

        workspace = OSQPProjectionWorkspace(max_cached_patterns=2)
        geometric = np.array(
            [[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]
        )
        geometric_lower = np.array([-0.2, 0.1])
        limits = np.array([0.7, 0.5, 0.6])
        identity = np.eye(3)
        explicit_matrix = np.vstack(
            (geometric, identity, -identity)
        )
        explicit_lower = np.concatenate(
            (geometric_lower, -limits, -limits)
        )
        sparse_matrix = sparse.vstack(
            (
                sparse.csc_matrix(geometric),
                sparse.eye(3, format="csc"),
            ),
            format="csc",
        )
        sparse_lower = np.concatenate((geometric_lower, -limits))
        sparse_upper = np.concatenate(
            (np.full(len(geometric_lower), np.inf), limits)
        )
        for nominal in (
            np.array([-0.8, 0.9, -0.7]),
            np.array([0.4, -0.6, 0.8]),
        ):
            expected = project_halfspaces_osqp(
                nominal,
                explicit_matrix,
                explicit_lower,
            )
            actual = workspace.project(
                nominal,
                sparse_matrix,
                sparse_lower,
                sparse_upper,
            )
            self.assertTrue(actual.converged)
            np.testing.assert_allclose(
                actual.value,
                expected.value,
                atol=2.0e-7,
            )

    def test_osqp_feasibility_is_separate_from_solver_termination(self) -> None:
        from scipy import sparse

        identity = sparse.eye(2, format="csc")
        lower = np.full(2, -1.0)
        upper = np.full(2, 1.0)
        feasible_limited = None
        for max_iterations in range(1, 10):
            workspace = OSQPProjectionWorkspace(max_cached_patterns=1)
            workspace.project(np.zeros(2), identity, lower, upper)
            result = workspace.project(
                np.full(2, 2.0),
                identity,
                lower,
                upper,
                max_iterations=max_iterations,
            )
            if result.command_feasible and not result.optimizer_converged:
                feasible_limited = result
                break
        self.assertIsNotNone(feasible_limited)
        self.assertIsNotNone(feasible_limited.solver_status)

    def test_relaxed_halfspace_projection_has_same_fixed_point(self) -> None:
        dimension = 12
        basis = np.eye(dimension)
        chain = np.asarray(
            [basis[index] - basis[index + 1] for index in range(dimension - 1)]
        )
        # Repeated rows mimic the nearly dependent pair constraints in the
        # exact Swap task.
        matrix = np.repeat(chain, 3, axis=0)
        nominal = np.linspace(-1.0, 1.0, dimension)
        baseline = project_halfspaces(
            nominal,
            matrix,
            np.zeros(len(matrix)),
            tolerance=1.0e-10,
            max_sweeps=5_000,
            relaxation=1.0,
            batching_minimum_rows=10_000,
        )
        accelerated = project_halfspaces(
            nominal,
            matrix,
            np.zeros(len(matrix)),
            tolerance=1.0e-10,
            max_sweeps=5_000,
            relaxation=1.5,
        )
        self.assertTrue(baseline.converged)
        self.assertTrue(accelerated.converged)
        self.assertLess(accelerated.sweeps, baseline.sweeps)
        np.testing.assert_allclose(
            accelerated.value,
            baseline.value,
            atol=1.0e-8,
        )

        nearby_nominal = nominal + np.linspace(0.0, 1.0e-3, dimension)
        cold = project_halfspaces(
            nearby_nominal,
            matrix,
            np.zeros(len(matrix)),
            tolerance=1.0e-10,
            max_sweeps=5_000,
        )
        warm = project_halfspaces(
            nearby_nominal,
            matrix,
            np.zeros(len(matrix)),
            tolerance=1.0e-10,
            max_sweeps=5_000,
            initial_dual=accelerated.dual,
        )
        self.assertTrue(cold.converged)
        self.assertTrue(warm.converged)
        self.assertLessEqual(warm.sweeps, cold.sweeps)
        np.testing.assert_allclose(warm.value, cold.value, atol=1.0e-8)

    def test_halfspaces_and_exact_ball(self) -> None:
        result = project_halfspaces_and_balls(
            np.array([-2.0, 2.0]),
            np.array([[1.0, 0.0]]),
            np.array([0.0]),
            n_robots=1,
            speed_limit=1.0,
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.value[0], -1.0e-8)
        self.assertLessEqual(np.linalg.norm(result.value), 1.0 + 1.0e-8)


class GuidanceTests(unittest.TestCase):
    def test_route_goes_around_blocking_rectangle(self) -> None:
        protocol = Protocol(horizon=1.0)
        arena = Arena(
            protocol.half_width,
            (Rectangle(np.zeros(2), np.array([2.0, 4.0])),),
        )
        start = np.array([-3.0, 0.0])
        goal = np.array([3.0, 0.0])
        path = GridPlanner(arena, protocol, resolution=0.25).path(start, goal)
        self.assertGreater(len(path), 2)
        for left, right in zip(path[:-1], path[1:]):
            self.assertTrue(
                segment_is_free(
                    left,
                    right,
                    arena,
                    protocol.robot_clearance,
                    sample_step=0.05,
                )
            )

    def test_cost_field_target_strictly_descends_geodesic_cost(self) -> None:
        protocol = Protocol(horizon=1.0)
        arena = Arena(
            protocol.half_width,
            (Rectangle(np.zeros(2), np.array([2.0, 4.0])),),
        )
        planner = GridPlanner(arena, protocol, resolution=0.25)
        start = np.array([[-3.0, 0.0]])
        goal = np.array([[3.0, 0.0]])
        field = planner.cost_field_plan(goal)
        target = field.targets(start)[0]
        start_node = field._nearest_free(start[0])
        target_node = field._nearest_free(target)
        self.assertIsNotNone(start_node)
        self.assertIsNotNone(target_node)
        self.assertLess(field.costs[0][target_node], field.costs[0][start_node])


class ControllerStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = Protocol(horizon=4.0, dt=0.025)
        self.controller = CLEARController(self.protocol)
        self.arena = Arena(self.protocol.half_width)

    def test_pair_circulation_is_reciprocal_and_tangent(self) -> None:
        x = np.array([[-0.4, 0.0], [0.4, 0.0]])
        goals = -x
        goal = self.controller.goal_field(x, goals)
        circulation, count, _ = self.controller.circulation_field(
            x, goals, self.arena, goal
        )
        self.assertEqual(count, 1)
        np.testing.assert_allclose(circulation.sum(axis=0), 0.0, atol=1.0e-12)
        normal = (x[0] - x[1]) / np.linalg.norm(x[0] - x[1])
        self.assertAlmostEqual(float(normal @ (circulation[0] - circulation[1])), 0.0)

    def test_terminal_capture_suppresses_only_captured_pair_circulation(self) -> None:
        controller = CLEARController(
            self.protocol,
            ControllerConfig(terminal_capture_hysteresis=True),
        )
        positions = np.array([[-0.4, 0.0], [0.4, 0.0]])
        both_captured = controller.command(
            positions, positions.copy(), self.arena
        )
        np.testing.assert_allclose(
            both_captured.raw_circulation, 0.0, atol=1.0e-12
        )

        controller.reset()
        one_captured_goals = np.array([[-0.4, 0.0], [-0.4, 0.0]])
        one_captured = controller.command(
            positions, one_captured_goals, self.arena
        )
        self.assertGreater(
            float(np.linalg.norm(one_captured.raw_circulation)), 0.0
        )

    def test_terminal_capture_uses_wider_latch_only_without_obstacles(
        self,
    ) -> None:
        config = ControllerConfig(
            terminal_capture_hysteresis=True,
            terminal_capture_radius=0.22,
            terminal_open_capture_radius=0.60,
        )
        positions = np.array([[-0.4, 0.0], [0.4, 0.0]])
        goals = np.array([[0.1, 0.0], [-0.1, 0.0]])

        open_controller = CLEARController(self.protocol, config)
        open_audit = open_controller.command(
            positions,
            goals,
            self.arena,
        )
        np.testing.assert_allclose(
            open_audit.raw_circulation,
            0.0,
            atol=1.0e-12,
        )

        clutter_controller = CLEARController(self.protocol, config)
        clutter = Arena(
            self.protocol.half_width,
            (Circle(np.array([5.0, 5.0]), 0.1),),
        )
        clutter_audit = clutter_controller.command(
            positions,
            goals,
            clutter,
        )
        self.assertGreater(
            float(np.linalg.norm(clutter_audit.raw_circulation)),
            0.0,
        )

    def test_permutation_equivariance(self) -> None:
        x = np.array([[-0.8, 0.0], [0.8, 0.0], [0.0, 1.1]])
        goals = np.array([[0.8, 0.0], [-0.8, 0.0], [0.0, -1.1]])
        permutation = np.array([2, 0, 1])
        original = self.controller.command(x, goals, self.arena).executed_command
        permuted = self.controller.command(
            x[permutation], goals[permutation], self.arena
        ).executed_command
        np.testing.assert_allclose(permuted, original[permutation], atol=2.0e-7)

    def test_rotation_equivariance(self) -> None:
        angle = 0.37
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        x = np.array([[-0.7, -0.1], [0.8, 0.1]])
        goals = -x
        original = self.controller.command(x, goals, self.arena).executed_command
        rotated = self.controller.command(
            x @ rotation.T, goals @ rotation.T, self.arena
        ).executed_command
        np.testing.assert_allclose(rotated, original @ rotation.T, atol=2.0e-7)

    def test_reflection_requires_handedness_flip(self) -> None:
        reflection = np.diag([-1.0, 1.0])
        x = np.array([[-0.7, -0.1], [0.8, 0.1]])
        goals = -x
        original = self.controller.command(x, goals, self.arena).executed_command
        flipped = CLEARController(
            self.protocol, ControllerConfig(handedness=-1)
        )
        reflected = flipped.command(
            x @ reflection.T, goals @ reflection.T, self.arena
        ).executed_command
        np.testing.assert_allclose(reflected, original @ reflection.T, atol=2.0e-7)

    def test_cbf_command_is_safe_at_contact(self) -> None:
        d = self.protocol.pair_clearance
        x = np.array([[-0.5 * d, 0.0], [0.5 * d, 0.0]])
        goals = -x
        audit = self.controller.command(x, goals, self.arena)
        normal = (x[0] - x[1]) / d
        derivative = float(normal @ (audit.executed_command[0] - audit.executed_command[1]))
        self.assertGreaterEqual(derivative, -1.0e-7)
        self.assertTrue(audit.cbf_converged)
        self.assertTrue(audit.tangent_converged)

    def test_global_tangent_margin_matches_nominal_inner_product(self) -> None:
        d = self.protocol.pair_clearance
        positions = np.array([[-0.5 * d, 0.0], [0.5 * d, 0.0]])
        goals = -positions
        audit = self.controller.command(positions, goals, self.arena)
        expected = float(
            audit.nominal_command.reshape(-1)
            @ audit.cone_circulation.reshape(-1)
        )
        self.assertAlmostEqual(audit.global_tangent_margin, expected)
        self.assertAlmostEqual(
            audit.tangent_norm,
            float(np.linalg.norm(audit.cone_circulation)),
        )
        np.testing.assert_allclose(
            audit.executed_command,
            audit.common_scale * audit.projected_command,
        )

    def test_adaptive_gain_certifies_each_nonzero_component(self) -> None:
        x = np.array([[-0.6, 0.0], [0.6, 0.0], [4.0, -7.6]])
        goals = np.array([[0.6, 0.0], [-0.6, 0.0], [4.0, 6.0]])
        adaptive = CLEARController(
            self.protocol,
            ControllerConfig(adaptive_certificate_gain=True),
        )
        audit = adaptive.command(x, goals, self.arena)
        self.assertGreater(audit.tangent_margin, 0.0)
        self.assertTrue(np.all(audit.circulation_gains >= 0.72))

    def test_progress_boundary_circulation_aligns_with_guide(self) -> None:
        arena = Arena(
            self.protocol.half_width,
            (Circle(np.zeros(2), 1.0),),
        )
        controller = CLEARController(
            self.protocol,
            ControllerConfig(boundary_progress_aligned=True),
        )
        positions = np.array([[1.3, 0.0]])
        guide = np.array([[1.3, -2.0]])
        goal_command = controller.goal_field(positions, guide)
        circulation, _, boundaries = controller.circulation_field(
            positions, guide, arena, goal_command
        )
        self.assertGreater(boundaries, 0)
        self.assertGreater(float(circulation[0] @ goal_command[0]), 0.0)

    def test_rigid_cluster_direction_respects_opposed_boundaries(self) -> None:
        direction = self.controller._common_tangent_direction(
            [np.array([0.0, 1.0]), np.array([0.0, -1.0])],
            np.array([1.0, 0.2]),
        )
        self.assertIsNotNone(direction)
        np.testing.assert_allclose(direction, np.array([1.0, 0.0]), atol=1.0e-12)

    def test_rigid_cluster_mode_is_common_and_pair_tangent(self) -> None:
        arena = Arena(
            self.protocol.half_width,
            (Rectangle(np.array([0.0, -1.0]), np.array([4.0, 1.0])),),
        )
        controller = CLEARController(
            self.protocol,
            ControllerConfig(cluster_escape_gain=0.55),
        )
        clearance = self.protocol.pair_clearance
        positions = np.array([[-0.5 * clearance, -0.28], [0.5 * clearance, -0.28]])
        goals = np.array([[2.0, -0.28], [1.0, -0.28]])
        goal_command = controller.goal_field(positions, goals)
        circulation, cluster_count = controller.cluster_escape_field(
            positions,
            goal_command,
            arena,
            np.zeros((0, 2 * len(positions))),
        )
        self.assertGreater(cluster_count, 0)
        pair_normal = np.array([-1.0, 0.0])
        self.assertAlmostEqual(
            float(pair_normal @ (circulation[0] - circulation[1])),
            0.0,
            places=10,
        )

    def test_cluster_token_holds_direction_through_goal_sign_flip(self) -> None:
        arena = Arena(
            self.protocol.half_width,
            (Rectangle(np.array([0.0, -1.0]), np.array([4.0, 1.0])),),
        )
        controller = CLEARController(
            self.protocol,
            ControllerConfig(
                cluster_escape_gain=0.55,
                cluster_escape_hysteresis=True,
            ),
        )
        clearance = self.protocol.pair_clearance
        positions = np.array([[-0.5 * clearance, -0.28], [0.5 * clearance, -0.28]])
        right_goals = np.array([[2.0, -0.28], [1.0, -0.28]])
        right_command = controller.goal_field(positions, right_goals)
        controller.cluster_escape_field(
            positions,
            right_command,
            arena,
            np.zeros((0, 2 * len(positions))),
        )
        first_token = controller._cluster_tokens[(0, 1)].copy()

        left_goals = np.array([[-2.0, -0.28], [-1.0, -0.28]])
        left_command = controller.goal_field(positions, left_goals)
        controller.cluster_escape_field(
            positions,
            left_command,
            arena,
            np.zeros((0, 2 * len(positions))),
        )
        np.testing.assert_allclose(
            controller._cluster_tokens[(0, 1)], first_token, atol=1.0e-12
        )


class BehavioralProbeTests(unittest.TestCase):
    def test_component_ablation_keeps_clear_tangent_band(self) -> None:
        common = {
            "handedness": 1,
            "boundary_mode": "progress",
            "terminal_capture_radius": 0.22,
            "terminal_open_capture_radius": 0.60,
            "terminal_release_radius": 0.80,
        }
        clear = controller_config(
            SimpleNamespace(variant="clear", **common)
        )
        ablated = controller_config(
            SimpleNamespace(variant="component-free", **common)
        )
        self.assertEqual(clear.tangent_band, 0.08)
        self.assertEqual(ablated.tangent_band, clear.tangent_band)
        self.assertGreater(clear.cluster_escape_gain, 0.0)
        self.assertEqual(ablated.cluster_escape_gain, 0.0)

    def test_two_robot_exact_swap_moves_transversely_and_stays_safe(self) -> None:
        protocol = Protocol(horizon=4.0, dt=0.025)
        d = 2.0
        scenario = Scenario(
            "swap",
            2,
            0,
            np.array([[-d, 0.0], [d, 0.0]]),
            np.array([[d, 0.0], [-d, 0.0]]),
            Arena(protocol.half_width),
            protocol,
        )
        rollout = simulate(scenario, record_stride=8)
        self.assertGreater(float(np.max(np.abs(rollout.trajectory[:, :, 1]))), 0.05)
        self.assertGreaterEqual(
            rollout.minimum_pair_distance, protocol.pair_clearance - 2.0e-5
        )

    def test_scenario_generation_is_reproducible(self) -> None:
        first = make_scenario("circ15", 8, 7, Protocol(horizon=1.0))
        second = make_scenario("circ15", 8, 7, Protocol(horizon=1.0))
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_unicycle_virtual_clearance_inflation(self) -> None:
        physical = Protocol()
        lookahead = 0.10
        virtual = inflated_unicycle_protocol(physical, lookahead)
        self.assertAlmostEqual(
            virtual.pair_clearance,
            physical.pair_clearance + 2.0 * lookahead,
        )
        self.assertAlmostEqual(
            virtual.robot_clearance,
            physical.robot_clearance + lookahead,
        )

    def test_dense_unicycle_swap_initialization_is_safe(self) -> None:
        physical = Protocol()
        virtual = inflated_unicycle_protocol(physical, 0.10)
        scenario = make_scenario("swap", 80, 0, virtual)
        left, right = np.triu_indices(80, k=1)
        minimum = float(
            np.min(
                np.linalg.norm(
                    scenario.starts[left] - scenario.starts[right],
                    axis=1,
                )
            )
        )
        self.assertGreaterEqual(minimum, virtual.pair_clearance)
        self.assertTrue(np.allclose(scenario.goals, -scenario.starts))

    def test_unicycle_yaw_scale_has_positive_analytic_floor(self) -> None:
        physical = Protocol(speed_limit=0.80)
        config = UnicycleConfig(
            lookahead=0.10,
            yaw_rate_limit=6.0,
        )
        self.assertAlmostEqual(
            unicycle_yaw_scale_floor(physical, config),
            0.75,
        )

    def test_robotarium_unicycle_mode_validates(self) -> None:
        config = UnicycleConfig(
            lookahead=0.05,
            yaw_rate_limit=np.pi / 2.0,
            inner_substeps=3,
            actuation_mode="robotarium-clip",
        )
        self.assertEqual(config.actuation_mode, "robotarium-clip")
        self.assertEqual(
            UnicycleConfig(actuation_mode="component-scale").actuation_mode,
            "component-scale",
        )
        self.assertEqual(UnicycleConfig().actuation_mode, "native-cbf")
        with self.assertRaises(ValueError):
            UnicycleConfig(actuation_mode="unknown")
        with self.assertRaises(ValueError):
            UnicycleConfig(bridge_progress_retention=1.0)

    def test_native_unicycle_projection_enforces_cbf_and_input_bounds(
        self,
    ) -> None:
        physical = Protocol(horizon=1.0)
        config = UnicycleConfig()
        control = inflated_unicycle_protocol(
            physical,
            config.lookahead,
        )
        controller = CLEARController(control, ControllerConfig())
        positions = np.array(
            [[0.0, 0.0], [control.pair_clearance, 0.0]]
        )
        headings = np.array([0.0, np.pi])
        nominal = np.array([[0.8, 0.0], [-0.8, 0.0]])
        linear, yaw, _, result = _project_native_unicycle(
            controller,
            positions,
            headings,
            Arena(physical.half_width),
            nominal,
            physical,
            config,
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(
            result.minimum_residual,
            -config.projection_tolerance,
        )
        self.assertLessEqual(float(np.max(np.abs(linear))), 0.8 + 1.0e-8)
        self.assertLessEqual(
            float(np.max(np.abs(yaw))),
            np.pi / 2.0 + 1.0e-8,
        )
        self.assertLessEqual(
            float(np.max(np.abs(linear))),
            config.projection_tolerance,
        )

    def test_native_component_projection_matches_global_qp(self) -> None:
        physical = Protocol(horizon=1.0)
        config = UnicycleConfig(projection_solver_tolerance=1.0e-9)
        control = inflated_unicycle_protocol(
            physical,
            config.lookahead,
        )
        controller = CLEARController(control, ControllerConfig())
        positions = np.array(
            [
                [-2.0, 0.0],
                [-2.0 + control.pair_clearance, 0.0],
                [2.0, 0.0],
                [2.0 + control.pair_clearance, 0.0],
            ]
        )
        headings = np.array([0.0, np.pi, 0.0, np.pi])
        nominal = np.array(
            [[0.8, 0.1], [-0.8, -0.1], [0.7, 0.2], [-0.7, -0.2]]
        )
        global_output = _project_native_unicycle(
            controller,
            positions,
            headings,
            Arena(physical.half_width),
            nominal,
            physical,
            config,
            workspace=OSQPProjectionWorkspace(max_cached_patterns=2),
        )
        timing: dict = {}
        component_output = _project_native_unicycle(
            controller,
            positions,
            headings,
            Arena(physical.half_width),
            nominal,
            physical,
            config,
            component_workspaces={},
            timing=timing,
        )
        for global_value, component_value in zip(
            global_output[:3],
            component_output[:3],
        ):
            np.testing.assert_allclose(
                component_value,
                global_value,
                atol=2.0e-7,
                rtol=2.0e-7,
            )
        global_objective = float(
            np.sum((global_output[2] - nominal) ** 2)
        )
        component_objective = float(
            np.sum((component_output[2] - nominal) ** 2)
        )
        self.assertLessEqual(
            abs(global_objective - component_objective),
            1.0e-8,
        )
        self.assertTrue(global_output[3].converged)
        self.assertTrue(component_output[3].converged)
        self.assertEqual(timing["component_sizes"], [2, 2])
        self.assertEqual(len(timing["local_unit_ns"]), 2)
        self.assertEqual(
            timing["critical_path_ns"],
            timing["shared_ns"] + max(timing["local_unit_ns"]),
        )

    def test_component_partition_uses_final_qp_row_support(self) -> None:
        matrix = np.zeros((3, 8))
        matrix[0, 0] = 1.0
        matrix[0, 2] = -1.0
        matrix[1, 5] = 1.0
        matrix[1, 7] = -1.0
        matrix[2, 1] = 1.0
        self.assertEqual(
            _component_partition(matrix, 4),
            [(0, 1), (2, 3)],
        )

    def test_native_projection_reports_solver_and_restoration_separately(
        self,
    ) -> None:
        physical = Protocol(horizon=1.0)
        config = UnicycleConfig(
            projection_max_sweeps=1,
            projection_tolerance=1.0e-8,
            projection_solver_tolerance=1.0e-8,
        )
        control = inflated_unicycle_protocol(
            physical,
            config.lookahead,
        )
        controller = CLEARController(control, ControllerConfig())
        _, _, _, result = _project_native_unicycle(
            controller,
            np.array(
                [[0.0, 0.0], [control.pair_clearance, 0.0]]
            ),
            np.array([0.0, np.pi]),
            Arena(physical.half_width),
            np.array([[0.8, 0.0], [-0.8, 0.0]]),
            physical,
            config,
        )
        self.assertFalse(result.solver_converged)
        self.assertTrue(result.command_feasible)
        self.assertIn(
            result.restoration_type,
            (None, "common-contraction"),
        )

    def test_native_unicycle_projection_saturates_yaw_directly(self) -> None:
        physical = Protocol(horizon=1.0)
        config = UnicycleConfig()
        controller = CLEARController(
            inflated_unicycle_protocol(physical, config.lookahead),
            ControllerConfig(),
        )
        _, yaw, _, result = _project_native_unicycle(
            controller,
            np.array([[0.0, 0.0]]),
            np.array([0.0]),
            Arena(physical.half_width),
            np.array([[0.0, 0.8]]),
            physical,
            config,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(yaw[0]), np.pi / 2.0, places=7)

    def test_native_projection_accepts_one_certified_progress_row(
        self,
    ) -> None:
        physical = Protocol(horizon=1.0)
        config = UnicycleConfig()
        controller = CLEARController(
            inflated_unicycle_protocol(physical, config.lookahead),
            ControllerConfig(),
        )
        _, yaw, realized, result = _project_native_unicycle(
            controller,
            np.array([[0.0, 0.0]]),
            np.array([np.pi / 2.0]),
            Arena(physical.half_width),
            np.zeros((1, 2)),
            physical,
            config,
            certified_progress_world_rows=np.array([[[1.0, 0.0]]]),
            certified_progress_lower=np.array([0.05]),
            certified_progress_candidate=np.array([[0.06, 0.0]]),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.certified_progress_rows_requested, 1)
        self.assertEqual(result.certified_progress_rows_accepted, 1)
        self.assertGreaterEqual(
            result.minimum_certified_progress_residual,
            -config.projection_tolerance,
        )
        self.assertGreaterEqual(float(realized[0, 0]), 0.05 - 1.0e-7)
        self.assertLessEqual(
            abs(float(yaw[0])),
            config.yaw_rate_limit + 1.0e-8,
        )

    def test_native_projection_rejects_infeasible_progress_witness(
        self,
    ) -> None:
        physical = Protocol(horizon=1.0)
        config = UnicycleConfig()
        controller = CLEARController(
            inflated_unicycle_protocol(physical, config.lookahead),
            ControllerConfig(),
        )
        _, _, realized, result = _project_native_unicycle(
            controller,
            np.array([[0.0, 0.0]]),
            np.array([np.pi / 2.0]),
            Arena(physical.half_width),
            np.zeros((1, 2)),
            physical,
            config,
            certified_progress_world_rows=np.array([[[1.0, 0.0]]]),
            certified_progress_lower=np.array([0.09]),
            certified_progress_candidate=np.array([[0.10, 0.0]]),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.certified_progress_rows_requested, 1)
        self.assertEqual(result.certified_progress_rows_accepted, 0)
        np.testing.assert_allclose(realized, 0.0, atol=1.0e-8)

    def test_native_hqp_retains_first_level_structural_progress(self) -> None:
        physical = Protocol(horizon=1.0)
        config = UnicycleConfig(
            hierarchical_progress=True,
            hqp_progress_retention=0.995,
        )
        controller = CLEARController(
            inflated_unicycle_protocol(physical, config.lookahead),
            ControllerConfig(),
        )
        linear, yaw, _, result = _project_native_unicycle(
            controller,
            np.array([[0.0, 0.0]]),
            np.array([0.0]),
            Arena(physical.half_width),
            np.array([[0.1, 0.0]]),
            physical,
            config,
            progress_virtual_command=np.array([[1.0, 0.0]]),
        )
        self.assertTrue(result.converged)
        self.assertTrue(result.hierarchy_active)
        self.assertTrue(result.hierarchy_converged)
        self.assertGreaterEqual(
            result.progress_retention,
            config.hqp_progress_retention
            - 20.0 * config.projection_tolerance,
        )
        self.assertGreater(float(linear[0]), 0.79)
        self.assertAlmostEqual(float(yaw[0]), 0.0, places=7)

    def test_component_yaw_scale_is_shared_only_across_pair_rows(self) -> None:
        positions = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
        scales = _component_yaw_scales(
            positions,
            np.array([10.0, 1.0, 4.0]),
            yaw_rate_limit=2.0,
            pair_row_distance=1.0,
        )
        np.testing.assert_allclose(scales, [0.2, 0.2, 0.5])


if __name__ == "__main__":
    unittest.main()
