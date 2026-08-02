"""Geometry and baseline-structure tests for Social Mini-Games."""

from __future__ import annotations

import unittest

import numpy as np

from clear_nav import (
    ControllerConfig,
    Protocol,
    SMGGeometry,
    VanillaCBFController,
    inflated_unicycle_protocol,
    make_smg_scenario,
)


class SMGGeometryTests(unittest.TestCase):
    def test_doorway_width_controls_safe_queue_lanes(self):
        protocol = Protocol()
        narrow = SMGGeometry(doorway_width=0.8)
        medium = SMGGeometry(doorway_width=1.2)
        self.assertEqual(
            narrow.metadata(protocol, lookahead=0.05)[
                "doorway_nominal_safety_lanes"
            ],
            1,
        )
        self.assertEqual(
            medium.metadata(protocol, lookahead=0.05)[
                "doorway_nominal_safety_lanes"
            ],
            2,
        )

    def test_smg_sites_are_initially_safe_for_n16(self):
        physical = Protocol()
        for family, geometry in (
            ("doorway", SMGGeometry(doorway_width=0.8)),
            ("doorway", SMGGeometry(doorway_width=1.2)),
            (
                "intersection",
                SMGGeometry(intersection_corridor_width=2.4),
            ),
        ):
            scenario = make_smg_scenario(
                family,
                16,
                0,
                physical,
                geometry,
            )
            left, right = np.triu_indices(16, k=1)
            distance = np.linalg.norm(
                scenario.starts[left] - scenario.starts[right],
                axis=1,
            )
            self.assertGreaterEqual(
                float(np.min(distance)),
                physical.pair_clearance + 0.10 - 1.0e-12,
            )
            for point in scenario.starts:
                self.assertTrue(
                    scenario.arena.free(
                        point,
                        physical.robot_clearance + 0.05,
                    )
                )
            for point in scenario.goals:
                self.assertTrue(
                    scenario.arena.free(
                        point,
                        physical.robot_clearance,
                    )
                )

    def test_twenty_paired_instances_are_distinct_and_safe(self):
        physical = Protocol()
        for family, geometry in (
            ("doorway", SMGGeometry(doorway_width=1.2)),
            (
                "intersection",
                SMGGeometry(intersection_corridor_width=2.4),
            ),
        ):
            fingerprints = set()
            for count in (8, 16):
                for seed in range(20):
                    scenario = make_smg_scenario(
                        family, count, seed, physical, geometry
                    )
                    fingerprints.add((count, scenario.fingerprint()))
                    left, right = np.triu_indices(count, k=1)
                    distances = np.linalg.norm(
                        scenario.starts[left] - scenario.starts[right],
                        axis=1,
                    )
                    self.assertGreaterEqual(
                        float(np.min(distances)),
                        physical.pair_clearance + 0.10 - 1e-12,
                    )
                    for point in scenario.starts:
                        self.assertTrue(
                            scenario.arena.free(
                                point,
                                physical.robot_clearance + 0.05,
                            )
                        )
                self.assertEqual(
                    len([item for item in fingerprints if item[0] == count]),
                    20,
                )


class VanillaCBFTests(unittest.TestCase):
    def test_vanilla_controller_has_no_transverse_terms(self):
        physical = Protocol()
        control = inflated_unicycle_protocol(physical, 0.05)
        controller = VanillaCBFController(
            control,
            ControllerConfig(),
        )
        scenario = make_smg_scenario(
            "doorway",
            8,
            0,
            physical,
            SMGGeometry(doorway_width=1.2),
        )
        positions = scenario.starts.copy()
        goal = controller.goal_field(positions, scenario.goals)
        circulation, pairs, boundaries = controller.circulation_field(
            positions,
            scenario.goals,
            scenario.arena,
            goal,
        )
        cluster, active = controller.cluster_escape_field(
            positions,
            goal,
            scenario.arena,
            np.zeros((0, 2 * len(positions))),
        )
        np.testing.assert_array_equal(circulation, 0.0)
        np.testing.assert_array_equal(cluster, 0.0)
        self.assertEqual((pairs, boundaries, active), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
