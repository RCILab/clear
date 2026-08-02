from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from timing_v2_common import aggregate_record, save_samples


class TimingV2Tests(unittest.TestCase):
    def test_critical_path_identity_and_aggregate(self) -> None:
        record = aggregate_record(
            batch_step_ms=[5.0, 6.0],
            shared_coordination_ms=[1.0, 1.5],
            local_unit_ms=[[2.0, 4.0], [3.0, 4.5]],
            critical_path_ms=[5.0, 6.0],
            controller_backend="test",
            worker_count=1,
            warmup_steps=2,
            scenario_fingerprint="abc",
        )
        self.assertEqual(record["batch_step_ms"]["sample_count"], 2)
        self.assertEqual(record["local_unit_ms"]["per_call"]["sample_count"], 4)
        self.assertAlmostEqual(record["critical_path_ms"]["maximum"], 6.0)

    def test_alternate_batch_and_ego_paths_are_explicit(self) -> None:
        record = aggregate_record(
            batch_step_ms=[1.0],
            shared_coordination_ms=[0.0],
            local_unit_ms=[[1.2, 0.9]],
            critical_path_ms=[1.2],
            controller_backend="test",
            worker_count=1,
            warmup_steps=0,
            scenario_fingerprint="abc",
            batch_contains_local_units=False,
        )
        self.assertFalse(record["batch_contains_local_units"])

    def test_ragged_local_samples_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.npz"
            digest = save_samples(
                path,
                batch_step_ms=[3.0, 4.0],
                shared_coordination_ms=[1.0, 1.0],
                local_unit_ms=[[1.0, 2.0], [3.0]],
                critical_path_ms=[3.0, 4.0],
            )
            self.assertEqual(len(digest), 64)
            with np.load(path) as data:
                np.testing.assert_array_equal(
                    data["local_unit_offsets"],
                    [0, 2, 3],
                )
                np.testing.assert_allclose(
                    data["local_unit_ms"],
                    [1.0, 2.0, 3.0],
                )


if __name__ == "__main__":
    unittest.main()
