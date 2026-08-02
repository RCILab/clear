"""Validate the 12 Rect15 MGR timing records and emit timing-v2 JSON."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> None:
    source = Path("results/timing_v2/mgr_rect15.jsonl")
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 12:
        raise RuntimeError(f"expected 12 MGR records, found {len(records)}")
    expected = {
        (count, seed)
        for count in (20, 40, 60, 80)
        for seed in (0, 7, 13)
    }
    actual = {
        (int(record["num_agents"]), int(record["scenario_seed"]))
        for record in records
    }
    if actual != expected:
        raise RuntimeError(
            f"MGR timing matrix mismatch: missing={expected-actual}, "
            f"extra={actual-expected}"
        )
    required = (
        "batch_step_ms",
        "shared_coordination_ms",
        "local_unit_ms",
        "critical_path_ms",
        "controller_backend",
        "worker_count",
        "cpu_model",
        "gpu_model",
        "timing_warmup_steps",
        "scenario_fingerprint",
    )
    for record in records:
        missing = [key for key in required if key not in record]
        if missing:
            raise RuntimeError(f"MGR timing fields missing: {missing}")
        record["n_robots"] = int(record["num_agents"])
        record["seed"] = int(record["scenario_seed"])
        record["warmup_steps"] = int(record["timing_warmup_steps"])
    records.sort(key=lambda item: (item["n_robots"], item["seed"]))
    payload = {
        "schema_version": "timing-v2.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "MGR",
        "protocol": {
            "family": "rect15",
            "dt_s": 0.03,
            "horizon_s": 60.0,
            "seeds": [0, 7, 13],
            "warmup_steps": 40,
        },
        "records": records,
    }
    output = Path("results/timing_v2/timing_v2_mgr_rect15.json")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
