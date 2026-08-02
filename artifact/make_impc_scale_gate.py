"""Turn the measured N=8 IMPC-DR gate into explicit larger-N records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


SOURCE = Path(
    "validation/timing_v2/timing_v2_impc_rect15_n8_gate_raw.json"
)
OUTPUT = Path("validation/timing_v2/timing_v2_impc_scale_gate.json")


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    records = payload["records"]
    seeds = sorted(int(record["seed"]) for record in records)
    if seeds != [0, 7, 13]:
        raise RuntimeError(f"expected N=8 seeds 0/7/13, found {seeds}")
    if any(int(record["n_robots"]) != 8 for record in records):
        raise RuntimeError("scale-gate source must contain only N=8")
    worst_p95 = max(
        record["critical_path_ms"]["p95"] for record in records
    )
    if worst_p95 <= 30.0:
        raise RuntimeError(
            "N=8 critical path passed; continue the scale ladder"
        )
    canonical = [
        record["phase_profile_per_agent_call_ms"][
            "canonicalization_and_interface"
        ]["mean"]
        for record in records
    ]
    solver = [
        record["phase_profile_per_agent_call_ms"][
            "solver_reported"
        ]["mean"]
        for record in records
    ]
    reason = (
        f"N=8 worst-seed local critical-path p95 is {worst_p95:.3f} ms, "
        "above the 30 ms period; mean canonicalization/interface time "
        f"({sum(canonical)/len(canonical):.3f} ms per agent call) exceeds "
        f"reported solver time ({sum(solver)/len(solver):.3f} ms), so the "
        "GPU-solver entry condition is also false."
    )
    gate_records = [
        {
            "method": "IMPC-DR",
            "family": "rect15",
            "n_robots": count,
            "seed": None,
            "status": "not_run_scale_gate",
            "reason": reason,
            "gate_n_robots": 8,
            "gate_worst_seed_critical_p95_ms": worst_p95,
            "control_period_ms": 30.0,
        }
        for count in (20, 40, 60, 80)
    ]
    output = {
        "schema_version": "timing-v2.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "IMPC-DR",
        "scale_gate_source": SOURCE.as_posix(),
        "scale_gate_source_sha256": hashlib.sha256(
            SOURCE.read_bytes()
        ).hexdigest(),
        "gpu_decision": "not implemented: solver is not the dominant phase",
        "reason": reason,
        "gate_records": records,
        "records": gate_records,
    }
    OUTPUT.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
