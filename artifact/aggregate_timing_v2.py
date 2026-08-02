"""Aggregate timing-v2 shards into the paper-facing summary and report."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCES = (
    (
        "CLEAR",
        ROOT
        / "validation/timing_v2/timing_v2_clear_rect15.json",
    ),
    (
        "Vanilla CBF-QP",
        ROOT
        / "validation/timing_v2/timing_v2_vanilla_rect15.json",
    ),
    (
        "MGR",
        ROOT
        / "baselines/mgr-harness/results/timing_v2/"
        "timing_v2_mgr_rect15.json",
    ),
    (
        "ORCA",
        ROOT
        / "baselines/comparison-harness/results/timing_v2/"
        "timing_v2_orca_rect15.json",
    ),
    (
        "NH-ORCA",
        ROOT
        / "baselines/comparison-harness/results/timing_v2/"
        "timing_v2_nhorca_rect15.json",
    ),
    (
        "GCBF+ CPU",
        ROOT
        / "baselines/comparison-harness/results/timing_v2/"
        "timing_v2_gcbfplus_cpu_rect15.json",
    ),
    (
        "GCBF+ GPU",
        ROOT
        / "baselines/comparison-harness/results/timing_v2/"
        "timing_v2_gcbfplus_gpu_rect15.json",
    ),
    (
        "IMPC-DR",
        ROOT
        / "validation/timing_v2/"
        "timing_v2_impc_scale_gate.json",
    ),
)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def aggregate_group(records: list[dict]) -> dict:
    seeds = sorted(int(record["seed"]) for record in records)
    if len(seeds) != 3 or seeds != [0, 7, 13]:
        raise RuntimeError(f"expected seeds 0/7/13, found {seeds}")

    def aggregate_field(field: str) -> dict:
        return {
            "mean_of_seed_means_ms": float(
                np.mean([record[field]["mean"] for record in records])
            ),
            "mean_of_seed_medians_ms": float(
                np.mean([record[field]["median"] for record in records])
            ),
            "worst_seed_p95_ms": float(
                max(record[field]["p95"] for record in records)
            ),
            "maximum_ms": float(
                max(record[field]["maximum"] for record in records)
            ),
            "sample_count": int(
                sum(
                    record[field]["sample_count"]
                    for record in records
                )
            ),
        }

    fingerprints = [record["scenario_fingerprint"] for record in records]
    if len(set(fingerprints)) != len(fingerprints):
        raise RuntimeError("duplicate scenario fingerprint in timing group")
    return {
        "seeds": seeds,
        "scenario_fingerprints": fingerprints,
        "batch": aggregate_field("batch_step_ms"),
        "critical_path": aggregate_field("critical_path_ms"),
        "controller_backend": sorted(
            {record["controller_backend"] for record in records}
        ),
        "cpu_model": sorted({record["cpu_model"] for record in records}),
        "gpu_model": sorted({record["gpu_model"] for record in records}),
        "worker_count": sorted(
            {int(record["worker_count"]) for record in records}
        ),
        "warmup_steps": sorted(
            {int(record["warmup_steps"]) for record in records}
        ),
    }


def display(summary: dict | None) -> str:
    if summary is None:
        return "not run: scale gate"
    mean = summary["mean_of_seed_means_ms"]
    p95 = summary["worst_seed_p95_ms"]
    return f"{mean:.3f} / {p95:.3f}"


def main() -> None:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    not_run: dict[tuple[str, int], dict] = {}
    source_hashes = {}
    import hashlib

    for method, path in SOURCES:
        records = load_records(path)
        if not records:
            continue
        source_hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for record in records:
            count = int(record.get("n_robots", record.get("num_agents")))
            if record.get("status") == "not_run_scale_gate":
                not_run[(method, count)] = record
            else:
                groups[(method, count)].append(record)

    summaries = []
    for (method, count), records in sorted(groups.items()):
        summary = aggregate_group(records)
        summaries.append(
            {
                "method": method,
                "n_robots": count,
                **summary,
            }
        )
    lookup = {
        (item["method"], item["n_robots"]): item
        for item in summaries
    }
    payload = {
        "schema_version": "timing-v2.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "family": "rect15",
            "dt_s": 0.03,
            "horizon_s": 60.0,
            "seeds": [0, 7, 13],
            "warmup_steps": 40,
            "blas_threads": 1,
        },
        "aggregation": (
            "mean is the mean of seed means; p95 is the largest seed p95"
        ),
        "source_sha256": source_hashes,
        "groups": summaries,
        "not_run_scale_gate": list(not_run.values()),
    }
    output_dir = ROOT / "validation/timing_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "timing_v2_summary.json"
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    methods = [
        "CLEAR",
        "Vanilla CBF-QP",
        "MGR",
        "ORCA",
        "NH-ORCA",
        "GCBF+ CPU",
        "GCBF+ GPU",
        "IMPC-DR",
    ]
    lines = [
        "# Architecture-aware timing v2",
        "",
        "Rect15, 60 s, dt=0.03 s, seeds 0/7/13, 40 warm-up "
        "steps, one timing container at a time, and one BLAS/OpenMP "
        "thread. Entries are mean of seed means / worst-seed p95 in ms.",
        "",
        "## Single-machine batch latency",
        "",
        "| N | " + " | ".join(methods) + " |",
        "|" + "---:|" + "---:|" * len(methods),
    ]
    for count in (20, 40, 60, 80):
        cells = []
        for method in methods:
            item = lookup.get((method, count))
            cells.append(
                display(item["batch"] if item is not None else None)
            )
        lines.append(f"| {count} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Deployment critical-path latency",
            "",
            "| N | " + " | ".join(methods) + " |",
            "|" + "---:|" + "---:|" * len(methods),
        ]
    )
    for count in (20, 40, 60, 80):
        cells = []
        for method in methods:
            if method == "GCBF+ GPU":
                cells.append("batch only")
                continue
            item = lookup.get((method, count))
            cells.append(
                display(
                    item["critical_path"] if item is not None else None
                )
            )
        lines.append(f"| {count} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "GCBF+ CPU includes validated induced one-hop ego-graph "
            "timing; GCBF+ GPU is the synchronized official full-graph "
            "batch path and therefore has no deployment critical-path "
            "entry. IMPC-DR uses the documented scale gate when its "
            "local critical-path p95 exceeds the 30 ms control period.",
            "",
        ]
    )
    report = output_dir / "TIMING_V2_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(output_json)
    print(report)


if __name__ == "__main__":
    main()
