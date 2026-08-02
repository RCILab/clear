"""Common schema and artifact helpers for architecture-aware timing."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Iterable

import numpy as np


SCHEMA_VERSION = "timing-v2.1"
REQUIRED_RECORD_FIELDS = (
    "batch_step_ms",
    "shared_coordination_ms",
    "local_unit_ms",
    "critical_path_ms",
    "controller_backend",
    "worker_count",
    "cpu_model",
    "gpu_model",
    "warmup_steps",
    "scenario_fingerprint",
)


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unreported"


def gpu_model() -> str:
    value = os.environ.get("TIMING_V2_GPU_MODEL")
    return value if value else "none"


def stats_ms(values: Iterable[float]) -> dict:
    samples = np.asarray(list(values), dtype=float)
    if samples.size == 0:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
            "sample_count": 0,
        }
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
        "maximum": float(np.max(samples)),
        "sample_count": int(samples.size),
    }


def aggregate_record(
    *,
    batch_step_ms: Iterable[float],
    shared_coordination_ms: Iterable[float],
    local_unit_ms: Iterable[Iterable[float]],
    critical_path_ms: Iterable[float],
    controller_backend: str,
    worker_count: int,
    warmup_steps: int,
    scenario_fingerprint: str,
    cpu: str | None = None,
    gpu: str | None = None,
    batch_contains_local_units: bool = True,
    **metadata,
) -> dict:
    batch = list(batch_step_ms)
    shared = list(shared_coordination_ms)
    local = [list(values) for values in local_unit_ms]
    critical = list(critical_path_ms)
    count = len(batch)
    if not (
        len(shared) == count
        and len(local) == count
        and len(critical) == count
    ):
        raise ValueError("timing streams must have equal step counts")
    for index, (batch_value, shared_value, local_values, critical_value) in enumerate(
        zip(batch, shared, local, critical)
    ):
        expected = shared_value + (max(local_values) if local_values else 0.0)
        if abs(expected - critical_value) > max(1.0e-6, 1.0e-6 * expected):
            raise ValueError(
                f"critical-path identity failed at step {index}: "
                f"{critical_value} != {expected}"
            )
        if (
            batch_contains_local_units
            and batch_value + 1.0e-6 < critical_value
        ):
            raise ValueError(
                f"batch latency is below critical path at step {index}"
            )
    record = {
        **metadata,
        "batch_step_ms": stats_ms(batch),
        "shared_coordination_ms": stats_ms(shared),
        "local_unit_ms": {
            "per_step_max": stats_ms(
                max(values) if values else 0.0 for values in local
            ),
            "per_call": stats_ms(
                value for values in local for value in values
            ),
            "unit_count": stats_ms(len(values) for values in local),
        },
        "critical_path_ms": stats_ms(critical),
        "controller_backend": controller_backend,
        "worker_count": int(worker_count),
        "cpu_model": cpu if cpu is not None else cpu_model(),
        "gpu_model": gpu if gpu is not None else gpu_model(),
        "warmup_steps": int(warmup_steps),
        "scenario_fingerprint": scenario_fingerprint,
        "batch_contains_local_units": bool(batch_contains_local_units),
    }
    validate_record(record)
    return record


def validate_record(record: dict) -> None:
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise ValueError(f"timing record missing fields: {missing}")
    if (
        record["batch_step_ms"]["sample_count"]
        != record["critical_path_ms"]["sample_count"]
    ):
        raise ValueError("batch and critical-path sample counts differ")


def save_samples(
    path: Path,
    *,
    batch_step_ms: Iterable[float],
    shared_coordination_ms: Iterable[float],
    local_unit_ms: Iterable[Iterable[float]],
    critical_path_ms: Iterable[float],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    local = [np.asarray(values, dtype=np.float64) for values in local_unit_ms]
    offsets = np.zeros(len(local) + 1, dtype=np.int64)
    for index, values in enumerate(local):
        offsets[index + 1] = offsets[index] + len(values)
    flattened = (
        np.concatenate(local)
        if local and offsets[-1]
        else np.zeros(0, dtype=np.float64)
    )
    np.savez_compressed(
        path,
        batch_step_ms=np.asarray(list(batch_step_ms), dtype=np.float64),
        shared_coordination_ms=np.asarray(
            list(shared_coordination_ms),
            dtype=np.float64,
        ),
        local_unit_ms=flattened,
        local_unit_offsets=offsets,
        critical_path_ms=np.asarray(
            list(critical_path_ms),
            dtype=np.float64,
        ),
    )
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_payload(path: Path, records: list[dict], **metadata) -> None:
    for record in records:
        validate_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **metadata,
        "records": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
