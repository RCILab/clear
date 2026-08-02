"""Hash and validate every retained timing-v2 JSON/NPZ artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DIRECTORIES = (
    ROOT / "validation/timing_v2",
    ROOT / "baselines/mgr-harness/results/timing_v2",
    ROOT / "baselines/comparison-harness/results/timing_v2",
)
OUTPUT = ROOT / "validation/timing_v2/timing_v2_manifest.json"


def validate_npz(path: Path) -> dict:
    with np.load(path) as data:
        batch = np.asarray(data["batch_step_ms"], dtype=float)
        shared = np.asarray(data["shared_coordination_ms"], dtype=float)
        critical = np.asarray(data["critical_path_ms"], dtype=float)
        if not (len(batch) == len(shared) == len(critical)):
            raise RuntimeError(f"sample count mismatch: {path}")
        if not np.all(
            np.isfinite(np.concatenate((batch, shared, critical)))
        ):
            raise RuntimeError(f"nonfinite timing sample: {path}")
        if "local_unit_offsets" in data:
            flattened = np.asarray(data["local_unit_ms"], dtype=float)
            offsets = np.asarray(data["local_unit_offsets"], dtype=int)
            if len(offsets) != len(batch) + 1:
                raise RuntimeError(f"local offset mismatch: {path}")
            local_max = np.asarray(
                [
                    np.max(flattened[offsets[index] : offsets[index + 1]])
                    if offsets[index + 1] > offsets[index]
                    else 0.0
                    for index in range(len(batch))
                ]
            )
            local_count = int(len(flattened))
        else:
            local = np.asarray(data["local_unit_ms"], dtype=float)
            if local.ndim != 2 or local.shape[0] != len(batch):
                raise RuntimeError(f"local matrix mismatch: {path}")
            local_max = np.max(local, axis=1)
            local_count = int(local.size)
        if not np.allclose(
            critical,
            shared + local_max,
            atol=1.0e-6,
            rtol=1.0e-6,
        ):
            raise RuntimeError(f"critical-path identity failed: {path}")
        return {
            "step_samples": int(len(batch)),
            "local_samples": local_count,
        }


def main() -> None:
    entries = []
    npz_steps = 0
    npz_local = 0
    for directory in DIRECTORIES:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            comparison_smoke_sample = (
                directory
                == ROOT
                / "baselines/comparison-harness/results/timing_v2"
                and path.parent.name == "samples"
                and "_s99" in path.stem
            )
            if (
                not path.is_file()
                or path == OUTPUT
                or "smoke" in path.as_posix().lower()
                or comparison_smoke_sample
                or path.suffix not in (".json", ".jsonl", ".npz", ".md")
            ):
                continue
            metadata = {}
            if path.suffix == ".npz":
                metadata = validate_npz(path)
                npz_steps += metadata["step_samples"]
                npz_local += metadata["local_samples"]
            entries.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                    **metadata,
                }
            )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "npz_file_count": sum(
            entry["path"].endswith(".npz") for entry in entries
        ),
        "npz_step_samples": npz_steps,
        "npz_local_samples": npz_local,
        "files": entries,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
