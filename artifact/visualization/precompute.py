"""Precompute a static run bank for GitHub Pages deployment.

Each run is produced by the same ``generate_run`` used by the live server,
so the deployed page replays genuine canonical-pipeline trajectories.

Usage (inside Docker):
    docker run --rm -v "${PWD}:/work" -w /work clear-nav \
        python visualization/precompute.py --robots 20 40 60 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from server import FAMILIES, generate_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--robots", nargs="+", type=int, default=[20, 40, 60])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest: list[dict] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    def has_entry(family: str, n: int, seed: int) -> bool:
        return any(
            e["family"] == family and e["n"] == n and e["seed"] == seed
            for e in manifest
        )

    jobs = [
        (family, n, seed)
        for family in args.families
        for n in args.robots
        for seed in args.seeds
    ]
    for index, (family, n, seed) in enumerate(jobs, start=1):
        name = f"{family}_n{n}_s{seed}.json"
        target = args.output_dir / name
        if target.exists() and has_entry(family, n, seed) and not args.force:
            print(f"[{index}/{len(jobs)}] skip {name} (exists)")
            continue
        started = time.time()
        print(f"[{index}/{len(jobs)}] run  {name} ...", flush=True)
        run = generate_run(family, n, seed)
        target.write_text(json.dumps(run))
        manifest = [
            e
            for e in manifest
            if not (
                e["family"] == family and e["n"] == n and e["seed"] == seed
            )
        ]
        manifest.append(
            {
                "family": family,
                "n": n,
                "seed": seed,
                "file": name,
                "completed": run["completed"],
                "makespan": run["makespan"],
            }
        )
        manifest.sort(key=lambda e: (e["family"], e["n"], e["seed"]))
        manifest_path.write_text(json.dumps(manifest, indent=1))
        size_mb = target.stat().st_size / 1e6
        print(
            f"          done in {time.time() - started:.1f}s, "
            f"{size_mb:.2f} MB, completed={run['completed']}"
        )
    print(f"bank: {len(manifest)} runs in {args.output_dir}")


if __name__ == "__main__":
    main()
