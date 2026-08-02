"""Resumable sharded runner for the 320-scenario CLEAR/MGR pairing manifest."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import sys
import traceback

from run_headless import configure_imports, run_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/mgr"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--completed-glob",
        action="append",
        default=[],
        help=(
            "Additional JSONL glob whose successful fingerprints are skipped. "
            "Repeatable; useful when repartitioning an interrupted run."
        ),
    )
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--robots", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--arrival-threshold", type=float, default=0.22)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--safe-ratio", type=float, default=1.1)
    parser.add_argument("--metric-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--env-scale", type=float, default=16.0)
    parser.add_argument("--sim-scale", type=int, default=600)
    parser.add_argument(
        "--smg-resource",
        choices=("none", "doorway", "intersection"),
        default="none",
    )
    parser.add_argument("--smg-resource-width", type=float, default=1.2)
    parser.add_argument("--smg-doorway-thickness", type=float, default=0.8)
    parser.add_argument(
        "--termination-mode",
        choices=("official", "mission", "horizon"),
        default="mission",
    )
    parser.add_argument(
        "--audit-pruning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable semantics-preserving evaluator-side safety pruning.",
    )
    parser.add_argument(
        "--trace-digest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record full state/control SHA-256 digests in each record.",
    )
    parser.add_argument(
        "--timing-warmup-steps",
        type=int,
        default=40,
        help="Initial controller iterations excluded from mean/p95 timing.",
    )
    parser.add_argument(
        "--timing-v2-dir",
        type=Path,
        help="Optional directory for architecture-aware raw NPZ samples.",
    )
    args = parser.parse_args()
    args.trace_npz = None
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must lie in [0, shard-count)")
    return args


def upstream_commit(repo: Path) -> str:
    git_dir = repo / ".git"
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text(encoding="ascii").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="ascii").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            commit, name = line.split(" ", 1)
            if name == reference:
                return commit
    raise RuntimeError(f"cannot resolve upstream Git reference {reference}")


def selected(args: argparse.Namespace, record: dict) -> bool:
    return (
        (args.families is None or record["family"] in args.families)
        and (args.robots is None or record["n_robots"] in args.robots)
        and (args.seeds is None or record["seed"] in args.seeds)
    )


def completed_fingerprints(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid JSONL at {path}:{line_number}"
            ) from error
        if "error" not in record:
            completed.add(record["scenario_fingerprint"])
    return completed


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.parent.resolve()
    candidates = [
        record
        for record in manifest["records"]
        if selected(args, record)
    ]
    candidates = [
        record
        for index, record in enumerate(candidates)
        if index % args.shard_count == args.shard_index
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_fingerprints(args.output)
    for pattern in args.completed_glob:
        for path in sorted(glob.glob(pattern)):
            completed.update(completed_fingerprints(Path(path)))
    pending = [
        record
        for record in candidates
        if record["fingerprint"] not in completed
    ]

    (
        repo,
        QPointF,
        detect_env_type,
        load_config,
        scale_coordinates,
        build_simulator,
    ) = configure_imports(args.repo)
    commit = upstream_commit(repo)
    failures = 0
    print(
        f"shard {args.shard_index}/{args.shard_count}: "
        f"{len(completed)} complete, {len(pending)} pending",
        flush=True,
    )
    with args.output.open("a", encoding="utf-8") as output:
        for run_index, manifest_record in enumerate(pending, start=1):
            yaml_path = (base / manifest_record["relative_yaml"]).resolve()
            actual_sha = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
            if actual_sha != manifest_record["yaml_sha256"]:
                raise RuntimeError(f"YAML checksum mismatch: {yaml_path}")
            try:
                result = run_instance(
                    yaml_path,
                    args,
                    QPointF,
                    detect_env_type,
                    load_config,
                    scale_coordinates,
                    build_simulator,
                )
                result.update(
                    {
                        "algorithm": "MGR",
                        "upstream_commit": commit,
                        "scenario_fingerprint": (
                            manifest_record["fingerprint"]
                        ),
                        "scenario_family": manifest_record["family"],
                        "scenario_seed": manifest_record["seed"],
                        "scenario_yaml_sha256": actual_sha,
                        **{
                            key: manifest_record[key]
                            for key in (
                                "record_kind",
                                "parent_fingerprint",
                                "parent_n_robots",
                                "robot_index",
                            )
                            if key in manifest_record
                        },
                    }
                )
            except Exception as error:
                failures += 1
                result = {
                    "algorithm": "MGR",
                    "upstream_commit": commit,
                    "scenario_fingerprint": manifest_record["fingerprint"],
                    "scenario_family": manifest_record["family"],
                    "num_agents": manifest_record["n_robots"],
                    "scenario_seed": manifest_record["seed"],
                    "scenario_yaml_sha256": actual_sha,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            output.write(
                json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
            )
            output.flush()
            state = "ERROR" if "error" in result else (
                "success" if result["success"] else "failure"
            )
            print(
                f"[{run_index}/{len(pending)}] "
                f"{manifest_record['family']} "
                f"N={manifest_record['n_robots']} "
                f"seed={manifest_record['seed']}: {state}, "
                f"wall={result.get('wall_time_s', float('nan')):.2f}s",
                flush=True,
            )

    if failures:
        print(f"{failures} runs failed with evaluator errors", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
