"""Headless smoke test for the streaming endpoint logic."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import stream_run  # noqa: E402


def main() -> None:
    family = sys.argv[1] if len(sys.argv) > 1 else "swap"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    started = time.time()
    events: list[dict] = []
    first_seen: dict[str, float] = {}

    def emit(obj: dict) -> None:
        events.append(obj)
        first_seen.setdefault(obj["type"], time.time() - started)

    stream_run(family, n, seed, emit)
    frames = [e for e in events if e["type"] == "frame"]
    done = events[-1]
    print(
        f"meta at {first_seen['meta']:.2f}s, "
        f"first frame at {first_seen['frame']:.2f}s, "
        f"{len(frames)} frames, last t={frames[-1]['t']:.2f}s"
    )
    print(
        f"done: completed={done['completed']} makespan={done['makespan']} "
        f"min_pair={done['min_pair']:.4f}"
    )
    wall = time.time() - started
    print(
        f"wall {wall:.1f}s, mean compute speed "
        f"{frames[-1]['t'] / wall:.2f}x real-time"
    )


if __name__ == "__main__":
    main()
