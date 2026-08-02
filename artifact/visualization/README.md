# CLEAR visualization

This directory contains the live viewer and the static-run generator for the
project-page playground. Both call the same `clear_nav` implementation used by
the paper artifact. The live server uses the reported 30 ms passage update and
three native projection substeps, and is capped at N=60 to keep browser-facing
latency practical.

From `artifact/`:

```bash
python visualization/server.py --port 8765
```

Open `http://localhost:8765`. To regenerate a static replay bank:

```bash
python visualization/precompute.py --robots 20 40 60 --seeds 0 1 2 \
  --output-dir ../playground/runs --force
```

The viewer is qualitative. Paper completion, clearance, and timing values are
computed by the experiment and validation entry points, not by browser replay.
