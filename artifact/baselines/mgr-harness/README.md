# Public MGR evaluation and optimization harness

This directory provides an isolated headless runtime for the publicly released
MGR implementation. The checkout contains behavior-preserving geometry and
simulator optimizations while retaining the public CVXOPT numerical path.

- Upstream: `https://github.com/Aiden-wjLee/merry-go-round`
- License: Apache-2.0 (see the upstream `LICENSE` and `NOTICE`)
- Official protocol: `dt=0.05 s`, `120 s` timeout, `0.10 m` arrival radius

Build from the `baselines` directory:

```powershell
docker build -f mgr-harness/Dockerfile.official -t mgr-official:84aaefd .
```

The official GUI runner can execute without a display using Qt's offscreen
backend:

```powershell
docker run --rm `
  --mount "type=bind,source=<absolute-path-to-merry-go-round>,target=/mgr" `
  mgr-official:84aaefd `
  python main.py --instance <instance-path>
```

Outputs are written by the upstream code to `experiment_result/` and
`Video_MGR/` inside the checkout.

For repeated trials, run the separate headless evaluator. It imports the
official controller/simulator and preserves the update order in
`MainWindow.update_simulation`, but skips Qt timing, rendering, video, and
spreadsheet export:

```powershell
docker run --rm `
  --mount "type=bind,source=<absolute-path-to-merry-go-round>,target=/mgr,readonly" `
  --mount "type=bind,source=<absolute-path-to-mgr-harness>,target=/harness" `
  mgr-official:84aaefd `
  python /harness/run_headless.py `
    --instance instances/Free_maps/agents20/Free_0_20_0.yaml
```

Each trial emits one JSON object with the official arrival metrics plus
evaluator-side minimum robot, obstacle, and workspace clearances. A negative
physical clearance is counted as a collision.

For a matched-protocol run, add:

```text
--dt 0.03 --timeout 60 --arrival-threshold 0.22 --termination-mode mission
```

`official` termination preserves upstream time labels and early stopping.
`horizon` termination runs exactly `round(timeout/dt)` steps, reports true
first-entry times separately from the upstream re-entry-overwriting metric,
and evaluates arrivals at the final horizon. Safety minima are sampled at
control nodes; physical collision and violation of MGR's declared
`SAFE_RATIO=1.1` margin are distinct fields.
The paper comparison uses `mission` termination: it stops at the first
simultaneous all-arrived state or at the common finite horizon.

## Canonical 320-mission pairing

The paper's canonical CLEAR scenarios are exported from their deterministic
generator, checked against all retained fingerprints, converted to MGR's YAML
coordinate convention, and checked again after the YAML round trip:

```powershell
docker run --rm -v "<experiments>:/work" -w /work clear-nav:paired `
  python export_mgr_pairing_instances.py
```

Run the public MGR implementation in resumable shards:

```powershell
docker run --rm `
  --mount "type=bind,source=<merry-go-round>,target=/mgr,readonly" `
  --mount "type=bind,source=<mgr-harness>,target=/harness" `
  --mount "type=bind,source=<mgr_paired_instances>,target=/paired,readonly" `
  mgr-official:84aaefd `
  python /harness/run_batch.py --manifest /paired/manifest.json `
    --output /harness/results/mgr_clear320_final_shard0of6.jsonl `
    --shard-index 0 --shard-count 6
```

The batch defaults implement the common protocol (`dt=0.03 s`, `60 s`,
`0.22 m`) and stop at the first simultaneous all-arrived state, matching the
finite-horizon mission definition. Every record includes a scenario
fingerprint, YAML checksum, node and exact swept safety minima, solver
diagnostics, and controller-loop timing. Existing successful JSONL records are
skipped when a shard is resumed.

After every shard completes, validate the one-to-one fingerprint pairing and
generate paired bootstrap intervals, the exact McNemar success test, and the
exact paired arrival-time sign test:

```powershell
python aggregate_clear320.py
```

The long corrected-safety run can be stopped at any time because each record
is flushed immediately. Resume existing containers (or create missing ones)
from the workspace with:

```powershell
.\baselines\mgr-harness\resume_clear320.ps1
.\baselines\mgr-harness\status_clear320.ps1
```

`run_batch.py` reads the existing JSONL and skips every completed scenario
fingerprint. The final aggregation intentionally reads only
`mgr_clear320_final*.jsonl`.

The computationally optimized path uses cached/vectorized geometry and
communication operations, constant QP-data reuse, and evaluator pruning. Its
exact trace-equivalence audit and the rejected alternative-solver experiment
are documented in `results/OPTIMIZATION_EQUIVALENCE.md`.

For a low-contention timing audit across all four families and team sizes, run
`run_mgr_timing_audit.ps1` and then `aggregate_mgr_timing.py`.

The architecture-aware Rect15 audit additionally passes
`--timing-v2-dir /harness/results/timing_v2/samples`.  `run_headless.py`
records `sim.update()` as shared communication/circle coordination and each
robot's `update_escape_status()+decide_velocity()` call as a local unit.
The serial batch is their sum; the deployment path is the shared phase plus
the slowest local unit.  After the 12 N=20/40/60/80, seed 0/7/13 JSONL
records finish, run `aggregate_mgr_timing_v2.py` from this directory to
validate the matrix and emit `results/timing_v2/timing_v2_mgr_rect15.json`.
