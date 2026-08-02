# Native bridge progress-row validation

Date: 2026-07-31

Environment:

- Python 3.13.5
- OSQP 1.1.3
- Windows, Intel64 Family 6 Model 183

## Correctness

`python -B -m unittest tests.test_core`

- 34 tests passed.
- The added tests verify that a world-frame progress row is enforced through
  a rotated native input box and that an infeasible rigid witness causes the
  row to be rejected without changing the original native QP.

Twenty four-robot native StraightBridge runs, seeds 0--19:

- 3,342 certified outer bridge samples;
- 10,026 accepted progress-constrained inner projections;
- 0 progress-row rejections;
- 0 certified progress-bound violations;
- 0 nonconverged native projections;
- 18/20 task arrivals, unchanged from the prior audit.  Seeds 3 and 11 exit
  the bridge cell but remain outside the separate physical arrival ball at
  the 10 s timeout.

One `N=20`, seed-0 replay of each main family produced no false activation:

- Free: 0 certified bridge rows;
- Swap: 0;
- Circ15: 0;
- Rect15: 0.

## Full Docker rerun

The complete main matrix was rerun from the final audited code in one
`clear-nav` Docker container (Python 3.11) and promoted directly to the
canonical main artifacts:

- four task families (`free`, `swap`, `circ15`, and `rect15`);
- team sizes 20, 40, 60, and 80;
- seeds 0--19;
- 320 unique `(family, N, seed)` records and 320 unique fingerprints;
- 310/320 physical mission successes (96.875%);
- zero changes in controller behavior, mission outcome, makespan, or safety
  metrics relative to the pre-audit canonical records;
- 297 nonsolved optimizer calls in 83 missions;
- 270 feasibility restorations in 79 missions: 46 actuator contractions,
  0 certified-witness restorations, 224 common contractions, and 0 HQP
  fallbacks;
- 0 final command-infeasible steps;
- minimum physical pair distance 0.440000836 m;
- minimum physical obstacle clearance \(7.4389\times10^{-6}\) m.

The exact straight-bridge detector does not activate in these four general
benchmark families, so the certified bridge fields remain zero as expected;
the exact theorem domain is exercised separately above.

The corrected state-wise audit reports 638,745 positive-witness candidates.
It excludes 103 samples because a solver limit or feasibility restoration
means that the exact-projection premise is not satisfied.  All 638,642
applicable samples have zero conclusion violations.

Earlier bridge-progress artifacts called 31 samples violations: 30 in Rect15
at \(N=80\), seed 18, and one in Swap at \(N=80\), seed 15.  Those samples
were solver-limited exact-projection premise failures, not violations of the
theorem conclusion.  They are now contained in the explicit exclusion count;
the historical claim is not retained as a current result.

Artifacts:

- `results/unicycle_clear_all_sizes_raw.json`
  (SHA-256
  `E2606D1DE06CD01C98286DF683C448CB2470B76C37B883FD68FCA5088E6F2FB0`);
- `results/unicycle_clear_all_sizes_summary.json`
  (SHA-256
  `04E9AF03A2BC5AFC1598B856D703FD13CD2E6CBC3004D60248DC7E38833DCA5A`).

## Timing

Persistent-workspace microbenchmark at one fixed 12-robot bridge state,
after 30 warm-up calls and over 500 calls per variant:

| Variant | Mean native projection | p95 native projection |
|---|---:|---:|
| Original native QP | 0.467 ms | 0.697 ms |
| One certified row | 0.571 ms | 0.711 ms |

The mean per-projection increase was 0.105 ms.  The row is applied at three
inner nodes per 30 ms outer control step.

Seven complete eight-second 12-robot bridge replays per variant:

| Variant | Median replay time | Approx. time per 30 ms step |
|---|---:|---:|
| Certified row disabled | 690.45 ms | 2.60 ms |
| Certified row enabled | 804.03 ms | 3.02 ms |

The median end-to-end increase is approximately 0.43 ms per outer step,
or 1.4% of the 30 ms control-period budget.  These are local development
measurements, not replacements for the paper's controlled timing protocol.
