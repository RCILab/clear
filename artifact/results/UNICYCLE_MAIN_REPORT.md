# CLEAR native bounded-unicycle evaluation

## Protocol

- 20 seeds for each of Free, Swap, Circ15, and Rect15 at
  N = 20, 40, 60, 80 (320 missions).
- First-order look-ahead unicycle kinematics with
  ell = 0.05 m, |v| <= 0.8 m/s, and |omega| <= pi/2 rad/s.
- 33.3 Hz integration, three inner substeps, and a 60 s timeout.
- The explicit feasibility audit uses a 1e-6 tolerance; the internal sparse
  solver termination tolerance is 1e-7.
- One sparse OSQP projection directly enforces pair/boundary CBF rows and the
  bounded unicycle input box in coordinates [v, ell*omega].
- The physical-body audit uses the exact look-ahead transfer inflation.
- Terminal capture uses virtual-point distance to the current guidance target:
  0.60 m in obstacle-free Free and Swap, 0.22 m in Circ15 and Rect15, and a
  common 0.80 m release radius.
  This suppresses only pair circulation; the 0.22 m mission-arrival test and
  the final safety projection are unchanged.

This removes the former component-wide actuation scaling. Structural
components remain part of CLEAR's nominal liveness construction; they no
longer determine a common actuator scale.

## Main result

| Family | N | Arrived + certified | Ever all arrived | Makespan (s) |
|:---|---:|---:|---:|---:|
| Free | 20 | 20/20 | 20/20 | 21.32 +/- 2.25 |
| Free | 40 | 20/20 | 20/20 | 22.94 +/- 1.43 |
| Free | 60 | 20/20 | 20/20 | 25.89 +/- 1.67 |
| Free | 80 | 20/20 | 20/20 | 26.99 +/- 2.21 |
| Swap | 20 | 20/20 | 20/20 | 21.47 +/- 0.10 |
| Swap | 40 | 20/20 | 20/20 | 24.00 +/- 0.34 |
| Swap | 60 | 20/20 | 20/20 | 26.85 +/- 0.60 |
| Swap | 80 | 20/20 | 20/20 | 30.40 +/- 1.24 |
| Circ15 | 20 | 20/20 | 20/20 | 22.77 +/- 2.61 |
| Circ15 | 40 | 18/20 | 18/20 | 26.47 +/- 2.79 |
| Circ15 | 60 | 18/20 | 18/20 | 29.10 +/- 2.84 |
| Circ15 | 80 | 19/20 | 19/20 | 32.15 +/- 4.42 |
| Rect15 | 20 | 20/20 | 20/20 | 22.66 +/- 2.58 |
| Rect15 | 40 | 20/20 | 20/20 | 26.36 +/- 3.78 |
| Rect15 | 60 | 18/20 | 18/20 | 29.47 +/- 3.17 |
| Rect15 | 80 | 17/20 | 17/20 | 33.38 +/- 6.74 |

Overall, 310/320 missions finish with every robot inside the 0.22 m arrival
set at 60 s. All 320 runs preserve the audited physical safety clearances,
and every executed command passes the final feasibility audit. The minimum
physical pair distance is 0.44000084 m and the minimum physical obstacle
clearance is 7.439e-6 m. The transferred virtual-point quantities remain
within the declared 1e-6 feasibility-audit tolerance.

Optimizer termination and final command feasibility are reported separately.
There are 297 nonsolved optimizer calls in 83 missions, but zero
command-infeasible steps. The feasibility audit invokes 270 restorations in
79 missions: 46 actuator contractions, 0 certified-witness restorations,
224 common contractions, and 0 HQP fallbacks. The actuator count here is the
restoration branch, not routine one-ulp inward rounding of an already feasible
input.

The state-wise theorem audit finds 638,745 positive-witness candidates.
It excludes 103 solver- or restoration-limited samples for which the exact
projection premise is not satisfied. All 638,642 applicable samples satisfy
the conclusion, giving zero static-certificate violations.

The 10 liveness failures are confined to clutter: Circ15 misses two seeds at
N=40, two at N=60, and one at N=80; Rect15 misses two seeds at N=60 and three
at N=80. Free and Swap complete in all 160 obstacle-free runs, including
Swap N=80 at 30.40 +/- 1.24 s. The result supports a safety claim and strong
finite-benchmark completion, not global liveness or a paired
performance-dominance claim.

## HQP development result

The optional two-level progress hierarchy is not enabled in the official
matrix. A pure structural-progress priority produced persistent circulation
in Swap N=20, and a complete-nominal progress priority recovered that case but
reached only 63.75% robot arrival in the Swap N=80 seed-0 diagnostic. The
reported controller instead keeps the single native safety projection and
uses the fixed geometry-aware terminal latch above.

## Paired structural-component ablation

The paired ablation removes CLEAR's structural component nominal field while
retaining the same 0.08 m tangent band and native bounded-input CBF projection.

| Family | N | Component-free | CLEAR | Joint-success mean CLEAR - base (s) |
|:---|---:|---:|---:|---:|
| Circ15 | 20 | 20/20 | 20/20 | -0.20 |
| Rect15 | 20 | 19/20 | 20/20 | +0.09 |
| Circ15 | 40 | 17/20 | 18/20 | -0.19 |
| Rect15 | 40 | 18/20 | 20/20 | -0.71 |

CLEAR recovers four component-free failures and introduces no paired
regressions. The recovery traces are Circ15 N=40 seed 12, Rect15 N=20
seed 16, and Rect15 N=40 seeds 3 and 12.

## Computation

The architecture-aware Rect15 rerun separates observed single-worker batch
cost from the component-parallel deployment critical path. At N=80, CLEAR is
33.635 / 39.360 ms in the batch panel and 8.786 / 12.177 ms in the critical
panel (mean / worst-seed p95). Thus the deployment critical path remains
inside the 30 ms period through N=80, while the one-worker batch p95 crosses
it at N=60. MGR shows the same distinction: its N=80 batch p95 is 68.752 ms,
but its agent-parallel critical-path p95 is 4.305 ms.

The older family-wide controller scaling diagnostic remains in
`../validation/unicycle_controller_scaling.md`; the paper-facing,
cross-method architecture-aware table and raw-sample hashes are in
`../validation/timing_v2/TIMING_V2_REPORT.md` and
`../validation/timing_v2/timing_v2_manifest.json`.

## Direct paired official-MGR comparison

On the same 320 scenario fingerprints and common 0.03 s, 60 s, and 0.22 m
protocol, CLEAR completes 310 missions and the pinned official MGR
implementation completes 149. The discordant successes are 161 CLEAR-only
and 0 MGR-only (exact McNemar p=6.842e-49).

Across the 149 joint successes, CLEAR is 10.49 s faster on average
(95% bootstrap CI: 9.09--11.91 s faster), wins 130 paired arrival-time
comparisons, and loses 19 (exact sign p=1.585e-21). With failures assigned the
60 s horizon, the paired mean advantage is 20.79 s
(95% bootstrap CI: 19.37--22.20 s).

All 149 successful MGR missions are physically collision-free. The detailed
group table and raw-source manifest are in
`../baselines/results/MGR_CLEAR320_PAIRED_REPORT.md`.

## Published timing context

Among the 15 task-size rows with a finite published MGR makespan, CLEAR is
faster in 14, with a 43.0% median reduction. Across the 11 interaction-heavy
Swap and clutter rows, the median reduction is 45.3%; Free N=20 is the only
near-tie at 21.32 versus 21.19 s. CLEAR is also faster in all nine rows with
a finite GCBF+ time. ORCA is faster in the four reported Free rows, while
CLEAR is faster in all five reported ORCA Swap or clutter rows. These are
practically meaningful published-value differences, not paired statistical
significance claims, because the scenario generators differ.

## Artifacts

- Canonical raw records: `unicycle_clear_all_sizes_raw.json`
- Main aggregate: `unicycle_clear_all_sizes_summary.json`
- Canonical ablation records: `unicycle_component_free_n20_n40_raw.json`
- Ablation aggregate: `unicycle_component_free_n20_n40_summary.json`
- Paired traces: `unicycle_diagnostics_clear/` and
  `unicycle_diagnostics_component_free/`
- GIF collection: `paper_gifs/`
