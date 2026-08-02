# Runtime theorem and native-unicycle audit

## Exact StraightBridge theorem domain

The exact local theorem-domain audit contains 60 runs: component sizes
2, 4, and 8 over 20 seeds each.

- 60/60 make the required longitudinal exit.
- 3,780 theorem-domain steps are certified.
- There are zero progress-bound, projection-axis, static-certificate, or
  token-flip violations.
- The largest observed exit-time / certified-bound ratio is 0.2333.

The perturbation suites intentionally test the antecedent boundary. All 30
guide-rotation and all 20 curved-boundary runs exit longitudinally, but none
is counted as a theorem-domain run because the exact straight,
progress-aligned antecedents do not hold.

## StraightBridge scale audit used in the paper

The paper-facing scale audit contains 20 deterministic seeds at each of
(N=8) and (N=16). All 40 continuous-planar trials make the required
longitudinal exit. Across 2,520 theorem-domain steps, there are zero
progress-bound, projection-axis, static-certificate, or token-flip
violations.

The paired bounded-unicycle audit uses the common 30 ms outer update, three
native projections, and a 10 s horizon. All 40 missions arrive and pass the
physical-clearance and sampled-command audit. All 19,953 requested native
progress rows are accepted, with zero row rejections, progress violations,
nonconverged projections, or feasibility restorations. Mean makespan is
4.743 s at both sizes; the minimum transferred pair lower bound is
0.447075 m and the minimum physical obstacle-clearance residual is
0.062612 m.

## Native bounded-unicycle transfer

Every row uses ell = 0.05 m, |v| <= 0.8 m/s,
|omega| <= pi/2 rad/s, three native projections per 33.3 Hz outer period,
and the same native
OSQP CBF/input-box formulation as the 60 s main study. The four-robot bridge
retains its separate 10 s task horizon.

| Task | Success | Mean successful time (s) | Min physical pair (m) | Transferred pair lower bound (m) | Min virtual obstacle clearance (m) |
|:---|---:|---:|---:|---:|---:|
| Free N=20 | 20/20 | 21.32 | 0.470438 | 0.440732 | 0.003103 |
| Swap N=20 | 20/20 | 21.47 | 0.515625 | 0.440000 | 0.230071 |
| Circ15 N=20 | 20/20 | 22.77 | 0.464683 | 0.440001 | 6.65e-6 |
| Rect15 N=20 | 20/20 | 22.66 | 0.467561 | 0.440000 | -3.50e-10 |
| StraightBridge N=4 | 18/20 | 4.74 | 0.547420 | 0.447420 | 0.013240 |

All 80 benchmark runs have zero primal-infeasible steps and preserve the
physical and transferred safety bounds. Sparse OSQP reaches its configured
iteration limit in 7 individual projection calls across the N=20 benchmark
rows, but their primal residuals already satisfy the 1e-6 certification
tolerance; the iteration-limit count is therefore reported separately from
feasibility.

StraightBridge seeds 3 and 11 make the certified longitudinal exit but finish
just outside the separate physical arrival ball at the 10 s task timeout.
Thus the theorem-domain exit result is 60/60 while the task-level unicycle
row is 18/20.

The native bridge progress row is active over 3,342 outer samples and is
accepted in all 10,026 inner projections.  There are zero row rejections,
progress-bound violations, and nonconverged native projections.  This audit
supports the controller-specific bounded-input finite-exit corollary for the
exact bridge stratum; it does not claim global bounded-input liveness outside
that stratum.
