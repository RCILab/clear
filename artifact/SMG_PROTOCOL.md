# Social Mini-Game protocol

This extension uses the same physical robots and controller limits as the
main evaluation: radius 0.20 m, safety margin 0.02 m, speed at most 0.80 m/s,
yaw rate at most pi/2 rad/s, 0.03 s control period, and 60 s horizon.  Every
reported method must execute bounded unicycle kinematics to enter the common
comparison table.
All robots start at yaw 0, matching the main benchmark protocol.

## Geometry

| Scenario | Parameter | Main value | Role |
|---|---:|---:|---|
| Doorway | wall thickness | 0.80 m | short, thick bottleneck |
| Doorway-tight | opening width | 0.80 m | one safety lane; stress/limit case |
| Doorway-medium | opening width | 1.20 m | two safety lanes; primary comparison |
| Intersection-standard | corridor width | 2.40 m | six robot diameters, close to SMGLib's relative scale |
| Intersection-tight | corridor width | 1.20 m | optional stress/limit case |

The 0.8 m doorway has only 0.06 m nominal margin over the recorded 0.74 m
local-model requirement of the affected baseline.  Any table containing that
condition must state this fact.  It is not the sole doorway comparison; the
1.2 m condition is reported beside it.

Doorway approaches use one queue lane at 0.8 m and two at 1.2 m.  Starts are
at least the common pair clearance plus 0.10 m apart.  Goals lie in a
post-resource dispersal region, with the leading robot assigned the farthest
goal.  This prevents a robot that has already completed the bottleneck from
parking in front of a follower.  Intersection goals preserve the same queue
order.

The canonical sizes are N=8 and N=16, with seeds 0--19.  Seed 0 retains the
base layout.  The other seeds vary paired queue gaps, approach asymmetry, and
goal assignment without reducing the initial 0.10 m pair margin or changing
the shared resource geometry.  This yields 20 distinct fingerprints per
scenario-size row, and every method receives the identical fingerprint.

## Metrics

In addition to mission success, arrival fraction, makespan, safety, and
controller time, the SMG extension reports:

1. **Parallel throughput:** `|S| / (t_end - t_start)` robots/s, where
   `t_start` is the first assigned robot's resource entry and `t_end` is the
   last assigned robot's exit beyond the post-resource boundary.
2. **SMG flow rate:** parallel throughput divided by the physical opening or
   corridor width, in robots/(m s).
3. **Interference delay:** for every agent,
   `D_i = TTG_multi_i - TTG_solo_i`.  The solo counterfactual reruns the same
   method with the same dynamics, map, start, and goal.  Mean, p95, and the
   mean normalized by solo TTG are stored.

## Claim boundary

Doorway and intersection are empirical SMG scenarios.  A doorway is
geometrically suggestive of a short straight bridge, but Theorem 3 is invoked
only for time steps where the existing straight-bridge auditor verifies every
antecedent and the native projection accepts its certified progress row.
Otherwise no theorem claim is made.  In particular, a nonzero
`certified_bridge_steps` count is necessary before discussing local theorem
coverage.

## Baselines

`Vanilla CBF-QP` uses the identical cost-field guidance, geometric CBF rows,
native bounded-unicycle projection, and numerical tolerances as CLEAR.  It
removes circulation, rigid-component motion, event-held tokens, and every
deadlock-resolution term.

The IMPC-DR comparison uses the validated adapter with state
`[x,y,v,theta]`, input `[a,omega]`, nonlinear unicycle execution, and two
sequential affine solves.  MBVC, warning-band, terminal-stop, and
deadlock-resolution mechanisms are retained.  Returned solutions are audited
before execution, and an invalid solve follows the shifted last feasible
nonlinear plan.  The original double-integrator proof is not asserted for
this adapter.

The reported method set is CLEAR, MGR, ORCA, NH-ORCA, GCBF+, Vanilla CBF-QP,
and IMPC-DR.  A missing large-scale IMPC-DR row is recorded as a computational
limit rather than produced by lowering the common 33.3 Hz control protocol.
