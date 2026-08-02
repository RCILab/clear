# IMPC-DR common-unicycle validation

The public double-integrator path remains unchanged.  The comparison adapter
uses state `[x,y,v,theta]`, input `[a,omega]`, nonlinear plant execution, a
24-step horizon at 0.03 s, and two sequential affine solves per replan.

## Regression checks

- The affine model is exact at every linearization point.
- Condensed and recursive affine rollouts agree to `1e-12`.
- The original SMGLib constraint assembly remains numerically identical.
- The terminal warning-band/deadlock coefficient is active in the adapter.
- A stopped dynamic robot retains the full pair clearance and is not
  classified as a stationary wall point.
- N=2/4/8 solver smoke tests satisfy the input, speed, terminal, and pair
  constraints.

## Doorway pilot

The final Doorway 1.2 m, N=8, 10 s result is stored in
`impc_unicycle_smg_doorway_n8_10s_final.json`.

- minimum physical pair distance: 0.440200 m;
- minimum physical obstacle clearance: 0.000100 m;
- maximum yaw rate: 1.570796327 rad/s;
- five of eight robots cleared the doorway resource by 10 s;
- mean full decentralized replan wall time: 299.7 ms.

The robots have not yet reached the far goals at 10 s, so this pilot is not a
task-completion result.  Returned solutions are explicitly audited.  An
infeasible or numerically invalid solve follows the shifted last feasible
nonlinear plan.

## Scale probe

All first-step problems were feasible:

| Scenario | N | Replan wall time |
|---|---:|---:|
| Doorway 1.2 m | 8 | 203.9 ms |
| Doorway 1.2 m | 16 | 503.4 ms |
| Intersection 2.4 m | 8 | 189.5 ms |
| Intersection 2.4 m | 16 | 426.4 ms |

These timings show that the adapter is suitable as an MPC comparison but not
a 33.3 Hz real-time baseline at N=8 or N=16.  Long-horizon completion runs
must report this computational result rather than hiding it with a different
table label.

