# Public-baseline comparison harness

The harness evaluates public navigation implementations on the same canonical
CLEAR scenarios.  The retained protocol is:

- physical unicycle centers with initial heading zero;
- `dt = 0.03 s`, horizon `60 s`;
- body radius `0.20 m`, safety margin `0.02 m`;
- `|v| <= 0.80 m/s`, `|omega| <= pi/2 rad/s`;
- arrival radius `0.22 m`;
- the same starts, goals, obstacles, and continuous swept-motion audit.

The primary task metric is simultaneous all-robot arrival.  Each public
controller stops at its first simultaneous team completion, as the released
MGR runner does.  `Certified completion` additionally requires the common
`0.44 m` pair separation, inflated obstacle clearance, and workspace-boundary
margin to hold under the swept-motion audit.  Keeping these two fields
separate avoids conflating goal-reaching performance with a method's native
safety-margin convention.

The main comparison families are MGR, ORCA/NH-ORCA, and GCBF+.  ORCA and
NH-ORCA use the public RVO2/NH-ORCA implementation with the common static
cost-to-go guidance needed by every local controller in clutter.  The
PathPlanning ORCA repository was a reading reference and is not bundled or
counted as a second independent ORCA algorithm.  SRL-ORCA is discussed in
related work but is not bundled because its ROS/Gazebo policy was not part of
the common large-scale matrix.
The ORCA image fixes OpenMP and BLAS to one internal worker per evaluation
process.
This removes nested-thread oversubscription without changing deterministic
controller outputs and permits independent seeds to be sharded across
containers.

## NH-ORCA image

From `baselines`:

```powershell
docker build -f comparison-harness/Dockerfile.nhorca `
  -t nh-orca-benchmark:latest .
```

The image builds the public pybind11 RVO2 implementation in Release mode.

`run_nhorca.py` exposes both the public NH-ORCA effective-point mapping and a
standard ORCA velocity command tracked by the same bounded unicycle.  Any
tracking padding is selected only on seeds 20--21; the paper test set remains
seeds 0--19.  The global guidance remains common to all methods; robustness
padding is confined to the local collision-avoidance geometry.

The retained outcome settings are:

```powershell
# Repeat with --robots 20, 40, 60, and 80 in separate shards.
python baselines/comparison-harness/run_nhorca.py `
  --mode nh-orca --families free swap circ15 rect15 `
  --robots 20 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --obstacle-inflation 0.10 `
  --output baselines/comparison-harness/results/final_nhorca_n20.json

python baselines/comparison-harness/run_nhorca.py `
  --mode orca --families free swap circ15 rect15 `
  --robots 20 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --heading-gain 4.0 --pair-inflation 0.05 --obstacle-inflation 0.15 `
  --output baselines/comparison-harness/results/final_orca_n20.json
```

## GCBF+ GPU image

```powershell
docker build -f baselines/comparison-harness/Dockerfile.gcbfplus-gpu `
  -t gcbfplus-benchmark:gpu .
```

The image keeps the public GCBF+ dependency versions and replaces only the
CPU JAX wheel with its matching CUDA build.  `train_gcbfplus_common.py` keeps
the public network, losses, replay buffers, QP layer, and trainer while
matching the common speed, yaw-rate, body-radius, time-step, and arrival
bounds from the start of training.  It trains on separately generated random
instances rather than any paper test seed.  Its default Dubins-car training
budget, environment batch, ray count, obstacle count, seed, and learning
hyperparameters match the released Dubins-car checkpoint configuration.
`run_gcbfplus.py` then evaluates the saved policy on the canonical
fingerprints.
The public 2-D Dubins environment represents rectangular obstacles, so GCBF+
is evaluated on Free, Swap, and Rect15; Circ15 is not converted into a
different geometry.

The released training batch of 12 environments exceeds an 8 GB GPU during the
QP-label update.  On that hardware, use a batch of 6 for 2,000 steps; this
preserves the released configuration's total number of collected rollouts:

```powershell
python baselines/comparison-harness/train_gcbfplus_common.py `
  --n-env-train 6 --steps 2000 --eval-interval 20 --save-interval 200 --seed 0
```

Select a checkpoint only with held-out seeds 20--21.  The ray count must match
the training configuration; the evaluator also applies the public Dubins-car
acceleration clipping before integrating the common bounded unicycle:

```powershell
python baselines/comparison-harness/run_gcbfplus.py `
  --checkpoint <training-run-directory> --checkpoint-step 2000 `
  --families free swap rect15 --robots 20 --seeds 20 21 --n-rays 32 `
  --output baselines/comparison-harness/results/gcbfplus_validate_step2000.json
```

After selection, keep seeds 0--19 disjoint from that decision and run each
team size as a separate, incrementally written shard:

```powershell
python baselines/comparison-harness/run_gcbfplus.py `
  --checkpoint <training-run-directory> --checkpoint-step 1600 `
  --families free swap rect15 --robots 20 `
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 `
  --n-rays 32 `
  --output baselines/comparison-harness/results/final_gcbfplus_n20.json
```

Timing results must be collected with one controller process at a time.
Concurrent tuning and training runs are suitable for outcome screening but
not for the paper's control-time table.

The retained artifact uses step 1600, selected only on seeds 20--21.  Final
test records are `final_{orca,nhorca,gcbfplus}_n{20,40,60,80}.json`; the
cross-method aggregate is `public_baselines_all.json` and its flat companion
is `public_baselines_all.csv`.  `gcbfplus_checkpoint_selection.json` records
the held-out selection audit.

### GCBF+ Social Mini-Games

The same selected checkpoint can be evaluated on the common yaw-zero
doorway/intersection instances. The runner records SMG entry-to-exit
throughput and, by default, reruns every agent alone for counterfactual delay:

```powershell
python baselines/comparison-harness/run_gcbfplus.py `
  --checkpoint <training-run-directory> --checkpoint-step 1600 `
  --families doorway12 intersection24 --robots 8 16 --seeds 0 `
  --isolated-reference `
  --output baselines/comparison-harness/results/gcbfplus_smg_yaw0_seed0.json
```

Use `doorway08` only for the declared tight-doorway stress condition.
The public policy remains named GCBF+ in tables; the common physical bounds
and SMG geometry are described in the experimental protocol rather than in
the method label.

## Architecture-aware timing hooks

The local RVO2 build exposes `stepTiming()`, which is behavior-equivalent to
`step()` and returns KD-tree, state-update, and per-agent
`computeNeighbors()+computeNewVelocity()` times.  `run_nhorca.py` combines
the Python snapshot/command conversion with the shared phase and stores one
raw NPZ per case.  A hook-on/off rollout must have an identical final-state
SHA before the timing image is accepted.

`run_gcbfplus.py` synchronizes every timed JAX result.  On CPU it also builds
the induced one-hop sensing graph for every ego robot and validates that
action against the official full-graph batch.  The smaller padded graph
changes float32 segment-reduction order, so the explicit local parity bound is
`5e-6`.  GPU records use the synchronized official batch path only; CPU ego
timing supplies the validated decentralized critical path.  JIT compilation,
static planning, target lookup, integration, and safety audits remain outside
all timing intervals.
