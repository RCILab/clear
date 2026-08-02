# CLEAR reproducibility artifact

This directory is the canonical code and retained evidence for the 310/320
manuscript configuration of CLEAR. The reported bounded-unicycle protocol uses
a 60 s horizon, a 30 ms passage-update period, and three sequential native
projection substeps.

## Directory map

| Path | Contents |
|:---|:---|
| `clear_nav/` | CLEAR controller, geometry, CBF projection, scenarios, and simulator |
| `tests/` | controller, shared-resource, timing, and adapter tests |
| `results/` | final paper-facing records, summaries, and experiment manifest |
| `validation/` | numerical parity, certificate, and architecture-aware timing audits |
| `baselines/` | local evaluation harnesses, retained final records, and upstream revisions |
| `visualization/` | live viewer and static-run generator used by the project-page playground |

The executable Python files at this directory's root are kept beside
`clear_nav/` so their imports work identically in a checkout and in Docker.
They fall into four groups:

- experiment entry points: `run_unicycle.py`, `benchmark_smg.py`,
  `benchmark_theorem_extensions.py`, and `benchmark_timing_v2_clear.py`;
- baseline adapters: `run_clear_on_mgr.py`, `run_impc_smg.py`, and the two
  instance exporters;
- aggregation and validation: `aggregate_results.py`,
  `aggregate_timing_v2.py`, `canonicalize_results.py`, and the timing helpers;
- figures and animations: `make_paper_figures.py`, `make_paper_gifs.py`, and
  the two GIF entry points.

## Quick verification

From this directory:

```bash
docker build -t clear-nav .
docker run --rm -v "${PWD}:/work" -w /work clear-nav
```

This runs the default CLEAR, shared-resource, and timing tests. The IMPC-DR
adapter tests require the separate dependency image:

```bash
docker build -f Dockerfile.smg-impc -t clear-impc .
docker run --rm -v "${PWD}:/work" -w /work clear-impc \
  python -m unittest tests.test_impc_unicycle -v
```

## Main benchmark

The final 320-mission matrix uses four task families, four team sizes, and 20
seeds per task--size cell:

```bash
python run_unicycle.py \
  --families free swap circ15 rect15 \
  --robots 20 40 60 80 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --dt 0.03 --inner-substeps 3 \
  --output results/reproduced_clear_320.json
```

Use `--variant component-free` or `--variant vanilla-cbf-qp` for the two
nested internal baselines. `benchmark_smg.py` evaluates Doorway and
Intersection, and `benchmark_theorem_extensions.py --sections bridge-scale`
reproduces the N=8/16 StraightBridge audit.

## Retained evidence

`results/PAPER_EXPERIMENT_MANIFEST.md` is the authoritative claim inventory.
The three audited 320-run records are in `results/headline30/`; aggregate
reports and theorem/shared-resource records are beside them. Timing samples
and their SHA-256 manifest are under `validation/timing_v2/`. Large GIFs and
duplicate diagnostic traces are not stored here: the public replay bank is
already maintained once in `../playground/runs/`.

## Baselines

The repository includes only our adapters and evaluation harnesses. Public
third-party source trees are intentionally not copied into this directory.
Their exact upstream revisions and local adaptation boundary are recorded in
`baselines/README.md`; place those checkouts beside this artifact when a full
external-baseline rerun is required.
