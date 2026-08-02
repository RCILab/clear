# CLEAR project page and code

This repository hosts the CLEAR project page and its reproducibility artifact.

| Path | Purpose |
|:---|:---|
| `index.html` | project page |
| `playground/` | browser replay of retained CLEAR executions |
| `static/` | project-page styles, scripts, and images |
| `artifact/` | canonical 310/320 implementation, tests, validation, and final records |

The experiment inventory and hashes are in
`artifact/results/PAPER_EXPERIMENT_MANIFEST.md`. Start with
`artifact/README.md` to reproduce the controller, internal comparisons,
shared-resource experiments, StraightBridge audit, and latency analysis.

## Local preview

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Release checklist

- add the final submitted paper and enable the Paper button;
- replace the marked hardware placeholders and add the final video;
- enable the Code button when the anonymous-review restriction is lifted.
