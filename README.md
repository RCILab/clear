# CLEAR — project page

Anonymous project page for **CLEAR: Certified Constraint-Compatible Motion for
Multi-Robot Passage** (submitted to IEEE T-RO).

- `index.html` — project page (template shared with the other RCILab project pages)
- `playground/` — in-browser replay of precomputed runs from the actual CLEAR
  pipeline (static run bank; no server required)
- `static/` — page assets

## Local preview

```bash
python -m http.server 8000
# open http://localhost:8000
```

## TODO before announcement

- [ ] `static/pdfs/clear_paper.pdf` (final submitted PDF) + enable the Paper button
- [ ] hardware section and video clips after the 40-trial batch
- [ ] extend `playground/runs/` bank (`visualization/precompute.py` in the main repo)
- [ ] de-anonymize the author block after acceptance
