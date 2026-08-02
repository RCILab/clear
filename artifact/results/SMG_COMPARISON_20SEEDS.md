# Social Mini-Game comparison

| Method | Scenario | N | Arr./cert. | Resource | Flow | Delay (s) |
|---|---|---:|---:|---:|---:|---:|
| CLEAR | Doorway | 8 | 20/20 | 100.0% | 0.848 | 3.610 |
| CLEAR | Doorway | 16 | 20/20 | 100.0% | 1.006 | 6.330 |
| CLEAR | Intersection | 8 | 20/20 | 100.0% | 0.692 | 1.970 |
| CLEAR | Intersection | 16 | 20/20 | 100.0% | 0.918 | 4.010 |
| GCBF+ | Doorway | 8 | 0/0 | 0.0% | 0.000 | 16.014 |
| GCBF+ | Doorway | 16 | 0/0 | 0.0% | 0.000 | 26.617 |
| GCBF+ | Intersection | 8 | 0/0 | 0.0% | 0.000 | 22.779 |
| GCBF+ | Intersection | 16 | 0/0 | 0.0% | 0.000 | 18.431 |
| IMPC-DR | Doorway | 8 | 1/0 | 5.0% | 0.022 | 3.168 |
| IMPC-DR | Doorway | 16 | 0/0 | 0.0% | 0.000 | 6.641 |
| IMPC-DR | Intersection | 8 | 4/0 | 60.0% | 0.267 | 0.825 |
| IMPC-DR | Intersection | 16 | 2/0 | 50.0% | 0.262 | 2.131 |
| MGR | Doorway | 8 | 0/0 | 0.0% | 0.000 | 5.171 |
| MGR | Doorway | 16 | 0/0 | 0.0% | 0.000 | 8.002 |
| MGR | Intersection | 8 | 19/19 | 95.0% | 0.406 | 3.651 |
| MGR | Intersection | 16 | 13/13 | 65.0% | 0.353 | 6.096 |
| NH-ORCA | Doorway | 8 | 11/11 | 65.0% | 0.170 | 20.210 |
| NH-ORCA | Doorway | 16 | 12/12 | 70.0% | 0.313 | 16.606 |
| NH-ORCA | Intersection | 8 | 20/20 | 100.0% | 0.381 | 2.297 |
| NH-ORCA | Intersection | 16 | 20/19 | 100.0% | 0.414 | 5.706 |
| ORCA | Doorway | 8 | 11/10 | 55.0% | 0.202 | 10.324 |
| ORCA | Doorway | 16 | 0/0 | 5.0% | 0.013 | 15.254 |
| ORCA | Intersection | 8 | 20/17 | 100.0% | 0.194 | 5.937 |
| ORCA | Intersection | 16 | 11/9 | 85.0% | 0.130 | 28.667 |
| Vanilla CBF-QP | Doorway | 8 | 17/17 | 85.0% | 0.297 | 11.502 |
| Vanilla CBF-QP | Doorway | 16 | 18/18 | 90.0% | 0.474 | 15.918 |
| Vanilla CBF-QP | Intersection | 8 | 20/20 | 100.0% | 0.314 | 4.753 |
| Vanilla CBF-QP | Intersection | 16 | 20/20 | 100.0% | 0.386 | 11.025 |

## Paired CLEAR comparison

| Baseline | Scenario | N | C-only | B-only | Flow delta | Delay delta (s) |
|---|---|---:|---:|---:|---:|---:|
| GCBF+ | Doorway | 8 | 20 | 0 | 0.848 | -12.165 |
| GCBF+ | Doorway | 16 | 20 | 0 | 1.006 | -20.241 |
| GCBF+ | Intersection | 8 | 20 | 0 | 0.692 | -21.448 |
| GCBF+ | Intersection | 16 | 20 | 0 | 0.918 | -14.254 |
| IMPC-DR | Doorway | 8 | 20 | 0 | 0.826 | 0.785 |
| IMPC-DR | Doorway | 16 | 20 | 0 | 1.006 | -0.510 |
| IMPC-DR | Intersection | 8 | 20 | 0 | 0.425 | 1.169 |
| IMPC-DR | Intersection | 16 | 20 | 0 | 0.656 | 1.885 |
| MGR | Doorway | 8 | 20 | 0 | 0.848 | -1.522 |
| MGR | Doorway | 16 | 20 | 0 | 1.006 | -1.743 |
| MGR | Intersection | 8 | 1 | 0 | 0.287 | -1.664 |
| MGR | Intersection | 16 | 7 | 0 | 0.565 | -2.036 |
| NH-ORCA | Doorway | 8 | 9 | 0 | 0.678 | -17.179 |
| NH-ORCA | Doorway | 16 | 8 | 0 | 0.694 | -11.058 |
| NH-ORCA | Intersection | 8 | 0 | 0 | 0.311 | -0.327 |
| NH-ORCA | Intersection | 16 | 1 | 0 | 0.504 | -1.696 |
| ORCA | Doorway | 8 | 10 | 0 | 0.646 | -7.129 |
| ORCA | Doorway | 16 | 20 | 0 | 0.993 | -12.649 |
| ORCA | Intersection | 8 | 3 | 0 | 0.498 | -3.967 |
| ORCA | Intersection | 16 | 11 | 0 | 0.788 | -24.578 |
| Vanilla CBF-QP | Doorway | 8 | 3 | 0 | 0.551 | -7.272 |
| Vanilla CBF-QP | Doorway | 16 | 2 | 0 | 0.533 | -9.143 |
| Vanilla CBF-QP | Intersection | 8 | 0 | 0 | 0.379 | -2.783 |
| Vanilla CBF-QP | Intersection | 16 | 0 | 0 | 0.532 | -7.015 |
