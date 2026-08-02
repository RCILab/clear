# Architecture-aware timing v2

Rect15, 60 s, dt=0.03 s, seeds 0/7/13, 40 warm-up steps, one timing container at a time, and one BLAS/OpenMP thread. Entries are mean of seed means / worst-seed p95 in ms.

## Single-machine batch latency

| N | CLEAR | Vanilla CBF-QP | MGR | ORCA | NH-ORCA | GCBF+ CPU | GCBF+ GPU | IMPC-DR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 12.714 / 13.998 | 12.729 / 14.205 | 14.054 / 16.259 | 0.091 / 0.147 | 0.086 / 0.144 | 4.432 / 4.659 | 1.917 / 3.000 | not run: scale gate |
| 40 | 20.444 / 23.610 | 20.860 / 24.039 | 29.306 / 32.826 | 0.156 / 0.245 | 0.150 / 0.222 | 7.123 / 7.440 | 2.054 / 3.276 | not run: scale gate |
| 60 | 27.814 / 32.344 | 27.273 / 32.065 | 45.604 / 49.508 | 0.226 / 0.339 | 0.219 / 0.303 | 7.414 / 8.645 | 2.198 / 3.438 | not run: scale gate |
| 80 | 33.635 / 39.360 | 32.108 / 37.264 | 62.078 / 68.752 | 0.304 / 0.438 | 0.296 / 0.403 | 10.112 / 11.647 | 2.842 / 5.165 | not run: scale gate |

## Deployment critical-path latency

| N | CLEAR | Vanilla CBF-QP | MGR | ORCA | NH-ORCA | GCBF+ CPU | GCBF+ GPU | IMPC-DR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 4.025 / 4.660 | 3.955 / 4.602 | 1.412 / 2.445 | 0.084 / 0.134 | 0.079 / 0.131 | 1.364 / 1.980 | batch only | not run: scale gate |
| 40 | 5.418 / 6.920 | 5.303 / 6.381 | 1.714 / 2.976 | 0.133 / 0.212 | 0.129 / 0.191 | 2.198 / 3.169 | batch only | not run: scale gate |
| 60 | 7.177 / 9.944 | 6.800 / 8.866 | 2.059 / 3.433 | 0.182 / 0.285 | 0.176 / 0.249 | 3.145 / 5.181 | batch only | not run: scale gate |
| 80 | 8.786 / 12.177 | 8.583 / 11.796 | 2.559 / 4.305 | 0.232 / 0.352 | 0.226 / 0.322 | 4.025 / 5.276 | batch only | not run: scale gate |

GCBF+ CPU includes validated induced one-hop ego-graph timing; GCBF+ GPU is the synchronized official full-graph batch path and therefore has no deployment critical-path entry. IMPC-DR uses the documented scale gate when its local critical-path p95 exceeds the 30 ms control period.
