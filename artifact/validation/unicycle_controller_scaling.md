# Controller computation-time scaling

CLEARController.command plus three bounded native-unicycle CBF projections per 33.3 Hz step; excludes scenario/planner/target lookup, state integration, and swept-distance audits

CPU: Intel(R) Core(TM) i7-14700F

| N | Family | Mean command (ms) | Worst seed p95 (ms) | Max (ms) |
|---:|:---|---:|---:|---:|
| 20 | circ15 | 1.961 | 2.600 | 8.175 |
| 20 | free | 1.726 | 2.642 | 7.529 |
| 20 | rect15 | 2.501 | 3.387 | 9.002 |
| 20 | swap | 1.591 | 2.578 | 4.604 |
| 40 | circ15 | 2.498 | 3.465 | 13.378 |
| 40 | free | 2.181 | 3.302 | 6.513 |
| 40 | rect15 | 2.960 | 4.289 | 29.438 |
| 40 | swap | 2.113 | 4.721 | 10.157 |
| 60 | circ15 | 3.193 | 5.447 | 55.376 |
| 60 | free | 2.485 | 3.732 | 8.510 |
| 60 | rect15 | 4.058 | 5.950 | 67.519 |
| 60 | swap | 3.191 | 9.249 | 42.705 |
| 80 | circ15 | 4.115 | 7.579 | 62.885 |
| 80 | free | 3.277 | 5.211 | 12.274 |
| 80 | rect15 | 4.690 | 7.972 | 61.376 |
| 80 | swap | 7.229 | 15.205 | 44.184 |
| 120 | free | 5.859 | 9.957 | 21.235 |
| 120 | swap | 10.060 | 31.646 | 96.250 |

The p95 column is the largest per-seed p95 among the three paired state streams. The maximum is retained as an outlier diagnostic, not as the real-time claim.
