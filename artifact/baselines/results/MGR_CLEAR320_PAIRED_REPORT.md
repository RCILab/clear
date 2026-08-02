# Paired CLEAR–MGR comparison

This report summarizes
`mgr_clear320_optimized_paired_summary.json`. The comparison uses 320
identical scenario fingerprints, a 0.03 s control step, a 60 s horizon, and a
0.22 m arrival radius. Bootstrap intervals use 20,000 resamples with seed
20260729.

## Completion

| Outcome | Count |
|:---|---:|
| Both complete | 149 |
| CLEAR only | 161 |
| MGR only | 0 |
| Neither | 10 |

CLEAR completes 310/320 missions and MGR completes 149/320. All 161
discordant outcomes favor CLEAR; the exact McNemar p-value is
`6.842277657836021e-49`.

## Arrival time

Across the 149 joint successes, CLEAR is 10.4936 s faster on average
(95% bootstrap interval: 9.0908--11.9100 s faster). CLEAR wins 130 paired
arrival-time comparisons, MGR wins 19, and there are no ties; the exact sign
p-value is `1.5845286212458737e-21`.

When failures are assigned the 60 s horizon, CLEAR is 20.7875 s faster on
average (95% bootstrap interval: 19.3698--22.1977 s faster).

## Safety convention

All 149 successful MGR missions are physically collision-free. Under the
stricter common declared-margin audit, 3 MGR missions are both complete and
certified, compared with 310 for CLEAR. Task completion and common-margin
certification are therefore reported separately in the paper.

The machine-readable source records the implementation provenance, shard
list, optimization-equivalence status, per-family summaries, and safety
minima.
