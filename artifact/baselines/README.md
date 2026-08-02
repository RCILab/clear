# Baseline provenance and local harnesses

This artifact retains our common-protocol adapters and final paper-facing
records, but does not vendor third-party repositories. Full reruns use the
following public implementations at the recorded revisions.

| Method | Upstream implementation | Revision |
|:---|:---|:---|
| GCBF+ | `MIT-REALM/gcbfplus` | `fb449907bdbf981aa10f0edfecca02663ddc8037` |
| MGR | `Aiden-wjLee/merry-go-round` | `b166d2a8cf3f22f5a708976c6adba50ac0ee5af4` |
| ORCA / NH-ORCA | `dongfangliu/NH-ORCA-python` | `20202e1fe7427f1499200853b5bb1e606d6fb8b4` |
| IMPC-DR | `CRAL-UVA/SMGLib` | `2e901ae5d6b8e920e30e3787a0e04feefa698a85` |

`comparison-harness/` contains our ORCA, NH-ORCA, and GCBF+ runners.
`mgr-harness/` contains the official-instance MGR runner and aggregation
utilities. The timing-v2 inputs required to regenerate the paper-facing
latency aggregate are retained below each harness. `results/` contains the
final external comparison records referenced by the experiment manifest.

Local instrumentation is limited to the documented execution boundary. The
NH-ORCA checkout exposes per-agent timing hooks while preserving legacy output
parity. The SMGLib checkout contains the bounded-unicycle IMPC-DR adapter and
persistent-solver instrumentation. Exact parity checks are retained under
`../validation/`.
