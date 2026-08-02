# CLEAR paper experiment manifest

Validated on 2026-08-02 (Asia/Seoul).

## Official configuration

- Dynamics: bounded first-order look-ahead unicycle
- Main horizon / control step: 60 s / 0.03 s (33.3 Hz)
- Speed / yaw limits: 0.8 m/s / pi/2 rad/s
- Look-ahead / inner projections: 0.05 m / 3
- Terminal capture: 0.22 m in clutter, 0.60 m in open scenes
- Terminal release: 0.80 m
- Native bounded-input CBF projection: enabled
- Declared feasibility audit / internal solver tolerance: 1e-6 / 1e-7
- Hierarchical progress (HQP): disabled

## Completed experiments

| Experiment | Runs | Result |
|:---|---:|:---|
| Main Free/Swap/Circ15/Rect15, N=20/40/60/80 | 320 | 310 complete; 0 command-infeasible steps; 0/638,642 applicable static-certificate violations |
| Vanilla CBF-QP main matrix | 320 | 263 complete; safety audit 320/320 |
| Component-free main matrix | 320 | 301 complete; safety audit 320/320 |
| Same-protocol algorithm records | 1,840 | paired fingerprints verified across six completed methods |
| Doorway/Intersection SMGs, N=8/16 | 560 | seven methods, 20 paired seeds/cell; CLEAR 80/80 |
| IMPC-DR SMG subset | 80 | 7/80 all-arrived, 0/80 certified; 0.504--1.242 s mean step |
| IMPC-DR persistent Rect15 gate, N=8 | 3 | 172.387 ms mean batch; 31.760 ms worst-seed local critical-path p95 |
| IMPC-DR gated Rect15 rows, N=20/40/60/80 | 4 | not run: N=8 already exceeds the 30 ms critical-path gate |
| Paired nested internal comparison | 960 | Vanilla 263, component-free 301, full CLEAR 310; all 320 fingerprints shared by all variants |
| Exact StraightBridge theorem domain | 60 | 60/60 longitudinal exits |
| StraightBridge antecedent perturbations | 50 | 30 guide + 20 curvature probes |
| StraightBridge N=8/16 scale audit | 80 | 40/40 planar exits and 40/40 bounded-unicycle audited completions; zero certificate violations |
| Native-unicycle N=20 input audit | 80 | 80/80 complete |
| Native-unicycle four-robot bridge | 20 | 18/20 task completion; 20/20 theorem exit |
| Architecture-aware Rect15 timing streams | 87 | 84 N=20/40/60/80 baseline streams plus three IMPC N=8 gate streams |
| Timing/reference parity cases | 20 | CLEAR/Vanilla 6, NH-ORCA 1, IMPC 1, GCBF+ CPU 12; all pass declared tolerances |
| High-resolution paired diagnostic traces | 8 | 4 CLEAR + 4 component-free |
| Paper GIFs | 13 | all multi-frame, infinite loop; four paired recoveries and four SMG animations |
| Qualitative N=20 comparison GIFs | 16 | 14 individual + 2 synchronized Doorway/Intersection grids; 120 frames each |
| Unit tests | 51 | 46 CLEAR/SMG/timing + 5 IMPC-DR pass in their dependency images |

The architecture-aware Rect15 study reports two distinct panels. CLEAR's
N=80 one-worker batch result is 33.635 / 39.360 ms, whereas its
component-parallel deployment critical path is 8.786 / 12.177 ms
(mean / worst-seed p95). The deployment critical path therefore remains
inside the 30 ms period through N=80, while the one-worker batch p95 crosses
the period at N=60. The older family-wide diagnostic still records Swap N=80
at 15.205 ms and the separate Swap N=120 stress row at 31.646 ms.

Persistent IMPC-DR matches the reference final positions within
2.85e-9 m. Its N=8 Rect15 worst-seed critical-path p95 is 31.760 ms;
canonicalization/interface time averages 10.993 ms per agent call versus
4.450 ms solver-reported time, so larger rows are scale-gated and the
GPU-solver entry condition is false.

The audited main rerun preserves all 320 fingerprints and every controller
behavior, mission, makespan, and safety field from the pre-audit canonical
records. It separately reports 297 nonsolved optimizer calls in 83 missions,
270 feasibility restorations in 79 missions (46 actuator contractions,
0 certified witnesses, 224 common contractions, and 0 HQP fallbacks), and
0 final command-infeasible steps. Of 638,745 positive-witness theorem
candidates, 103 are exact-projection exclusions and 638,642 are applicable,
with zero conclusion violations.

## Principal artifacts and SHA-256

- `headline30/clear_full_320.json`  
  `98AA3CD1C5E451BC92EA883160F01830E758BAC904F079217094F888A6775047`
- `headline30/component_free_320.json`  
  `514310D3D390DDB8D784DD71D1017301390EDFCC3C40BD9043E3F2C82B45C368`
- `headline30/vanilla_320.json`  
  `B51AC4BF0EECF1F8BB98DC1CCCA6C0EC1D5D9DB35262AC90774E4E478B469BAC`

- `unicycle_clear_all_sizes_raw.json`  
  `E2606D1DE06CD01C98286DF683C448CB2470B76C37B883FD68FCA5088E6F2FB0`
- `unicycle_clear_all_sizes_summary.json`  
  `04E9AF03A2BC5AFC1598B856D703FD13CD2E6CBC3004D60248DC7E38833DCA5A`
- `unicycle_component_free_n20_n40_raw.json`  
  `7199F2024C7EDBA35581C98C8C68A55BDF1DE46131707FCE4D82789FA8E7A96C`
- `unicycle_component_free_n20_n40_summary.json`  
  `5A602F72F3193DB624A79526888A9B4EE6F25290DF42DD8F73D2D9323A254C14`
- `theorem_extension_audit.json`  
  `FD65AB67ADACDE3117B9F84C6DA707D2B29AF69727171805B1FF3F63D008DE6C`
- `straight_bridge_scale_n8_n16.json`  
  `83FC3C90A52F780416FF3B4028F97441FC1CB9A989080D44FD404C34ABBADDB0`
- `../validation/unicycle_controller_scaling.json`  
  `00E57453B21972DF4C167ED64A0A73C2AA8B536C4D49BDE00360A90C1FF70EEE`
- `../validation/timing_v2/timing_v2_summary.json`  
  `BDB0ADB794341526987C710DFEF22FC9AF86F2935CC3C4BA0EC3B722236E0B3F`
- `../validation/timing_v2/TIMING_V2_REPORT.md`  
  `AC591C84D96BA8BAA437095A472FB7D9D013D4F7E28758DBB40B36C8C0A75C16`
- `../validation/timing_v2/timing_v2_manifest.json`  
  `DDBCBD50BAE8E2AED20F45A73762D68C1696FE375AAFA91F891B1ACD93706145`
- `smg_comparison_20seeds.json`  
  `C73D152F9C58A9D929DC22D88B823A8ABAF1A057025E26BF14D89435322F6B0E`
- `../baselines/results/public_baselines_with_vanilla.json`  
  `212EF195145EE53E31BD98799EC9022F799DBE74ED16508665B1A795927EBE83`
- `../baselines/results/mgr_clear320_optimized_paired_summary.json`  
  `FE4F7E7DCC290DED7A31A319A7FE79CCEF6D57AE3BCCA0130C84AF55132DEF81`
- `../baselines/results/MGR_CLEAR320_PAIRED_REPORT.md`  
  `50111A260B6CD8A0C13B47E4C26CD6FCB99A8C3D9B5626A0D9D0D67654BA9A41`
Large paper GIFs and duplicated diagnostic traces are not included in the
code artifact. The project-page replay bank is retained separately under
`../../playground/runs/` and is qualitative rather than part of the official
outcome matrices.
