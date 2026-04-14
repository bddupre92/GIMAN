# Phase 5 Task 7 — NASEM Digital Twin Criteria Audit RUN MANIFEST

**Date:** 2026-04-13
**Script:** `scripts/mechanistic_twin/phase5_nasem_audit.py`
**Tests:** `tests/mechanistic_twin_v2/test_nasem_audit.py` (11 tests)
**Output:** `outputs/mechanistic_twin/paper10_mech_vs_giman/nasem_audit.json`

## Purpose

Score Paper 10 honestly against the 7 NASEM 2024 digital twin criteria (An & Cockrell 2024 arXiv:2405.05301 operationalization). Own partial implementation explicitly — no overclaim, no evasion. Output feeds Paper 10 Discussion and motivates Phase 6 / Paper 11 future work.

## Scoring

| Score | Meaning |
|---|---|
| 0 | Absent |
| 1 | Partial / acknowledged limitation |
| 2 | Substantial |
| 3 | Complete |

## Results

**Aggregate: 16/21 (76.2%), mean score 2.29. Zero criteria at 0 or 1.**

| Criterion | Score | Rationale |
|---|---|---|
| Virtual representation | 2 | Patient-specific ODE (aggregation + neuron death + Hill PK/PD); not full 5-module |
| Bidirectional flow | 2 | Task 5 SIR updater works (episodic); not continuous sensor-stream |
| Predictive capability | 2 | Task 6 counterfactual calibration passes; wearing-off honest null |
| **Uncertainty quantification** | **3** | Paper 4 IPCW conformal (91.1% coverage) + Phase 2 posterior CIs + bootstrap |
| Validation | 2 | Task 3 cross-sectional external + Task 6 observational counterfactual + Task 4 head-to-head |
| Fitness for purpose | 2 | Research-grade context declared; not regulatorily qualified |
| **Governance** | **3** | Closed-Loop v1.5 + Doc Lifecycle v1.0 + per-task RUN_MANIFEST + chain SHAs |

## Honest Framing

Paper 10 sits at the **"bidirectional-ready, episodically updated"** tier of NASEM DT maturity.

- **Exceeds** prior PD mechanistic one-shot calibrations (Véronneau-Veilleux 2020/2021; Nair 2026 review: no published PD DT demonstrates bidirectional updating).
- **Matches** cardiac DT episodic tier (Corral-Acero 2020 Eur Heart J; Coorey 2021 Nat Rev Cardiol).
- **Falls short of** sensor-continuous + prospective-interventional tier, correctly scoped as Phase 6 (MindMend biosensor) future work.

## Implications for Paper 10 Positioning

1. **Primary claim:** methodological contribution (bidirectional SIR + PosteriorStore architecture + observational counterfactual validity), not benchmark victory
2. **Secondary claim:** Task 6 slope CI [0.88, 1.29] contains 1.0 — observationally calibrated
3. **Defensive framing:** cross-sectional external validation (Task 3) is current field standard given field-wide longitudinal public PD DaT data gap
4. **Future work:** longitudinal external decay validation pending DeNoPa / SURE-PD3 / ICEBERG collaborations
5. **Venue target:** npj Parkinson's Disease or Journal of Parkinson's Disease (per Agent C systematic sweep: no CPT:PSP Bayesian-updating-for-PD precedent 2023-2026)

## Literature Anchors

| Criterion source | Citation |
|---|---|
| NASEM framework | NASEM 2024 *Foundational Research Gaps for Digital Twins* — 10.17226/26894 |
| Operationalization template | An & Cockrell 2024 — arXiv:2405.05301 |
| VVUQ framework | Viceconti et al. 2021 *Methods* 185:120 — 10.1016/j.ymeth.2020.01.011 |
| VVUQ ML extension | Viceconti et al. 2025 IEEE JBHI — 10.1109/JBHI.2025.3552320 |
| Credibility matrix (risk-informed) | Musuamba et al. 2021 CPT:PSP 10(8):804 — 10.1002/psp4.12669 |
| QSP-MQM qualification ladder | Friedrich 2016 CPT:PSP — 10.1002/psp4.12056 |
| PD DT ancestors (one-shot) | Véronneau-Veilleux 2020/2021; Nair 2026 Brain Sci |
| Cardiac DT peer tier | Corral-Acero 2020 — 10.1093/eurheartj/ehaa159; Coorey 2021 — 10.1038/s41569-021-00630-4 |
| Regulatory MIDD on-ramp | Galluppi 2024 — 10.1002/cpt.3245; ICH M15 2024 Fed Reg 89 FR 106745 |

Full BibTeX in `phase5_literature_bibliography.bib`.

## Verification

- **Tests:** 11/11 passing
- **Test coverage:** file exists; 7 expected criteria present; scores in [0,3]; every criterion has evidence+gaps; aggregate math correct; zero criteria absent; governance = 3; UQ = 3; bidirectional ≥ 2; honest framing present; venue documented

## Artifacts

- `scripts/mechanistic_twin/phase5_nasem_audit.py` (~200 lines)
- `tests/mechanistic_twin_v2/test_nasem_audit.py` (11 tests)
- `outputs/mechanistic_twin/paper10_mech_vs_giman/nasem_audit.json`

## Closed-Loop v1.5 — Stage 6.5 Cycle A

- Stage 1 (Literature): NASEM 2024 + An & Cockrell 2024 + Viceconti VVUQ + Musuamba credibility + Friedrich MQM
- Stage 4 (Post-exec review): 11/11 tests pass; score distribution = {2: 5, 3: 2}; zero absent or partial-only
- Stage 5 (Independent validation): each criterion's evidence cross-referenced to a committed Task 1-6 output artifact or Phase 2/4 JSON
- Stage 6 (Decision gate): APPROVE — 76.2% compliance at "bidirectional-ready episodic" tier is defensible for a methodological PhD contribution
- Stage 6.5 Cycle A: this RUN_MANIFEST + nasem_audit.json

**Next:** Task 8 — 9 publication figures synthesising Tasks 1-7.
