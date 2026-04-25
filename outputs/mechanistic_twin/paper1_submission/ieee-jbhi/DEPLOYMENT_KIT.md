# Paper 1 Deployment Kit

**Target:** IEEE Journal of Biomedical and Health Informatics submission
"NSD-ISS Biological-Stage Prediction with Calibrated Uncertainty"
**Authors:** Blair Dupre, Department of Biomedical Engineering, University of
North Dakota
**Repository root:** `CSCI-FALL-2025/` (branch `feat/ch9-6-multichannel`)
**Companion document:** [REPRODUCIBILITY_PACKAGE.md](REPRODUCIBILITY_PACKAGE.md)
**Reviewer query addressed:** R5-Q9 — minimal deployable artefact set + licensing

This kit is the minimal artefact bundle required to (a) install a CatBoost
NSD-ISS staging model behind a clinical decision-support dashboard, (b)
calibrate it to a local cohort, and (c) emit conformal prediction sets a
clinician can act on. It is intentionally narrower than the full
reproducibility package: only the deployable subset is described.

---

## 1. Model card (Mitchell et al. 2019 schema)

| Field | Value |
|---|---|
| **Model name + version** | `CatBoost-default-21feat-NSDISS-v1.0` |
| **Date** | 2026-04-24 |
| **Type** | Gradient-boosted decision-tree ensemble (CatBoost 1.2.10, library defaults). |
| **Primary intended use** | Decision support for NSD-ISS biological-stage classification in adult Parkinson's disease cohorts. Two deployment surfaces are recommended: (i) **binary NSD$+$ detection** (NSD$+$ vs NSD$-$) at the patient's first qualifying visit; (ii) **NSD$+$ sub-staging** (Stage 1 / 2B / 3 / 4) at follow-up visits, contingent on a previously confirmed NSD$+$ flag. |
| **Out-of-scope use** | (i) Stand-alone diagnosis without clinician oversight; (ii) screening unselected community populations; (iii) deployment in cohorts outside the 30 PPMI sites without local recalibration; (iv) any commercial use of the optional TabPFN v2 sensitivity variant (see §5). |
| **Training data** | PPMI ($n{=}2{,}201$ patients), NSD-ISS staged 2024-2025 per Simuni et al. (Lancet Neurol 2024) and Russo et al. (npj PD 2025) replication; SAA from PPMI Project 222 + 9000 (released 2024-2025); DaT-SPECT from PPMI (08Oct2025 release). |
| **Features (21, strict-circularity Path 3)** | Demographics (3): `AGE_AT_BASELINE`, `SEX`, `HANDED`. UPDRS subscales (5): `UPDRS1_TOTAL`, `UPDRS2_TOTAL`, `UPDRS3_TREMOR`, `UPDRS3_RIGIDITY`, `UPDRS3_BRADYKINESIA`, `UPDRS3_AXIAL`, `UPDRS4_TOTAL`. Cognitive (1): `MOCA_TOTAL`. Sleep (2): `RBD_TOTAL`, `ESS_TOTAL`. Autonomic (1): `SCOPA_AUT_TOTAL`. DaT imaging (4): `CAUDATE_R_SBR`, `CAUDATE_L_SBR`, `CAUDATE_MEAN_SBR`, `CAUDATE_ASYMMETRY`. Genetics (3): `LRRK2_CARRIER`, `GBA_CARRIER`, `APOE_E4_CARRIER`. **Excluded by Path-3 strict circularity:** putamen SBR (NSD-ISS D anchor), MDS-UPDRS-III total (functional staging threshold), NP1 cognition (functional staging threshold), `CAUDATE_PUTAMEN_RATIO`. |
| **Evaluation protocol** | 5-fold stratified cross-validation, seed 42, with 1,000-bootstrap patient-level 95\% CIs. External validation on BioFIND (n=103 NSD-ISS staged + 15 SAA$-$ PD = 118). |
| **Headline performance (internal, 5-fold CV)** | Binary AUC 0.901 [0.887, 0.915]; three-class macro-AUC 0.897 [0.884, 0.910]; full-ordinal macro-AUC 0.915 [0.903, 0.926]; NSD$+$ sub-staging macro-AUC 0.908 [0.889, 0.925]. |
| **Headline performance (external, BioFIND)** | Binary balanced accuracy 0.516 (PPMI NSD$-$ class blends healthy controls and prodromals; PD-only retraining mitigates this — see §V.E of manuscript); three-class AUC 0.703; NSD$+$ sub-staging AUC moderate but n=103. |
| **Limitations** | (i) **External multiclass under-coverage**: BioFIND three-class conformal coverage drops to 0.499 vs ~0.90 internal; ECE reliability degrades in the small-n external regime. (ii) **Site-level transferability**: pre-registered site-LOSO fails on the MRI-covered subsample (n=657, 30 sites) with pooled mean AUC 0.832 ± 0.091 — class imbalance in small held-out sites. (iii) **No prospective deployment study has been performed.** All performance numbers are retrospective. (iv) **Training cohort enrichment**: PPMI is research-grade, urban, with high genetic-carrier representation (LRRK2 + GBA enrichment cohorts); community-clinic case-mix may differ. |
| **Ethical considerations** | (i) PPMI DUA constrains data redistribution; site identifiers (LONI IDA `siteKey`) must be suppressed. (ii) NSD-ISS is itself a research framework subject to active critique (Espay et al. Mov Disord 2025) — clinician-facing outputs should frame predictions as biological-stage estimates, not clinical diagnoses. (iii) Domain-shift risks for non-PD-research populations; recalibration is mandatory. |
| **Versioning** | This card describes v1.0. Future versions must update the §IV.B headline table and re-run the temperature-scaling fit (§3 of this kit). |

---

## 2. Decision thresholds

All thresholds below are computed on the 21-feature Path-3 model with
per-target temperature scaling applied (see §4 for $T^{\star}$ values).
Source data: `outputs/paper1_calibration/results/per_fold_probs.npz`,
post-temperature-scaling per-target operating points enumerated by
`.venv/bin/python` over 91 candidate thresholds in $[0.05, 0.95]$ at the
balanced-accuracy maximum.

### 2.1 Binary NSD$+$ detection (primary deployable)

| Threshold strategy | Cutoff $p^\star$ | Bal. Acc. | Sens. (NSD$+$) | Spec. (NSD$-$) | PPV | NPV |
|---|---|---|---|---|---|---|
| **Recommended (max bal. acc., post-$T^\star$)** | **0.560** | **0.951** | **0.920** | **0.982** | **0.966** | **0.957** |
| Default ($p \geq 0.5$) | 0.500 | 0.948 | 0.924 | 0.971 | 0.954 | 0.959 |

Confusion matrix at $p^\star = 0.560$ (n=2,201; NSD$+$ n=779, NSD$-$
n=1,422): TP=717, FP=25, TN=1,397, FN=62.

### 2.2 Three-class (Early 0/1, Clinical 2B, Impaired 3/4)

Default behaviour: argmax over softmax probabilities, then LAC conformal
abstention at the configured confidence level (90% by default → set
inclusion if $p_k \geq 1 - \tau_{3c}$). No alternative operating point is
recommended; clinical interpretation belongs to the conformal set, not a
margin tweak.

### 2.3 Full ordinal (Stages 0, 1, 2B, 3, 4)

Argmax + LAC conformal abstention. Reportable only when the conformal set
is a singleton OR a contiguous pair (e.g., {2B, 3}); non-contiguous sets
should escalate per §6.

### 2.4 NSD$+$ sub-staging (Stages 1, 2B, 3, 4)

Argmax + LAC conformal abstention; intended only after a confirmed binary
NSD$+$ flag (two-stage deployment). Argmax-Stage-3 is the modal
prediction (PPMI base rate dominates Stage 3 within NSD$+$).

---

## 3. Calibration curves

### 3.1 Per-target temperature ($T^\star$) and ECE

Source: `outputs/paper1_r2_responses/q4_temperature_scaling.json`.
$T^\star$ fit by minimising NLL on 5-fold validation logits; pooled mean
across folds.

| Target | $T^\star$ pooled | Pre-cal ECE | Post-cal ECE | Pre-cal Brier | Post-cal Brier |
|---|---|---|---|---|---|
| Binary | 1.433 | 0.0534 | **0.0196** | 0.2520 | 0.2454 |
| Three-class | 1.146 | 0.0420 | **0.0313** | 0.3002 | 0.2970 |
| Full ordinal | 1.009 | 0.0604 | 0.0569 | 0.3949 | 0.3945 |
| NSD$+$ sub-stage | 1.028 | 0.0418 | 0.0477 | 0.3083 | 0.3084 |

Binary calibration improves most (ECE $\downarrow$ 63%); ordinal targets
near-perfectly calibrated already ($T \approx 1$). Reliability diagrams:
`outputs/paper1_figures/fig7_calibration.pdf`.

### 3.2 Shared $T$ vs per-target $T$

A single shared $T = 1.150$ is also fit and reported in
`q4_temperature_scaling.json`. Per-target $T$ outperforms shared $T$ on
binary and full-ordinal targets but the gap is < 0.025 ECE. **Deployment
recommendation:** use **per-target $T^\star$**; shared $T$ is acceptable
when local calibration data are scarce ($n < 50$ per target).

### 3.3 Hosmer–Lemeshow

Reported in `outputs/paper1_calibration/results/internal_*.json`. The
22-feat-full pipeline shows HL $p < 10^{-4}$; the 21-feat strict-Path-3
pipeline reduces miscalibration substantially after $T^\star$.

---

## 4. Example conformal outputs (5 representative patients)

All five patients drawn from the PPMI internal CV pool. PATNOs anonymised
to `PT-XXXX` (last 4 digits modulo 10,000). Probabilities are
post-temperature-scaling. Conformal sets use LAC scoring at 90% CL with
the post-$T^\star$ thresholds reported in §V.D of the manuscript:
$\tau_{\text{bin}}=0.657$, $\tau_{3c}=0.783$, $\tau_{\text{ord}}=0.812$,
$\tau_{\text{nsd+}}=0.824$ (i.e. include class $k$ if $p_k \geq 1 - \tau$).

### Patient PT-2925 — high-confidence NSD$+$, true Stage 3

| Feature | Value | Feature | Value |
|---|---|---|---|
| AGE_AT_BASELINE | 67.0 | UPDRS3_AXIAL | 2.0 |
| SEX | 1 (F) | UPDRS4_TOTAL | 0.0 |
| HANDED | 1 (R) | MOCA_TOTAL | 28.0 |
| UPDRS1_TOTAL | 11.0 | RBD_TOTAL | 5.0 |
| UPDRS2_TOTAL | 4.0 | ESS_TOTAL | 6.0 |
| UPDRS3_TREMOR | 5.0 | SCOPA_AUT_TOTAL | 8.0 |
| UPDRS3_RIGIDITY | 4.0 | CAUDATE_MEAN_SBR | 1.34 |
| UPDRS3_BRADYKINESIA | 11.0 | CAUDATE_ASYMMETRY | 0.13 |
| LRRK2_CARRIER | 0 | GBA_CARRIER | 0 |
| APOE_E4_CARRIER | 0 |  |  |

- Binary: $p_+ = 0.9998$ → singleton **{NSD+}**
- Three-class: (0.004, 0.050, 0.947) → singleton **{Impaired (3-4)}**
- NSD$+$ sub-stage: (Stage 1: 0.017, 2B: 0.264, 3: 0.678, 4: 0.041) → set **{Stage 2B, Stage 3}**

**Clinician interpretation:** All three layers agree NSD$+$ + impaired.
The sub-staging set spans 2B–3, indicating residual uncertainty between
clinical-parkinsonism and mild functional impairment. Reportable as
"NSD$+$, sub-stage 2B or 3" — present alternatives.

### Patient PT-0717 — high-confidence NSD$-$, true Stage 0

| Feature | Value | Feature | Value |
|---|---|---|---|
| AGE_AT_BASELINE | 64.7 | UPDRS3_AXIAL | 0.0 |
| SEX | 1 (F) | UPDRS4_TOTAL | (NA) |
| HANDED | 1 (R) | MOCA_TOTAL | 28.0 |
| UPDRS1_TOTAL | 4.0 | RBD_TOTAL | 1.0 |
| UPDRS2_TOTAL | 0.0 | ESS_TOTAL | 5.0 |
| UPDRS3_TREMOR | 0.0 | SCOPA_AUT_TOTAL | 4.0 |
| UPDRS3_RIGIDITY | 0.0 | CAUDATE_MEAN_SBR | 3.20 |
| UPDRS3_BRADYKINESIA | 0.0 | CAUDATE_ASYMMETRY | 0.045 |
| LRRK2_CARRIER | 0 | GBA_CARRIER | 0 |
| APOE_E4_CARRIER | 0 |  |  |

- Binary: $p_+ = 0.0012$ → singleton **{NSD-}**
- Three-class: (0.998, 0.001, 0.001) → singleton **{Early (0-1)}**

**Clinician interpretation:** Decisive NSD$-$ with normal motor and
imaging features. Reportable singleton — no further escalation. Patient
not eligible for §V NSD$+$ sub-staging stage of the two-stage pipeline.

### Patient PT-3003 — borderline three-class abstention, true Stage 3

| Feature | Value | Feature | Value |
|---|---|---|---|
| AGE_AT_BASELINE | 66.0 | UPDRS3_AXIAL | 1.0 |
| SEX | 1 (M) | UPDRS4_TOTAL | (NA) |
| HANDED | 1 (R) | MOCA_TOTAL | 30.0 |
| UPDRS1_TOTAL | 1.0 | RBD_TOTAL | 4.0 |
| UPDRS2_TOTAL | 5.0 | ESS_TOTAL | 6.0 |
| UPDRS3_TREMOR | 6.0 | SCOPA_AUT_TOTAL | 9.0 |
| UPDRS3_RIGIDITY | 4.0 | CAUDATE_MEAN_SBR | 1.96 |
| UPDRS3_BRADYKINESIA | 7.0 | CAUDATE_ASYMMETRY | 0.13 |
| LRRK2_CARRIER | 0 | GBA_CARRIER | 0 |
| APOE_E4_CARRIER | 0 |  |  |

- Binary: $p_+ = 0.251$ → singleton **{NSD-}** (under threshold 0.560)
- Three-class: (0.661, 0.041, 0.297) → **{Early (0-1), Impaired (3-4)}** — non-contiguous multi-label set
- NSD$+$ sub-stage (computed for diagnostic interest): (Stage 1: 0.012, 2B: 0.056, 3: 0.927, 4: 0.005) → singleton **{Stage 3}**

**Clinician interpretation:** Disagreement across layers. Binary says
NSD$-$ at the recommended cutoff, but three-class returns a multi-label
set spanning Early and Impaired (skipping the middle Clinical 2B class) —
classic conformal abstention signal. Manuscript guidance (§IV.D)
escalates non-contiguous multi-label sets to clinician adjudication.
The NSD$+$ sub-staging (which assumes the patient is NSD$+$, contrary to
the binary call) returns a confident Stage-3 prediction. **Action:**
collect SAA + DaT-SPECT to break the binary-vs-substaging tie before
acting; do not auto-route.

### Patient PT-3002 — NSD$+$ sub-staging mid-confidence, true Stage 3

| Feature | Value | Feature | Value |
|---|---|---|---|
| AGE_AT_BASELINE | 56.7 | UPDRS3_AXIAL | 1.0 |
| SEX | 1 (M) | UPDRS4_TOTAL | (NA) |
| HANDED | 2 (L) | MOCA_TOTAL | 30.0 |
| UPDRS1_TOTAL | 9.0 | RBD_TOTAL | 8.0 |
| UPDRS2_TOTAL | 4.0 | ESS_TOTAL | 1.0 |
| UPDRS3_TREMOR | 8.0 | SCOPA_AUT_TOTAL | 7.0 |
| UPDRS3_RIGIDITY | 1.0 | CAUDATE_MEAN_SBR | 2.45 |
| UPDRS3_BRADYKINESIA | 6.0 | CAUDATE_ASYMMETRY | 0.07 |
| LRRK2_CARRIER | 0 | GBA_CARRIER | 0 |
| APOE_E4_CARRIER | 0 |  |  |

- Binary: $p_+ = 0.010$ → singleton **{NSD-}** (binary disagrees with ground truth — known PPMI HC contamination of NSD$-$ class)
- Three-class: (0.972, 0.014, 0.014) → singleton **{Early (0-1)}**
- NSD$+$ sub-stage: (Stage 1: 0.048, 2B: 0.279, 3: 0.661, 4: 0.012) → set **{Stage 2B, Stage 3}**

**Clinician interpretation:** Two-stage deployment behaves as designed:
binary layer mis-routes this patient to the NSD$-$ branch (one of the
known failure modes — see §V.E PD-only retraining sensitivity). If a
clinician overrides the binary gate and forces the NSD$+$ sub-staging
arm, the conformal set {Stage 2B, Stage 3} brackets the true Stage 3.
**Lesson:** the two-stage deployment is not infallible; clinician
override remains essential for ambiguous binary outputs.

### Patient PT-3607 — rare Stage 4 (n=17 in PPMI), true Stage 4

| Feature | Value | Feature | Value |
|---|---|---|---|
| AGE_AT_BASELINE | 63.4 | UPDRS3_AXIAL | 7.0 |
| SEX | 1 (M) | UPDRS4_TOTAL | 5.0 |
| HANDED | 1 (R) | MOCA_TOTAL | 28.0 |
| UPDRS1_TOTAL | 12.0 | RBD_TOTAL | 5.0 |
| UPDRS2_TOTAL | 13.0 | ESS_TOTAL | 9.0 |
| UPDRS3_TREMOR | 9.0 | SCOPA_AUT_TOTAL | 19.0 |
| UPDRS3_RIGIDITY | 8.0 | CAUDATE_MEAN_SBR | 1.66 |
| UPDRS3_BRADYKINESIA | 12.0 | CAUDATE_ASYMMETRY | 0.07 |
| LRRK2_CARRIER | 0 | GBA_CARRIER | 0 |
| APOE_E4_CARRIER | 0 |  |  |

- Binary: $p_+ = 0.892$ → singleton **{NSD+}**
- Three-class: (0.201, 0.298, 0.501) → **{Clinical (2B), Impaired (3-4)}** — multi-label
- NSD$+$ sub-stage: (Stage 1: 0.004, 2B: 0.371, 3: 0.606, 4: 0.019) → set **{Stage 2B, Stage 3}**

**Clinician interpretation:** Binary is decisive. Three-class returns a
plausible adjacent-class set. NSD$+$ sub-staging set bridges 2B-3 but
misses the true Stage 4 — a known limitation given Stage 4 prevalence is
0.8% (17/2,201) in PPMI. **Action:** report sub-staging set as 2B-3
with explicit clinician note that Stage 4 cannot be ruled out at base
rate <1%; recommend in-person motor assessment for functional
impairment.

---

## 5. Licensing matrix

| Component | Version | Licence | Clinical-deployment status |
|---|---|---|---|
| **CatBoost** (primary model) | 1.2.10 | Apache 2.0 | **OK** — permissive, commercial deployment unrestricted. |
| **TabPFN v2 (sensitivity variant only)** | 0.1.x client + v2 weights | Client: Apache 2.0; **weights: CC-BY-NC-SA 4.0** (non-commercial) | **RESTRICTED.** Commercial clinical deployment requires a separate PriorLabs licence. **The recommended deployable does NOT include TabPFN v2.** TabPFN appears in the manuscript's SOTA-equivalence comparison only. |
| **AutoGluon** | 1.5 | Apache 2.0 | **OK.** Used as a SOTA equivalence comparator; not in the recommended deployable. |
| **MAPIE** (conformal prediction) | 1.3.0 | BSD-3-Clause | **OK.** |
| **PyTorch** | 2.8.0 (2.9.1 sidecar) | BSD-3-Clause | **OK.** Used by GAT comparators; not in the CatBoost deployable. |
| **PyTorch Geometric** | 2.6.1 | MIT | **OK.** GAT comparator only. |
| **scikit-learn** | 1.5 | BSD-3-Clause | **OK.** Cross-validation, metrics, preprocessing. |
| **LightGBM, XGBoost** | 4.6.0 / 3.2.0 | MIT / Apache 2.0 | **OK.** Tabular comparators. |
| **NumPy, pandas, SciPy** | latest | BSD-3-Clause | **OK.** |
| **PPMI training data** | 2024-2025 release | Michael J. Fox Foundation DUA — research, non-commercial | DUA governs data redistribution; trained model weights derived from PPMI may be redistributed per PPMI policy if no patient-level data leaks through model inversion. **Confirm with PPMI Data Coordination Center before any commercial redistribution of trained CatBoost weights.** |

**Net licensing posture for the recommended deployable
(CatBoost-default-21feat-NSDISS-v1.0):** Apache-2.0 / BSD-3-Clause /
MIT throughout the software stack — **fully permissive for both
academic and commercial clinical deployment**, subject only to the PPMI
DUA constraint on the training data lineage.

**Key risk to flag to integrators:** if a downstream team re-runs the
SOTA-equivalence sensitivity in §IV.D using TabPFN v2, they cannot ship
that variant in a commercial product without separately licensing the
TabPFN v2 model weights from PriorLabs (CC-BY-NC-SA 4.0, "NC" = no
commercial use). The CatBoost-default deployable avoids this entirely.

---

## 6. Recommended workflow (3-step deployable)

### Step 1 — Site preparation (one-time per deployment site)

1. **Feature plumbing.** Ensure the local EHR / research database can
   emit the 21-feature Path-3 vector (§1) per patient per visit. If the
   site lacks SAA + DaT-SPECT, fall back to the 12-feature
   clinical-only subset (`AGE_AT_BASELINE`, `SEX`, `UPDRS1_TOTAL`,
   `UPDRS2_TOTAL`, `UPDRS3_{TREMOR,RIGIDITY,BRADYKINESIA,AXIAL}`,
   `UPDRS4_TOTAL`, `MOCA_TOTAL`, `ESS_TOTAL`, `RBD_TOTAL`); accept the
   $-17.6$pp binary-AUC penalty documented in §V.B.
2. **Local recalibration.** Collect $n \geq 50$ labelled local cases per
   target. Refit the per-target temperature scalar $T^\star$ via NLL
   minimisation on local validation logits (script:
   `scripts/paper1/run_temperature_scaling.py`). Persist as
   `local_T_star.json` keyed by target name.
3. **Conformal calibration.** Refit the LAC quantile $\tau$ on local
   labelled hold-out via cross-conformal CV+ (script:
   `scripts/run_conformal_benchmark.py`). Persist as
   `local_tau.json`.

### Step 2 — Per-patient inference

```
features_21d = build_path3_features(patient_record)
raw_probs    = catboost_model.predict_proba(features_21d)
cal_probs    = temperature_scale(raw_probs, T_star=load("local_T_star.json"))
pred_set     = lac_set(cal_probs, tau=load("local_tau.json"), alpha=0.10)
```

Surface to the dashboard: per-target `cal_probs`, the LAC set
`pred_set`, the recommended threshold-0.560 binary call, and the model
version string `CatBoost-default-21feat-NSDISS-v1.0`.

### Step 3 — Clinician action protocol

| LAC set behaviour | Recommended action |
|---|---|
| **Singleton** | Report the predicted class as a calibrated estimate with the 90% CL annotation. |
| **Multi-label, contiguous** (e.g. {2B, 3}) | Present alternatives to the clinician; do not auto-route. |
| **Multi-label, non-contiguous** (e.g. {0-1, 3-4}) | Escalate per §IV.D abstention semantics — collect missing biomarkers (SAA, DaT-SPECT) before acting. |
| **Empty set** | Escalate; insufficient evidence for a 90% CL prediction at this confidence. (Empty sets are rare on binary by construction; more common on three-class and full-ordinal targets.) |

---

## 7. Limitations and out-of-scope warnings

1. **No prospective validation.** Every metric in §1 is retrospective
   on PPMI / BioFIND. A prospective deployment study is a precondition
   for any FDA-equivalent clearance pathway.
2. **External multiclass calibration is degraded.** The marginal
   conformal coverage on BioFIND falls to 0.499 for three-class — local
   recalibration (Step 1.2) is mandatory before clinician-facing
   deployment.
3. **PPMI cohort enrichment.** LRRK2 and GBA carriers are
   over-represented vs community PD prevalence; model behaviour on
   sporadic PD without genetic carrier status is well within the
   training distribution, but calibration may need adjustment for
   prodromal and at-risk cohorts.
4. **NSD-ISS itself is contested.** Espay et al. (Mov Disord 2025) raise
   medication-confound and clinical-validity concerns. Predictions
   should be framed as biological-stage estimates, not clinical
   diagnoses.

---

## 8. Point-of-contact

Correspondence to Blair Dupre <blair.dupre@und.edu>. Trained CatBoost
weights, $T^\star$ calibration scalars, and local-deployment templates
available on request, subject to the PPMI DUA chain.
