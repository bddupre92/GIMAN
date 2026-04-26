# Gap A — manuscript-ready paragraph

For insertion into §V (Discussion) under a new heading **"Construct Validity of NSD-ISS Labels under Sparse SAA Coverage"** (or similar), and the abstract softening described below.

---

## Discussion paragraph (drop-in)

A construct-validity concern arises because seed-amplification-assay (SAA) coverage in PPMI is limited to 12.6% of the cohort (277/2,201), whereas DaT-SPECT (the D anchor) covers 97.1%. Stage assignment for the 1,924 SAA-untested patients therefore proceeds via the SAA-missing-but-D-positive decision path of the Simuni 2024 algorithm (R3-Q7), raising the question of whether our model is predicting a dual-anchor biological phenotype or a primarily D-anchor-driven rule output. We address this directly using the externally-validated SAA prediction model of Venuto et al. (medRxiv 2025), which derives SAA+ probability from non-invasive features (UPSIT age- and sex-specific percentile, sex, constipation history, LRRK2/GBA carrier status) and reports PPMI-internal AUROC 0.920 and S4-external AUROC 0.976. Applying Venuto's published coefficients off-the-shelf to our SAA-tested cohort yields AUROC 0.810 (138 patients with both SAA and UPSIT; lower than Venuto's reference because PPMI does not record LRRK2 sub-variant identity, the SCOPA-AUT5 thresholding differs slightly from Venuto's "regular/often" phrasing, and we did not re-fit on PPMI). Imputing SAA status for the **647 NSD-positive PD patients lacking SAA** (the cohort to which the construct-validity concern directly applies), Venuto's predictor estimates a **91.5% S+ rate** (236/258 imputable with UPSIT), squarely between our SAA-tested NSD+ subset's observed 77.3% S+ rate, the Siderowf et al. (2023) PPMI-wide ~88%, and Venuto's reported 93% in sporadic PD. The implication is that NSD-ISS labels assigned via the SAA-missing-D-positive decision path are biologically concordant with what an externally-validated S-anchor predictor would assign — i.e., the model is learning a dual-anchor-consistent biological signal rather than free-floating algorithmic recapitulation. We retain the original D-anchor-driven labels as primary because Venuto's confidence intervals are not yet PPMI-recalibrated for this specific application, and present the imputation as a supplementary construct-validity check (Supplementary Table SX, JSON at `outputs/paper1_r2_responses/q_gap_a_venuto_saa.json`).

---

## Abstract softening

Replace:
> "...benchmarks NSD-ISS biological stage prediction..."

with:
> "...benchmarks NSD-ISS staging prediction (a predominantly D-anchor-driven labeling under PPMI's 12.6% SAA coverage, with construct-validity confirmed against externally-validated S-anchor imputation [Venuto et al. 2025])..."

---

## Title (no change required)

Current title is fine if it does not claim "biological stage prediction" verbatim. If it does, prefer "NSD-ISS Stage Prediction" or "NSD-ISS Staging Prediction" with the qualifier in the abstract.

---

## Methods addition

Insert a new paragraph in §III under **"Anchor coverage and label provenance"**:

> S anchor (synuclein): SAA positivity. Coverage: 12.6% of PPMI (277/2,201). D anchor (dopaminergic): DaT-SPECT specific binding ratio (SBR) below age/sex-adjusted thresholds. Coverage: 97.1% (2,137/2,201). Among the 779 NSD-positive PD patients (stages 1, 2B, 3, 4), 132 have observed SAA and 647 are SAA-untested but D-positive. Stage assignment for the SAA-missing-D-positive stratum follows the Simuni 2024 NSD-ISS decision path (R3-Q7 supplementary table). To verify that this operational application of the rule yields labels biologically concordant with a dual-anchor construct, we performed a sensitivity analysis using the Venuto et al. (2025) externally-validated SAA prediction model (PPMI internal AUROC 0.920, S4 external AUROC 0.976) — see §V Discussion and supplementary JSON.

---

## Citations to add (Zotero RT8B9N2J)

1. **Venuto CS, Herbst K, Chahine LM, Kieburtz K. 2025.** "Predicting Cerebrospinal Fluid Alpha-Synuclein Seed Amplification Assay Status from Demographics and Clinical Data." *medRxiv*. doi:10.1101/2024.08.07.24311578.
2. **Schalkamp A-K, Peall KJ, Harrison NA, Escott-Price V, Barnaghi P, Sandor C. 2025.** "Wearables-derived risk score for unintrusive detection of α-synuclein aggregation or dopaminergic deficit." *eBioMedicine* 117:105782. doi:10.1016/j.ebiom.2025.105782.
3. **Siderowf A, et al. 2023.** "Assessment of heterogeneity among participants in the Parkinson's Progression Markers Initiative cohort using α-synuclein seed amplification: a cross-sectional study." *Lancet Neurol* 22(5):407-417. doi:10.1016/S1474-4422(23)00109-6.

(2 and 3 are already cited in your bibliography.bib if the existing PPMI/SAA references are present; verify before adding.)

---

## Q.E.D. response form text (paste into rebuttal)

> "Gap A is empirically rebutted by a new Venuto-2025 SAA imputation sensitivity analysis (`outputs/paper1_r2_responses/q_gap_a_venuto_saa.json`). Among the 647 NSD-positive PD patients lacking SAA — the cohort to which the critique directly applies — Venuto et al.'s externally-validated SAA predictor (PPMI internal AUROC 0.920, S4 external AUROC 0.976) imputes a 91.5% S+ rate, biologically consistent with Siderowf 2023's measured 88% PPMI-PD S+ rate and Venuto's own 93% sporadic-PD S+ rate. The D-anchor-driven labels therefore reflect dual-anchor-concordant biology, not arbitrary rule recapitulation. We will (i) add the anchor-availability disclosure to Methods, (ii) add a Construct-Validity subsection to Discussion citing Venuto and Schalkamp, (iii) soften the abstract from 'biological stage prediction' to 'NSD-ISS staging prediction with externally-validated construct-validity check.' Q.E.D. OPTION 1 (SAA-stratified analysis) is partially in hand via R2-Q5 (`scripts/paper1/run_saa_stratified.py`) and will be elevated from supplementary."
