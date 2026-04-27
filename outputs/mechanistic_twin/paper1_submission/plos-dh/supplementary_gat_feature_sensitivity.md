# Supplementary S-3 — Multimodal GAT Feature-Count Sensitivity Variant

**Companion to:** IEEE JBHI submission `main.tex` (chapter_content.tex §IV-D Graph-Based Models + §V-B Discussion)

**Question addressed:** The 2-modality Multimodal GAT reported in the submission's Table IV underperforms CatBoost by 8–13 percentage points in balanced accuracy on all four NSD-ISS targets. Is this gap driven by the limited 22-feature input (feature-count ceiling) or by the graph-attention architecture itself (architectural ceiling)? This supplementary tests the hypothesis by admitting 10 additional observed PPMI features into a third modality stream and retraining the same architecture with identical hyperparameters.

---

## S-3.1 Purpose and pre-registration

Pre-registered decision gate (set before the benchmark was run, 2026-04-22):

- If mean gap (3-mod vs CatBoost) ≤ 3 pp across 4 targets → FLAG for user review; feature-count is the binding constraint.
- If 3 pp < mean gap ≤ 8 pp → Narrative shift: GNNs become competitive with additional modalities; reframe submission §V-B Discussion.
- If mean gap > 8 pp → Current narrative holds; sensitivity confirms Grinsztajn-2022 finding that trees dominate on tabular clinical data regardless of input dimensionality.

Primary metric: balanced accuracy, 5-fold stratified CV, seed 42, 1,000-resample bootstrap 95% CIs.

## S-3.2 Architecture

A fork of the 2-modality MM-GAT reported in the main submission (§III-E-ii). The input is partitioned into three modality streams; each stream has its own encoder and its own per-modality GAT stack; pairwise cross-modal multi-head attention then fuses them before a classification head.

- Each modality $m \in \{\text{Clinical}, \text{Biomarker}, \text{Extended}\}$: `input_d_m → FC(128) → LayerNorm → ReLU → Dropout(0.3) → FC(128) → LayerNorm → ReLU → Dropout(0.3)`.
- Per-modality GAT stack: 3-layer PyG GATConv, 4 attention heads per layer, hidden 128, residual connections. Same k=10 cosine-similarity kNN patient-similarity graph as the 2-modality model, built on the concatenated 32-dim standardised vector, training-only per fold, with inductive test-node extension via nearest-training-neighbour projection.
- Cross-modal attention: 3 `nn.MultiheadAttention(embed=128, heads=4)` blocks — Clinical↔Biomarker, Clinical↔Extended, Biomarker↔Extended (the B.i pairwise variant). Each block applies residual + LayerNorm.
- Fusion: `concat([h_C_updated, h_B_updated, h_S_updated]) → Linear(384, 128) → LayerNorm → ReLU → Dropout(0.3)`.
- Classifier: `Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, K)`.

All hyperparameters match the 2-modality MM-GAT exactly: seed 42, 150 epochs with early-stop patience 15 on 10% internal validation split, AdamW lr 1e-3 wd 1e-4, CrossEntropyLoss with balanced class weights, MPS backend.

## S-3.3 Feature partition (32 features across 3 modalities)

| Modality | Count | Features | Original source / citation |
|---|---|---|---|
| Clinical | 14 | AGE_AT_BASELINE, SEX, HANDED, UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS4_TOTAL, UPDRS3_TREMOR, UPDRS3_RIGIDITY, UPDRS3_BRADYKINESIA, UPDRS3_AXIAL, MOCA_TOTAL, RBD_TOTAL, ESS_TOTAL, SCOPA_AUT_TOTAL | Main Table II — PPMI clinical / cognitive / sleep / autonomic routine assessments |
| Biomarker | 8 | CAUDATE_L_SBR, CAUDATE_R_SBR, CAUDATE_MEAN_SBR, CAUDATE_ASYMMETRY, CAUDATE_PUTAMEN_RATIO, LRRK2_CARRIER, GBA_CARRIER, APOE_E4_CARRIER | Main Table II — DaT-SPECT (`DaTScan_SBR_Analysis_*.csv`) + PPMI genetic-consensus LRRK2/GBA/APOE flags |
| Extended | 10 | CTH_ENTORHINAL_L/R, CTH_POSTCINGULATE_L/R, CTH_PRECENTRAL_L/R, CSF_ASYN, CSF_ABETA42, CSF_PTAU181, CSF_TTAU | Cortical thickness: Fischl 2012 (FreeSurfer); CSF biomarkers: Mollenhauer 2017 (PPMI CSF methodology canonical) |
| **Total** | **32** | | |

**GRS_TOTAL deliberately omitted.** The pre-registered target was 11 extra features including a Nalls-2019 weighted polygenic risk score. The PPMI `iu_genetic_consensus_20250515_*.csv` file publishes 7 carrier flags and APOE genotype strings, but not a pre-computed weighted PRS. Computing a Nalls-2019 PRS from raw PPMI genotypes would require the full chromosome-level genotype release and is out of scope for this Paper 1 first-journal submission. The Extended stream is therefore 10-d and the total feature count is 32, not 33.

**Coverage (pre-imputation).** Cortical thickness: 1,086 / 2,201 patients (49.3%) at baseline visit. CSF biomarkers: 757–860 / 2,201 patients (34.4–39.1%), earliest of BL/SC/V01/V02/V04 per biomarker per patient. Patients with all 10 extra features fully observed: 581. Missing values are imputed with per-feature training-fold median (same protocol as the 2-modality MM-GAT).

## S-3.4 Per-target per-fold results

Source JSONs: `outputs/paper1_enhanced_gat_3mod/{binary,three_class,full_ordinal,nsd_positive}_results.json` (2026-04-22).

### Binary (n=2,201; K=2)

| Fold | bal_acc | AUC | QWK |
|---|---|---|---|
| 1 | 0.9262 | 0.9683 | 0.8599 |
| 2 | 0.9265 | 0.9751 | 0.8467 |
| 3 | 0.9542 | 0.9792 | 0.9150 |
| 4 | 0.9369 | 0.9812 | 0.8713 |
| 5 | 0.9266 | 0.9839 | 0.8472 |
| **Mean ± SD** | **0.934 ± 0.011** | **0.978 ± 0.005** | **0.868 ± 0.025** |
| Bootstrap 95% CI | [0.923, 0.945] | [0.969, 0.983] | [0.846, 0.890] |

### Three-class (n=2,197; K=3)

| Fold | bal_acc | AUC (macro OvR) | QWK |
|---|---|---|---|
| 1 | 0.7298 | 0.9073 | 0.6561 |
| 2 | 0.7789 | 0.9405 | 0.8592 |
| 3 | 0.7571 | 0.9261 | 0.7619 |
| 4 | 0.7636 | 0.9331 | 0.7826 |
| 5 | 0.7370 | 0.9140 | 0.7556 |
| **Mean ± SD** | **0.753 ± 0.018** | **0.924 ± 0.012** | **0.763 ± 0.065** |
| Bootstrap 95% CI | [0.730, 0.779] | [0.907, 0.933] | [0.737, 0.788] |

### Full ordinal (n=2,197; K=5)

| Fold | bal_acc | AUC (macro OvR) | QWK |
|---|---|---|---|
| 1 | 0.5790 | 0.8774 | 0.6294 |
| 2 | 0.3687 | 0.8325 | 0.6462 |
| 3 | 0.4388 | 0.8435 | 0.6391 |
| 4 | 0.6390 | 0.8998 | 0.7053 |
| 5 | 0.4870 | 0.8468 | 0.6013 |
| **Mean ± SD** | **0.502 ± 0.097** | **0.860 ± 0.025** | **0.644 ± 0.034** |
| Bootstrap 95% CI | [0.449, 0.550] | [0.839, 0.875] | [0.616, 0.673] |

### NSD-positive subgroup (n=779; K=4; Stage 0 excluded)

| Fold | bal_acc | AUC (macro OvR) | QWK |
|---|---|---|---|
| 1 | 0.3458 | 0.6739 | 0.1495 |
| 2 | 0.4981 | 0.7708 | 0.4470 |
| 3 | 0.2853 | 0.6028 | 0.1312 |
| 4 | 0.3837 | 0.6389 | 0.1090 |
| 5 | 0.5706 | 0.8004 | 0.3357 |
| **Mean ± SD** | **0.417 ± 0.104** | **0.697 ± 0.076** | **0.234 ± 0.133** |
| Bootstrap 95% CI | [0.347, 0.485] | [0.640, 0.725] | [0.180, 0.295] |

## S-3.5 Gap analysis and decision-gate verdict

Comparing the 3-modality 32-feature variant against the main submission's 2-modality 22-feature MM-GAT and CatBoost (22), all on 5-fold stratified CV seed 42:

| Target | CatBoost (22) | MM-GAT 2-mod (22) | MM-GAT 3-mod (32) | Gap 2-mod vs CB | **Gap 3-mod vs CB** | Δ (3-mod − 2-mod) |
|---|---|---|---|---|---|---|
| Binary | 0.951 | 0.825 ± 0.013 | **0.934 ± 0.011** | −12.6 pp | **−1.7 pp** | **+10.9 pp** |
| Three-class | 0.783 | 0.705 ± 0.033 | **0.753 ± 0.018** | −7.8 pp | **−3.0 pp** | +4.8 pp |
| Full ordinal | 0.658 | 0.549 ± 0.044 | 0.502 ± 0.097 | −10.9 pp | −15.6 pp | −4.7 pp |
| NSD+ subgroup | 0.671 | 0.544 ± 0.060 | 0.417 ± 0.104 | −12.7 pp | −25.4 pp | −12.7 pp |

**Mean gap (3-mod vs CatBoost) = 11.4 pp** → the pre-registered decision gate (mean > 8 pp) is satisfied, so the submission's current narrative (trees dominate graph attention on tabular PD data, per Grinsztajn et al. 2022) is preserved. However the per-target pattern is heterogeneous and informative:

1. **Binary detection and three-class triage (clinically the two most important targets)** — the 3-modality variant closes almost the entire 2-modality gap: +10.9 pp on binary (−12.6 → −1.7) and +4.8 pp on three-class (−7.8 → −3.0). At −1.7 pp binary and −3.0 pp three-class, the MM-GAT is within bootstrap-CI overlap of CatBoost. This directly rebuts the "GAT architecturally loses to trees" reading of the main Table IV: on the targets where NSD-ISS prediction is most clinically valuable, adding observed PPMI imaging and CSF features closes almost the entire architectural gap.

2. **Full ordinal (5 classes, 17 Stage 4 patients) and NSD+ subgroup (4 classes, 17 Stage 4 within 779 patients)** — the 3-modality variant *widens* the gap: −15.6 pp and −25.4 pp, respectively. The most parsimonious explanation is that the Extended modality's ~50% CTH and ~35% CSF coverage rates translate into median-imputed features in exactly the minority classes where the stratified folds contain only 3–4 patients per class. The Extended stream then contributes signal noise that outweighs signal, penalising rare-class balanced accuracy. This hypothesis is falsifiable — restricting the analysis to the 581 patients with fully-observed Extended features would remove imputation noise, at the cost of reducing the Stage 4 subset from 17 to approximately 10. We mark that analysis as future work.

Given this heterogeneity, the correct reading of the sensitivity is: **the 2-modality MM-GAT's apparent architectural gap to CatBoost is partly a feature-count effect (closes to 1.7–3.0 pp on binary and three-class once imaging + CSF features are admitted) and partly a coverage-noise effect (worsens on rarer-class targets where imputation is unreliable).** Grinsztajn 2022's conclusion still holds in aggregate, but on clinically important coarse-grained targets with sufficient observed extended-feature support the GAT is genuinely competitive.

## S-3.6 Reproducibility

| Step | Script | Output |
|---|---|---|
| Data assembly | `scripts/paper1/assemble_extended_33_feat.py` | `data/05_features/paper1_features_extended_33.csv` (2,201 × 48) + `paper1_features_extended_33_metadata.json` |
| Benchmark (4 targets × 5 folds) | `scripts/run_enhanced_gat_3modality_benchmark.py` | `outputs/paper1_enhanced_gat_3mod/{target}_results.json` + `summary.json` |

Random seed 42 for StratifiedKFold, PyTorch manual_seed, NumPy RandomState, and bootstrap resampling. Total runtime 110 seconds on Apple M-series with MPS backend. The benchmark script writes per-fold probability matrices (`probs`), predicted-class arrays (`preds`), and ground-truth arrays (`y_true`) into the per-target JSONs, so the full bootstrap and CI analysis is post-hoc reconstructible from the JSONs without re-running the model.

---

**Citations:**

- Fischl, B. (2012). FreeSurfer. *NeuroImage*, 62(2), 774–781. doi:10.1016/j.neuroimage.2012.01.021 — for cortical thickness regions.
- Mollenhauer, B., Caspell-Garcia, C.J., Coffey, C.T., et al. (2017). Longitudinal CSF biomarkers in patients with early Parkinson disease and healthy controls. *Neurology*, 89(19), 1959–1969. doi:10.1212/WNL.0000000000004609 — for PPMI CSF methodology.
- Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS — for the aggregate-level "trees beat DL on tabular" finding this supplementary confirms.
