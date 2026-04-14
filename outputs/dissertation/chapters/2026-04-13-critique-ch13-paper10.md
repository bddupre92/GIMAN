# Manuscript Critique — Chapter 13 (Paper 10)

**Date:** 2026-04-13
**Paper:** *Bidirectional-Ready Mechanistic Patient-Specific Model with External Validation and NASEM Audit* (dissertation Ch 13; target venue *npj Parkinson's Disease* or *J Parkinson's Disease*)

## Summary

The chapter presents a patient-specific mechanistic Parkinson's disease model that ingests new DaT-SPECT observations via Sequential Importance Resampling (SIR), updates its posterior without full Julia re-calibration, and applies the Phase~4 Path~B interaction coefficients without retuning to predict observational LEDD-escalation responses. Four empirical claims are advanced: (1) held-out SBR MAE drops 33\% monotonically with SIR updates; (2) cross-sectional external validation on the LCC cohort places the HC-vs-PD gap inside the 40--200\% literature range; (3) paired-bootstrap head-to-head vs Graph-DT on time-to-wearing-off gives both models near-random $C$-indices (mech 0.472, Graph-DT 0.518; paired $p = 0.046$); (4) observational counterfactual on 481 LEDD-escalation events yields calibration slope $1.074$ with 95\% CI $[0.88, 1.29]$ containing $1.0$. A seventh-criterion NASEM self-audit scores 16/21 with no absent criteria.

## Top findings

### Top risks

1. **External validation is cross-sectional only.** Reviewers at both target venues will ask why longitudinal external DaT-SPECT was not used. The chapter correctly flags this as a field-wide data-access gap (no public PD cohort with longitudinal DaT-SPECT), but the claim as written ("cross-sectional external validation is field standard") will still attract pushback because it reads like an excuse. Strengthen by (a) listing the five cohorts we explicitly audited and why each is unusable (LCC HC-only; PDBP SPECT is DLB-only; BioFIND and HBS no longitudinal DaT; SURE-PD3 pending DUA); (b) citing published DaT-SPECT cross-sectional validation papers (Chahine 2020, Wakasugi 2024) as precedent; (c) adding a one-paragraph table summarising the data-access state of the field.

2. **The head-to-head result is a near-wash on wearing-off.** Reviewers will ask "what did this benchmark establish?" The chapter's complementarity-not-competition framing is correct but reads as reactive unless Chapter~\ref{ch:discussion} fully develops the three-lines-of-evidence argument (Path~C in Chapter~\ref{ch:paper9} + Chapter~\ref{ch:paper10} head-to-head + Graph-DT transition-timing win in Paper~3). Consider one additional paragraph in §13.6 explicitly framing the head-to-head as a \emph{confirmatory null} that bounds the shared predictive territory, not as a failure of the mechanistic model. Reference: Nair 2026 review explicitly documents no published PD DT demonstrates bidirectional updating --- add this citation early in §13.1 to preempt "is this really novel?" critique.

3. **Mean-vs-median empirical inversion is undiscussed in the chapter body.** The figure caption of Fig 3 notes the weighted mean empirically outperforms the weighted median on MAE, which contradicts the classical Gelman BDA3 §2.5 prescription. The chapter body does not explain why. Reviewers at methodological venues will flag this. Add a short methodological subsection (§13.2.3 or a paragraph in §13.3.1) citing Vehtari \& Ojanen 2012 §3 on heavy-tailed predictive distributions and noting that for lognormal-prior PD posteriors the mean is more stable than the quantile median. The reasoning is already in the RUN\_MANIFEST; copy it verbatim into the chapter.

### What to preserve

- **The three-pathway → observational-counterfactual bridge.** The inference chain (Phase~4 fits $\beta_{\text{interaction}} = 1.41$ on one data slice → Paper~10 applies the same $\beta$ without retuning on 481 new LEDD-escalation events → calibration slope 95\% CI contains 1.0) is the strongest single argument for the mechanistic twin's counterfactual validity. Keep this structure intact; if anything, foreground it earlier in §13.1.
- **The NASEM self-audit with honest scoring.** Scoring 2/3 on five criteria and 3/3 on only two is credible, not disappointing. Preserving "no absent criteria" language is critical.
- **The negative-result framing for wearing-off.** Treating Path~C and the head-to-head as *informative negatives* rather than failures is the right scientific framing.
- **The regulatory framing (MIDD + ICH M15).** This positions the contribution inside field-accepted pharmacometric practice and should stay.

### Anticipated reviewer questions

- **R1:** "Why is R^2 = 0.245 on the calibration plot acceptable?" *Answer in chapter:* the relevant statistic is slope, not R^2, because individual visit-level $\Delta$gap has ~6-UPDRS-point measurement noise; R^2 is variance-accounting, slope is calibration.
- **R2:** "How do you know the model isn't overfit given Phase 4 is fit on PPMI and Paper 10 tests on a subset of PPMI?" *Answer in chapter:* the 481 LEDD-escalation events are new observational units (visit pairs with $\geq 200$~mg LEDD change) that were not in the Phase~4 training data except incidentally. Strengthen by noting the first-difference patient-anchoring absorbs fixed effects.
- **R3:** "What does 'episodic' bidirectional mean vs 'continuous' and why is episodic enough?" *Answer:* episodic = per-visit update; continuous = sensor-stream. The chapter cites Corral-Acero 2020 + Coorey 2021 cardiac twins as peers at the episodic tier. Reinforce this comparison.
- **R4:** "Is SIR regulatorily acceptable?" *Answer:* Dosne 2016 NONMEM SIR + Marshall 2016/2019/2023 MIDD + ICH M15 2024 are all cited — but consider adding one sentence clarifying that SIR is not a novel statistical method here, only novel in the PD DT application.
- **R5:** "Why Paper~10 not CPT:PSP?" *Answer:* Agent C's systematic sweep showed no CPT:PSP Bayesian-updating-for-PD precedent 2023--2026; therefore a PD-specialist venue (npj PD / J Parkinson's Dis) is a better fit. Consider stating this explicitly in a Discussion or Data Availability subsection.

## Detailed comments

### Literature and novelty

**Strengths.** The chapter cites 25+ verified works including the critical NASEM 2024 report, An \& Cockrell 2024 template, Viceconti 2021/2025 VVUQ, Musuamba 2021 credibility matrix, Friedrich 2016 MQM, and the full Chopin/Del Moral/Dosne/Vehtari methodology arc.

**Weaknesses.**
- §13.1 paragraph 2 states "no published PD mechanistic-model literature operates above the one-shot-fit tier" — cite Nair 2026 review (\verb|nair2026computationalpd|) explicitly here rather than in §13.6 Discussion. This is the headline novelty claim and should be supported at first statement.
- The MIDD regulatory framing (§13.2.2) is strong but the ICH M15 citation (\verb|ich_m15_2024|) is a 2024 draft. By the time this chapter is submitted, the draft may have been finalized or superseded. Check ICH website for post-draft updates before final submission.

### Methodological rigor

**Strengths.** Three-parameter posterior well below Beskos 2014 dimensional-stability threshold. PSIS-$\hat{k}$ + split-$\hat{R}$ diagnostic gate is pre-specified. RNG seeds pinned. 10/10 unit tests pass.

**Weaknesses.**
- **Reporting gap on bootstrap variance assumption.** The paired-bootstrap in §13.3.3 assumes exchangeability of patients within the shared cohort; reviewers may ask whether pre-vs-post-MIDS subgroup bootstrap separately would change the sign of the head-to-head result. Consider a sensitivity analysis in an appendix.
- **Observation noise $\sigma = 0.20$ is inherited from Phase 2.** Reviewers may ask whether the SIR update is robust to misspecification of $\sigma$. One sensitivity analysis (repeat with $\sigma \in \{0.15, 0.25\}$) would close this.

### Causal claims and confounders

**Strengths.** The chapter explicitly frames LEDD as a "severity proxy, not a causal modifier of $N(t)$" (citing Verschuur 2019 LEAP and Frequin 2024). The observational counterfactual analysis uses patient-anchored first-differences, which absorbs fixed effects.

**Weaknesses.**
- **Reverse causation in LEDD escalation.** Patients escalate LEDD *because* they worsen. The observed $\Delta$gap may partly reflect progression, not drug response. The chapter controls for severity via $\Delta$UPDRS3-OFF in the prediction formula, but a plot of predicted vs observed $\Delta$gap stratified by severity trajectory would close this concern definitively.
- **Confounding by indication.** Physicians who escalate LEDD earlier may differ from those who escalate later. Consider a paragraph in §13.6 acknowledging this limitation and noting it does not invalidate the slope=1 finding (which is mechanistic, not causal).

### Data quality and limitations

**Strengths.** The PPMI → LCC external validation is honest about the cross-sectional constraint. All limitations (PPMI-only training, no interventional data, episodic updates) are explicitly named in §13.6.

**Weaknesses.**
- **Domain shift to non-PPMI-like cohorts.** The chapter mentions the PPMI→LCC HC gap is 17.5\% (scanner/site effects) but does not discuss whether this shift biases the head-to-head or counterfactual results. Add a one-sentence note that internal and external findings may drift when deployed on different scanners.
- **Mean scan interval.** Figure 3's $x$-axis is "Scans used" but doesn't specify the time interval between scans. Consider adding: "In PPMI, scans are approximately annual until ~year 5, then biennial" to the figure caption or Methods.

### Generalizability

**Weaknesses.**
- The chapter does not explicitly discuss how generalizable the bidirectional-update infrastructure is to other PD cohorts. A one-paragraph "deployment consideration" subsection noting (a) PosteriorStore is cohort-agnostic if Phase~2 priors are re-fit, (b) the SIR updater is a pure Python module, (c) the forward model's pinned constants are literature-anchored not PPMI-fit, would strengthen generalizability.

### Mechanism

**Strengths.** The chapter ties bidirectional updates to specific mechanistic parameters ($k_n, \alpha_{\text{tox}}, T_{\text{tox}}$) rather than treating the twin as a black box.

**Weaknesses.**
- The linkage between $N(t)$ modulating Levodopa benefit and the biological mechanism (surviving AADC-expressing neurons → dopamine synthesis capacity) is implicit. Add 1--2 sentences in §13.1 making this explicit so the Path~B positive result in Paper~9 reads as mechanistically expected, not merely statistical.

### Clarity and presentation

**Strengths.** The chapter has a clear structure: Introduction → Methods (bidirectional arch + stats + external + h2h + counterfactual) → Results (4 empirical + NASEM) → Discussion. Tables are well-labeled. Figure references resolve.

**Weaknesses.**
- Table 13.4 (NASEM scorecard) is dense. Consider converting to a radar-chart subfigure (Fig 2 in the Paper 10 figure package already exists) as a visual complement. The chapter *refers* to `figures/fig2_nasem_radar.pdf` but does not embed it. Add the figure.
- §13.5 has a single long paragraph. Split into four paragraphs, one per NASEM tier finding (compliance summary, cardiac-peer comparison, gaps, forward-looking statement).

## Specific fixes to apply before submission

1. **Add Nair 2026 citation to §13.1 paragraph 2** supporting "no published PD DT demonstrates bidirectional updating."
2. **Expand footnote or paragraph in §13.5** to split the long NASEM-results paragraph into four.
3. **Embed Fig 2 (NASEM radar)** in §13.5 alongside Table 13.4.
4. **Add 3--5 sentences in §13.6** on reverse causation of LEDD escalation and the first-difference defense.
5. **Add 1 sentence in §13.3.1** explaining mean-vs-median choice per Vehtari \& Ojanen 2012.
6. **Check ICH M15 status** closer to submission date.

## Overall assessment

This chapter is in strong shape for venue submission. Its core empirical result (calibration slope CI contains 1.0 on 481 unseen LEDD escalations) is substantial, its methodological novelty (first published PD bidirectional-updating model) is defensible per the systematic literature sweep, and its honest framing (partial NASEM compliance, cross-sectional external) will survive hostile review. The six suggested fixes above are polish, not structural rework. Proceed to submission after applying them and re-running `/claude-scholar:check-refs` and `/claude-scholar:critique-figures` on the 9 Paper 10 figures.
