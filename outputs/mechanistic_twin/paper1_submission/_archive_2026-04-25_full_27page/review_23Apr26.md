The paper proposes the first machine-learning benchmark for predicting NSD-ISS biological stages in Parkinson’s disease from routinely collected PPMI data, with explicit uncertainty quantification via cross-conformal prediction (CV+). Across four clinically motivated target formulations, CatBoost and other tree ensembles substantially outperform a Multimodal Graph Attention Network; DaT-SPECT features are shown to be critical for binary NSD detection while clinical features alone suffice for NSD-positive sub-staging. The study also identifies and addresses a key training-label confound (healthy controls mixed into NSD-negative in PPMI), evaluates external transportability on BioFIND, and outlines a two-stage deployment strategy with calibrated set-valued predictions.
Strengths

Technical novelty and innovation
Introduces the first computational benchmark targeting NSD-ISS staging (binary, three-class, full ordinal, NSD+ sub-staging) with distribution-free uncertainty quantification via CV+ conformal prediction.
Proposes a clinically pragmatic two-stage deployment paradigm: (i) one-time biological confirmation requiring DaT-SPECT/SAA; (ii) subsequent clinical-only sub-staging for sites without imaging access.
Transparently surfaces and mitigates a training-label confound (NSD- negative class containing healthy controls), including a PD-only retraining protocol and a hierarchical Stage-A/Stage-B framing.
Experimental rigor and validation
Systematic comparison of eight models (tree ensembles, linear models, SVM, and a Multimodal GAT) under consistent 5-fold stratified CV with bootstrapped CIs.
External evaluation on BioFIND using a common-feature subset; additional prediction-only application to PDBP and a candid cautionary report for HBS with missing features.
Multiple sensitivity analyses: split vs. CV+ conformal calibration, extended feature-sets for both tabular and graph models, and ablation of imaging features.
Clarity of presentation
Clear articulation of clinical targets and their intended utility spectrum; useful justification for feature engineering and a “circularity audit.”
Honest reporting of negative or ambiguous findings (e.g., near-chance external binary result due to label structure, null effect of 33-feature extension).
Significance of contributions
Addresses an important, emergent biological staging paradigm (NSD-ISS) with high clinical interest, offering a reproducible baseline and calibrated uncertainty useful for deployment.
Provides evidence-based guidance on the necessity of DaT-SPECT for binary biological detection and feasibility of clinical-only NSD+ sub-staging.
Weaknesses

Technical limitations or concerns
Potential leakage risk in the graph-based pipeline: unclear whether k-NN graphs and standardization were constructed strictly within training folds in CV to avoid information spillover.
Conformal prediction reporting shows coverage greatly exceeding nominal levels and mean set sizes <1; the introduction of abstentions and the CL-level consistency are not fully specified, causing confusion.
Severe class imbalance (Stage 4 n=17) limits robust conclusions for the full ordinal setting; no ordinal-aware modeling or class-weighting explored as principled alternatives.
Medication-status confounding is flagged in related work, but implementation details of how medication status was handled (stratification, adjustment, covariates) are insufficiently specified.
Experimental gaps or methodological issues
No hyperparameter optimization for any model (especially the GAT), which may understate non-tree baselines; a limited, fair HPO for top contenders would strengthen conclusions.
External coverage and efficiency of conformal sets are not reported; only internal coverage is shown, leaving uncertainty about calibration under distribution shift.
Calibration metrics (e.g., ECE, Brier) and per-class coverage are not presented in the main text, despite claims of poor calibration on BioFIND three-class predictions.
Limited exploration of ordinal-aware objectives (e.g., CORAL, ordinal CatBoost, or ordinal CP scoring) for the 5-class target.
Clarity or presentation issues
Several table typos/incomplete CIs (e.g., Table III QWK brackets) and inconsistent narrative around confidence levels in conformal sections (“90% level rather than the 95% used here”) reduce clarity.
The rationale for dropping MOCA and UPDRS4 in full-feature CatBoost due to high missingness but retaining them in the external common-feature subset (with imputation) needs clearer explanation.
Missing related work or comparisons
The related work could better situate against DaT-SPECT-based DL diagnostics and manifold/classical pipelines (e.g., diffusion maps + LDA) and modern tabular DL/foundation models (e.g., TabPFNv2, AutoGluon) from recent benchmarks.
Ordinal modeling literature and ordinal conformal prediction are not discussed.
Detailed Comments

Technical soundness evaluation
The fundamental design—four target formulations aligned with workflow needs, exclusion of direct staging thresholds to reduce circularity, internal CV with bootstrapped CIs—is sound and clinically motivated.
Explicitly verify and describe foldwise preprocessing for every step that can leak information (standardization, graph construction, neighbor search, similarity metrics). If graphs were built on the entire dataset rather than foldwise, re-run with foldwise graphs.
Consider ordinal-specific models or loss functions for the 5-class target and ordinal-aware nonconformity scores for conformal prediction. This can better reflect the graded nature of NSD-ISS stages and potentially improve QWK and per-class performance.
The medication-status critique is important; detail exactly how medication information entered modeling (as covariates, exclusions, or stratification) and perform sensitivity analyses where feasible.
Experimental evaluation assessment
Internal benchmarking is thorough, but the lack of HPO—while justified to avoid bias—may particularly handicap deep methods. A constrained, nested-CV HPO for a small subset of hyperparameters (e.g., k in k-NN graph, learning rates/heads/layers) for the top 2–3 models would yield a more balanced assessment without overfitting.
External validation is valuable; however, expand reporting to include calibration (ECE, Brier), per-class confusion matrices, and class-conditional conformal coverage on BioFIND. This will substantiate claims about “poor calibration” and clarify conformal performance under shift.
The conformal results would benefit from reporting efficiency-coverage trade-offs across CLs for each target in the main text, and specifying whether empty-set abstentions are allowed by design. If abstentions are present, state their frequency and clinical triage plan.
Given the extreme scarcity of Stage 4, consider ordinal grouping sensitivity (e.g., merging 3–4 for robustness) or report per-class metrics with exact binomial CIs; emphasize descriptive status where inferential power is low (as you already do).
Comparison with related work (using the summaries provided)
The observed dominance of tree ensembles on medium-sized tabular data aligns with Grinsztajn et al. and Shwartz-Ziv & Armon; nonetheless, recent large-scale re-evaluations (e.g., 2402.03970) show competitive or superior performance by tuned meta-learned tabular DL models (TabPFNv2, AutoGluon). A brief discussion acknowledging this evolving landscape and explaining your choice to fix defaults would strengthen positioning.
Prior DaT-SPECT classification studies (e.g., 1909.04142; 2104.02066) demonstrate high diagnostic performance; positioning your binary NSD-ISS target against these (distinct) diagnostic settings would clarify why NSD+ vs NSD− is a harder problem, especially under the PPMI label structure.
The multimodal graph fusion literature (e.g., 2311.14902) reports gains by co-attention/contrastive mechanisms; your GAT underperformance fits the tabular-setting literature but could also reflect under-tuning. Consider referencing such designs and clarifying how your cross-modal attention differs.
The use of conformal prediction in PD (Diaz-Rincon et al.) is aptly connected; extending this to ordinal NSD-ISS via ordinal CP variants would be a natural next step.
Discussion of broader impact and significance
The two-stage deployment guidance is actionable and responsive to real-world resource variability (DaT-SPECT/sAA availability). The conformal wrapper’s set-valued outputs are well-suited to clinical triage and abstention policies.
Fairness and subgroup analyses (age, sex, genetic carrier status) are important for clinical translation; consider adding subgroup performance and coverage analyses to preempt potential disparities.
The transparent handling of label confounds and negative external results is exemplary and will help the community avoid common pitfalls in PPMI-trained models.
Questions for Authors

How were patient-similarity graphs, feature scaling, and neighbor searches implemented within cross-validation for the GAT models? Were these constructed strictly on training folds to prevent leakage into validation folds?
In the conformal setup, do you allow empty-set abstentions? If so, please quantify their frequency per target and clarify the clinical policy for abstentions. Also, why do mean set sizes fall below 1 while coverage substantially exceeds the nominal 90%?
Can you report external (BioFIND) conformal coverage and efficiency, including class-conditional coverage, to assess calibration under distribution shift?
How exactly was medication status handled (e.g., as covariates, stratification, exclusion) given Espay et al.’s critique? Can you provide sensitivity analyses showing its effect on internal and external results?
For the NSD+ sub-staging external result where Logistic Regression outperformed trees, can you provide additional diagnostics (calibration plots, confusion matrices, feature coefficients) to support the claim of an approximately linear mapping?
Did you consider ordinal-specific models/losses or ordinal conformal scores for the 5-class target? If tested, how did they compare?
Could you provide SHAP (or equivalent) feature-importance analyses for CatBoost on each target, and subgroup performance stratified by age/sex to assess fairness?
Will you release code and exact data processing scripts (including SQL extracts, fold assignments, and conformal calibration details) to enhance reproducibility?
Overall Assessment

This is a timely and valuable first benchmark for NSD-ISS biological staging with calibrated uncertainty, offering clear clinical takeaways: DaT-SPECT is indispensable for binary NSD detection, whereas clinical features suffice for NSD+ sub-staging; gradient-boosted trees dominate on these tabular tasks; and conformal prediction provides practical, set-valued outputs. The paper is commendably transparent about a critical training-label confound and proposes a defensible PD-only retraining and a two-stage deployment strategy. To reach top-tier impact, the authors should tighten methodological assurances (especially leakage controls for graph models), clarify the conformal implementation and external calibration, and modestly expand experiments (limited HPO for top models, ordinal-aware methods, subgroup/fairness, and feature-importance analyses). Cleaning presentation artifacts and adding external conformal metrics will further strengthen credibility. With these refinements, the work would provide a robust, clinically meaningful baseline and a principled uncertainty framework for the community. 