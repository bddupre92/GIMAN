# Paper 12 (phys-GIMIN) — Method Blueprint (D4)

**Deliverable:** D4 — atomic-decomposition method blueprint.
**Produced:** 2026-04-18.
**Upstream inputs:** approved plan `research-goal-onsider-using-jolly-matsumoto.md`, `litreview_synthesis.md` (53 entries after 2026-04-19 corrections), `novelty_verdict.md` (CONDITIONAL / MEDIUM).
**Scope constraint (user-locked, non-negotiable):** phys-GIMIN lives in a **standalone `paper12_phys_gimin/` directory**. It imports from `src/giman_pipeline/` as a consumer and does NOT modify any existing project code. Every concept below lands in a file under `paper12_phys_gimin/src/phys_gimin/`.

This blueprint decomposes phys-GIMIN into **18 concepts**. Each concept has: (1) a mathematical statement with variable glossary, (2) a code binding (exact class/function name in the standalone package), (3) provenance (imported vs new), and (4) one acceptance test. The appendix estimates an H100-hour compute budget.

---

## §0 Manuscript-framing preamble

This section answers the load-bearing reviewer objection surfaced in `scholar_eval_report.md` §2.2 / Objection 3 — "Paper 2 §V.E temperature scaling already closes 82% of the raw-decoder calibration gap; what does physics add?" — and makes the plan's generalization, Paper-11 dependency, and tautology-audit-framing claims explicit before any §1-§18 math is read.

### 0.1 Pre-registered expected marginal benefit over Paper 2 §V.E

> **Pre-registered hypothesis (H0 / H1 for Paper 12):**
>
> Paper 2 §V.E reports raw-decoder coverage 0.664 → 0.858 at γ=0.90 after temperature scaling, and 0.682 → 0.940 at γ=0.95. phys-GIMIN is expected to deliver:
>
> - **Coverage at γ=0.90:** baseline (T-only) = 0.858; expected with physics = **0.89 ± 0.02** (marginal gain ≥ +3 pp). **H0:** physics ≤ T-only; **H1:** physics > T-only by ≥ 3 pp.
> - **Coverage at γ=0.95:** baseline = 0.940; expected with physics = **0.948 ± 0.01** (marginal gain ≥ +0.8 pp).
> - **Absolute RMSE** on 33-feature continuous schema: baseline (Paper 2 StageDecoder) RMSE_median ≈ 0.35 on normalized scale; expected with physics = **0.33 ± 0.02** (marginal gain ≥ 5%, matching the Revision-1 2% MDE floor).
> - **Downstream Graph-DT C-td:** baseline (Paper 3 StageDecoder-imputed) C-td ≈ 0.91; expected with physics = **0.915 ± 0.005** (marginal gain ≥ 0.3 pp).
>
> **If NONE of these is met at p < 0.05 in the pilot:** Paper 12 pivots to σ-calibration-only contribution (Fix #3 abort trigger).

The coverage baselines cited above (raw 0.664; temperature-scaled 0.858; conformal 0.909 at γ=0.90) are the exact numbers from CLAUDE.md "Session 2026-04-18 Summary" §1 Paper-2 Calibration-Ablation §V.E. Expected-with-physics targets are pre-registered as *intermediate* — physics is predicted to help but not dominate post-hoc calibration, which is the honest story: temperature scaling + split-conformal is already strong, and physics' contribution is (a) to preserve σ-quality *during training* for downstream Bayesian updating and (b) to add an informative prior on out-of-sample patients whose calibration-set residuals are not available.

### 0.2 What generalizes beyond PD + priority acknowledgement of de Rooij 2025

> **Priority acknowledgement (updated 2026-04-19).** **de Rooij et al. 2025 PLOS Comp Biol** (DOI 10.1371/journal.pcbi.1012198) holds priority on physiology-informed UDE regularisation in biology — non-negativity + area-under-curve penalties applied to training a Universal Differential Equation system, validated on Michaelis-Menten and the glucose minimal model with real PREDICT UK meal-response data. phys-GIMIN extends the pattern to **multimodal imputation with σ-preserving posterior integration** — axes that remain genuinely novel after acknowledging de Rooij. The primary novelty claim is narrowed accordingly: phys-GIMIN is the **first physics-regularized multimodal heteroscedastic imputer for PD with σ-preserving integration into Bayesian posterior updating**, NOT the first physiology-informed regularised UDE.
>
> **Reusable contributions (independent of PD).** Three components of phys-GIMIN generalize to any hybrid ML+mechanistic imputation problem:
>
> 1. The `PriorProvider` interface that cleanly separates literature-anchored from self-derived ODE priors — portable to any disease with a calibratable mechanistic model.
> 2. The β-NLL + stop-gradient-on-σ-inside-L_physics recipe that prevents physics-induced σ collapse — applicable to any heteroscedastic predictive model with an auxiliary physics loss.
> 3. The pre-registered tautology-audit protocol — applicable whenever a method both generates and consumes mechanistic posteriors.
>
> These three are the "methods paper" core. The PD-specific α-synuclein + neuron-death ODE + PPMI evaluation is the "application" vehicle. Note that de Rooij's physiology-informed regularisation pattern is orthogonal to these three and is imported — phys-GIMIN's lit variant vendors `github.com/Computational-Biology-TUe/ude-regularization` directly (CC-BY).

### 0.3 Paper 11 (L1 integration) dependency

> **Paper 11 (L1 integration) dependency.** Q4 (bidirectional posterior-update ESS) is the single research question that REQUIRES Paper 11's L1 integration pipeline to exist before phys-GIMIN can test it. If L1 slips past the 9-15 month novelty window:
>
> - **Option A:** scope-restrict Paper 12 to Q1-Q3 + Q5 (drop the bidirectional-updating arm).
> - **Option B:** implement a minimal L1-equivalent inside phys-GIMIN's standalone dir for a single 10-patient demonstration (not scaled validation).
>
> Monitored at weeks 3, 8, 12 per the risk register.

### 0.4 Tautology-audit framing (honesty line)

> **Novelty of disclosure, not of method.** Multiple papers (Philipps 2025, Giampiccolo 2024 [previously cited as "Horvát 2025"]) have documented hybrid-ML identifiability problems on synthetic data. Paper 12's tautology-audit contribution is **publishing the negative result on a real disease cohort with a mechanistic ODE that was actually fit to that cohort** — not inventing the audit methodology. Frame accordingly.

### 0.5 Competitor landscape after 2026-04-19 corrections

> **Top-3 physics-regularised competitors (REVISED):**
>
> 1. **de Rooij et al. 2025 PLOS Comp Biol** — physiology-informed UDE regularisation for biology (glucose minimal model). THE closest methodological prior. Open-source CC-BY code: vendor directly.
> 2. **Wang et al. 2025 CNODE PPMI** (arXiv 2511.04789) — conditional neural ODE for PD progression on PPMI MRI. Data-driven, no mechanistic prior, forecasting not imputation.
> 3. **Xiao et al. 2025 TD-HNODE** — hypergraph + neural ODE on T2D progression. Pending independent verification.
>
> **NOT a physics-regularised competitor:** LagCNN (Li et al. 2024 CIKM, DOI 10.1145/3627673.3679672) was originally mis-labelled "Liang 2024 HSPGNN" by a lit-review agent hallucination. The actual paper is a CNN + Time Lag + FFT time-series imputer — it belongs in the DL-imputer baseline zoo (SAITS / GAIN / MIWAE / BRITS / CSDI), NOT in the physics-regularised competitor list.

---

## Forward Pass

### Concept 1 — Input encoding (GIMIN backbone unchanged)

**Math.**

$$
\mathbf{z}_i \;=\; \mathrm{GNN}_{\theta_{\mathrm{GIMIN}}}\!\bigl(\mathbf{x}_i^{\mathrm{obs}},\, \mathbf{m}_i,\, s_i,\, \mathcal{G}(\beta=0.3)\bigr), \qquad \mathbf{z}_i \in \mathbb{R}^{d}
$$

**Glossary.** $\mathbf{x}_i^{\mathrm{obs}}$: observed feature vector (33 dims, one per feature; NaNs masked). $\mathbf{m}_i \in \{0,1\}^{33}$: observed-entry mask. $s_i \in \{0,1,2\mathrm{B},3,4\}$: NSD-ISS stage. $\mathcal{G}$: stage-aware k-NN patient graph with same-stage affinity bonus $\beta=0.3$. $\mathbf{z}_i$: node embedding (GIMIN base $d=128$).

**Code binding.** `phys_gimin.model.PhysGIMIN.encode(...)` delegates to `giman_pipeline.imputation.StageConditionedGIMIN.encode(...)` via inheritance. No override.

**Provenance.** **Imported** verbatim from `GIMImpN_imputation/gimin/model/gimin_core.py:39–250` and `src/giman_pipeline/imputation/stage_conditioned_gimin.py:120–290`. phys-GIMIN does not touch the GNN.

**Test.** Feed identical `(x_obs, mask, stage, graph)` through `PhysGIMIN.encode` and `StageConditionedGIMIN.encode`; assert element-wise equality (`torch.allclose(rtol=0, atol=0)`).

---

### Concept 2 — Heteroscedastic decoder (inherited)

**Math.** For each continuous feature $f$ and patient $i$,

$$
\mu_{i,f},\ \log \sigma^2_{i,f} \;=\; \mathrm{Dec}_{\phi}\bigl(\mathbf{z}_i,\, s_i\bigr)_f, \qquad \sigma_{i,f} \;=\; \exp\!\bigl(\tfrac{1}{2}\log \sigma^2_{i,f}\bigr).
$$

Binary feature SEX uses a logit head (out of physics scope per feature taxonomy; continuous-only assertion elsewhere).

**Glossary.** $\mu_{i,f}$: predicted mean (feature-native units, post inverse-scaler). $\log \sigma^2_{i,f}$: heteroscedastic log-variance head output. Decoder $\mathrm{Dec}_{\phi}$ is `StageConditionedGIMIN`'s decoder with `nn.Embedding(6,16)` stage embedding concatenated to $\mathbf{z}_i$.

**Code binding.** `phys_gimin.model.PhysGIMIN.decode(...)` — passthrough to `StageConditionedGIMIN.decode`.

**Provenance.** **Imported.** Heteroscedastic head exists in GIMIN base (per Paper 2 §V.E).

**Test.** Assert `decode(z, s)` returns a 2-tuple; assert `sigma > 0` element-wise (`torch.all(sigma > 0)`); assert `logvar` has `requires_grad=True` during training.

---

### Concept 3 — Physics residual computation

**Math.** For twin-observable feature $f \in \mathcal{F}_{\mathrm{twin}} = \{\text{SBR\_CAUDATE\_L, SBR\_CAUDATE\_R, SBR\_PUTAMEN\_L, SBR\_PUTAMEN\_R}\}$,

$$
\mathbf{y}_{i,f}^{\mathrm{ODE}} \;=\; \mathrm{ODE}_{\mathcal{P}}\!\bigl(\mathrm{patno}_i,\, \mathbf{t}_i\bigr), \qquad \mathcal{P} \in \{\mathrm{Lit},\mathrm{Self}\}.
$$

$\mathcal{P}$ is the `PriorProvider` instance injected at construction time. Trajectories are pre-computed **once per epoch** and cached; per-batch cost is an HDF5 lookup.

**Glossary.** $\mathrm{patno}_i$: PPMI patient ID. $\mathbf{t}_i \in \mathbb{R}^{T_i}$: visit times in years from baseline. $\mathrm{ODE}_{\mathcal{P}}$: the α-syn + dopaminergic-neuron-death ODE family (Véronneau-Veilleux 2020, Phase 2 forward model), integrated with `rtqichen/torchdiffeq` (MIT, drop-in, no autodiff needed here because trajectories are not parameters of phys-GIMIN).

**Code binding.** `phys_gimin.regularizer.PhysicsRegularizer.__call__(patnos, times)` → calls `self.provider.ode_trajectory(patno, t_years)` for each patient, caches per-epoch in `self._cache: Dict[Tuple[int, epoch_id], np.ndarray]`.

**Provenance.** **New** wrapper. Imports `src/giman_pipeline/mechanistic_twin_v2/forward_model.py` (read-only); never writes.

**Test.** With `provider=LiteraturePriorProvider(...)`, compute a trajectory at $t=[0,1,5]$ years for a synthetic patient with $N_0 = 400{,}000$, $\gamma=0.7$; assert monotonic decay and final value $\in [0.30\,N_0,\, 0.85\,N_0]$ (matches Fearnley-Lees 1991 + Lee 2019 envelope).

---

### Concept 4 — Blended output

**Math.**

$$
\hat{x}_{i,f} \;=\; m_{i,f}\,x_{i,f}^{\mathrm{obs}} \;+\; (1 - m_{i,f})\,\mu_{i,f},\qquad \hat{\sigma}_{i,f} \;=\; (1 - m_{i,f})\,\sigma_{i,f}.
$$

Observed entries pass through with $\hat\sigma = 0$; imputed entries use the heteroscedastic $\sigma$. This is the Paper 2 contract and MUST NOT change or Paper 6/Paper 10 downstream wrappers break.

**Code binding.** `phys_gimin.model.PhysGIMIN.impute(...)` — inherited from base; no override unless a future-work flow-matching decoder is added.

**Provenance.** **Imported.**

**Test.** Set mask to all-observed; assert $\hat x = x^{\mathrm{obs}}$ exactly and $\hat\sigma = 0$ exactly.

---

## Loss Composition

### Concept 5 — Reconstruction β-NLL

**Math (Seitzer 2022, β=0.5).**

$$
\mathcal{L}_{\mathrm{recon}} \;=\; \frac{1}{|\mathcal{O}|}\sum_{(i,f)\in\mathcal{O}} \underbrace{\bigl(\sigma_{i,f}^2\bigr)^{\beta}}_{\text{stop-grad}} \cdot \left[ \frac{(x_{i,f}^{\mathrm{obs}} - \mu_{i,f})^2}{2\sigma_{i,f}^2} + \tfrac{1}{2}\log \sigma^2_{i,f} \right],
$$

with $\beta = 0.5$ and the leading $(\sigma^2)^\beta$ factor applied via `stop_gradient` so it rescales the per-sample weight without adding a gradient path through $\sigma$ that would encourage variance collapse. $\mathcal{O}$ is the observed-entry set.

**Glossary.** $\beta$: Seitzer β-NLL exponent; $\beta=0$ is plain Gaussian NLL (variance-collapse-prone), $\beta=1$ is MSE-like. $\beta=0.5$ is the ICLR 2022 recommendation.

**Code binding.** `phys_gimin.loss.beta_nll(mu, sigma, target, beta=0.5) -> Tensor`.

**Provenance.** **New** function; pure PyTorch; ~25 LOC.

**Test.** Set $\beta=0$ and confirm numerical equality to `torch.distributions.Normal(mu,sigma).log_prob(target).neg().mean()`. Set $\beta=0.5$, confirm gradient through the $(\sigma^2)^{0.5}$ factor is zero (`torch.autograd.grad(..., sigma)[0]` matches plain-NLL gradient, not amplified by $\sigma$-weight).

---

### Concept 6 — Physics β-NLL with stop-gradient on σ (the key blueprint addition)

**Math.** For $(i,f) \in \mathcal{T}$ (visits where a twin-observable prediction exists),

$$
\mathcal{L}_{\mathrm{physics}} \;=\; \frac{1}{|\mathcal{T}|}\sum_{(i,f)\in\mathcal{T}} \underbrace{\bigl[\mathrm{sg}(\sigma_{i,f})\bigr]^{2\beta}}_{\text{stop-grad factor}} \cdot \left[ \frac{(y_{i,f}^{\mathrm{ODE}} - \mu_{i,f})^2}{2\,[\mathrm{sg}(\sigma_{i,f})]^2} + \tfrac{1}{2}\log \mathrm{sg}(\sigma_{i,f})^2 \right],
$$

where $\mathrm{sg}(\cdot)$ denotes `tensor.detach()`. **$\sigma$ appears ONLY inside stop-gradient.** This is non-negotiable: it prevents physics from explaining-away aleatoric variance (the well-known PINN failure mode flagged by Seitzer 2022 and Wang-Perdikaris 2021).

**Why stop-grad on σ and not μ.** Physics should regularize the mean (the ODE speaks to signal, not noise). Letting $\sigma$ absorb ODE residual would inflate $\sigma$ wherever the ODE is wrong, undoing the whole calibration contract.

**Code binding.** `phys_gimin.regularizer.PhysicsRegularizer.forward(mu, sigma, ode_target) -> Tensor`. Internally calls `phys_gimin.loss.beta_nll(mu, sigma.detach(), ode_target, beta=0.5)`.

**Provenance.** **New.** The `detach()` call is the critical line.

**Test.** Set $\sigma_{i,f} = 1$ (constant), compute $\mathcal{L}_{\mathrm{physics}}$, call `.backward()`. Assert `sigma.grad is None or torch.allclose(sigma.grad, 0)` (stop-grad verification).

---

### Concept 7 — Base-inherited losses (unchanged from Paper 2)

**Math.** $\mathcal{L}_{\mathrm{distribution}}$ (distribution-matching between imputed and observed marginals), $\mathcal{L}_{\mathrm{cross\,modal}}$ (modality-consistency), $\mathcal{L}_{\mathrm{calibration}}$ (Paper 2 §V.E per-feature temperature-scaling auxiliary loss).

**Code binding.** `phys_gimin.loss.inherited_loss_terms(model_outputs, batch)` — imports `GIMImpN_imputation.gimin.model.loss.compute_auxiliary_losses` and returns the tuple unchanged.

**Provenance.** **Imported verbatim.** Not relitigated; phys-GIMIN does not claim contribution on these terms.

**Test.** Assert that for $\lambda_{\mathrm{dist}}=\lambda_{\mathrm{cross}}=\lambda_{\mathrm{cal}}=0,\, \lambda_{\mathrm{phys}}=0$, training reduces to vanilla $\mathcal{L}_{\mathrm{recon}}$ only; RMSE matches a no-auxiliary GIMIN run within 1e-4 on a 10-patient toy batch.

---

### Concept 8 — Total loss with clamp

**Math.**

$$
\mathcal{L} \;=\; \mathcal{L}_{\mathrm{recon}} \;+\; \lambda_{\mathrm{dist}}\mathcal{L}_{\mathrm{dist}} \;+\; \lambda_{\mathrm{cross}}\mathcal{L}_{\mathrm{cross}} \;+\; \lambda_{\mathrm{cal}}\mathcal{L}_{\mathrm{cal}} \;+\; \lambda_{\mathrm{phys}}(e)\,\mathcal{L}_{\mathrm{physics}},
$$

subject to the **σ-collapse clamp**:

$$
\text{if } \frac{\mathcal{L}_{\mathrm{recon}}}{\mathcal{L}} < 0.30,\ \ \text{then } \lambda_{\mathrm{phys}}(e) \leftarrow \lambda_{\mathrm{phys}}(e) \cdot 0.5\ \text{and reissue warning.}
$$

The clamp triggers at the batch level; if triggered 3× in one epoch, training halts with `PhysGIMINCalibrationError`.

**Code binding.** `phys_gimin.loss.TotalLoss.__call__(outputs, batch, epoch) -> Tuple[Tensor, Dict[str, float]]` returns total + diagnostic dict (includes the recon share). `phys_gimin.loss.CollapseClampError`.

**Provenance.** **New.** The clamp is phys-GIMIN's safety net against the failure mode predicted in the risk register.

**Test.** Inject $\lambda_{\mathrm{phys}}=100$ on a random batch (forces $\mathcal{L}_{\mathrm{physics}}$ to dominate); assert clamp fires (ratio < 0.30), assert $\lambda_{\mathrm{phys}}$ halved post-batch, assert no `NaN` in any loss component.

---

## Physics Regularizer Variants

### Concept 9 — LiteraturePriorProvider (zero-leakage lit variant)

**Math.** ODE trajectory uses **fixed literature rate constants**:

- $N_0 = 400{,}000$ neurons per hemisphere (Fearnley-Lees 1991)
- $\gamma = 0.7$ neuron-death rate scaling (Lee 2019)
- $k_{\mathrm{agg}}$, $k_{\mathrm{frag}}$, $k_{\mathrm{nuc}}$ from Iljina 2016 rate envelope
- SBR calibration from Phase 1 Paper 7 whole-striatum decay (3.29%/yr median, literature-consistent)

No per-patient parameters. Cohort-invariant trajectory per $t$ grid.

**Glossary.** "Zero-leakage" = zero overlap with Paper 7/9/10 downstream posteriors → safe against tautology for any downstream benchmark.

**Code binding.** `phys_gimin.priors.literature.LiteraturePriorProvider(config_path: Path)` — loads frozen YAML `configs/phys_gimin_lit.yaml` at construction.

**Provenance.** **New.** Reads `outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md` as source of anchors (READ-ONLY).

**Test.** Instantiate `LiteraturePriorProvider`, call `ode_trajectory(patno=3001, t_years=[0,5,10])`, call again with `patno=3002` — assert outputs identical (cohort-invariant, patno is ignored for lit variant). Assert `self.provider_type == 'literature'` in output JSON.

---

### Concept 10 — PosteriorStorePriorProvider (tautological self variant)

**Math.** Per-patient ODE trajectory from the bidirectional-SIR posterior:

$$
\mathbf{y}_{i,f}^{\mathrm{ODE,self}} \;=\; \mathrm{ODE}_{\hat{\boldsymbol{\theta}}_i}\!(\mathbf{t}_i), \qquad \hat{\boldsymbol{\theta}}_i = \mathbb{E}_{\mathrm{post}}[\boldsymbol{\theta}_i \mid \mathcal{D}_i^{\mathrm{scans}}].
$$

$\hat{\boldsymbol{\theta}}_i$ is read from the Paper 10 posterior HDF5. **Tautological** against any downstream target trained on the same posteriors (Papers 7, 9, 10).

**Code binding.** `phys_gimin.priors.posterior_store.PosteriorStorePriorProvider(hdf5_path: Path)` — opens existing `outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_combined_1065.h5` read-only.

**Provenance.** **New** wrapper. Imports `src/giman_pipeline/mechanistic_twin_v2/posterior_store.PosteriorStore.read()` (read-only).

**Test.**
(a) Instantiate `PosteriorStorePriorProvider` with `variant='self'`; confirm construction succeeds.
(b) Attempt to register `PosteriorStorePriorProvider` with a config whose `variant='lit'` → must raise `ValueError("PosteriorStorePriorProvider incompatible with variant='lit'")` via runtime assertion in `PhysicsRegularizer.__init__`.

---

## Training Schedule

### Concept 11 — Warmup ($\lambda_{\mathrm{phys}} = 0$)

**Math.** $\lambda_{\mathrm{phys}}(e) = 0$ for $e < N_{\mathrm{warmup}}$ (default $N_{\mathrm{warmup}}=20$).

Rationale: let the imputation path converge before imposing physics pressure — prevents physics from dominating an under-trained decoder.

**Code binding.** `phys_gimin.training.LambdaScheduler.value_at(epoch)` — returns `0.0` for `epoch < warmup_epochs`.

**Provenance.** **New.**

**Test.** At $e=0$, assert `scheduler.value_at(0) == 0.0`; at $e=N_{\mathrm{warmup}}$, assert `value_at > 0.0` (ramp begins).

---

### Concept 12 — Linear ramp

**Math.** For $e \in [N_{\mathrm{warmup}},\, N_{\mathrm{warmup}}+N_{\mathrm{ramp}}]$ ($N_{\mathrm{ramp}}=30$),

$$
\lambda_{\mathrm{phys}}(e) \;=\; \lambda_{\mathrm{phys}}^\star \cdot \frac{e - N_{\mathrm{warmup}}}{N_{\mathrm{ramp}}}.
$$

After $e > N_{\mathrm{warmup}} + N_{\mathrm{ramp}}$, $\lambda_{\mathrm{phys}}(e) = \lambda_{\mathrm{phys}}^\star$ (subject to LR-annealing rescaling, Concept 13). Default $\lambda_{\mathrm{phys}}^\star = 1.0$ before annealing.

**Code binding.** `phys_gimin.training.LambdaScheduler.linear_ramp(epoch)`.

**Provenance.** **New.**

**Test.** At $e = N_{\mathrm{warmup}} + N_{\mathrm{ramp}}/2$, assert `scheduler.value_at(e) ≈ λ* / 2` within 1e-6.

---

### Concept 13 — LR-annealing λ scheduler (Wang-Perdikaris 2021)

**Math.** Per-batch, compute the maximum gradient magnitude of each loss component,

$$
\mathsf{g}_{\mathrm{recon}}^{\max} = \max_{\theta}\bigl|\nabla_\theta \mathcal{L}_{\mathrm{recon}}\bigr|, \qquad \bar{\mathsf{g}}_{\mathrm{phys}} = \mathbb{E}_{\theta}\bigl[\,|\nabla_\theta \mathcal{L}_{\mathrm{physics}}|\,\bigr],
$$

and compute the target weight

$$
\hat{\lambda}_{\mathrm{phys}}^{(b)} \;=\; \frac{\mathsf{g}_{\mathrm{recon}}^{\max}}{\bar{\mathsf{g}}_{\mathrm{phys}} + \varepsilon}.
$$

Apply EMA smoothing:

$$
\lambda_{\mathrm{phys}}^{(b+1)} \;=\; \alpha\,\lambda_{\mathrm{phys}}^{(b)} \;+\; (1-\alpha)\,\hat{\lambda}_{\mathrm{phys}}^{(b)}, \qquad \alpha = 0.9.
$$

This balances gradient magnitudes so physics and reconstruction compete on equal footing regardless of absolute scale. NTK-based alternatives are $O(N^2)$ in patients and intractable at $n = 1{,}065$ — not used.

**Code binding.** `phys_gimin.training.GradNormBalancer(ema_alpha=0.9, eps=1e-8).step(losses, model) -> Dict[str, float]` — returns updated λ values.

**Provenance.** **New** (~200 LOC reimplementation of Wang 2021; cannot port `rbischof/relative_balancing` due to missing LICENSE file, per D2 inventory).

**Test.** Run 100 training steps with two synthetic losses of known gradient magnitudes (10× scale difference). Assert the EMA-smoothed $\lambda_{\mathrm{phys}}$ converges to a value whose product with the physics gradient matches the recon gradient within 5%.

---

### Concept 14 — Early stopping on dual-gate (val-RMSE + coverage)

**Math.** Training halts at epoch $e$ if either:

$$
\text{val-RMSE}(e) > \min_{e' \le e} \text{val-RMSE}(e') \ \text{for 10 consecutive epochs}, \quad \text{OR}
$$
$$
\text{val-coverage}(e, \gamma{=}0.90) < 0.87 \ (3\text{pp below target}).
$$

The coverage gate is the Paper 2 §V.E guarantee — if physics inflates losses and destroys calibration, stop before it gets worse.

**Code binding.** `phys_gimin.training.DualGateStopper(patience=10, coverage_target=0.90, coverage_tol=0.03).should_stop(history) -> bool`.

**Provenance.** **New.** Coverage gate is specific to phys-GIMIN; RMSE gate is standard.

**Test.** Feed a mock history with val-coverage dropping from 0.91 → 0.85 → 0.84 at epochs {20, 21, 22}; assert `should_stop()` returns `True` at epoch 21 (first sub-0.87). Feed a history where RMSE plateaus but coverage stays at 0.91; assert `should_stop()` triggers only after patience expires.

---

## σ-Contract (Continuous Features Only)

### Concept 15 — Aleatoric σ (inherited heteroscedastic head)

**Math.** $\sigma_{i,f}^{\mathrm{alea}} = \exp(\tfrac{1}{2}\log\sigma^2_{i,f})$ from Concept 2's decoder head.

**Code binding.** Accessible via `PhysGIMIN.forward(...)[1]`.

**Provenance.** **Imported.**

**Test.** Assert $\sigma^{\mathrm{alea}}_{i,f} > 0$ element-wise and has `requires_grad=True` during training.

---

### Concept 16 — Epistemic σ (MC-dropout)

**Math.**

$$
\sigma^{\mathrm{epis}}_{i,f} \;=\; \sqrt{\mathrm{Var}_{k=1}^{K}\bigl[\mu_{i,f}^{(k)}\bigr]}, \qquad K = 50 \text{ forward passes with dropout enabled.}
$$

**Code binding.** `phys_gimin.model.PhysGIMIN.mc_dropout_predict(x, mask, n_samples=50)` — delegates to `GIMImpN_imputation.gimin.model.uncertainty.MCDropoutWrapper` (inherited).

**Provenance.** **Imported.**

**Test.** Enable dropout with `p=0.5`, run 50 passes on fixed input; assert $\sigma^{\mathrm{epis}} > 0$ and monotonic: decreases as `n_samples → 200`.

---

### Concept 17 — Per-feature temperature scaling

**Math.** For feature $f$, the post-hoc scaler is

$$
\tilde\sigma_{i,f} \;=\; T_f \cdot \sigma_{i,f}^{\mathrm{total}}, \qquad \sigma^{\mathrm{total}}_{i,f} \;=\; \sqrt{(\sigma^{\mathrm{alea}}_{i,f})^2 + (\sigma^{\mathrm{epis}}_{i,f})^2}.
$$

$T_f$ is fit on a held-out **artificial-mask** split by minimizing NLL (no training labels leak). Paper 2 median $T_f = 0.87$.

**Code binding.** `phys_gimin.observation_adapters.temperature.fit_temperature_scaler(mu, sigma, target, mask) -> PerFeatureTemperatureScaler` — wraps `src.giman_pipeline.imputation.temperature_scaling.PerFeatureTemperatureScaler`.

**Provenance.** **Imported** (not reimplemented). The scaler is reused as-is from Paper 2 §V.E.

**Test.** Run the Paper 2 §V.E unit test on phys-GIMIN's σ outputs: assert per-feature NLL after temperature scaling is ≤ NLL before.

---

### Concept 18 — Conformal bands (split-conformal per feature, Podina 2024)

**Math.** For feature $f$, calibration set $\mathcal{C}_f$, and target coverage $\gamma$,

$$
\mathcal{R}_f = \left\{\frac{|\,x_{i,f} - \mu_{i,f}\,|}{\tilde\sigma_{i,f}} \,:\, (i,f) \in \mathcal{C}_f\right\}, \qquad q_f = \mathrm{Quantile}\!\left(\mathcal{R}_f,\, \lceil (n+1)\gamma \rceil / n\right).
$$

Prediction interval at test time:

$$
\bigl[\mu_{i,f} - q_f\,\tilde\sigma_{i,f},\ \mu_{i,f} + q_f\,\tilde\sigma_{i,f}\bigr].
$$

This is **split-conformal**, which has a finite-sample marginal coverage guarantee $\ge \gamma$ (Angelopoulos & Bates 2021 Thm 3.1; Podina 2024 extends to PINNs). It is **loss-agnostic** — the physics regularizer cannot break the coverage guarantee. This is the safety net when temperature scaling is fragile under $\lambda_{\mathrm{phys}} > 0$.

**Code binding.** `phys_gimin.observation_adapters.conformal.SplitConformalPerFeature(calibration_residuals).predict_interval(mu, sigma, gamma) -> Tuple[Tensor, Tensor]`. Imports `src.giman_pipeline.paper4.conformal_survival.split_conformal_quantile` if available.

**Provenance.** **New** wrapper; delegates to Paper 4 IPCW infrastructure for calibration quantile computation.

**Test.** On a held-out test split, assert empirical coverage $\hat\gamma \in [\gamma - 0.02,\, 1.0]$ across 100 random calibration/test splits (Hoeffding bound at $n=100$ cal points gives ±0.14 worst-case; with $n \ge 500$ expect ±0.03).

---

## File Binding Summary

| Concept(s) | File (under `paper12_phys_gimin/src/phys_gimin/`) |
|---|---|
| 1–2, 4 (forward pass wrappers) | `model.py` |
| 3, 6 (physics residual) | `regularizer.py` |
| 5, 7, 8 (loss composition + clamp) | `loss.py` |
| 9 (lit variant) | `priors/literature.py` |
| 10 (self variant) | `priors/posterior_store.py` |
| — (Protocol) | `priors/base.py` |
| 11–14 (schedule, balancer, stop) | `training.py` |
| 15 (aleatoric) | `model.py` (inherited) |
| 16 (epistemic) | `model.py` (inherited via MCDropoutWrapper) |
| 17 (temperature) | `observation_adapters/temperature.py` |
| 18 (conformal) | `observation_adapters/conformal.py` |
| σ → SBR-likelihood wiring | `observation_adapters/sbr.py` |
| σ → Ch 9.6 multi-channel | `observation_adapters/multichannel.py` |
| σ → Path B errors-in-vars | `observation_adapters/path_b.py` |

---

## Acceptance-Test Matrix (one-liner per concept)

| # | Concept | Test (one sentence) |
|---|---|---|
| 1 | Input encoding | `PhysGIMIN.encode` equals `StageConditionedGIMIN.encode` elementwise. |
| 2 | Heteroscedastic decoder | Decoder returns 2-tuple, σ > 0, log-variance has `requires_grad=True`. |
| 3 | Physics residual | Lit-variant trajectory at $t=\{0,1,5\}$ yr is monotonic decay with final value in Fearnley-Lees + Lee envelope. |
| 4 | Blended output | All-observed mask → $\hat x = x^{\mathrm{obs}}$, $\hat\sigma = 0$ exactly. |
| 5 | β-NLL | $\beta=0$ matches `torch.distributions.Normal.log_prob`; $\beta=0.5$ has zero gradient through the $(\sigma^2)^\beta$ factor. |
| 6 | **Stop-grad on σ** | Set $\sigma=1$; $\nabla_\sigma \mathcal{L}_{\mathrm{physics}} = 0$. |
| 7 | Inherited auxiliaries | All $\lambda=0$ reduces to vanilla recon loss. |
| 8 | **Clamp** | Inject $\lambda_{\mathrm{phys}}=100$; clamp fires, $\lambda$ halved, no NaN. |
| 9 | Lit variant | Two different patnos yield identical lit trajectory. |
| 10 | **Self variant cannot be instantiated with lit flag** | `PhysicsRegularizer(PosteriorStorePriorProvider, variant='lit')` raises `ValueError`. |
| 11 | Warmup | `scheduler.value_at(0) == 0.0`. |
| 12 | Ramp | Midway through ramp, λ ≈ λ*/2. |
| 13 | LR-annealing balancer | On two synthetic losses with 10× gradient gap, EMA-λ equalizes within 5%. |
| 14 | Dual-gate stop | Coverage dropping to 0.84 triggers stop; RMSE plateau waits for patience. |
| 15 | Aleatoric | σ > 0 and `requires_grad=True`. |
| 16 | Epistemic | K=50 MC variance > 0 and decreases with $K$. |
| 17 | Temperature | Per-feature NLL post-scaling ≤ NLL pre-scaling. |
| 18 | Conformal | Empirical marginal coverage at $\gamma=0.90$ is in $[0.88, 1.00]$ for $n_{\mathrm{cal}} \ge 500$. |

---

## Appendix — H100-Hour Compute Budget

**Assumptions.**

- **Paper 2 baseline wall time** (single GIMIN variant, 1 mask fraction, 1 seed, 4 NSD-ISS stages): **~2.5 H100-hours** per run on a single H100 80GB. Source: Paper 2 §IV.B training-history JSONs, full benchmark `20260222_160247/` run log.
- **Physics-integration overhead:** ×1.15 (15% — ODE trajectories pre-computed once per epoch from HDF5, per-batch cost is a cached lookup + a β-NLL forward; no per-batch `torchdiffeq` calls).
- **Number of variants:** 2 (lit + self).
- **Seeds per configuration:** 3 (per Paper-2 benchmark integrity governance).
- **Mask fractions:** 4 (10%, 20%, 30%, 50%).
- **Missingness regimes:** 4 (MCAR + MAR + MNAR + block-missing, per the research-goal plan).
- **PD-only ablation arm:** +1 full replication (~780 PD patients ≠ 2,201 full PPMI, training cost scales roughly linearly in batch-count → assume 0.4× of a full run; counted as 4 additional runs at 0.4× cost = 1.6 equivalent runs).

**Primary budget (main benchmark grid).**

$$
H_{\mathrm{main}} \;=\; 2.5 \times 1.15 \times 2 \times 3 \times 4 \times 4 \;=\; 276\ \text{H100-hours}.
$$

**PD-only ablation.**

$$
H_{\mathrm{PD-only}} \;=\; 2.5 \times 1.15 \times 2 \times 3 \times 4 \times 1 \times 0.4 \;=\; 27.6\ \text{H100-hours}.
$$

**Downstream Paper 3 Graph-DT re-eval under phys-GIMIN imputation.**

$$
H_{\mathrm{downstream}} \;=\; 1.2\ \text{hours (Graph-DT 5-fold CV)} \times 2\ \text{variants} \times 4\ \text{mask-fracs} \;\approx\; 10\ \text{H100-hours}.
$$

**Hyperparameter search** (λ_phys* star point, $\beta$ ablation $\{0.25,0.5,0.75\}$, $N_{\mathrm{warmup}} \in \{10,20,30\}$): **~40 H100-hours** (coarse grid at mask-frac 0.2 + seed 1 only).

**Conformal calibration sweep** (coverage at $\gamma \in \{0.80, 0.90, 0.95\}$, 10 random splits): **~5 H100-hours** (CPU-bound for split-conformal quantile computation; H100 time negligible).

**Contingency (15%)** for debug runs, failed fits that trigger the Q2 abort, and re-runs on MPS-nondeterminism reseeds: **~53 H100-hours**.

**Total.**

$$
\boxed{H_{\mathrm{total}} \;\approx\; 276 + 28 + 10 + 40 + 5 + 53 \;=\; 412\ \text{H100-hours}.}
$$

Rounded to **~420 H100-hours** for budgeting. At a typical cloud rate of \$2–\$4 per H100-hour, that is **\$840–\$1,680** in compute for the full Paper 12 benchmark grid. Four-month postdoc timeline accommodates this comfortably at ~3 H100-hours per day.

**Risk to budget.**

1. If ODE trajectory caching proves insufficient (>15% overhead), the self-variant — which reads per-patient posteriors — could inflate overhead to ×1.4 and push total to ~500 hours. Mitigation: pre-materialize all self-variant trajectories once at start-of-run as NumPy caches.
2. If the Q2 abort criterion triggers and the paper pivots to σ-calibration-only, the benchmark grid shrinks (no need to run self-variant × 4 mask-fracs × 3 seeds under full physics) → budget drops to **~180 H100-hours**. This is the cheap failure mode.
3. If a 2026 de Rooij-for-multimodal-clinical-imputation follow-up forces a direct head-to-head comparison, add ~80 H100-hours for adapting the vendored `Computational-Biology-TUe/ude-regularization` code to phys-GIMIN's 33-feature schema + matched benchmark (per D2 inventory + 2026-04-19 corrections).

---

*End of blueprint. All 18 concepts have: (a) a mathematical statement, (b) an exact code binding under `paper12_phys_gimin/src/phys_gimin/`, (c) imported-vs-new provenance, and (d) a one-sentence acceptance test. The compute budget is ~420 H100-hours across the full 4-month postdoc grid. This blueprint compiles against the existing GIMIN API without modifying any main-project file.*
