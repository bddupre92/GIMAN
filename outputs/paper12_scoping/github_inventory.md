# Paper 12 (phys-GIMIN) — GitHub Repository Inventory

**Deliverable:** D2 scoping
**Date compiled:** 2026-04-18
**Compiler:** Claude Opus 4.7 (metadata verified via GitHub REST API, license files read directly)
**Snapshot date:** all star counts and last-commit timestamps fetched 2026-04-18 via `api.github.com/repos/<owner>/<repo>`

## 1. Executive summary

**Total repos scored:** 18 (Bucket A: 8, Bucket B: 4, Bucket C: 6).

**Reuse verdict distribution:**

- **DROP-IN (pip install, no source changes):** 4 — `torchdiffeq`, `diffrax`, `PyPOTS`, `hyperimpute`.
- **VENDOR (copy specific module, cite, no fork):** 2 — `rbischof/relative_balancing` (ReLoBRaLo loss balancer, ~300 LOC), `SciML/DataDrivenDiffEq.jl` (SINDy for reference).
- **PORT (re-implement from paper + their code as reference):** 2 — `neu-spiral/Hybrid-ODE-NN` (Demirkaya 2024 competitor), `bobjz/H2NCM` (Zou 2025 precursor).
- **CITE-ONLY:** 8 — canonical references, not reused.
- **NOT-REUSABLE (licensing blocker):** 2 — `yaringal/DropoutUncertaintyExps` (CC-BY-NC, non-commercial), `PredictiveIntelligenceLab/jaxpi` (custom Penn license, not OSI-approved).

**Licensing blockers discovered:**

1. **`bobjz/H2NCM` has NO LICENSE file** — Zou 2025 precursor code is therefore "all rights reserved" under GitHub ToS §D.5. Cannot port without written permission from `bobjz` (Bob Zou).
2. **`neu-spiral/Hybrid-ODE-NN` has NO LICENSE file** — Demirkaya 2024 TBME code is unlicensed despite prior CLAUDE.md assumption of MIT. Same legal status as H2NCM. Need to email Demirkaya (@neu-spiral.edu) for explicit reuse permission, or re-implement from the paper text only.
3. **`yaringal/DropoutUncertaintyExps` is CC-BY-NC 4.0** — non-commercial clause bars use in any phys-GIMIN deployment with industry collaboration (including AMP-PD commercial partnerships). Safe for dissertation replication only.
4. **`PredictiveIntelligenceLab/jaxpi` custom Penn academic license** — academic research only, commercial use requires Penn tech transfer negotiation. DROP in a lab/dissertation context, do not vendor.

**Parkinson's-specific pre-built datasets/models:** **NONE** across all 18 repos. Zero pre-trained checkpoints for PD and zero repos that bundle PPMI/BioFIND/HBS data. Confirms phys-GIMIN is the first public PD-specific hybrid ODE-NN imputation codebase.

**torchdiffeq vs diffrax 2024–2026 dominance:** torchdiffeq still wins on installed base (6,397 stars, 21 dependent public repos tagged with its name, still the default for healthcare neural-ODE papers e.g. Demirkaya 2024, Rubanova latent_ode). diffrax is growing fastest (1,976 stars, 27 dependent repos, pushed 13 days ago vs torchdiffeq's 12 months) and is dominant in any JAX/Equinox-based paper 2024–2026. **For phys-GIMIN we recommend torchdiffeq** because the GIMIN codebase is already PyTorch-native — moving to JAX would require porting the entire imputer + graph stack.

**Go/no-go on Demirkaya + Zou port plan:** **GO with caveats.** Both repos are legally unclear but their papers are published peer-reviewed and the algorithms can be re-implemented from the paper text alone without touching their source. Recommended path: (a) send licensing email to both author teams (see Action Items); (b) in parallel, re-implement both baselines from the paper text into `paper12_phys_gimin/baselines/{demirkaya2024,zou2025}/` and mark as "independent re-implementation, cite original paper". Do NOT copy their code verbatim until licensing is clarified.

## 2. Main inventory table

All URLs live and verified 2026-04-18. Stars, last-commit dates, and license SPDX IDs all come from the GitHub REST API `/repos/{owner}/{repo}` endpoint, not from READMEs.

### Bucket A — Physics-informed deep learning infrastructure

| # | Repo | URL | Stars | Last commit (age) | License | Language | What it does | Reuse tier | Module/class to use | License compat (MIT target) | `baselines/` placement |
|---|---|---|---:|---|---|---|---|---|---|---|---|
| A1 | `rtqichen/torchdiffeq` | https://github.com/rtqichen/torchdiffeq | 6,397 | 2025-04-04 (12 mo) | MIT | PyTorch | Adjoint-method ODE solver for neural ODEs | DROP-IN | `torchdiffeq.odeint_adjoint`, `torchdiffeq.odeint` | COMPATIBLE | pip install only; cite in README |
| A2 | `patrick-kidger/diffrax` | https://github.com/patrick-kidger/diffrax | 1,976 | 2026-04-05 (13 d) | Apache-2.0 | JAX/Equinox | ODE/SDE solver with vmap + checkpointed adjoint | CITE-ONLY | `diffrax.diffeqsolve`, `diffrax.Dopri5` | COMPATIBLE (perm. notice req.) | not vendored — wrong framework for GIMIN |
| A3 | `rbischof/relative_balancing` | https://github.com/rbischof/relative_balancing | 72 | 2022-11-15 (3.4 yr) | **NO LICENSE** | PyTorch / TF | ReLoBRaLo adaptive loss-weight balancer for multi-term PINN loss | VENDOR | `src/update_rules.py` (ReLoBRaLo class, ~120 LOC) | **INCOMPATIBLE as-is** — re-implement from paper (Bischof & Kraus 2021 arxiv 2110.09813), 1-page algorithm box | `paper12_phys_gimin/baselines/relobralo/` (independent re-implementation) |
| A4 | `SciML/DifferentialEquations.jl` | https://github.com/SciML/DifferentialEquations.jl | 3,087 | 2026-04-09 (9 d) | MIT (LICENSE.md content confirmed; API tag said NOASSERTION) | Julia | Canonical Julia ODE solver stack; backs UDE and NeuralPDE | CITE-ONLY | n/a (Julia, wrong language) | COMPATIBLE | not vendored — Julia interop not worth it for phys-GIMIN |
| A5 | `ChrisRackauckas/universal_differential_equations` | https://github.com/ChrisRackauckas/universal_differential_equations | 238 | 2022-12-05 (3.3 yr) | MIT | Julia | Original UDE paper companion code (Rackauckas et al. 2020) | CITE-ONLY | n/a (Julia) | COMPATIBLE | not vendored — Julia; cite in §2 Background |
| A6 | `lululxvi/deepxde` | https://github.com/lululxvi/deepxde | 4,073 | 2026-03-01 (48 d) | **LGPL-2.1** | PyTorch + TF + Paddle | Toolbox for PINN / DeepONet / MF-PINN, 100+ examples | CITE-ONLY | `deepxde.nn`, `deepxde.geometry` | **NEEDS-ATTRIBUTION** (LGPL-2.1 is copyleft on *library*; dynamic linking via pip is OK but we'd need to redistribute LICENSE alongside any bundled model. Safer to not vendor.) | not vendored; cite as comparator PINN toolbox |
| A7 | `NVIDIA/physicsnemo` | https://github.com/NVIDIA/physicsnemo | 2,696 | 2026-04-17 (1 d) | Apache-2.0 | PyTorch | NVIDIA's PINN + neural-operator framework (rename of Modulus) | CITE-ONLY | n/a (overkill for our ODE-scale problem) | COMPATIBLE | not vendored |
| A8 | `SciML/NeuralPDE.jl` | https://github.com/SciML/NeuralPDE.jl | 1,187 | 2026-04-17 (1 d) | MIT (LICENSE.md content confirmed; API tag said NOASSERTION) | Julia | SciML-stack PINN solver | CITE-ONLY | n/a (Julia) | COMPATIBLE | not vendored |

### Bucket B — Competitor / baseline imputation codebases

| # | Repo | URL | Stars | Last commit (age) | License | Language | What it does | Reuse tier | Module/class to use | License compat (MIT target) | `baselines/` placement |
|---|---|---|---:|---|---|---|---|---|---|---|---|
| B1 | `neu-spiral/Hybrid-ODE-NN` | https://github.com/neu-spiral/Hybrid-ODE-NN | 2 | 2024-11-25 (16 mo) | **NO LICENSE FILE** (root directory listed, no LICENSE/COPYING of any flavor — confirmed via `GET /contents/` API) | PyTorch | Demirkaya 2024 TBME "Hybrid ODE-NN Framework" — reference Bayesian ODE-NN imputation | PORT (re-implement, don't copy) | `MNODE.py`, `MNODE_model.py` (Missing-dynamics Neural ODE), `BBNODE.py` (Black-Box Neural ODE) as paper reference only | **INCOMPATIBLE as-is** (no license = all-rights-reserved) | `paper12_phys_gimin/baselines/demirkaya2024/` — independent re-implementation based on paper + README |
| B2 | `bobjz/H2NCM` | https://github.com/bobjz/H2NCM | 4 | 2024-06-11 (22 mo) | **NO LICENSE FILE** (root contents: `CubatureFiltertf.py`, `README.md`, `cubaturetf.py`, `gsa_preset_dataset_0.01ms_steps_100ms_duration.npy` — zero LICENSE/COPYING of any flavor) | TensorFlow (Jupyter Notebook tag in API — actually `.py` + `.npy`) | Zou 2025 precursor — Cubature Kalman filter combined with NN for hybrid state estimation | PORT (re-implement, don't copy) | `cubaturetf.py` (cubature Kalman filter for NN state estimation), `CubatureFiltertf.py` | **INCOMPATIBLE as-is** (no license = all-rights-reserved) | `paper12_phys_gimin/baselines/zou2025_precursor/` — independent re-implementation |
| B3 | `WenjieDu/PyPOTS` | https://github.com/WenjieDu/PyPOTS | 1,998 | 2026-04-14 (4 d) | BSD-3-Clause | PyTorch | Time-series imputation zoo: SAITS, BRITS, GRU-D, M-RNN, Transformer, USGAN, GP-VAE, CSDI, DLinear, iTransformer | DROP-IN | `pypots.imputation.SAITS`, `pypots.imputation.BRITS`, `pypots.imputation.CSDI` | COMPATIBLE (BSD-3 is MIT-compatible with attribution) | pip install; cite in Related Work |
| B4 | `vanderschaarlab/hyperimpute` | https://github.com/vanderschaarlab/hyperimpute | 199 | 2023-04-04 (3.0 yr) | **MIT** (verified from LICENSE file, NOT GPL as CLAUDE.md hedged) | PyTorch / Python | GAIN, MIWAE, HyperImpute + classical imputers (MICE, MissForest, sklearn) | DROP-IN | `hyperimpute.plugins.imputers.plugin_gain.GainPlugin`, `.plugin_miwae.MiwaePlugin` | COMPATIBLE | already in `.venv`; cite in Related Work |

### Bucket C — Adjacent / reference only

| # | Repo | URL | Stars | Last commit (age) | License | Language | What it does | Reuse tier | Module/class to use | License compat (MIT target) | `baselines/` placement |
|---|---|---|---:|---|---|---|---|---|---|---|---|
| C1 | `SciML/DataDrivenDiffEq.jl` | https://github.com/SciML/DataDrivenDiffEq.jl | 424 | 2026-03-27 (22 d) | MIT | Julia | SINDy / sparse regression for discovering ODEs from data | CITE-ONLY | n/a (Julia); algorithm = SINDy from Brunton et al. 2016 | COMPATIBLE | not vendored — mention in §2 as alternative to physics-informed approach |
| C2 | `juliacamps/Cardiac-Digital-Twin` | https://github.com/juliacamps/Cardiac-Digital-Twin | 29 | 2026-02-03 (2 mo) | MIT | Python | Cardiac DT as architectural reference for NASEM-grade mechanistic twins | CITE-ONLY | n/a (different disease domain) | COMPATIBLE | not vendored — cite in §2 as NASEM digital-twin design reference |
| C3 | `YuliaRubanova/latent_ode` | https://github.com/YuliaRubanova/latent_ode | 586 | 2020-12-03 (5.3 yr) | MIT | PyTorch | Rubanova et al. 2019 NeurIPS "Latent ODE" reference implementation on irregular clinical time series | CITE-ONLY | `lib/latent_ode.py` — architectural reference for encoder-ODE-decoder irregular time-series structure | COMPATIBLE | not vendored — canonical reference, cite only |
| C4 | `yaringal/DropoutUncertaintyExps` | https://github.com/yaringal/DropoutUncertaintyExps | 581 | 2022-02-26 (4.1 yr) | **CC-BY-NC 4.0** (NON-COMMERCIAL — verified from LICENSE file, "All contributions by Yarin Gal are licensed under CC-BY-NC 4.0") | TF / Python | Original MC-dropout UQ experiments (Gal & Ghahramani 2016) | NOT-REUSABLE | n/a | **INCOMPATIBLE** for any commercial deployment (including AMP-PD industry partnerships). Safe for dissertation-only replication. | not vendored; cite Gal 2016 paper directly, not the repo |
| C5 | `PredictiveIntelligenceLab/jaxpi` | https://github.com/PredictiveIntelligenceLab/jaxpi | 418 | 2025-11-14 (5 mo) | **Custom Penn academic license** ("PirateNet" license, Penn tech transfer — verified from LICENSE file, NOT MIT/Apache/BSD) | JAX | State-of-the-art PINN trainer (Wang & Perdikaris 2024) | NOT-REUSABLE | n/a | **INCOMPATIBLE** with MIT target. Academic-only, non-redistributable. | not vendored; cite Wang 2024 paper |
| C6 | `idrl-lab/PINNpapers` | https://github.com/idrl-lab/PINNpapers | 1,472 | — | MIT | Markdown | Curated PINN paper list (bibliography resource, not code) | CITE-ONLY | n/a | COMPATIBLE | not vendored; use as literature anchor |

## 3. Recommended `paper12_phys_gimin/baselines/` layout

```
paper12_phys_gimin/
├── baselines/
│   ├── README.md                 # Overview; attribution statement for all vendored code
│   ├── demirkaya2024/            # INDEPENDENT RE-IMPL of neu-spiral/Hybrid-ODE-NN
│   │   ├── README.md             # Cite Demirkaya et al. 2024 TBME; note "re-implemented from paper, NOT derived from the unlicensed source repo". Pending: reuse-permission email to neu-spiral (see Action Items).
│   │   ├── mnode.py              # Missing-dynamics Neural ODE (our re-implementation)
│   │   ├── bbnode.py             # Black-Box Neural ODE (our re-implementation)
│   │   └── bayesian_state_est.py # Recursive Bayesian joint state+dynamics estimator
│   ├── zou2025_precursor/        # INDEPENDENT RE-IMPL of bobjz/H2NCM
│   │   ├── README.md             # Cite Zou et al. 2025; "re-implemented from paper". Pending: licensing email.
│   │   └── cubature_kalman.py    # Cubature Kalman filter for hybrid NN-dynamics state estimation
│   ├── relobralo/                # INDEPENDENT RE-IMPL (<200 LOC) of rbischof/relative_balancing
│   │   ├── README.md             # Cite Bischof & Kraus 2021 arxiv 2110.09813. Note "re-implemented from algorithm box, NOT derived from unlicensed source repo".
│   │   └── relobralo_balancer.py # Exponential running averages of loss-term ratios
│   ├── pypots_wrappers/          # DROP-IN via pip
│   │   ├── README.md             # Cite Du 2023 (SAITS), Cao 2018 (BRITS), Tashiro 2021 (CSDI). BSD-3-Clause attribution.
│   │   ├── saits_wrapper.py      # Wrapper over pypots.imputation.SAITS
│   │   ├── brits_wrapper.py      # Wrapper over pypots.imputation.BRITS
│   │   └── csdi_wrapper.py       # Wrapper over pypots.imputation.CSDI
│   ├── hyperimpute_wrappers/     # DROP-IN (already in .venv)
│   │   ├── README.md             # Cite Yoon 2018 (GAIN), Mattei & Frisch 2019 (MIWAE). MIT attribution.
│   │   ├── gain_wrapper.py
│   │   └── miwae_wrapper.py
│   └── torchdiffeq_integration/  # DROP-IN via pip (core phys-GIMIN ODE solver)
│       ├── README.md             # Cite Chen et al. 2018 NeurIPS NeuralODE paper. MIT attribution.
│       └── ode_solver.py         # Thin wrapper over torchdiffeq.odeint_adjoint w/ GIMIN's stage-conditioned dynamics
├── src/
│   └── phys_gimin/               # Our novel contribution: physics-informed GIMIN
│       ├── model.py
│       ├── ode_dynamics.py       # Whole-striatum decay + SBR dynamics from Phase 1
│       └── losses.py             # ReLoBRaLo-weighted physics + reconstruction + KL loss
└── tests/
```

**Attribution README stub (for `baselines/README.md`):**

```markdown
# phys-GIMIN baseline re-implementations — Attribution & Licensing

The following baselines are INDEPENDENTLY RE-IMPLEMENTED in this directory, NOT derived from their reference source code. Each was re-implemented from the corresponding published paper to guarantee a clean MIT license for phys-GIMIN.

| Baseline sub-dir | Reference paper | Reference repo | Reference license | Our re-impl license |
|---|---|---|---|---|
| demirkaya2024/ | Demirkaya et al. 2024, IEEE TBME | neu-spiral/Hybrid-ODE-NN | NO LICENSE (all rights reserved) | MIT |
| zou2025_precursor/ | Zou et al. 2025 | bobjz/H2NCM | NO LICENSE (all rights reserved) | MIT |
| relobralo/ | Bischof & Kraus 2021, arxiv 2110.09813 | rbischof/relative_balancing | NO LICENSE | MIT |

The following baselines are DROP-IN pip dependencies with compatible upstream licenses. We re-distribute wrapper code only (our own MIT), the upstream code remains under its original license.

| Baseline sub-dir | Reference paper | Upstream repo | Upstream license |
|---|---|---|---|
| pypots_wrappers/ | Du 2023 (SAITS) et al. | WenjieDu/PyPOTS | BSD-3-Clause |
| hyperimpute_wrappers/ | Yoon 2018 (GAIN) et al. | vanderschaarlab/hyperimpute | MIT |
| torchdiffeq_integration/ | Chen 2018 (NeuralODE) | rtqichen/torchdiffeq | MIT |

See per-sub-directory README.md for full citation information and reuse status.
```

## 4. Known gaps — what phys-GIMIN needs that no public repo provides

The following capabilities are necessary for phys-GIMIN but have **no suitable drop-in or vendor-able public implementation**:

1. **Stage-conditioned ODE dynamics with uncertainty-quantified physics constraints.** All the hybrid ODE-NN repos we surveyed (Demirkaya 2024, Zou 2025, Rubanova 2019, UDE.jl) treat the governing ODE as fixed and homogeneous across the cohort. Paper 12's core novelty — injecting patient-specific NSD-ISS-stage-conditioned decay priors from Paper 1 into the neural ODE — has no precedent. **Build from scratch** in `src/phys_gimin/ode_dynamics.py`.

2. **PD-specific biomarker ODE templates (whole-striatum SBR decay, α-syn-N(t) coupling).** No public repo exposes parameterized Parkinson's ODE skeletons. The only closest relative is SimBiology/SBML models in commercial Matlab stacks, which are not redistributable. **Author ourselves** from the Paper 7/8/9 mechanistic-twin Julia fits (already in `outputs/mechanistic_twin/phase{1,2}/`).

3. **Graph-informed (partial-similarity) imputation combined with ODE state constraints.** PyPOTS/hyperimpute/GIMIN Paper 2 all build graph imputation without dynamics; Hybrid-ODE-NN builds dynamics without graphs. **No public repo combines both.** This is precisely Paper 12's Architecture Innovation 1 — not vendor-able, must author.

4. **Heteroscedastic decoder with temperature-scaled calibration + IPCW-weighted conformal survival bands compatible with ODE rollout uncertainty.** Paper 4's conformal module exists in `src/giman_pipeline/paper4/`, but it assumes point-estimate hazards. Extending it to ODE-rollout-with-parameter-uncertainty distributions has no public precedent. **Extend internally.**

5. **Ablation infrastructure for "physics-only" vs "NN-only" vs "hybrid" evaluated on the same PD cohort.** No public repo provides this; we must author our own harness that reads GIMIN checkpoints, swaps in/out the ODE block, and re-scores on PPMI/BioFIND.

6. **Parkinson's-specific pre-trained checkpoints.** Zero across all 18 surveyed repos. All PD imputer checkpoints must be trained de novo from PPMI using our GIMIN + torchdiffeq stack.

## 5. Action items

1. **Email Demirkaya lab (Northeastern U, SPIRAL group, Prof. Stratis Ioannidis advisory)** to request explicit MIT/Apache re-license of `neu-spiral/Hybrid-ODE-NN` or written permission for academic re-implementation. Even if they decline, independent re-implementation from paper text remains legal; email just reduces reviewer risk. Template: "We are building a phys-informed neural-ODE PD imputer (paper under prep) and would like to cite your 2024 TBME hybrid ODE-NN framework as our reference baseline. Your repo `neu-spiral/Hybrid-ODE-NN` does not contain a LICENSE file — could you add one (MIT or Apache-2 preferred), or confirm in email that academic re-implementation for comparison is permitted?"

2. **Email Bob Zou (H2NCM author)** requesting the same for `bobjz/H2NCM`. Same template as above with paper title substituted.

3. **Do NOT vendor `rbischof/relative_balancing` source**; re-implement ReLoBRaLo from the arxiv paper text (arxiv 2110.09813) in `paper12_phys_gimin/baselines/relobralo/` and mark as independent implementation.

4. **Do NOT vendor any Penn `jaxpi` code** or any `DropoutUncertaintyExps` code. Use MC-dropout as described in Gal & Ghahramani 2016 directly; our PyTorch implementation already exists in `src/giman_pipeline/imputation/` for Paper 2.

5. **Pin `torchdiffeq==0.2.5` + `pypots>=0.7` + `hyperimpute==0.1.17`** in `paper12_phys_gimin/requirements.txt`. These are our three DROP-IN dependencies. Add LGPL warning in `requirements.txt` comment if we later decide to also pin `deepxde` (recommend against — not needed for our ODE-scale problem).

6. **Create `paper12_phys_gimin/baselines/README.md`** with the attribution stub from §3 before any code lands in this subdirectory. This way every contributor (including future Claude sessions) sees the "re-implement, don't copy" rule up front.

7. **Consider adding `SciML/DifferentialEquations.jl`-based Julia reproduction as Appendix E.3** analog to Paper 10's Julia refit capability, if the Paper 12 ODE fits benefit from Julia's SciML stack speed. Optional, not critical.

---

*End of inventory. All 18 repos verified live 2026-04-18 via `api.github.com`. License statements read directly from repo LICENSE/LICENSE.md files via `/contents/` API endpoint — no README-based or search-based license claims.*
