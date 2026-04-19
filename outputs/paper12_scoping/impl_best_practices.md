# Paper 12 — phys-GIMIN Implementation Best Practices (D4)

**Status:** Implementation-ready specification. Extends the locked decisions in the scoping plan (`~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md`) into copy-paste-grade PyTorch patterns. Companion to `method_blueprint.md` (D4 math side).

**Locked-decision recap (do not re-litigate):**

1. β-NLL (Seitzer ICLR 2022, β=0.5) as the aleatoric loss everywhere.
2. `stop_grad` on σ inside `L_physics` — physics cannot explain-away noise.
3. LR-annealing with EMA α=0.9 (Wang, Teng, Perdikaris 2021) — per-batch λ update.
4. Clamp `L_NLL ≥ 30%` of total loss (hard floor on λ_phys).
5. Physiology-informed Tikhonov on decoder-head parameters (Philipps 2024 PLOS CB).
6. Pre-compute ODE trajectories once per epoch from the HDF5 `PosteriorStore`.

**Codebase anchors (files to import, never modify):**

- `src/giman_pipeline/mechanistic_twin_v2/forward_model.py:30` — `SBR_SIGMA = 0.20`
- `src/giman_pipeline/mechanistic_twin_v2/observations.py:13` — `loglik_sbr(..., sigma=SBR_SIGMA)` already takes σ via kwarg.
- `src/giman_pipeline/mechanistic_twin_v2/posterior_store.py:88` — `class PosteriorStore` (HDF5-backed, read-only from phys-GIMIN).
- `GIMImpN_imputation/gimin/model/gimin_core.py:133-215` — GIMIN forward pass; `imputed_log_var` clamped to [−10, 10] at line 204.
- `src/giman_pipeline/imputation/temperature_scaling.py` — `PerFeatureTemperatureScaler` (post-hoc, reuse as-is).

Everything below slots into `paper12_phys_gimin/src/phys_gimin/` without touching these anchors.

---

## 1. Architectural patterns for `PhysicsRegularizer(provider: PriorProvider)`

**Chosen pattern: Strategy via `typing.Protocol`, with constructor injection + runtime type assertion + provenance logging.** Not a decorator, not a factory — those hide the variant at call sites and make contamination harder to audit.

**Why strategy-via-protocol beats alternatives:**

- **Decorator pattern** (e.g., `@with_lit_prior` wrapping a base regularizer) puts the variant choice at the *use* site. A reviewer reading `phys_regularizer.py` sees only "uses a prior" — has to chase decorators. Leaks are invisible in the module's own docstring.
- **Factory with runtime dispatch on a string flag** (`make_regularizer("lit")`) is exactly what Fix #2 in the plan rejects: a misconfigured config silently swaps variants.
- **Strategy + Protocol + construction-time `isinstance` check** puts the variant in the type system. `PhysicsRegularizer(provider=LiteraturePriorProvider(...))` is load-bearing Python syntax; there's no way to "accidentally" pass a `PosteriorStorePriorProvider` unless you *type the class name out*. The type is then serialized to the run's `config.json` (§7).

**Copy-paste skeleton:**

```python
# paper12_phys_gimin/src/phys_gimin/priors/base.py
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class PriorProvider(Protocol):
    """ODE-trajectory source for L_physics. Implementations MUST document
    their provenance so that config.json can record variant unambiguously."""
    variant_label: str  # "lit" or "self" — MUST be set by implementation
    prior_source_hash: str  # sha256 of the parameter source (lit constants or HDF5 file)

    def ode_trajectory(self, patno: int, t_years: np.ndarray) -> np.ndarray:
        """Return (T, K_twin) trajectory in observation-space units.
        K_twin = 4 (CAUDATE_L/R, PUTAMEN_L/R) for Paper 12 scope."""
        ...
```

```python
# paper12_phys_gimin/src/phys_gimin/priors/literature.py
class LiteraturePriorProvider:
    variant_label = "lit"
    # Fearnley-Lees 1991 N0, Lee 2019 γ=0.7, Iljina 2016 rate combination — FROZEN
    def __init__(self, constants_json: str):
        self._consts = json.load(open(constants_json))
        self.prior_source_hash = sha256(open(constants_json, "rb").read()).hexdigest()
    def ode_trajectory(self, patno, t_years):
        # Lit variant IGNORES patno (population-average trajectory)
        # This is a feature, not a bug — guarantees zero leakage from posteriors
        return _integrate_population_ode(self._consts, t_years)
```

```python
# paper12_phys_gimin/src/phys_gimin/priors/posterior_store.py
class PosteriorStorePriorProvider:
    variant_label = "self"
    def __init__(self, h5_path: str):
        self._store = PosteriorStore(h5_path, mode="r")  # read-only contract
        self.prior_source_hash = sha256(open(h5_path, "rb").read()).hexdigest()
    def ode_trajectory(self, patno, t_years):
        samples = self._store.load_posterior(patno)  # (n_samples, K)
        medians = np.median(samples, axis=0)
        return _integrate_per_patient_ode(medians, t_years)
```

```python
# paper12_phys_gimin/src/phys_gimin/regularizer.py
class PhysicsRegularizer(nn.Module):
    def __init__(self, provider: PriorProvider, beta: float = 0.5):
        super().__init__()
        assert isinstance(provider, PriorProvider), \
            f"provider must implement PriorProvider protocol, got {type(provider)}"
        assert hasattr(provider, "variant_label") and provider.variant_label in {"lit", "self"}, \
            "provider must set variant_label in {'lit','self'}"
        self.provider = provider
        self.beta = beta
        self.variant_label = provider.variant_label  # promoted to regularizer for logging
```

**Runtime type assertion is the belt-and-braces**: `@runtime_checkable` on the Protocol lets `isinstance(provider, PriorProvider)` actually enforce structural typing at construction. Combined with the `variant_label in {"lit","self"}` check, the set of reachable states is `{lit, self, TypeError}`.

**Similar-patterns citations (copy-paste the idiom):**

- [`torchdiffeq.odeint`](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py) dispatches solvers via a `method: str → SolverClass` table — we're rejecting that in favor of strategy-injection for exactly the reason the plan gives (string flag = stealth swap).
- [`diffrax.diffeqsolve`](https://github.com/patrick-kidger/diffrax) takes a `solver: AbstractSolver` argument — this is the strategy pattern we're copying. Kidger's API is the reference.
- [`pyro.distributions.TorchDistribution`](https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/torch_distribution.py) uses structural typing via mixins + `arg_constraints` — the provenance-fields-on-the-class idiom (`variant_label`, `prior_source_hash`) is borrowed from Pyro.

---

## 2. The β-NLL + stop-gradient recipe in PyTorch

**The one-line incantation:**

```python
L_physics = beta_nll(mu, sigma.detach(), ode_target, beta=0.5)
```

**The canonical implementation** (Seitzer 2022 §3.2, Eq. 8):

```python
def beta_nll(mu: Tensor, sigma: Tensor, target: Tensor, beta: float = 0.5) -> Tensor:
    """Seitzer et al. ICLR 2022. β=0 → NLL; β=1 → MSE; β=0.5 → balanced.
    sigma MUST be on the log-variance → stddev path already clamped to >=1e-3."""
    sigma = sigma.clamp(min=1e-3)  # numerical floor, see §6
    nll = 0.5 * ((target - mu) / sigma) ** 2 + torch.log(sigma)
    weight = sigma.detach() ** (2.0 * beta)  # inner detach: σ here is only a weight
    return (weight * nll).mean()
```

**Stop-gradient: the three footguns.**

| Pattern | Behavior | Verdict |
|---|---|---|
| `sigma.detach()` | Returns a new tensor with `requires_grad=False`, shares storage. Gradients through σ-producing layers are blocked at this tensor. | **Correct.** |
| `sigma.data` | Returns the underlying storage. Modifications to `.data` don't trigger autograd warnings. Use in forward → gradient is silently *lost*, not blocked. Future versions of PyTorch may deprecate. | **Wrong. Do not use.** |
| `with torch.no_grad(): sigma_sg = sigma` | Blocks gradient computation *during the block*, but `sigma_sg` re-enters the graph on next op. If `mu` is computed inside the block too, you just killed the mean-pathway gradient. | **Wrong (over-kill).** |
| `sigma.clone().detach()` | Equivalent to `.detach()` but with an extra memory copy. Unnecessary. | Suboptimal. |

**Use `.detach()`.** That's it. Seitzer's official repo ([martius-lab/beta-nll](https://github.com/martius-lab/beta-nll/blob/main/losses.py)) uses `.detach()`.

**Where to place stop-gradient in the training step:**

```python
# paper12_phys_gimin/src/phys_gimin/loss.py
def phys_gimin_loss_step(batch, model, regularizer, lam_phys):
    out = model(batch)  # out["imputed_mean"], out["imputed_log_var"]
    mu = out["imputed_mean"]
    sigma = torch.exp(0.5 * out["imputed_log_var"]).clamp(min=1e-3)

    # L_NLL: σ has gradient (aleatoric noise learned here)
    L_nll = beta_nll(mu[obs_mask], sigma[obs_mask], target[obs_mask], beta=0.5)

    # L_physics: σ is DETACHED — cannot absorb ODE-mean residual as "noise"
    ode_target = regularizer.get_ode_target(batch.patno, batch.t_years)
    twin_idx = [I_CAUDATE_L, I_CAUDATE_R, I_PUTAMEN_L, I_PUTAMEN_R]  # 4 features
    L_phys = beta_nll(
        mu[:, twin_idx],
        sigma[:, twin_idx].detach(),   # <-- the one line that matters
        ode_target,
        beta=0.5,
    )

    return L_nll + lam_phys * L_phys, {"L_nll": L_nll, "L_phys": L_phys}
```

**Unit-test pattern to verify stop-gradient actually stops gradients:**

```python
# paper12_phys_gimin/tests/test_stop_gradient.py
def test_physics_loss_does_not_flow_into_sigma_head():
    """Gradient of L_physics w.r.t. log_var_head params MUST be zero."""
    model = _make_toy_phys_gimin()
    batch = _make_toy_batch()
    reg = PhysicsRegularizer(provider=_make_toy_lit_provider())

    # Compute physics loss ALONE
    out = model(batch)
    sigma = torch.exp(0.5 * out["imputed_log_var"]).clamp(min=1e-3)
    L_phys = beta_nll(
        out["imputed_mean"][:, :4],
        sigma[:, :4].detach(),
        reg.get_ode_target(batch.patno, batch.t_years),
        beta=0.5,
    )
    L_phys.backward()

    # The log_var head (second half of decoder output projection) must have zero grad
    log_var_head_grad = model.decoder.out_proj.weight.grad[model.total_features:].abs().sum()
    assert log_var_head_grad.item() == 0.0, \
        f"σ head got gradient from L_physics (={log_var_head_grad.item():.3e}) — stop-grad is broken"

    # Sanity: μ head SHOULD have non-zero grad
    mu_head_grad = model.decoder.out_proj.weight.grad[:model.total_features].abs().sum()
    assert mu_head_grad.item() > 1e-6, "μ head got no grad — test is broken"
```

This is the single most important test in the repo. Run it in CI on every commit. It directly verifies the explain-away mitigation.

---

## 3. LR-annealing with EMA α=0.9

**Reference:** Wang, Teng, Perdikaris 2021 ([arXiv:2001.04536](https://arxiv.org/abs/2001.04536)), §3.3 "Learning-rate annealing for balancing loss terms." Reference PyTorch implementation: [`rbischof/relative_balancing`](https://github.com/rbischof/relative_balancing) — copy the λ-update loop (~60 LOC), ignore the NTK code (their NTK path is O(N²)-intractable at n=1,065).

**The math.**

At each training step, measure gradient magnitudes:

```
g_nll  = max_p |∂L_NLL / ∂θ_p|         # over all trainable params θ_p
g_phys = max_p |∂L_physics / ∂θ_p|
λ_hat  = g_nll / (g_phys + ε)           # the instantaneous balancing λ
λ      = α · λ_prev + (1 - α) · λ_hat  # EMA with α=0.9
```

Then clamp: `λ_phys = min(λ, λ_max)` where `λ_max` is chosen so `L_NLL ≥ 0.3 · (L_NLL + λ_phys · L_physics)`.

**Copy-paste implementation:**

```python
# paper12_phys_gimin/src/phys_gimin/scheduler.py
class LRAnnealLambda:
    """Wang-Teng-Perdikaris 2021 gradient-balancing λ scheduler with floor."""
    def __init__(self, ema_alpha: float = 0.9, lam_init: float = 0.0,
                 l_nll_floor_frac: float = 0.30, eps: float = 1e-8):
        self.alpha = ema_alpha
        self.lam = lam_init
        self.floor_frac = l_nll_floor_frac
        self.eps = eps
        self.history: list[dict] = []  # per-step diagnostics

    def step(self, L_nll: Tensor, L_phys: Tensor, model: nn.Module) -> float:
        """Call after loss.backward() BUT BEFORE optimizer.step(). Returns new λ."""
        # 1. Compute per-term gradients (retain_graph because we call backward twice)
        grads_nll  = torch.autograd.grad(L_nll,  model.parameters(),
                                         retain_graph=True, create_graph=False,
                                         allow_unused=True)
        grads_phys = torch.autograd.grad(L_phys, model.parameters(),
                                         retain_graph=True, create_graph=False,
                                         allow_unused=True)

        g_nll  = max((g.abs().max().item() for g in grads_nll  if g is not None), default=0.0)
        g_phys = max((g.abs().max().item() for g in grads_phys if g is not None), default=self.eps)

        # 2. NaN / degenerate-gradient protection — skip update, keep previous λ
        if not (math.isfinite(g_nll) and math.isfinite(g_phys)) or g_phys < self.eps:
            self.history.append({"skipped": True, "g_nll": g_nll, "g_phys": g_phys,
                                 "lam": self.lam})
            return self.lam

        # 3. Instantaneous λ and EMA
        lam_hat = g_nll / (g_phys + self.eps)
        self.lam = self.alpha * self.lam + (1.0 - self.alpha) * lam_hat

        # 4. Enforce L_NLL ≥ 30% of total loss (σ-collapse prevention)
        L_nll_v  = L_nll.item()
        L_phys_v = L_phys.item()
        if L_phys_v > 0:
            lam_max = (L_nll_v / self.floor_frac - L_nll_v) / L_phys_v
            self.lam = min(self.lam, max(lam_max, 0.0))

        self.history.append({"g_nll": g_nll, "g_phys": g_phys, "lam_hat": lam_hat,
                             "lam": self.lam, "L_nll": L_nll_v, "L_phys": L_phys_v})
        return self.lam
```

**Training-loop integration:**

```python
sched = LRAnnealLambda(ema_alpha=0.9, lam_init=0.0, l_nll_floor_frac=0.30)

for epoch in range(n_epochs):
    for batch in loader:
        opt.zero_grad()
        L_nll, L_phys, parts = model.loss(batch, regularizer)

        # Warmup: λ=0 for first `phys_warmup_epochs` — let imputation converge first
        if epoch < PHYS_WARMUP_EPOCHS:
            total = L_nll
        else:
            # Update λ BEFORE computing total loss
            lam = sched.step(L_nll, L_phys, model)
            total = L_nll + lam * L_phys

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # NaN insurance
        opt.step()
```

**When to trigger the λ update — the two valid cadences:**

1. **Every batch** (copied above): cheapest, most responsive. EMA α=0.9 means effective look-back of ~10 batches. Safe because Wang 2021's analysis is per-step.
2. **Every N batches** (e.g., N=10): reduces the `autograd.grad` overhead but lags behind loss-landscape changes. Only use if per-batch gradient-compute is >15% of step time (profile first).

**NaN strategy (§3 step 2 above) — full taxonomy:**

- `g_phys == 0` (patient has no valid ODE target): treated as "skip" — λ unchanged. Logged in `history` for audit.
- `g_nll == 0` (all observed values perfectly predicted — vanishingly rare): λ will naturally go to 0 via the ratio; physics loss dominates; but the *floor clamp* then blocks λ at 0 (since L_NLL_v/0.3 − L_NLL_v → 0 when L_NLL_v → 0). This is the right behavior: if reconstruction is perfect, no need for regularization.
- `g_nll == NaN` or `g_phys == NaN`: hard skip + log. If NaN persists >10 steps, abort training (assertion in training loop).

---

## 4. Pre-computing ODE trajectories from HDF5 `PosteriorStore`

**Design principle:** ODE integration is the single most expensive op per step if done naively. With 1,065 patients × 11 time bins × 4 twin-observable features, pre-computation is the no-brainer win the plan already identifies.

**Recommended data flow (once per epoch):**

```python
# paper12_phys_gimin/src/phys_gimin/ode_cache.py
class ODECacheBuilder:
    """Loads posterior samples from HDF5 once at epoch start, integrates ODE
    per-patient, caches (patno → trajectory tensor on device) dict.
    Replaces per-batch odeint calls with O(1) dict lookup."""

    def __init__(self, provider: PriorProvider, patnos: list[int],
                 t_years: np.ndarray, device: torch.device):
        self.provider = provider
        self.patnos = patnos
        self.t_years = t_years
        self.device = device
        self.cache: dict[int, Tensor] = {}  # patno → (T, K_twin) on device

    def rebuild(self):
        """Call once per epoch. Scales as O(n_patients × T_integration_steps)."""
        with torch.no_grad():
            for p in self.patnos:
                traj_np = self.provider.ode_trajectory(p, self.t_years)  # (T, K_twin)
                self.cache[p] = torch.from_numpy(traj_np).float().to(self.device)

    def lookup(self, patnos: Tensor) -> Tensor:
        """Per-batch, O(batch_size) lookup."""
        return torch.stack([self.cache[int(p)] for p in patnos.cpu().tolist()], dim=0)

# In the training loop:
cache = ODECacheBuilder(provider, train_patnos, t_years, device)
for epoch in range(n_epochs):
    cache.rebuild()  # ~10s for lit variant (population-avg, identical trajectory)
                     # ~2min for self variant (1,065 per-patient integrations on CPU)
    for batch in loader:
        ode_target = cache.lookup(batch.patno)  # ms
        ...
```

**Memory budget for H100 (80 GB):**

| Dimension | Value | Per-tensor bytes |
|---|---|---|
| Patients | 1,065 | |
| Time bins (T) | 11 (Paper 3 bins) | |
| Twin features (K) | 4 (CAUDATE_L/R, PUTAMEN_L/R) | |
| dtype | `float32` | 4 |
| Per-patient tensor | 11 × 4 × 4 B | 176 B |
| **Full cache** | 1,065 × 176 B | **≈ 188 KB** |

Trivially fits in L2. Even scaling to the full Paper 3 cohort (1,900) × 11 × 33 features × FP32 = 275 KB. This is not a memory-bound problem; integration wall-time is.

**Integration budget:** Lit variant runs the population-average ODE once per epoch (constant). Self variant runs 1,065 per-patient integrations per epoch. Using SciPy's `solve_ivp` with `method="LSODA"` (stiff-safe), each is ~2 ms on CPU → ~2 s per epoch for self. Negligible vs the GIMIN forward pass (~30 s/epoch).

**Why *not* `torchdiffeq.odeint_adjoint`:** autodiff through the ODE is only needed if we're fitting k_n and α_tox end-to-end. The plan expressly forbids that (it's the tautology audit's core constraint). Pre-compute + `torch.from_numpy` is 100× faster and architecturally cleaner.

**HDF5 read pattern (critical — do not hold the file handle open during training):**

```python
# GOOD: open once, materialize all posteriors to memory, close file, integrate at leisure.
with h5py.File(h5_path, "r") as f:
    posteriors = {patno: f[f"patient_{patno}/v1/samples"][...] for patno in patnos}
# File closed; trajectories integrated from in-memory dict; no file locks during training.

# BAD: holding the file open during `cache.rebuild()` — HDF5 locking will collide
#      with multiprocessing DataLoader workers.
```

---

## 5. σ-calibration coverage protocol

**The hypothesis under test:** β-NLL (β=0.5) + stop-gradient-on-σ-inside-`L_physics` preserves marginal coverage at γ=0.90 under λ_phys > 0, within ±3 pp of the λ_phys=0 baseline.

**Three-tier smoke test, in order of increasing stakes:**

### Tier 1 — Synthetic recovery (Seitzer 2022 Table 1 benchmark)

Generate `y = f(x) + σ_true · ε` on synthetic data with *known* heteroscedastic `σ_true(x)`. Use Seitzer's toy dataset ([martius-lab/beta-nll/tests/test_synthetic.py](https://github.com/martius-lab/beta-nll)) as the reference.

**Acceptance:** Predicted σ within ±5% of σ_true over the input domain, median across 10 seeds, for β ∈ {0, 0.5, 1.0}. Replicates Table 1, row β=0.5. If this fails, the β-NLL implementation itself is broken — stop before testing phys-GIMIN.

### Tier 2 — Physics-off ablation

Run phys-GIMIN with `λ_phys = 0` (i.e., reduces to vanilla GIMIN with β-NLL-replaced reconstruction loss). Measure marginal coverage at γ ∈ {0.50, 0.80, 0.90, 0.95} on Paper 2's held-out split.

**Acceptance:** Coverage matches the Paper 2 §V.E raw-decoder baseline within 2 pp per γ level. (Coverage at γ=0.90 should be ~0.664, matching the session 2026-04-18 measurement.) If this deviates more than 2 pp, the β-NLL formulation has drifted from GIMIN's existing Gaussian-NLL.

### Tier 3 — λ_phys sweep coverage monitor

Sweep λ_phys ∈ {0.0, 0.01, 0.05, 0.1, 0.5, 1.0, scheduled}. For each setting, measure:

- Marginal coverage at γ ∈ {0.50, 0.80, 0.90, 0.95}
- Per-feature coverage for the 4 twin-observable SBR features
- Median and 90%-quantile σ values across the evaluation cohort
- RMSE on held-out MCAR-0.25 masked entries

**Acceptance (pre-registered):**

1. Coverage at γ=0.90 drops by at most 3 pp vs λ_phys=0 (the plan's risk-register threshold).
2. Per-feature 90% coverage for all 4 SBR features stays ≥ 0.85.
3. Median σ for SBR features does not drop below 50% of its λ_phys=0 value (σ collapse canary).
4. RMSE improves monotonically or stays flat as λ_phys rises to the scheduled value.

If any (1)-(3) fails on the lit variant, pre-registered abort: pivot to σ-calibration-only contribution (plan Fix #3).

**Automation:** Ship this as `paper12_phys_gimin/scripts/run_coverage_smoke_test.py` and wire it to the benchmark pipeline as a "red-light" gate before full training completes.

---

## 6. Anti-patterns with concrete fixes

### AP-1: σ collapse at high λ_phys

**Symptom.** Coverage at γ=0.90 drops below 0.80; median σ for twin features halves vs the λ_phys=0 baseline; test RMSE looks superficially better because predicted distributions are over-confident.

**Root cause (diagnostic order).**

1. Is stop-gradient actually stopping? Run the unit test from §2. If the log-var head has non-zero gradient from L_physics, stop-gradient is broken.
2. Is the `L_NLL ≥ 30%` floor firing? Check `sched.history[-1]["lam"]` vs the unclamped `lam_hat`. If the floor is *never* clamping, raise `l_nll_floor_frac` to 0.40.
3. Is β too low? β=0 (plain NLL) is Seitzer's worst-case for variance underestimation. If tests show β=0.5 is borderline, step up to β=0.7 (still in Seitzer's safe range).

**Fix.** In priority: (1) re-run the stop-gradient unit test, (2) bump the floor to 40%, (3) step β to 0.7, (4) fall back to disabling `L_physics` in eval — temperature-scaling at λ_phys=0 as Paper 2 §V.E showed, closing 82% of the gap without any physics.

### AP-2: Physics loss dominance

**Symptom.** L_NLL plateaus early (epoch 10-15) while L_physics continues decreasing; training loss looks great, held-out RMSE degrades; `lam_phys > 10 · L_NLL` in the scheduler history.

**Root cause.** The EMA scheduler has no upper bound on λ unless the floor clamp is active — if L_physics has a systematic offset (e.g., ODE integrator's numerical precision > σ scale), λ_hat blows up.

**Fix.** Hard ceiling on λ: `lam_cap = 10.0`, enforced in `LRAnnealLambda.step` *before* the floor check. Log both `lam_hat` (pre-ceiling) and `lam` (post-ceiling) to every run's JSON so reviewers can see the clamp was active. If `lam_hat > 100` for more than 20 consecutive steps, abort training — the provider is producing nonsense trajectories.

### AP-3: Tautology leakage via config error

**Symptom.** Self-variant results look "too good" on downstream tasks (Paper 7/9/10-like targets). Silent confirmation that `config.variant_label == "self"` but the downstream evaluation pipeline was trained with knowledge of the same posteriors.

**Root cause.** `config.json` (§7) was hand-edited, or a YAML anchor silently pointed `self` at a lit provider. Either way, the audit trail is ambiguous.

**Fix.**

1. `PhysicsRegularizer.__init__` logs `(variant_label, prior_source_hash)` to stdout AND to the run's `config.json` — both entries MUST match.
2. Results tables in Paper 12 §V.C (negative results) carry the `⚠ partially tautological` glyph *generated* from `config.json`, not hand-typed. `scripts/run_section_c_negative.py` errors out if the tautology glyph is missing on any self-variant row.
3. Git pre-commit hook checks that `config.json` in any `outputs/paper12_benchmark/runs/*self*/` dir has `variant_label == "self"` AND `prior_source_hash` matches the project's canonical HDF5. (Analog of the existing `check_audit_freshness.py` hook.)

### AP-4: Unit mismatch (normalized vs absolute SBR)

**Symptom.** `L_physics` is ~100× its expected magnitude; the σ-aware weighting blows up; training NaNs after a few hundred steps.

**Root cause.** The GIMIN decoder outputs z-scored features (via `ModalityAwareScaler`); the ODE integrator outputs raw SBR (unitless ratio, ~0.5-2.5). One side or the other has to convert.

**Fix (canonical — enforce at the boundary).** All `L_physics` computation happens in **z-scored space**. The `PhysicsRegularizer` holds the `ModalityAwareScaler` and transforms the ODE trajectory *into* z-scored space at cache-build time:

```python
# paper12_phys_gimin/src/phys_gimin/ode_cache.py
class ODECacheBuilder:
    def __init__(self, ..., scaler: ModalityAwareScaler):
        self.scaler = scaler  # frozen — same scaler fit at phys-GIMIN training start

    def rebuild(self):
        for p in self.patnos:
            traj_abs = self.provider.ode_trajectory(p, self.t_years)  # absolute SBR
            traj_z = self.scaler.transform_features(traj_abs, feature_names=TWIN_FEATURES)
            self.cache[p] = torch.from_numpy(traj_z).float().to(self.device)
```

Assertion at the boundary: `assert abs(traj_z.mean()) < 5.0, f"ODE target z-score out of plausible range: mean={traj_z.mean()}"`. Any drift triggers a unit-check failure before training eats 10 epochs of compute.

### AP-5: Numerical instability in β-NLL at σ → 0

**Symptom.** Loss = Inf, followed by NaN gradients, typically in epoch 1-2 for patients with near-perfect reconstruction.

**Root cause.** `torch.log(sigma)` → −∞ as σ → 0; simultaneously `(y − μ)² / σ²` → ∞; their sum is ill-defined.

**Fix.** Double-clamp: clamp `log_var` to [−10, 10] in GIMIN (already done — see `gimin_core.py:204`) AND clamp `sigma` to ≥ 1e-3 at every β-NLL call site. The `sigma.clamp(min=1e-3)` in `beta_nll` (§2) is the defensive line. Verify with:

```python
def test_beta_nll_at_zero_sigma():
    mu = torch.zeros(4)
    sigma = torch.zeros(4)  # pathological
    target = torch.randn(4)
    loss = beta_nll(mu, sigma, target, beta=0.5)
    assert torch.isfinite(loss), "β-NLL not robust to σ=0"
```

---

## 7. Benchmark reproducibility harness

**The non-negotiable `config.json` schema** (dual-save pattern from Paper 2 incident):

```json
{
  "run_id": "phys_gimin_lit_20260419_143022",
  "timestamp": "2026-04-19T14:30:22Z",
  "git": {
    "sha": "e6c63b5",
    "branch": "feat/paper12-phys-gimin-lit-variant",
    "dirty": false,
    "diff_hash": null
  },
  "variant": {
    "label": "lit",
    "prior_provider_class": "LiteraturePriorProvider",
    "prior_source_hash": "sha256:a3f4..."
  },
  "seeds": {"torch": 42, "numpy": 42, "cuda": 42},
  "hyperparameters": {
    "beta": 0.5, "ema_alpha": 0.9, "lam_init": 0.0,
    "l_nll_floor_frac": 0.30, "lam_cap": 10.0,
    "phys_warmup_epochs": 10
  },
  "data": {
    "cohort_hash": "sha256:..",  // of train CSV + masks
    "features_used": ["AGE", "SEX", "..."],
    "mask_fraction": 0.25, "mask_strategy": "MCAR"
  },
  "upstream_artifacts": {
    "gimin_base_checkpoint": "outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/checkpoints/gimin_frac0.25_run0.pt",
    "gimin_base_checkpoint_hash": "sha256:...",
    "posterior_store": "outputs/mechanistic_twin/paper10_mech_vs_giman/posteriors.h5",
    "posterior_store_hash": "sha256:..."
  }
}
```

**Emit from a single writer:**

```python
# paper12_phys_gimin/src/phys_gimin/config_writer.py
def write_run_config(run_dir: Path, regularizer, args, data_ctx):
    cfg = {
        "run_id": run_dir.name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git": _git_info(),            # subprocess call to `git rev-parse`
        "variant": {
            "label": regularizer.provider.variant_label,
            "prior_provider_class": type(regularizer.provider).__name__,
            "prior_source_hash": regularizer.provider.prior_source_hash,
        },
        "seeds": {"torch": args.seed, "numpy": args.seed, "cuda": args.seed},
        "hyperparameters": _extract_hparams(args),
        "data": data_ctx.as_dict(),
        "upstream_artifacts": _upstream_hashes(args),
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
```

**Variant-level safety interlock.** Results-analysis scripts MUST filter by `config["variant"]["label"]` — never by directory name regex. Directory naming conventions drift; `config.json` is canonical. Cross-variant plots must be explicitly requested, never default behavior.

**Prior-source-hash firewall.** The regularizer's `prior_source_hash` is computed at construction (sha256 of either the lit constants JSON or the HDF5 posterior store). Any results file downstream of the regularizer embeds this hash. Post-hoc re-analysis scripts check:

```python
assert run_config["variant"]["prior_source_hash"] == \
       results_json["meta"]["prior_source_hash"], \
       "results file produced by a different prior source than the run config claims"
```

Mismatches trigger a loud error, not a warning.

---

## 8. SciML testing idioms worth copying

**All three tests below are MANDATORY in `paper12_phys_gimin/tests/` before the first benchmark run.**

### Test 8.1 — Identifiability parity with Phase 2 (Fisher Information Matrix overlap)

**Motivation.** If phys-GIMIN's effective likelihood is equivalent (up to a constant) to Phase 2 direct calibration on overlapping patients, the FIMs should have comparable rank + condition number. Divergence means phys-GIMIN has unintentionally regularized out an identifiable direction — the exact failure mode Villaverde 2016 warns about.

**Pattern** (following [`SciML/StructuralIdentifiability.jl`](https://github.com/SciML/StructuralIdentifiability.jl) and [`Philipps 2024 PLOS CB replication code`](https://github.com/philipps-biomed/pinn-identifiability)):

```python
# paper12_phys_gimin/tests/test_identifiability_parity.py
def test_fim_rank_matches_phase2():
    """On the 40-patient Phase 2 ∩ phys-GIMIN overlap, phys-GIMIN's effective FIM
    must have the same rank as Phase 2's direct FIM (within numerical tolerance)."""
    overlap_patnos = _load_phase2_phys_gimin_overlap()  # ~40 patients
    phase2_fim  = _compute_phase2_fim(overlap_patnos, posteriors)
    phys_fim    = _compute_phys_gimin_effective_fim(overlap_patnos, trained_model)
    rank_phase2 = np.linalg.matrix_rank(phase2_fim, tol=1e-6)
    rank_phys   = np.linalg.matrix_rank(phys_fim,   tol=1e-6)
    assert rank_phase2 == rank_phys, \
        f"phys-GIMIN changed identifiability structure: rank {rank_phase2} → {rank_phys}"
    cond_ratio = np.linalg.cond(phys_fim) / np.linalg.cond(phase2_fim)
    assert 0.1 < cond_ratio < 10.0, f"FIM condition number changed by {cond_ratio}×"
```

### Test 8.2 — Gradient check against numerical differentiation

**Motivation.** Standard SciML sanity check ([`torchdiffeq/tests/gradient_test.py`](https://github.com/rtqichen/torchdiffeq/blob/master/tests/gradient_tests.py), [`diffrax/tests/test_adjoint.py`](https://github.com/patrick-kidger/diffrax)). On a single minibatch, analytic gradients must match finite-difference gradients.

```python
# paper12_phys_gimin/tests/test_gradient_check.py
def test_gradient_check_single_minibatch():
    """Analytic ∂L/∂θ vs finite-difference, on one batch of 8 patients."""
    model = _make_small_phys_gimin()
    batch = _make_toy_batch(n=8)
    regularizer = _make_toy_lit_regularizer()

    # Analytic
    loss_fn = lambda: model.loss(batch, regularizer)[0]
    analytic = torch.autograd.grad(loss_fn(), model.parameters(), create_graph=False)

    # Finite difference over a small subset (first 50 params — full O(P) is expensive)
    eps = 1e-4
    for i, p in enumerate(list(model.parameters())[:5]):
        p_flat = p.view(-1)
        for j in range(min(10, p_flat.numel())):
            p_flat[j] += eps; f_plus  = loss_fn().item()
            p_flat[j] -= 2*eps; f_minus = loss_fn().item()
            p_flat[j] += eps  # restore
            fd = (f_plus - f_minus) / (2 * eps)
            assert abs(analytic[i].view(-1)[j].item() - fd) < 1e-2, \
                f"gradient mismatch at param {i}[{j}]: analytic={analytic[i].view(-1)[j]}, fd={fd}"
```

### Test 8.3 — Physics-off equivalence ablation

**Motivation.** Setting `λ_phys=0` MUST reduce phys-GIMIN to vanilla GIMIN (with β-NLL reconstruction) — bit-exact, modulo floating-point reproducibility.

```python
# paper12_phys_gimin/tests/test_physics_off_equivalence.py
def test_lambda_zero_matches_vanilla_gimin():
    """With λ_phys=0, phys-GIMIN must match the vanilla GIMIN β-NLL baseline
    on the same batch, same seed, same initialization."""
    torch.manual_seed(0); np.random.seed(0)
    vanilla = _make_vanilla_gimin_beta_nll()  # β-NLL version of existing gimin_core
    torch.manual_seed(0); np.random.seed(0)
    phys = PhysGIMIN(provider=_make_toy_lit_provider())

    batch = _make_toy_batch()
    with torch.no_grad():
        v_out = vanilla(batch)
        p_out = phys(batch, lam_phys=0.0)

    torch.testing.assert_close(v_out["imputed_mean"],   p_out["imputed_mean"],   rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(v_out["imputed_log_var"], p_out["imputed_log_var"], rtol=1e-6, atol=1e-6)
```

This test is the architectural contract with Paper 2 — if it fails, phys-GIMIN has diverged from the vanilla base for reasons unrelated to physics, and all Paper 2 §V.E comparisons are invalid.

**Reference patterns:**

- [`PyPOTS/tests/`](https://github.com/WenjieDu/PyPOTS/tree/main/tests) — imputer interface conformance tests; copy the interface-shape assertions (output dict keys, tensor shapes, dtype).
- [`SciML/NeuralPDE.jl tests`](https://github.com/SciML/NeuralPDE.jl/tree/master/test) — physics-loss convergence tests on known analytic solutions; adapt for the Phase 2 ODE's analytic limit (α_tox → 0 ⇒ N(t) = N₀, trivially verifiable).
- [`diffrax/tests/test_misc.py`](https://github.com/patrick-kidger/diffrax/blob/main/tests/) — ODE-integrator monotonicity + endpoint-at-t0 tests; copy these for the lit-variant population trajectory.

---

## Closing note: priority order if time-constrained

If postdoc execution is compressed, implement in this order:

1. Architecture (§1) + β-NLL + stop-grad unit test (§2) — no physics runs without these.
2. Physics-off equivalence test (Test 8.3) — architectural contract with Paper 2.
3. ODE cache + unit assertion (§4 + AP-4) — correctness gate.
4. LR-annealing scheduler (§3) with NaN guards.
5. Tier 1 + Tier 2 coverage smoke tests (§5) — catches σ collapse before the full sweep.
6. config.json writer + tautology interlock (§7 + AP-3) — leakage firewall.
7. FIM parity (Test 8.1) + Tier 3 sweep (§5) — methods-paper credibility.
8. Gradient check (Test 8.2) — reviewer-facing polish.

Everything else in this doc (AP-1, AP-2, AP-5 mitigations, full benchmark integrity) becomes non-optional once the benchmark run goes multi-day.
