# Lesson: Identifiability Validation Protocol

**Date:** 2026-04-10
**Context:** Phase 2.5 SAA kinetics integration
**Triggered by:** Permutation null p=0.70 initially interpreted as "generic tightening" — WRONG. Then reinterpreted as "insensitive test" — PARTIALLY RIGHT. Then decisive SBR-only vs SAA test (ρ=0.0) revealed the REAL answer: SBR cannot separate k_n from α_tox under the slow-fast collapse.

---

## The Lesson

When adding a new observable to break a parameter degeneracy, you MUST validate the claim with THREE tests in this order:

### Test 1: Structural Test (pre-computation)
**Question:** Does the new observable depend on the target parameter through a DIFFERENT pathway than the existing observable?

**Method:** Write out the observation equations:
- Observable 1 (SBR): depends on T_tox = k_n × α_tox × const
- Observable 2 (CSF): depends on M_ss + r_o × O_ss(k_n) ← k_n enters through O_ss
- Observable 3 (SAA): depends on F_ss(k_n) ← k_n enters through F_ss

**Check:** If both observables depend on k_n only through the SAME product (T_tox), adding the second observable cannot help. If they depend on k_n through DIFFERENT functions, there is structural potential for degeneracy-breaking.

**Our case:** CSF and SAA both depend on k_n INDEPENDENTLY of α_tox — this structural test PASSES. BUT the SBR observable constrains T_tox (the product), not k_n individually. So adding CSF/SAA constrains k_n within the joint model, but the SBR constraint on the other end (α_tox) is still only through the product.

### Test 2: Non-Circular Decisive Test (post-computation)
**Question:** Does the EXISTING-observable-only posterior predict the NEW observable?

**Method:** Compute posterior from Observable 1 ONLY. Then correlate posterior target parameter with Observable 2.

**Our case:** Spearman(k_n_SBR-only, SAA TTT) = ρ = -0.01, p = 0.95. FAIL. SBR-only posterior has NO information about k_n that SAA can validate.

**If this test fails:** The new observable is constraining the parameter WITHIN the joint model but not providing independently verifiable per-patient information. The tightening is real computation but not independently validated biology.

### Test 3: Population Consistency Test (only if Test 2 passes)
**Question:** Is the constraint stronger for correctly-paired than randomly-paired data?

**Method:** Conflict frequency test or consistency score (see consciousness council recommendations).

**Our case:** Skipped because Test 2 failed.

---

## When You Violated This Protocol (and what happened)

1. We ran the computation (CSF/SAA IS) BEFORE the decisive test
2. We saw tightening and assumed it was genuine
3. The permutation null (wrong test) gave an ambiguous result
4. We spent cycles diagnosing the permutation null instead of running the simple decisive test first
5. The mathematical review correctly identified the test limitation but didn't flag the circularity
6. The consciousness council's Contrarian and Empiricist caught the circularity concern
7. The decisive test (5 minutes to run) resolved everything

**The fix:** Always run Test 2 (non-circular decisive test) BEFORE investing in multi-observable IS. It takes 5 minutes and saves days.

---

## Codified Rule (add to closed-loop methodology)

**Change 5 — Mandatory non-circular decisive test before claiming degeneracy-breaking**

**Rule:** When adding observable Y to break a parameter degeneracy in an existing model fit to observable X, BEFORE running the joint X+Y posterior, compute:

```
ρ = Spearman(posterior_θ_from_X_only, Y_observed)
```

If ρ is not significantly correlated in the expected direction, the joint model will produce tightening that is model-internal, not independently verifiable. Report this honestly.

**Enforcement:** The decisive test runs in <1 minute. It MUST precede any joint IS run. The result determines the framing: "independently validated per-patient constraint" (if ρ significant) vs "model-internal computational constraint" (if ρ ≈ 0).
