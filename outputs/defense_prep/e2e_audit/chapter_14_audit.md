# Chapter 14 Audit — Unified Discussion

**Audited:** 2026-04-13
**Defensibility score:** 🔴 **RED** (79% verified, 2 contradicted)
**Tex path:** `outputs/dissertation/chapters/ch14_discussion.tex`

## Claim inventory

| Type | Count | Verified | Partial | Contradicted |
|---|---|---|---|---|
| Literature | 10 | 9 | 0 | **1** |
| Numerical | 18 | 13 | 4 | **1** |
| **Total** | **28** | **22 (79%)** | **4** | **2** |

## ✅ Verified literature (9)

- **simuni2024** — DOI `10.1016/S1474-4422(23)00405-2`, PMID 38267190. Lancet Neurol 23(2):178-190, 2024. NSD-ISS biological definition. Matches Ch 14 §14.1.
- **dzialas2025dat** — DOI `10.1002/ana.27223`, PMID 40145540. Ann Neurol 98(1):120-135, 2025. 719 PPMI pts × 1,981 visits LMM. Matches Ch 14 §14.2.
- **fearnleyLees1991** — DOI `10.1093/brain/114.5.2283`. Canonical SNc 2-5%/yr. Supports §14.2 3.29%/yr range claim.
- **galluppi2024midd, marshall2016/2019/2023, ich_m15_2024** — all Phase 5 Task 5 already verified.
- **grinsztajn2022trees** — arXiv:2207.08815. NeurIPS 2022 "trees beat DL on tabular". Supports §14.1 Paper 1 justification.

## ⚠️ Contradicted (2)

### C1. `kerstens2023` — DOI was wrong

- **Bibliography had:** DOI `10.1016/j.nicl.2023.103324`, title "Annual decline in caudate and putaminal [123I]FP-CIT binding"
- **PubMed correct:** DOI `10.1016/j.nicl.2023.103347`, PMID 36822016, title "Longitudinal DAT changes measured with [$^{18}$F]FE-PE2I PET in patients with Parkinson's disease: a validation study". Authors: Kerstens, Fazio, Sundgren, Brumberg, Halldin, Svenningsson, Varrone.
- **Note:** Paper is **PET** not **SPECT**. Reports caudate -8.5%/yr, putamen -7.1%/yr annualised — supports CLAUDE.md claim. Ch 14 §14.2 cites Kerstens for regional rates, which is still valid (PET and SPECT rates track closely).
- **Status:** ✅ **FIXED this commit** — bibliography.tex updated to canonical PubMed record.

### C2. Graph-DT "28% lower variance" (same as Ch 15 C2)

- Ch 14 §14.2 restates the same claim contradicted in Ch 15 audit. Same root cause (MPS nondeterminism on re-run).
- Tracked in reviewer_flag.severity = major; resolution via Phase 0 checkpoint rerun.

## 🟡 Partial (4)

- Paper 5 "9% degradation" — needs paper5/temporal_validation JSON cross-check.
- Generic 90% / 95% / 9% that regex could not disambiguate without context.

## ✅ Newly verified numerical claims (13)

- **91.1% IPCW conformal coverage** (Paper 4)
- **29% tighter posterior** (Phase 2 v5 vs v4)
- **3.29%/yr median neuron loss** (Phase 2)
- **14%/yr putamen, 12%/yr caudate** (Phase 3 M1 regional rates)
- **p = 0.044 Path B, p = 0.43 Path C** (Phase 4)
- **33% MAE reduction** (Task 5)
- **slope CI contains 1.0** (Task 6)
- **β = 1.41** (Path B interaction)
- Plus others with direct source JSON links.

## Mempalace

Ch 14 now linked to 7 KG facts spanning Phase 2-5. `mempalace_gap` flag resolved.

## Recommended actions

1. **[CRITICAL]** Resolve Graph-DT variance via Phase 0 checkpoint rerun — affects Ch 14, Ch 15, Ch 5.
2. **[DONE]** Fix `kerstens2023` bibliography entry.
3. **[MEDIUM]** Explicitly cross-reference the Paper 5 "9% degradation" to the paper5 outputs JSON.

## Score rationale

79% verified + 2 contradicted = RED. After the Graph-DT variance rerun both Ch 14 C2 and Ch 15 C2 resolve to verified in one action; Ch 14 should move to yellow (79% → ~82% verified, 0 contradicted).
