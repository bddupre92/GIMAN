# Paper 3 (Chapter 5) Semantic Audit

**Reviewed:** `outputs/dissertation/chapters/ch05_paper3.tex`
**Against:** `outputs/paper3_benchmark/benchmark_summary.json`, `outputs/paper3_graph_dt/graph_dt_results.json`, `outputs/paper3_deephit/deephit_results.json`, `outputs/paper3_markov/markov_results.json`

## Verdict: 3 VALID, 4 AMBIGUOUS, 3 CONTRADICTED

### 🔴 CRITICAL — 3 contradicted claims

**C5-1: Graph-DT C-td = 0.926** (abstract line 9; table line 296; conclusion line 423)
- `benchmark_summary.json` → Graph-DT C-td = **0.9199**. `graph_dt_results.json` (Phase 0) = **0.904**.
- The 0.926 value is **DeepHit's** number, not Graph-DT's.
- VERDICT: CONTRADICTED

**C5-2: "Paired t-test t = 0.03, p = 0.976"** (lines 282, 308, table footnote)
- `benchmark_summary.json::statistical_tests` → t = **2.0676**, p = **0.1075**.
- p = 0.976 implies the two models are indistinguishable by nearly any test. Actual p = 0.108 is non-significant but qualitatively different.
- VERDICT: CONTRADICTED

**C5-3: "Graph-DT std = 0.007; 62% lower SD; 82% lower variance"** (lines 282, 295–296, 308, 385, 423)
- `benchmark_summary.json` → Graph-DT std = **0.01271** (DeepHit = 0.01824). Actual reduction: 30% lower SD, 51% lower variance.
- `graph_dt_results.json` (Phase 0 rerun) → std = **0.03021**, meaning Graph-DT is *worse* in variance than DeepHit under checkpoint-validated run.
- **The dual-reporting required by the e2e audit flag is entirely absent.** No MPS caveat, no Phase 0 qualification, no footnote anywhere in the file.
- VERDICT: CONTRADICTED (and unmitigated over-claim)

### 🟡 AMBIGUOUS

| # | Claim | Issue |
|---|---|---|
| C5-4 | DeepHit std = 0.019 | JSON 0.01824 (rounds to 0.018). Upward-rounded by 0.001. |
| C5-5 | Per-transition →0: DeepHit 0.904, Graph-DT 0.882 | `benchmark_summary.json` gives 0.902 / 0.856 |
| C5-6 | Per-transition →2B DeepHit 0.944 | JSON 0.9428 (rounds to 0.943) |
| C5-7 | Gate activation ~0.15 mean, ~0.20 stage 0 | No JSON artifact stores these values — fails traceability |

### ✅ VALID

- KM medians 2B→3=1.0yr, 3→4=5.2yr, 4→5=9.2yr — exact match to `simuni_aligned_km`
- Markov sojourn: Stage 0=13.3, 2B=0.68, 3=1.85, 4=1.42, 5=1.38 — confirmed
- Transition matrix counts (2B→3=438, 3→2B=540, etc.) — exact match

## Action Priority

1. Replace Graph-DT C-td 0.926 → 0.920; add Phase 0 footnote (0.904).
2. Replace t = 0.03, p = 0.976 → t = 2.07, p = 0.108 everywhere.
3. Replace std 0.007 → 0.013 (v5) OR 0.030 (Phase 0) with MPS footnote in all five occurrences.
4. Fix DeepHit std 0.019 → 0.018.
5. Persist gate activation statistics to JSON under `outputs/paper3_graph_dt/`.
