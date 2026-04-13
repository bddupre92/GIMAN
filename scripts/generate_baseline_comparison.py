from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path('/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025')
TODAY = date.today().isoformat()

phase8_path = BASE / 'outputs/phase8_2_final_training/training_results.json'
phase9_report = BASE / 'Docs/audit/GIMAN_PHASE9_RUN_REPORT_2026-02-05.md'

out_dir = BASE / 'visualizations' / 'model_comparison' / f'baseline_vs_fuzzy_{TODAY}'
out_dir.mkdir(parents=True, exist_ok=True)
report_path = BASE / 'Docs/audit' / f'GIMAN_BASELINE_VS_FUZZY_{TODAY}.md'

with phase8_path.open('r', encoding='utf-8') as f:
    p8 = json.load(f)

p8_cv = p8['cross_validation']['mean_c_index']
p8_test = p8['final_model']['best_test_c_index']

text = phase9_report.read_text(encoding='utf-8') if phase9_report.exists() else ''

def pick(pattern: str) -> float | None:
    m = re.search(pattern, text, flags=re.DOTALL)
    return float(m.group(1)) if m else None

p9_full_auc = pick(r'Best test AUC observed during training: `([0-9.]+)`')
p9_simple_auc = pick(r"### 1\) `phase9_neuro_fuzzy_implementation\.py`.*?SAA AUC: `([0-9.]+)`")
p9_multi_auc = pick(r"### 3\) `phase9_multitask_learning\.py`.*?SAA AUC: `([0-9.]+)`")
p9_multi_cidx = pick(r"### 3\) `phase9_multitask_learning\.py`.*?Survival C-index: `([0-9.]+)`")

# Plot 1: Survival comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['Phase8 GIMAN (CV)', 'Phase8 GIMAN (Test)', 'Phase9 Fuzzy MT (C-index)'],
            [p8_cv, p8_test, p9_multi_cidx if p9_multi_cidx is not None else 0],
            color=['#4e79a7', '#59a14f', '#f28e2b'])
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('C-index')
axes[0].set_title('Survival Performance: Baseline vs Fuzzy')
axes[0].tick_params(axis='x', rotation=20)

# Plot 2: Classification comparison (Phase9 variants)
auc_labels = ['Phase9 Fuzzy Simple AUC', 'Phase9 Fuzzy Full AUC', 'Phase9 Fuzzy MT AUC']
auc_vals = [
    p9_simple_auc if p9_simple_auc is not None else 0,
    p9_full_auc if p9_full_auc is not None else 0,
    p9_multi_auc if p9_multi_auc is not None else 0,
]
axes[1].bar(auc_labels, auc_vals, color=['#76b7b2', '#edc948', '#e15759'])
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel('AUC')
axes[1].set_title('Phase9 Classification Performance')
axes[1].tick_params(axis='x', rotation=20)

fig.tight_layout()
fig_path = out_dir / 'phase8_vs_phase9_comparison.png'
fig.savefig(fig_path, dpi=300)
plt.close(fig)

lines = [
    f'# GIMAN Baseline vs Fuzzy Comparison ({TODAY})',
    '',
    '## Baseline (Phase 8, non-fuzzy)',
    f'- CV mean C-index: `{p8_cv:.4f}`',
    f'- Best test C-index: `{p8_test:.4f}`',
    '',
    '## Neuro-Fuzzy (Phase 9)',
    f"- Full neuro-fuzzy best AUC: `{p9_full_auc:.4f}`" if p9_full_auc is not None else '- Full neuro-fuzzy best AUC: unavailable',
    f"- Multi-task neuro-fuzzy AUC: `{p9_multi_auc:.4f}`" if p9_multi_auc is not None else '- Multi-task neuro-fuzzy AUC: unavailable',
    f"- Multi-task neuro-fuzzy C-index: `{p9_multi_cidx:.4f}`" if p9_multi_cidx is not None else '- Multi-task neuro-fuzzy C-index: unavailable',
    f"- Simple neuro-fuzzy AUC: `{p9_simple_auc:.4f}`" if p9_simple_auc is not None else '- Simple neuro-fuzzy AUC: unavailable',
    '',
    '## Interpretation',
    '- Survival baseline remains very strong in Phase 8.',
    '- Multi-task fuzzy variant is competitive on survival while adding high classification AUC.',
    '- This supports reporting fuzzy enhancement as complementary rather than replacing the survival baseline outright.',
    '',
    '## Artifacts',
    f'- Figure: `{fig_path}`',
]

report_path.write_text('\n'.join(lines), encoding='utf-8')
print(f'Wrote {report_path}')
print(f'Wrote {fig_path}')
