from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

BASE = Path('/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025')
PYTHON = BASE / '.venv/bin/python'
TODAY = date.today().isoformat()
REPORT = BASE / 'Docs/audit' / f'GIMAN_WORKFLOW_MATRIX_{TODAY}.md'
JSON_OUT = BASE / 'Docs/audit' / f'GIMAN_WORKFLOW_MATRIX_{TODAY}.json'

CASES = [
    ('phase1_validation', 'archive/development/phase1/task_1_6_cohort_validation.py', 180),
    ('phase2_integration', 'archive/development/phase2/giman_integration_test.py', 180),
    ('phase3_e2e', 'archive/development/phase3/phase3_0_end_to_end_giman_test.py', 180),
    ('phase4_quick', 'archive/development/phase4/phase4_quick_stabilization_test.py', 180),
    ('phase5_validation', 'archive/development/phase5/phase5_validation_test.py', 180),
    ('phase6_validation', 'archive/development/phase6/phase6_phase3_ultimate_validation.py', 180),
    ('phase7_optimization', 'archive/development/phase7/phase7_aggressive_optimization.py', 180),
    ('phase8_final_train', 'archive/development/phase8/subphase8_2_dynamic_endpoints/train_final_giman_survival.py', 240),
    ('phase9_neuro_fuzzy', 'archive/development/phase9/phase9_full_training.py', 240),
    ('phase9_multitask', 'archive/development/phase9/phase9_multitask_learning.py', 180),
    ('global_validate_complete_dataset', 'scripts/validate_complete_dataset.py', 180),
    ('global_validate_production_model', 'scripts/validate_production_model.py', 180),
]


def run_case(name: str, rel_script: str, timeout: int) -> dict:
    cmd = [str(PYTHON), str(BASE / rel_script)]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=BASE,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start
        status = 'PASS' if proc.returncode == 0 else 'FAIL'
        tail = '\n'.join((proc.stdout + '\n' + proc.stderr).strip().splitlines()[-12:])
        return {
            'name': name,
            'script': rel_script,
            'status': status,
            'exit_code': proc.returncode,
            'duration_sec': round(duration, 2),
            'tail': tail,
        }
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        tail = ''
        if e.stdout:
            tail += e.stdout
        if e.stderr:
            tail += ('\n' + e.stderr)
        tail = '\n'.join(tail.strip().splitlines()[-12:])
        return {
            'name': name,
            'script': rel_script,
            'status': 'TIMEOUT',
            'exit_code': None,
            'duration_sec': round(duration, 2),
            'tail': tail,
        }
    except Exception as e:
        duration = time.time() - start
        return {
            'name': name,
            'script': rel_script,
            'status': 'ERROR',
            'exit_code': None,
            'duration_sec': round(duration, 2),
            'tail': str(e),
        }


def main() -> None:
    results = [run_case(*case) for case in CASES]

    with JSON_OUT.open('w', encoding='utf-8') as f:
        json.dump({'date': TODAY, 'results': results}, f, indent=2)

    pass_n = sum(r['status'] == 'PASS' for r in results)
    fail_n = sum(r['status'] == 'FAIL' for r in results)
    timeout_n = sum(r['status'] == 'TIMEOUT' for r in results)
    error_n = sum(r['status'] == 'ERROR' for r in results)

    lines = [
        f'# GIMAN Workflow Execution Matrix ({TODAY})',
        '',
        f'- PASS: {pass_n}',
        f'- FAIL: {fail_n}',
        f'- TIMEOUT: {timeout_n}',
        f'- ERROR: {error_n}',
        '',
        '## Matrix',
        '| Check | Script | Status | Runtime (s) | Exit |',
        '|---|---|---:|---:|---:|',
    ]

    for r in results:
        exit_code = '' if r['exit_code'] is None else str(r['exit_code'])
        lines.append(
            f"| {r['name']} | `{r['script']}` | {r['status']} | {r['duration_sec']} | {exit_code} |"
        )

    lines.append('')
    lines.append('## Failure/Timeout Tails')
    for r in results:
        if r['status'] != 'PASS':
            lines.append(f"### {r['name']} ({r['status']})")
            lines.append('```text')
            lines.append(r['tail'] or '(no captured output)')
            lines.append('```')

    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {REPORT}')
    print(f'Wrote {JSON_OUT}')


if __name__ == '__main__':
    sys.exit(main())
