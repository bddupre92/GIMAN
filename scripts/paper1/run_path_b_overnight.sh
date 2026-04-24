#!/usr/bin/env bash
# Paper 1 R2 Path B overnight serial orchestrator.
#
# Purpose: close the full-rigor requirements for IEEE JBHI R2 submission by
# executing every 21-feat primary-spec compute workstream that crashed under
# parallel launch earlier today:
#
#   - Nested 5x3 CV HPO on CatBoost + LightGBM x 4 targets (8 combos, serial)
#   - AutoGluon sidecar full 5-fold runs x 4 targets (4 re-runs, rebuild results)
#   - TOST equivalence test on all 4 tabular-SOTA methods using per-fold AUCs
#   - SQL load of all produced artifacts into features.paper1_r2_sensitivity
#
# Execution model: strictly serial. One subprocess at a time. Survives Mac
# sleep via caffeinate. Logs everything to outputs/paper1_path_b/logs/.
# Each combo commits its outputs to disk immediately; if the machine crashes
# mid-combo, we lose <1 hour of work.
#
# Invocation:
#   caffeinate -dimsu nohup bash scripts/paper1/run_path_b_overnight.sh \
#     > /tmp/path_b_stdout.log 2>&1 &
#
# Status monitoring:
#   tail -f outputs/paper1_path_b/logs/status.log
#   tail -f outputs/paper1_path_b/logs/<combo>.log

set -u  # undefined variable = error; do NOT set -e (we want to continue past failures)

cd "$(dirname "$0")/../.."  # repo root
REPO_ROOT="$PWD"

VENV="$REPO_ROOT/.venv/bin/python"
AG_VENV="$REPO_ROOT/.venv-autogluon/bin/python"
LOG_DIR="$REPO_ROOT/outputs/paper1_path_b/logs"
STATUS_LOG="$LOG_DIR/status.log"
mkdir -p "$LOG_DIR"

log_status() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$STATUS_LOG"
}

log_status "===== PATH B OVERNIGHT START ====="
log_status "Repo:  $REPO_ROOT"
log_status "venv:  $VENV"
log_status "AG venv: $AG_VENV"
log_status "PID: $$"
log_status ""

# -----------------------------------------------------------------------------
# STAGE 1: Nested HPO, 8 combos serial
# -----------------------------------------------------------------------------
log_status "===== STAGE 1: Nested 5x3 CV HPO (8 serial combos) ====="

TARGETS=(binary 3class full_ordinal nsd_positive)
MODELS=(catboost lightgbm)

for target in "${TARGETS[@]}"; do
    for model in "${MODELS[@]}"; do
        combo="${model}_${target}"
        log_combo="$LOG_DIR/hpo_${combo}.log"
        log_status "START  hpo/$combo  (log: $log_combo)"

        "$VENV" scripts/paper1/run_nested_cv_hpo_21feat.py \
            --model "$model" \
            --target "$target" \
            > "$log_combo" 2>&1
        rc=$?

        if [[ $rc -eq 0 ]]; then
            log_status "DONE   hpo/$combo  (exit 0)"
        else
            log_status "FAIL   hpo/$combo  (exit $rc)  see $log_combo"
        fi
    done
done

# -----------------------------------------------------------------------------
# STAGE 2: AutoGluon sidecar, 4 targets serial
# -----------------------------------------------------------------------------
log_status ""
log_status "===== STAGE 2: AutoGluon sidecar (4 serial targets) ====="

for target in "${TARGETS[@]}"; do
    log_combo="$LOG_DIR/ag_${target}.log"
    log_status "START  ag/$target  (log: $log_combo)"

    "$AG_VENV" scripts/paper1/run_autogluon_sidecar_21feat.py \
        --target "$target" \
        > "$log_combo" 2>&1
    rc=$?

    if [[ $rc -eq 0 ]]; then
        log_status "DONE   ag/$target  (exit 0)"
    else
        log_status "FAIL   ag/$target  (exit $rc)  see $log_combo"
    fi
done

# -----------------------------------------------------------------------------
# STAGE 3: TOST equivalence test on all available methods
# -----------------------------------------------------------------------------
log_status ""
log_status "===== STAGE 3: TOST equivalence test (paired bootstrap) ====="
log_status "START  TOST"

"$VENV" scripts/paper1/run_paired_bootstrap_tost.py \
    > "$LOG_DIR/tost.log" 2>&1
rc=$?

if [[ $rc -eq 0 ]]; then
    log_status "DONE   TOST  (exit 0)"
else
    log_status "FAIL   TOST  (exit $rc)"
fi

# -----------------------------------------------------------------------------
# STAGE 4: SQL load of HPO + SOTA results into features.paper1_r2_sensitivity
# -----------------------------------------------------------------------------
log_status ""
log_status "===== STAGE 4: SQL load ====="
log_status "START  SQL load (if loader script exists)"

if [[ -f scripts/paper1/consolidate_21feat_hpo.py ]]; then
    "$VENV" scripts/paper1/consolidate_21feat_hpo.py > "$LOG_DIR/sql_load.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]]; then
        log_status "DONE   SQL load (consolidate_21feat_hpo)"
    else
        log_status "FAIL   SQL load (exit $rc)  see $LOG_DIR/sql_load.log"
    fi
else
    log_status "SKIP   consolidate_21feat_hpo.py not found; post-process manually tomorrow"
fi

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
log_status ""
log_status "===== PATH B OVERNIGHT COMPLETE ====="
log_status "Summary:"
log_status "  HPO results:       outputs/paper1_hpo_21feat/results/nested_{model}_{target}/"
log_status "  AutoGluon results: outputs/paper1_tabular_sota_21feat/results/ag_sidecar_{target}_fold*/"
log_status "  TOST output:       outputs/paper1_r2_responses/q_r2_w1_paired_bootstrap_tost.json"
log_status "  SQL status:        see $LOG_DIR/sql_load.log"
log_status "  This orchestrator: $STATUS_LOG"
log_status ""
log_status "Next: morning resume via Docs/NEXT_STEPS_2026-04-24.md"
