#!/bin/bash
set -u
cd /Users/blair.dupre/Projects/CSCI-FALL-2025
LOG=outputs/paper1_phase_a_batch/logs/master.log
echo "=== Phase A batch launch $(date -u '+%Y-%m-%dT%H:%M:%S') ===" >>"$LOG"

# Group 1 (cheap, independent) — launch first, finish in ~15-30 min
nohup .venv/bin/python scripts/paper1/run_calibration_analysis.py \
  >outputs/paper1_phase_a_batch/logs/calibration.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched calibration (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_external_conformal.py --target binary \
  >outputs/paper1_phase_a_batch/logs/extcp_binary.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched extcp_binary (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_external_conformal.py --target 3class \
  >outputs/paper1_phase_a_batch/logs/extcp_3class.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched extcp_3class (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_external_conformal.py --target nsd_positive \
  >outputs/paper1_phase_a_batch/logs/extcp_nsd.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched extcp_nsd (pid $!)" >>"$LOG"

# Group 2 (medium, independent) — ordinal benchmarks
nohup .venv/bin/python scripts/paper1/run_ordinal_benchmarks.py --method coral --target full_ordinal \
  >outputs/paper1_phase_a_batch/logs/coral_ordinal.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched coral_ordinal (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_ordinal_benchmarks.py --method corn --target full_ordinal \
  >outputs/paper1_phase_a_batch/logs/corn_ordinal.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched corn_ordinal (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_ordinal_benchmarks.py --method ord_catboost --target full_ordinal \
  >outputs/paper1_phase_a_batch/logs/ord_catboost.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched ord_catboost (pid $!)" >>"$LOG"

# Group 3 (ordinal conformal + medication) — 3 arms medication
nohup .venv/bin/python scripts/paper1/run_ordinal_conformal.py --target full_ordinal \
  >outputs/paper1_phase_a_batch/logs/ordcp_ordinal.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched ordcp_ordinal (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_medication_sensitivity.py --arm 1 --target binary \
  >outputs/paper1_phase_a_batch/logs/med_arm1_binary.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched med_arm1_binary (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_medication_sensitivity.py --arm 2 --target binary \
  >outputs/paper1_phase_a_batch/logs/med_arm2_binary.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched med_arm2_binary (pid $!)" >>"$LOG"

nohup .venv/bin/python scripts/paper1/run_medication_sensitivity.py --arm 3 --target binary \
  >outputs/paper1_phase_a_batch/logs/med_arm3_binary.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched med_arm3_binary (pid $!)" >>"$LOG"

# Group 4 (SHAP + subgroup) - single long-running
nohup .venv/bin/python scripts/paper1/run_shap_subgroup.py --part both --target binary \
  >outputs/paper1_phase_a_batch/logs/shap_binary.log 2>&1 &
echo "$(date -u '+%H:%M:%S') launched shap_binary (pid $!)" >>"$LOG"

echo "$(date -u '+%H:%M:%S') all 12 workers dispatched; orphan parent PID $$" >>"$LOG"
wait
echo "$(date -u '+%Y-%m-%dT%H:%M:%S') Phase A batch COMPLETE" >>"$LOG"
touch outputs/paper1_phase_a_batch/logs/DONE
