#!/bin/bash
# P2-Cal.4 + P2-Cal.5 automation: once P2-Cal.3a (A retrain) is done,
# this script measures A variant raw coverage, regenerates the comparison
# figure, and prints the decision table.
#
# Usage:
#   bash scripts/paper2/run_calibration_comparison.sh

set -e

cd "$(dirname "$0")/../.."

PYTHON=.venv/bin/python3
A_CKPT_DIR=outputs/paper2_benchmark/runs/cal_retune_lambda0.1_warmup0/checkpoints
OUT_DIR=outputs/paper2_benchmark/calibration_retune

# Ensure A retrain produced all 48 checkpoints
N_A=$(ls "$A_CKPT_DIR" 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_A" -ne 48 ]; then
    echo "ERROR: expected 48 A checkpoints, found $N_A at $A_CKPT_DIR"
    echo "Ensure run_paper2_experiments.py --lambda-cal 0.1 --cal-warmup-epochs 0 --num-runs 3 --skip-baselines --run-name cal_retune_lambda0.1_warmup0 completed."
    exit 1
fi

echo "======================================================================"
echo "P2-Cal.4: Measure A variant raw coverage"
echo "======================================================================"
$PYTHON scripts/paper2/measure_raw_coverage.py \
    --checkpoint-dir "$A_CKPT_DIR" \
    --output-name A_variant_raw_coverage.json \
    --summary-name A_variant_raw_coverage_summary.json

echo ""
echo "======================================================================"
echo "P2-Cal.5: Generate calibration comparison figure + decision table"
echo "======================================================================"
$PYTHON scripts/paper2/generate_calibration_comparison.py

echo ""
echo "======================================================================"
echo "DONE"
echo "======================================================================"
echo "  Decision table: $OUT_DIR/calibration_comparison_table.csv"
echo "  Summary JSON:   $OUT_DIR/calibration_comparison_summary.json"
echo "  Figure:         $OUT_DIR/figures/fig_calibration_compare.{pdf,png}"
