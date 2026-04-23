#!/bin/bash
# Orphan process — runs independent of any parent shell
cd /Users/blair.dupre/Projects/CSCI-FALL-2025
mkdir -p outputs/paper1_hpo/logs
echo "$(date +%Y-%m-%dT%H:%M:%S) Starting tree HPO orphan (PID $$)" >> outputs/paper1_hpo/logs/master.log
for model in catboost lightgbm; do
  for target in binary 3class full_ordinal nsd_positive; do
    logfile="outputs/paper1_hpo/logs/hpo_${model}_${target}.log"
    .venv/bin/python scripts/paper1/run_nested_cv_hpo.py --model $model --target $target > $logfile 2>&1 &
    echo "$(date +%H:%M:%S) launched $model x $target (pid $!)" >> outputs/paper1_hpo/logs/master.log
  done
done
wait
echo "$(date +%Y-%m-%dT%H:%M:%S) All 8 tree HPO jobs COMPLETED" >> outputs/paper1_hpo/logs/master.log
# write a DONE sentinel
date +%Y-%m-%dT%H:%M:%S > outputs/paper1_hpo/logs/DONE
