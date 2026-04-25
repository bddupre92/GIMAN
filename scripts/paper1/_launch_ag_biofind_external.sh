#!/usr/bin/env bash
# Launcher for AutoGluon BioFIND external validation (R4-Q4).
# Called one target at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
TARGET="${1:?target required (binary|three_class|nsd_positive)}"
exec .venv-autogluon/bin/python scripts/paper1/run_biofind_sota_external.py \
    --method autogluon --target "$TARGET"
