#!/usr/bin/env bash
# Launcher for AutoGluon sidecar 21-feat runs. Called one target at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
TARGET="${1:?target required}"
exec .venv-autogluon/bin/python scripts/paper1/run_autogluon_sidecar_21feat.py \
    --target "$TARGET"
