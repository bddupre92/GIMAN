#!/usr/bin/env bash
# Entrypoint for the giman service: wait for Postgres, verify DB restored,
# then exec the CMD (Jupyter by default).

set -euo pipefail

PG_HOST="${PGHOST:-postgres}"
PG_USER="${PGUSER:-giman}"
PG_DATABASE="${PGDATABASE:-giman_research}"

echo "[entrypoint] Waiting for PostgreSQL at ${PG_HOST}:5432..."
until PGPASSWORD="${PGPASSWORD}" psql -h "${PG_HOST}" -U "${PG_USER}" -d "${PG_DATABASE}" -c 'SELECT 1' >/dev/null 2>&1; do
    sleep 1
done
echo "[entrypoint] PostgreSQL is ready."

echo "[entrypoint] Verifying giman_research DB restored from db_dump/schema_and_data.sql..."
TABLE_COUNT=$(PGPASSWORD="${PGPASSWORD}" psql -h "${PG_HOST}" -U "${PG_USER}" -d "${PG_DATABASE}" -tAc \
    "SELECT count(*) FROM information_schema.tables WHERE table_schema IN ('ppmi_raw','biofind_raw','pdbp_raw','hbs_raw','staging','features','longitudinal','paper3','ledd','mechanistic')")

if [ "${TABLE_COUNT}" -lt 140 ]; then
    echo "[entrypoint] WARNING: expected 146 tables across 10 schemas, found ${TABLE_COUNT}."
    echo "[entrypoint] The Postgres volume may be stale. Run 'docker compose down -v' and rebuild"
    echo "[entrypoint] to force a fresh restore from db_dump/schema_and_data.sql."
else
    echo "[entrypoint] DB OK (${TABLE_COUNT} tables)."
fi

echo "[entrypoint] Spot-check: features.paper1_features_with_targets row count..."
ROW_COUNT=$(PGPASSWORD="${PGPASSWORD}" psql -h "${PG_HOST}" -U "${PG_USER}" -d "${PG_DATABASE}" -tAc \
    "SELECT count(*) FROM features.paper1_features_with_targets" 2>/dev/null || echo "0")
echo "[entrypoint] paper1_features_with_targets: ${ROW_COUNT} rows (expected 2,201)."

echo "[entrypoint] Starting: $*"
exec "$@"
