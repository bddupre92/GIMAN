# GIMAN dissertation Docker stack

One-command rebuild of the entire `giman_research` environment: PostgreSQL 17 with all 146 tables restored from `db_dump/schema_and_data.sql`, Python 3.10 with PyTorch 2.8, PyTorch Geometric 2.6, MAPIE 1.3, CatBoost, and Julia 1.11 with the full mechanistic_twin project precompiled.

## Prerequisites

- [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/) ≥ 4.30 (macOS, Windows, Linux) or equivalent (OrbStack, Colima, Podman with compose v2).
- Disk: ~25 GB free (image ~6 GB, pg volume ~1 GB, data mount read-only ~18 GB host-side).
- RAM: 8 GB minimum, 16 GB recommended.
- `db_dump/schema_and_data.sql` present on host (190 MB, gitignored). Regenerate with `pg_dump giman_research > db_dump/schema_and_data.sql` if missing.

## One-command rebuild

From the repository root:

```bash
cd docker
docker compose up --build
```

On the first run:

1. Docker builds the `giman` image (~15 min — most of it is Julia precompile).
2. Postgres restores `db_dump/schema_and_data.sql` on first container start (~3 min).
3. Jupyter Lab starts on http://localhost:8888 (token disabled for local use — do NOT expose this port publicly).

Subsequent `up` calls reuse the cached image and persistent `pgdata` volume (restart is ~20 s).

## Verifying the rebuild

In a second terminal:

```bash
docker compose exec giman python -c \
    "from giman_pipeline.data.db import read_sql; print(read_sql('SELECT count(*) AS n FROM staging.nsd_iss_staging_results'))"
```

Expected output:

```
        n
0  2201
```

Julia precompile check:

```bash
docker compose exec giman julia --project=src/mechanistic_twin -e 'using MechanisticTwin; println("OK")'
```

Expected: `OK` (after ~1 s if precompile was successful at build time, ~2 min otherwise).

## Reproducing a specific paper

All paper-running scripts are in `scripts/` and work unchanged inside the container (they read from the mounted `giman_research` DB). Examples:

```bash
# Paper 1 benchmark (7 models × 4 targets, ~10 min)
docker compose exec giman python scripts/run_paper1_benchmark.py

# Paper 3 Graph-DT reproduction from checkpoints (~30 sec; no training)
docker compose exec giman python scripts/paper3/validate_checkpoints.py

# Paper 4 conformal analysis (~5 min)
docker compose exec giman python scripts/paper4/run_conformal_survival.py

# Paper 10 Task 5 bidirectional demo (~3 min)
docker compose exec giman python scripts/mechanistic_twin/phase5_bidirectional_demo.py

# Phase 2 Julia refit (OPTIONAL — requires Julia; heavy, several hours)
docker compose exec giman julia --project=src/mechanistic_twin \
    src/mechanistic_twin/scripts/is_v5_wavea.jl
```

Results land in `outputs/` on the host (write-through mount).

## Teardown

```bash
# Stop containers, keep pg volume + data
docker compose down

# Stop + wipe pg volume (forces a fresh DB restore on next up)
docker compose down -v

# Stop + remove images (forces a full rebuild on next up)
docker compose down --rmi all
```

## Updating the image

Triggers and procedures:

| Change | Procedure |
|---|---|
| Python dep added (`pyproject.toml` / `poetry.lock`) | `docker compose build giman` (reuses Julia layer cache) |
| Julia dep added (`src/mechanistic_twin/Project.toml`) | `docker compose build --no-cache giman` (~15 min) |
| DB schema or data refresh | `pg_dump giman_research > db_dump/schema_and_data.sql`, then `docker compose down -v && docker compose up` |
| Python minor version bump (3.10 → 3.11) | Edit `docker/Dockerfile` `FROM` line, `docker compose build --no-cache giman` |
| PostgreSQL version bump | Edit `docker/docker-compose.yml` postgres `image:` tag, drop volume, rebuild |

## Troubleshooting

**`db_dump/schema_and_data.sql: no such file or directory`**
→ You are missing the DB dump. Generate it with `pg_dump giman_research > db_dump/schema_and_data.sql` on the machine that has the live DB, then rerun.

**Postgres container keeps restarting**
→ Usually the restore failed mid-way. `docker compose logs postgres` will show the SQL error. Common cause: a corrupted / partial dump. Regenerate and rebuild.

**Jupyter shows `ERR_CONNECTION_REFUSED` on localhost:8888**
→ The entrypoint waits for PG to be healthy before starting Jupyter. Check `docker compose logs giman` — if you see `[entrypoint] Waiting for PostgreSQL...` for >60 s, PG is not starting. Check `docker compose logs postgres`.

**Julia scripts fail with `Package not found`**
→ The build-time precompile was skipped. Run inside the container:
```bash
julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.instantiate()'
```

**Platform mismatch on Apple Silicon**
→ The Dockerfile is multi-arch via `TARGETARCH`. If `docker compose up` errors out with platform warnings, explicitly pass the platform:
```bash
DOCKER_DEFAULT_PLATFORM=linux/arm64 docker compose build
```

## What the image does NOT include

- **Connectome atlases** (`data/00_raw/connectome/`, 3 GB). Mounted read-only from the host when present; download separately from DSI Studio / Melbourne subcortex upstream if needed (see `outputs/dissertation/appendix_e/E1b_disk_artifact_manifest.md` §5).
- **GPU acceleration.** The base image is CPU-only. For CUDA inference, derive a child image with `FROM` set to an NVIDIA PyTorch base (post-defense work — not needed for defense reproduction because all Paper 3 / Paper 11 checkpoints are committed).
- **MPS (Apple GPU) passthrough.** Docker on macOS runs a Linux VM; MPS is not available inside the container. Use native Python on macOS if you need MPS acceleration; use the container for canonical reproduction.
