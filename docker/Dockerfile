# syntax=docker/dockerfile:1.6
#
# GIMAN dissertation reproducibility image.
# Base: Python 3.10 slim (Debian bookworm) + Julia 1.11 for mechanistic_twin refits.
#
# Multi-stage build:
#   1) julia-base  — downloads Julia 1.11.3 tarball (cached separately from Python deps)
#   2) python-deps — installs Poetry-locked Python deps (cached unless pyproject/poetry.lock change)
#   3) runtime     — copies both layers + project source, instantiates Julia project

###############################################################################
# Stage 1: Julia
###############################################################################
FROM debian:bookworm-slim AS julia-base

ARG JULIA_VERSION=1.11.3
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Julia provides prebuilt tarballs for x86_64 and aarch64. Map Docker's TARGETARCH.
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64)  julia_arch=x64; julia_arch_dir=x64 ;; \
        arm64)  julia_arch=aarch64; julia_arch_dir=aarch64 ;; \
        *) echo "Unsupported arch: ${TARGETARCH}" && exit 1 ;; \
    esac; \
    curl -fsSL "https://julialang-s3.julialang.org/bin/linux/${julia_arch_dir}/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-${julia_arch}.tar.gz" \
        -o /tmp/julia.tar.gz; \
    mkdir -p /opt/julia; \
    tar -xzf /tmp/julia.tar.gz -C /opt/julia --strip-components=1; \
    rm /tmp/julia.tar.gz; \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia

###############################################################################
# Stage 2: Python deps
###############################################################################
FROM python:3.10-slim-bookworm AS python-deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/opt/poetry/bin:${PATH}"

WORKDIR /giman
COPY pyproject.toml poetry.lock ./
COPY src/giman_pipeline/__init__.py src/giman_pipeline/__init__.py
COPY README.md ./

# Install main deps only (skip dev). --no-root skips installing the giman-pipeline
# package itself; that happens after src/ is copied in the runtime stage.
RUN poetry install --only main --no-root

###############################################################################
# Stage 3: Runtime
###############################################################################
FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    JULIA_DEPOT_PATH=/opt/julia-depot \
    PATH="/opt/poetry/bin:/opt/julia/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        postgresql-client \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Poetry installer output + site-packages from the python-deps stage
COPY --from=python-deps /opt/poetry /opt/poetry
COPY --from=python-deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy Julia runtime
COPY --from=julia-base /opt/julia /opt/julia
RUN ln -sf /opt/julia/bin/julia /usr/local/bin/julia

WORKDIR /giman

# Copy the project. Because of docker/.dockerignore, data/, outputs/, db_dump/,
# and .venv/ are excluded; everything else (src/, scripts/, tests/, pyproject, etc.)
# is copied.
COPY . .

# Install the giman-pipeline package itself (src/giman_pipeline) in editable mode
# so scripts importing `giman_pipeline` work without PYTHONPATH hacks.
RUN poetry install --only main

# Precompile the Julia mechanistic_twin project so reviewers don't pay the
# first-call JIT cost. Heavy: ~10-15 min on amd64, ~8-12 min on arm64.
RUN mkdir -p /opt/julia-depot \
    && julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
    || echo "WARNING: Julia precompile encountered issues; will lazy-compile on first run"

# Entrypoint waits for Postgres, verifies the DB restored from db_dump, then
# forwards to the CMD (Jupyter by default).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
