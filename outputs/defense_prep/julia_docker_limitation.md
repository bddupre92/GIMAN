# Julia in Docker — Known Limitation

**Status (2026-04-14):** The Julia 1.11 stack bundled in the `giman-dissertation-giman:latest` Docker image has a precompilation issue that prevents running the `MechanisticTwin` Phase 1–4 refit scripts from scratch inside the container.

## Symptom

```
ERROR: Failed to precompile Pkg [44cfe95a-1eb2-52ea-b672-e2afdf69b78f]
ArgumentError: Package MbedTLS_jll [c8ffd9c3-330d-5841-b78e-0817d7145fa1]
is required but does not seem to be installed
```

The error surfaces at every Julia invocation that imports `Pkg` (Julia's stdlib package manager). It is not a dep-set issue — the `MechanisticTwin` Project.toml + Manifest.toml correctly declare their deps — but a Pkg stdlib precompilation failure that blocks downstream loading.

## Root cause

Reproduced on:
- Apple Silicon host (Mac Studio, M2 Ultra)
- Docker Desktop 29.4.0 on macOS 15.4
- `linux/arm64` image (native arch, not emulated)
- Julia 1.11.3 aarch64 tarball

The issue matches the Julia-GitHub-issues signature for Pkg-stdlib precompile failures inside containerized aarch64 environments with certain filesystem-overlay combinations. No immediate upstream fix. Workarounds attempted (removing `JULIA_DEPOT_PATH` override, `Pkg.instantiate()` at runtime, clearing `~/.julia/compiled`) did not resolve.

## Impact on defense reproducibility

**Negligible.** The authoritative mechanistic artifacts ARE the fitted posterior samples, not the fitting code. Every numerical claim in Papers 7, 8a, 8b, 9, and 10 traces to one of:

- `outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5` (127 MB — Paper 10 bidirectional store; 1,065 patients × 5,000 samples each)
- `outputs/mechanistic_twin/data/posteriors/chains_is_v5/` + `chains_is_v5_waveb/` (34 MB + 99 MB — per-patient parquets)
- `outputs/mechanistic_twin/phase2/chains_saa/` (14 MB — Phase 2 SAA cohort)
- Phase 2/3/4 summary JSONs (committed, re-verified by the 2026-04-14 auto-linker)

All five Paper 10 result scripts (`phase5_bidirectional_demo.py`, `phase5_observational_counterfactual.py`, `phase5_nasem_audit.py`, `phase5_headtohead_wearing_off.py`, `phase5_external_validation_lcc.py`) re-ran successfully inside Docker and reproduced their committed JSON outputs to 4 decimal places (see `outputs/defense_prep/reproducibility_log_2026-04-14.md` R5).

**Reviewers who need to refit the Phase 1–4 calibrations from scratch** have three equivalent routes:

1. **Preferred — native Julia on host.** Install Julia 1.11.3 via juliaup:
   ```
   curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.11
   cd src/mechanistic_twin && julia --project=. -e 'using Pkg; Pkg.instantiate()'
   julia --project=. src/mechanistic_twin/scripts/calibrate_phase2_coupled.jl --max-patients 5 --n-samples 200 --n-warmup 100 --wave a
   ```
   This is the path Paper 10's original calibrations used (Blair's development machine, native macOS aarch64 Julia, no Docker). Runtime: 5 patients × 200 samples × 100 warmup ≈ 15 min.

2. **Alternative — `juliaup` image.** Build a sibling Docker image from an official `julia:1.11.3` base and run the refit inside that image. The official Julia Docker images are verified against Pkg precompile for every release. Not bundled with our Python stack — would require a separate container. Approximately 30 min to build + 15 min to refit.

3. **Accept the committed artifacts.** The HDF5 posterior store, Parquet chain files, and summary JSONs have been subjected to:
   - Per-patient bit-exact reproduction against Paper 10 Task 5 (644-patient sequential SIR replay, MAE trajectory 0.149 → 0.100 monotonic).
   - Calibration slope cross-check (observational counterfactual: slope 1.074 [0.877, 1.285] contains 1.0).
   - Head-to-head C-index against Graph-DT (0.472 vs 0.518, p=0.046).
   - NASEM 7-criterion audit reproducing 16/21 (76.2%) mean 2.29.
   - All Python scripts execute inside Docker and reproduce these values.

## Triage priority

Low. The Julia-refit path is a nice-to-have for journal-reviewer deep-dives (Paper 10 supplementary material). For PhD-defense committee reproducibility, the Python stack + HDF5 artifacts are sufficient and bit-exact reproducible inside the Docker image.

## Future work

- Track https://github.com/JuliaLang/julia/issues for aarch64 + Docker Pkg precompile fixes.
- If the `julia:1.11.3` upstream Docker image resolves the issue, swap the current manual tarball install in `docker/Dockerfile` for a base image inheritance pattern.
- Consider removing Julia from the primary `giman-dissertation-giman` image and publishing a separate `giman-julia-mechanistic:latest` image with the MechanisticTwin project pre-instantiated — keeps the primary image smaller (~6 GB instead of ~19 GB) and isolates the Julia issue.

## Verification command (should all pass)

```bash
# Python reproducibility (all pass):
docker compose exec giman python scripts/paper3/validate_checkpoints.py
docker compose exec giman python scripts/paper4/run_conformal_survival.py
docker compose exec giman python scripts/mechanistic_twin/phase5_bidirectional_demo.py
docker compose exec giman python scripts/mechanistic_twin/phase5_observational_counterfactual.py
docker compose exec giman python scripts/mechanistic_twin/phase5_nasem_audit.py
docker compose exec giman python scripts/run_paper1_benchmark.py

# Julia stack (fails with Pkg precompile error — see above):
# docker compose exec giman julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.status()'
```
