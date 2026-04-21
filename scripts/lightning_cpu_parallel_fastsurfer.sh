#!/bin/bash
# Paper 12 W6 — FULL FastSurfer pipeline (seg + surface recon), CPU parallel.
#
# Designed for 64+ CPU Lightning Studio. Each scan uses 4 threads, N parallel
# scans where N = n_cpu / 4. Full pipeline gives Wang 2025 features:
#   - stats/aseg+DKT.stats (68 subcortical volumes)
#   - stats/aparc+DKTatlas.ctab + per-hemi thickness (cortical thickness)
#
# Expected throughput:
#   64 cores: 16 parallel × 15 min × 26 batches = ~6.5 hrs, ~$8-13
#   96 cores: 24 parallel × 15 min × 17 batches = ~4 hrs, ~$16-20
#
# Usage:
#   bash lightning_cpu_parallel_fastsurfer.sh install
#   bash lightning_cpu_parallel_fastsurfer.sh process

PHASE="${1:-process}"
WORK="${WORK:-/teamspace/studios/this_studio}"
cd "$WORK"

install_phase() {
  echo "[install] Installing system deps..."
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq libsuitesparse-dev libopenblas-dev liblapack-dev \
      tcsh bc libgomp1 perl python3 2>&1 | tail -3
  fi

  if [ ! -d "$WORK/freesurfer" ]; then
    echo "[install] Downloading FreeSurfer 7.4.1 (~2 GB)..."
    wget -q -O fs.tar.gz \
      "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
    tar -xzf fs.tar.gz && rm fs.tar.gz
  fi

  if [ ! -d "$WORK/FastSurfer" ]; then
    echo "[install] Cloning FastSurfer..."
    git clone --depth 1 --branch stable https://github.com/Deep-MI/FastSurfer.git
  fi

  echo "[install] Installing PyTorch CPU + FastSurfer deps..."
  pip install -q --upgrade pip
  pip install -q "torch>=2.2" "torchvision>=0.17" --index-url https://download.pytorch.org/whl/cpu
  pip install -q \
    "nibabel>=4.0" "numpy<2.0" "scipy>=1.10" "scikit-image>=0.19" \
    "torchio>=0.19" "tqdm>=4.60" "pyyaml>=6.0" "yacs>=0.1.8" \
    "h5py>=3.8" "matplotlib>=3.6" "pandas>=2.0" "lapy>=1.0" \
    "simpleitk>=2.3" "requests>=2.30"

  if [ -f "$WORK/license.txt" ]; then
    cp "$WORK/license.txt" "$WORK/freesurfer/.license"
  fi

  echo "[install] DONE. Run: bash $0 process"
}

process_phase() {
  [ -f "$WORK/license.txt" ] || { echo "ERROR: license.txt missing"; exit 1; }
  mkdir -p "$WORK/freesurfer"
  cp "$WORK/license.txt" "$WORK/freesurfer/.license"

  export FREESURFER_HOME="$WORK/freesurfer"
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh" > /dev/null 2>&1
  export FASTSURFER_HOME="$WORK/FastSurfer"

  # Extract niftis if needed
  if [ ! -d "$WORK/nifti" ] || [ "$(find "$WORK/nifti" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "[process] Extracting nifti tarball..."
    mkdir -p nifti
    tar -xzf "$WORK/nifti_wang_n161.tar.gz" -C nifti/ 2>&1 | tail -3 || true
    if [ -d "nifti/nifti_wang_n161" ]; then
      mv nifti/nifti_wang_n161/*.nii.gz nifti/ 2>/dev/null
      rmdir nifti/nifti_wang_n161 2>/dev/null
    fi
  fi

  local n_niftis n_cpus n_parallel
  n_niftis=$(find "$WORK/nifti" -maxdepth 1 -name '*.nii.gz' | wc -l)
  n_cpus=$(nproc)
  n_parallel=$(( n_cpus / 4 ))
  [ "$n_parallel" -lt 1 ] && n_parallel=1
  echo "[process] $n_niftis scans · $n_cpus CPUs · $n_parallel parallel workers · 4 threads each"
  echo "[process] Expected wall-clock: $(( (n_niftis * 15 + n_parallel - 1) / n_parallel )) minutes for full pipeline"

  mkdir -p "$WORK/fastsurfer_out"
  export SUBJECTS_DIR="$WORK/fastsurfer_out"

  run_one() {
    local nifti=$1
    local subj_id
    subj_id=$(basename "$nifti" .nii.gz)
    if [ -f "$SUBJECTS_DIR/$subj_id/stats/aseg+DKT.stats" ] && \
       [ -f "$SUBJECTS_DIR/$subj_id/stats/lh.aparc.DKTatlas.mapped.stats" ]; then
      echo "[SKIP] $subj_id (full pipeline done)"; return 0
    fi
    local t0=$(date +%s)
    echo "[START] $subj_id ($(date -u +%H:%M:%S))"
    if "$FASTSURFER_HOME/run_fastsurfer.sh" \
        --t1 "$nifti" --sid "$subj_id" --sd "$SUBJECTS_DIR" \
        --device cpu --threads 4 --parallel \
        --fs_license "$FREESURFER_HOME/.license" \
        > "$SUBJECTS_DIR/${subj_id}.log" 2>&1; then
      local dt=$(($(date +%s) - t0))
      echo "[OK] $subj_id (${dt}s = $((dt/60))m)"
    else
      echo "[FAIL] $subj_id — see ${subj_id}.log"
    fi
  }
  export -f run_one
  export FASTSURFER_HOME SUBJECTS_DIR FREESURFER_HOME

  find "$WORK/nifti" -maxdepth 1 -name '*.nii.gz' | sort | \
    xargs -I{} -P"$n_parallel" bash -c 'run_one "$@"' _ {}

  echo "[process] Packaging /stats + aparc+aseg.mgz..."
  cd "$SUBJECTS_DIR"
  find . \( -path '*/stats/*' -o -name 'aparc+aseg.mgz' -o -name 'aparc.DKTatlas+aseg.deep.mgz' \) -type f \
    | tar -czf "$WORK/fastsurfer_full_stats.tar.gz" -T -
  echo "[DONE] $WORK/fastsurfer_full_stats.tar.gz ($(du -h "$WORK/fastsurfer_full_stats.tar.gz" | cut -f1))"
}

case "$PHASE" in
  install) install_phase ;;
  process) process_phase ;;
  *) echo "Usage: $0 {install|process}"; exit 1 ;;
esac
