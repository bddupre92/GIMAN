#!/bin/bash
# Paper 12 W6 Step 4 — FastSurfer batch on Lightning AI Studio
#
# Two phases so the Lightning driver can run INSTALL on free CPU, then switch
# to A100 for PROCESS. Idempotent: re-running skips completed steps.
#
# Usage (from inside Lightning Studio):
#   bash lightning_fastsurfer_setup.sh install   # CPU-OK, ~3-5 min
#   bash lightning_fastsurfer_setup.sh process   # needs GPU, ~2.5-7 hrs
#
# TIMING: 1× A100 ≈ 5-7 hrs for 401 scans · 2× A100 ≈ 2.5-3.5 hrs
# COST:   ~$7-10 total at Lightning A100 rates

set -euo pipefail

PHASE="${1:-process}"
WORK="${WORK:-/teamspace/studios/this_studio}"
cd "$WORK"

install_freesurfer() {
  if [ -d "$WORK/freesurfer" ]; then
    echo "[install] FreeSurfer already present — skip"
    return 0
  fi
  echo "[install] Downloading FreeSurfer 7.4.1 (~2 GB)..."
  wget -q -O fs.tar.gz \
    "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
  tar -xzf fs.tar.gz
  rm fs.tar.gz
}

install_fastsurfer() {
  if [ -d "$WORK/FastSurfer" ]; then
    echo "[install] FastSurfer already present — skip git clone"
  else
    echo "[install] Cloning FastSurfer..."
    git clone --depth 1 --branch stable https://github.com/Deep-MI/FastSurfer.git
  fi
  echo "[install] Installing Python deps..."
  cd FastSurfer
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  pip install -q "torch>=2.0" "torchvision>=0.15"
  cd "$WORK"
}

verify_license() {
  if [ ! -f "$WORK/license.txt" ]; then
    echo "ERROR: license.txt not found at $WORK/license.txt"
    echo "Register free at https://surfer.nmr.mgh.harvard.edu/registration.html"
    exit 1
  fi
  mkdir -p "$WORK/freesurfer"
  cp "$WORK/license.txt" "$WORK/freesurfer/.license"
}

extract_niftis() {
  if [ -d "$WORK/nifti" ] && [ "$(find "$WORK/nifti" -name "*.nii.gz" | wc -l)" -gt 0 ]; then
    local n=$(find "$WORK/nifti" -name "*.nii.gz" | wc -l)
    echo "[process] nifti/ exists with $n .nii.gz files — skip extract"
    return 0
  fi
  if [ ! -f "$WORK/nifti_wang_n161.tar.gz" ]; then
    echo "ERROR: nifti_wang_n161.tar.gz not found"
    exit 1
  fi
  echo "[process] Extracting nifti tarball..."
  mkdir -p nifti
  tar -xzf nifti_wang_n161.tar.gz -C nifti/
  # Tarball structure: nifti/nifti_wang_n161/PATNO*.nii.gz — flatten
  mv nifti/nifti_wang_n161/* nifti/ 2>/dev/null || true
  rmdir nifti/nifti_wang_n161 2>/dev/null || true
}

run_fastsurfer_batch() {
  export FREESURFER_HOME="$WORK/freesurfer"
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh" > /dev/null 2>&1
  export FASTSURFER_HOME="$WORK/FastSurfer"

  mkdir -p "$WORK/fastsurfer_out"
  export SUBJECTS_DIR="$WORK/fastsurfer_out"

  local n_gpus
  n_gpus=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
  if [ "$n_gpus" -eq 0 ]; then
    echo "ERROR: no GPU detected. Switch Studio to A100 before running 'process'."
    exit 1
  fi
  echo "[process] Detected $n_gpus GPU(s)"

  local nifti_list
  nifti_list=$(find "$WORK/nifti" -name "*.nii.gz" | sort)
  local n_total
  n_total=$(echo "$nifti_list" | wc -l)
  echo "[process] Processing $n_total scans..."

  run_one() {
    local nifti=$1
    local gpu=$2
    local subj_id
    subj_id=$(basename "$nifti" .nii.gz)
    if [ -f "$SUBJECTS_DIR/$subj_id/stats/aseg.stats" ]; then
      echo "[SKIP] $subj_id (stats already present)"
      return 0
    fi
    mkdir -p "$SUBJECTS_DIR/$subj_id"
    if CUDA_VISIBLE_DEVICES=$gpu "$FASTSURFER_HOME/run_fastsurfer.sh" \
        --t1 "$nifti" --sid "$subj_id" --sd "$SUBJECTS_DIR" \
        --parallel --threads 8 \
        --fs_license "$FREESURFER_HOME/.license" \
        > "$SUBJECTS_DIR/${subj_id}.log" 2>&1; then
      echo "[OK] $subj_id"
    else
      echo "[FAIL] $subj_id — see $SUBJECTS_DIR/${subj_id}.log"
    fi
  }
  export -f run_one
  export FASTSURFER_HOME SUBJECTS_DIR FREESURFER_HOME

  if [ "$n_gpus" -eq 2 ]; then
    echo "$nifti_list" | awk 'NR%2==1' | xargs -I{} -P1 bash -c 'run_one "$@" 0' _ {} &
    echo "$nifti_list" | awk 'NR%2==0' | xargs -I{} -P1 bash -c 'run_one "$@" 1' _ {} &
    wait
  else
    echo "$nifti_list" | xargs -I{} -P1 bash -c 'run_one "$@" 0' _ {}
  fi
}

package_results() {
  echo "[process] Packaging /stats outputs..."
  cd "$SUBJECTS_DIR"
  # Include /stats subdir + aparc+aseg.mgz for each subject
  find . \( -path "*/stats/*" -o -name "aparc+aseg.mgz" \) -type f \
    | tar -czf "$WORK/fastsurfer_stats.tar.gz" -T -
  local size
  size=$(du -h "$WORK/fastsurfer_stats.tar.gz" | cut -f1)
  echo "[DONE] Output: $WORK/fastsurfer_stats.tar.gz ($size)"
}

case "$PHASE" in
  install)
    install_freesurfer
    install_fastsurfer
    echo "[install] DONE. Switch to A100, then run: bash lightning_fastsurfer_setup.sh process"
    ;;
  process)
    verify_license
    extract_niftis
    run_fastsurfer_batch
    package_results
    ;;
  *)
    echo "Usage: $0 {install|process}"
    exit 1
    ;;
esac
