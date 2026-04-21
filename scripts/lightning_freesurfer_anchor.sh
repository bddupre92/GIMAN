#!/bin/bash
# Paper 12 W6 Step 4b — FreeSurfer concordance anchor on Lightning AI CPU-16
#
# Runs vanilla FreeSurfer recon-all on 20 random Wang scans to establish
# in-cohort concordance with FastSurfer (Dice + thickness correlation).
#
# Target machine: Lightning Machine.CPU_X_16 (~$0.60/hr × ~25 hrs ≈ $15)
# Runs 4 scans in parallel with -openmp 4 to fit 16 vCPU budget.
# Each scan ~4-6 hrs serial; 20/4 = 5 batches × ~5 hrs = ~25 hrs wall-clock.
#
# Usage:
#   bash lightning_freesurfer_anchor.sh install   # CPU install, ~5 min
#   bash lightning_freesurfer_anchor.sh process   # ~25 hrs CPU-16 recon-all

set -euo pipefail

PHASE="${1:-process}"
WORK="${WORK:-/teamspace/studios/this_studio}"
cd "$WORK"

install_freesurfer() {
  if [ -d "$WORK/freesurfer" ]; then
    echo "[install] FreeSurfer already present"; return 0
  fi
  echo "[install] Downloading FreeSurfer 7.4.1..."
  wget -q -O fs.tar.gz \
    "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
  tar -xzf fs.tar.gz; rm fs.tar.gz
  sudo apt-get update -q && sudo apt-get install -y -q tcsh bc libgomp1 perl python3 2>&1 | tail -5
}

verify_license() {
  if [ ! -f "$WORK/license.txt" ]; then
    echo "ERROR: license.txt missing"; exit 1
  fi
  mkdir -p "$WORK/freesurfer"
  cp "$WORK/license.txt" "$WORK/freesurfer/.license"
}

extract_niftis() {
  if [ -d "$WORK/nifti_anchor" ] && [ "$(find "$WORK/nifti_anchor" -name "*.nii.gz" | wc -l)" -gt 0 ]; then
    echo "[process] nifti_anchor/ exists — skip extract"; return 0
  fi
  if [ ! -f "$WORK/nifti_freesurfer_anchor.tar.gz" ]; then
    echo "ERROR: nifti_freesurfer_anchor.tar.gz not found"; exit 1
  fi
  mkdir -p nifti_anchor
  tar -xzf nifti_freesurfer_anchor.tar.gz -C nifti_anchor/
  mv nifti_anchor/nifti_freesurfer_anchor/* nifti_anchor/ 2>/dev/null || true
  rmdir nifti_anchor/nifti_freesurfer_anchor 2>/dev/null || true
}

run_recon_all_batch() {
  export FREESURFER_HOME="$WORK/freesurfer"
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh" > /dev/null 2>&1
  mkdir -p "$WORK/freesurfer_anchor_out"
  export SUBJECTS_DIR="$WORK/freesurfer_anchor_out"

  local nifti_list
  nifti_list=$(find "$WORK/nifti_anchor" -name "*.nii.gz" | sort)
  local n_total
  n_total=$(echo "$nifti_list" | wc -l)
  echo "[process] Running recon-all on $n_total scans (4 parallel, -openmp 4)..."

  run_one() {
    local nifti=$1
    local subj_id
    subj_id=$(basename "$nifti" .nii.gz)
    if [ -f "$SUBJECTS_DIR/$subj_id/stats/aseg.stats" ]; then
      echo "[SKIP] $subj_id (stats already present)"; return 0
    fi
    echo "[START] $subj_id ($(date -u +%H:%M:%S))"
    if recon-all -i "$nifti" -s "$subj_id" -sd "$SUBJECTS_DIR" \
         -all -openmp 4 -parallel \
         > "$SUBJECTS_DIR/${subj_id}.log" 2>&1; then
      echo "[OK] $subj_id ($(date -u +%H:%M:%S))"
    else
      echo "[FAIL] $subj_id — see ${subj_id}.log"
    fi
  }
  export -f run_one
  export SUBJECTS_DIR FREESURFER_HOME

  echo "$nifti_list" | xargs -I{} -P4 bash -c 'run_one "$@"' _ {}
}

package_results() {
  echo "[process] Packaging /stats..."
  cd "$SUBJECTS_DIR"
  find . \( -path "*/stats/*" -o -name "aparc+aseg.mgz" \) -type f \
    | tar -czf "$WORK/freesurfer_anchor_stats.tar.gz" -T -
  echo "[DONE] $WORK/freesurfer_anchor_stats.tar.gz"
}

case "$PHASE" in
  install)
    install_freesurfer
    echo "[install] DONE. Now run: bash lightning_freesurfer_anchor.sh process"
    ;;
  process)
    verify_license
    extract_niftis
    run_recon_all_batch
    package_results
    ;;
  *)
    echo "Usage: $0 {install|process}"; exit 1;;
esac
