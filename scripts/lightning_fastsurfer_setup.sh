#!/bin/bash
# Paper 12 W6 Step 4 — FastSurfer batch on Lightning AI A100 Studio
#
# This script installs FreeSurfer + FastSurfer, then runs parallel per-scan
# processing on 1 or 2 A100 GPUs.
#
# BEFORE RUNNING ON LIGHTNING:
#   1. Launch a Lightning AI Studio (Machine Learning template)
#   2. Select A100 40GB (or 2x A100 if available in your tier)
#   3. Upload `nifti_wang_n161.tar.gz` to the Studio (use Files panel or scp)
#   4. Upload your FreeSurfer license (license.txt) — register free at:
#         https://surfer.nmr.mgh.harvard.edu/registration.html
#   5. Run: bash lightning_fastsurfer_setup.sh
#
# TIMING (single A100): ~5-7 hours for 401 scans (Wang N=161)
# TIMING (2x A100):     ~2.5-3.5 hours for 401 scans

set -euo pipefail

WORK=${WORK:-/teamspace/studios/this_studio}
cd "$WORK"

# -------- 1. FreeSurfer v7.4.1 (required for surface recon path) --------
if [ ! -d "$WORK/freesurfer" ]; then
  echo "=== Installing FreeSurfer 7.4.1 ==="
  wget -q -O fs.tar.gz "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
  tar -xzf fs.tar.gz
  rm fs.tar.gz
fi

# License file is required - user must upload license.txt first
if [ ! -f "$WORK/license.txt" ]; then
  echo "ERROR: license.txt not found at $WORK/license.txt"
  echo "Register at https://surfer.nmr.mgh.harvard.edu/registration.html and upload the license file."
  exit 1
fi
cp "$WORK/license.txt" "$WORK/freesurfer/.license"

export FREESURFER_HOME="$WORK/freesurfer"
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

# -------- 2. FastSurfer v2.3+ --------
if [ ! -d "$WORK/FastSurfer" ]; then
  echo "=== Installing FastSurfer ==="
  git clone --depth 1 --branch stable https://github.com/Deep-MI/FastSurfer.git
  cd FastSurfer
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
  pip install -q "torch>=2.0" "torchvision>=0.15"
  cd ..
fi

export FASTSURFER_HOME="$WORK/FastSurfer"

# -------- 3. Extract NIfTI tarball --------
if [ ! -d "$WORK/nifti" ]; then
  echo "=== Extracting NIfTI tarball ==="
  mkdir -p nifti
  tar -xzf nifti_wang_n161.tar.gz -C nifti/
fi

# -------- 4. Output directory --------
mkdir -p "$WORK/fastsurfer_out"
SUBJECTS_DIR="$WORK/fastsurfer_out"
export SUBJECTS_DIR

# -------- 5. Batch run --------
# Each NIfTI is named: PATNO{patno}_{date}_{proto}.nii.gz
# Subject ID = filename without .nii.gz extension
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "=== Detected $N_GPUS GPU(s) ==="

NIFTI_LIST=$(find "$WORK/nifti" -name "*.nii.gz" | sort)
N_TOTAL=$(echo "$NIFTI_LIST" | wc -l)
echo "=== Processing $N_TOTAL scans ==="

run_one() {
  local nifti=$1
  local gpu=$2
  local subj_id=$(basename "$nifti" .nii.gz)
  if [ -d "$SUBJECTS_DIR/$subj_id/stats" ]; then
    echo "[SKIP] $subj_id (already done)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$FASTSURFER_HOME/run_fastsurfer.sh" \
    --t1 "$nifti" \
    --sid "$subj_id" \
    --sd "$SUBJECTS_DIR" \
    --parallel \
    --threads 8 \
    --fs_license "$FREESURFER_HOME/.license" \
    > "$SUBJECTS_DIR/${subj_id}.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "[OK] $subj_id"
  else
    echo "[FAIL] $subj_id (check $SUBJECTS_DIR/${subj_id}.log)"
  fi
}
export -f run_one
export FASTSURFER_HOME SUBJECTS_DIR FREESURFER_HOME

# Split across GPUs
if [ "$N_GPUS" -eq 2 ]; then
  echo "$NIFTI_LIST" | awk 'NR%2==1' | xargs -I{} -P1 bash -c 'run_one "$@" 0' _ {} &
  echo "$NIFTI_LIST" | awk 'NR%2==0' | xargs -I{} -P1 bash -c 'run_one "$@" 1' _ {} &
  wait
else
  echo "$NIFTI_LIST" | xargs -I{} -P1 bash -c 'run_one "$@" 0' _ {}
fi

# -------- 6. Collect /stats into a tarball for download --------
echo "=== Packaging /stats outputs ==="
cd "$SUBJECTS_DIR"
find . -path "*/stats/*" -o -name "aparc+aseg.mgz" | tar -czf "$WORK/fastsurfer_stats.tar.gz" -T -
echo "=== DONE ==="
echo "Output: $WORK/fastsurfer_stats.tar.gz"
echo "Download with: scp or Lightning Files panel"
