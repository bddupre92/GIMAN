#!/bin/bash
# Paper 12 W6 — FastSurfer native install on Apple Silicon Mac (no Docker, MPS backend).
#
# Skips Docker amd64 emulation overhead. Uses PyTorch MPS (Metal Performance Shaders)
# on M-series chips. FastSurfer --seg_only doesn't need FreeSurfer at runtime, so this
# works without the Linux-only FreeSurfer install.
#
# Expected speedup vs Docker amd64: 5-10x (~5-10 min/scan instead of 50 min).
# Caveat: FastSurfer isn't officially macOS-supported; some deps may need tweaking.
#
# Usage:
#   bash scripts/mac_native_fastsurfer_setup.sh install   # one-time setup
#   bash scripts/mac_native_fastsurfer_setup.sh test      # run on 1 scan to validate
#   bash scripts/mac_native_fastsurfer_setup.sh batch     # run full Wang N=161

set -o pipefail

PHASE="${1:-install}"
VENV=/tmp/fastsurfer_venv
FASTSURFER_DIR=/tmp/FastSurfer
NIFTI_DIR=/Users/blair.dupre/Projects/CSCI-FALL-2025/data/01_processed/GIMAN/t1_expansion_nifti
OUT_DIR=/Users/blair.dupre/Projects/CSCI-FALL-2025/data/02_fastsurfer_mac
LICENSE=/Users/blair.dupre/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin/paper12_phys_gimin/data/license.txt
MANIFEST=/Users/blair.dupre/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin/paper12_phys_gimin/data/wang_n161_manifest.csv

install_deps() {
  # Prefer 3.11 (official FastSurfer), fall back to 3.12 (tested works)
  local PY=""
  for v in 3.11 3.12; do
    local cand=/opt/homebrew/bin/python${v}
    if [ -x "$cand" ]; then PY=$cand; break; fi
    cand=$(command -v python${v})
    if [ -x "$cand" ]; then PY=$cand; break; fi
  done
  if [ -z "$PY" ]; then
    echo "ERROR: need Python 3.11 or 3.12 (brew install python@3.11)"; exit 1
  fi
  echo "[install] Using $PY"
  echo "[install] Creating venv at $VENV..."
  if [ ! -d "$VENV" ]; then
    "$PY" -m venv "$VENV" || { echo "ERROR: venv creation failed"; exit 1; }
  fi
  source "$VENV/bin/activate"
  pip install -q --upgrade pip

  if [ ! -d "$FASTSURFER_DIR" ]; then
    echo "[install] Cloning FastSurfer..."
    git clone --depth 1 --branch stable https://github.com/Deep-MI/FastSurfer.git "$FASTSURFER_DIR"
  fi

  echo "[install] Installing PyTorch (Apple Silicon MPS-capable, no CUDA)..."
  # Apple Silicon wheels — torch auto-selects MPS on M-series
  pip install -q "torch>=2.2" "torchvision>=0.17"

  echo "[install] Installing FastSurfer runtime deps (CUDA pins filtered out)..."
  # FastSurfer's requirements.txt pins nvidia-*-cu12 packages that don't exist
  # on macOS ARM64. Install the actually-needed CPU/MPS-compatible deps directly.
  pip install -q \
    "nibabel>=4.0" \
    "numpy<2.0" \
    "scipy>=1.10" \
    "scikit-image>=0.19" \
    "torchio>=0.19" \
    "tqdm>=4.60" \
    "pyyaml>=6.0" \
    "yacs>=0.1.8" \
    "h5py>=3.8" \
    "matplotlib>=3.6" \
    "pandas>=2.0" \
    "lapy>=1.0" \
    "simpleitk>=2.3" \
    "requests>=2.30"

  echo "[install] DONE. Test with: bash $0 test"
}

run_one() {
  local nifti=$1
  local subj_id=$2
  source "$VENV/bin/activate"
  export PYTHONPATH="$FASTSURFER_DIR:$PYTHONPATH"
  mkdir -p "$OUT_DIR"
  echo "[run] $subj_id starting ($(date -u +%H:%M:%S))"
  python "$FASTSURFER_DIR/FastSurferCNN/run_prediction.py" \
    --t1 "$nifti" \
    --asegdkt_segfile "$OUT_DIR/$subj_id/mri/aparc.DKTatlas+aseg.deep.mgz" \
    --conformed_name "$OUT_DIR/$subj_id/mri/orig.mgz" \
    --brainmask_name "$OUT_DIR/$subj_id/mri/mask.mgz" \
    --aseg_name "$OUT_DIR/$subj_id/mri/aseg.auto_noCCseg.mgz" \
    --sid "$subj_id" --sd "$OUT_DIR" \
    --device mps --batch_size 1 --threads 8 \
    --seg_log "$OUT_DIR/$subj_id/scripts/deep-seg.log" \
    --vox_size min --viewagg_device auto
  local rc=$?
  echo "[run] $subj_id rc=$rc ($(date -u +%H:%M:%S))"
  return $rc
}

test_one() {
  local first_nifti
  first_nifti=$(find "$NIFTI_DIR" -name "*.nii.gz" | sort | head -1)
  if [ -z "$first_nifti" ]; then echo "ERROR: no NIfTIs at $NIFTI_DIR"; exit 1; fi
  local subj_id
  subj_id=$(basename "$first_nifti" .nii.gz)
  run_one "$first_nifti" "$subj_id"
  echo
  echo "[test] output files for $subj_id:"
  ls -la "$OUT_DIR/$subj_id/mri" 2>&1 | head
}

batch_run() {
  [ -f "$MANIFEST" ] || { echo "ERROR: missing manifest $MANIFEST"; exit 1; }
  source "$VENV/bin/activate"
  local start=$(date +%s)
  local n_ok=0 n_fail=0 n_total
  n_total=$(tail -n +2 "$MANIFEST" | awk -F, '{print $1"_"$3"_"$4}' | sort -u | wc -l)
  echo "[batch] Wang N=161 cohort manifest → $n_total unique scans"

  # Iterate through manifest rows (patno,stratum,n_visits,visit_date,protocol,dicom_path,nifti_filename... approximate)
  while IFS=, read -r patno stratum n_visits visit_date protocol rest; do
    [ "$patno" = "patno" ] && continue
    local subj_dir="$NIFTI_DIR/PATNO_${patno}/${visit_date}_${protocol}"
    local nifti
    nifti=$(ls "$subj_dir"/*.nii.gz 2>/dev/null | head -1)
    [ -z "$nifti" ] && { echo "[batch] SKIP $patno $visit_date — no nifti"; continue; }
    local subj_id="PATNO${patno}_${visit_date}_${protocol}"
    if [ -f "$OUT_DIR/$subj_id/stats/aseg+DKT.stats" ]; then
      echo "[batch] SKIP $subj_id (already done)"
      continue
    fi
    if run_one "$nifti" "$subj_id"; then
      n_ok=$((n_ok+1))
    else
      n_fail=$((n_fail+1))
    fi
    local elapsed=$(( $(date +%s) - start ))
    echo "[batch] done=$n_ok fail=$n_fail elapsed=${elapsed}s"
  done < "$MANIFEST"
  echo "[batch] DONE  OK=$n_ok  FAIL=$n_fail"
}

case "$PHASE" in
  install) install_deps ;;
  test)    test_one ;;
  batch)   batch_run ;;
  *) echo "Usage: $0 {install|test|batch}"; exit 1 ;;
esac
