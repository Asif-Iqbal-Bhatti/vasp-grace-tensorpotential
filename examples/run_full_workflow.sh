#!/usr/bin/env bash
# =============================================================================
#  run_full_workflow.sh
#  End-to-end example:  VASP → dataset → GRACE training → MD → validation
#
#  Prerequisites
#  -------------
#    pip install tensorpotential python-ace ase pymatgen matplotlib tqdm
#    VASP calculations already completed in ./vasp_runs/
#
#  Usage:
#    chmod +x examples/run_full_workflow.sh
#    ./examples/run_full_workflow.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$SCRIPT_DIR/../scripts"

VASP_DIR="./vasp_runs"          # Directory of VASP calculation sub-folders
DATASET="collected.pkl.gz"
MODEL_DIR="seed/1/saved_model"
ELEMENTS="Li Ni Mn O"
STRUCTURE="POSCAR"

# ─── Step 1: Convert VASP OUTCARs to extxyz ──────────────────────────────────
echo "=== Step 1: VASP → extxyz ==="
python "$SCRIPTS/vasp_to_extxyz.py" \
    --input-dir "$VASP_DIR" \
    --output    "dataset_all.extxyz" \
    --selection all \
    --energy-key REF_energy \
    --forces-key REF_forces \
    --stress-key REF_virial \
    --verbose

# ─── Step 2: Build gracemaker pkl.gz dataset ──────────────────────────────────
echo ""
echo "=== Step 2: Collect VASP dataset (grace_collect) ==="
python "$SCRIPTS/collect_vasp_dataset.py" \
    --working-dir     "$VASP_DIR" \
    --output          "$DATASET" \
    --elements        $ELEMENTS \
    --selection       all \
    --check-convergence \
    --verbose

# ─── Step 3: Generate gracemaker input.yaml ───────────────────────────────────
echo ""
echo "=== Step 3: Generate gracemaker input.yaml ==="
python "$SCRIPTS/generate_gracemaker_input.py" \
    --train-file  "$DATASET" \
    --model-type  GRACE-2L \
    --complexity  medium \
    --elements    $ELEMENTS \
    --cutoff      6.0 \
    --output      input.yaml

# ─── Step 4: Train GRACE model ────────────────────────────────────────────────
echo ""
echo "=== Step 4: Train GRACE-2L model ==="
echo "  (run this on a GPU node; monitor with 'tensorboard --logdir seed/1/')"
gracemaker input.yaml

# ─── Step 5: Export to SavedModel format ──────────────────────────────────────
echo ""
echo "=== Step 5: Export model ==="
gracemaker input.yaml -r -s    # export best checkpoint to saved_model

# ─── Step 6: Single-point validation ─────────────────────────────────────────
echo ""
echo "=== Step 6: Structure relaxation with GRACE ==="
python "$SCRIPTS/grace_ase_calculator.py" \
    --structure "$STRUCTURE" \
    --model     "$MODEL_DIR" \
    --task      relax \
    --fmax      0.01 \
    --relax-cell \
    --output    relaxed_grace.extxyz

# ─── Step 7: NVT MD ───────────────────────────────────────────────────────────
echo ""
echo "=== Step 7: NVT MD at 1000K ==="
python "$SCRIPTS/grace_ase_calculator.py" \
    --structure  relaxed_grace.extxyz \
    --model      "$MODEL_DIR" \
    --task       nvt \
    --temperature 1000 \
    --timestep   2.0 \
    --steps      50000 \
    --log-interval 500 \
    --write-interval 500 \
    --output     nvt_1000K.extxyz

# ─── Step 8: Generate LAMMPS input ───────────────────────────────────────────
echo ""
echo "=== Step 8: Generate LAMMPS input files ==="
python "$SCRIPTS/grace_lammps_input.py" \
    --structure     "$STRUCTURE" \
    --model         "$MODEL_DIR" \
    --elements      $ELEMENTS \
    --ensemble      nvt \
    --temperature   1000 \
    --steps         200000 \
    --output-dir    lammps_run/ \
    --generate-submit \
    --n-gpu 1 \
    --partition gpu

# ─── Step 9: Validate model ───────────────────────────────────────────────────
echo ""
echo "=== Step 9: Validate GRACE model ==="
python "$SCRIPTS/validate_grace_model.py" \
    --model       "$MODEL_DIR" \
    --test-data   dataset_all.extxyz \
    --energy-key  REF_energy \
    --forces-key  REF_forces \
    --output-dir  validation/

echo ""
echo "=== Workflow complete ==="
echo "  Trained model     : $MODEL_DIR"
echo "  Relaxed structure : relaxed_grace.extxyz"
echo "  NVT trajectory    : nvt_1000K.extxyz"
echo "  LAMMPS input      : lammps_run/"
echo "  Validation plots  : validation/"
