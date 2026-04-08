# vasp-grace

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![gracemaker](https://img.shields.io/badge/GRACE-gracemaker-orange.svg)](https://gracemaker.readthedocs.io)

A complete, professional workflow for training and deploying **GRACE (Graph Atomic Cluster Expansion)** machine-learning interatomic potentials (MLIPs) from VASP ab initio data.

This repository covers the full pipeline:

```
VASP calculations
      │
      ▼
 OUTCAR/vasprun.xml  ──►  extxyz / pkl.gz dataset
      │
      ▼
 gracemaker training  (GRACE/FS · GRACE-1L · GRACE-2L)
      │
      ▼
 saved_model / YAML   ──►  ASE calculator  or  LAMMPS pair_style grace
      │
      ▼
 MD · relaxation · NEB · property prediction
```

---

## Contents

| Path | Description |
|---|---|
| `scripts/vasp_to_extxyz.py` | Convert VASP OUTCAR / vasprun.xml → extended XYZ with energies, forces, stresses |
| `scripts/collect_vasp_dataset.py` | Walk a directory tree of VASP runs and produce a `collected.pkl.gz` for gracemaker |
| `scripts/generate_gracemaker_input.py` | Interactively generate a `input.yaml` for gracemaker training |
| `scripts/grace_ase_calculator.py` | Run ASE-based MD/relaxation with a fitted GRACE model |
| `scripts/grace_lammps_input.py` | Generate LAMMPS input files using the GRACE pair style |
| `scripts/validate_grace_model.py` | Parity plots and error statistics for a trained GRACE model |
| `examples/` | Ready-to-run example input files |
| `tests/` | Unit tests |

---

## Installation

### 1. Python environment

```bash
conda create -n vasp-grace python=3.10
conda activate vasp-grace
```

### 2. Core dependencies

```bash
pip install ase pymatgen numpy scipy matplotlib pandas tqdm pyyaml
```

### 3. gracemaker / TensorPotential (GRACE backend)

Follow the [official installation guide](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/):

```bash
pip install tensorpotential          # gracemaker package
pip install python-ace               # required for GRACE/FS YAML models
```

> **GPU strongly recommended** for GRACE-1L / GRACE-2L training.  
> TensorFlow ≥ 2.12 is required.

### 4. (Optional) LAMMPS with GRACE support

See the [LAMMPS+GRACE build guide](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/#lammps-with-grace).  
The `grace` pair style is enabled via the `ML-PACE` package in LAMMPS.

---

## Quick Start

### Step 1 — Convert VASP output to extxyz

```bash
python scripts/vasp_to_extxyz.py \
    --input-dir  /path/to/vasp_runs \
    --output     dataset.extxyz \
    --selection  all \
    --energy-key REF_energy \
    --forces-key REF_forces \
    --stress-key REF_stress \
    --verbose
```

### Step 2 — Build gracemaker dataset (pkl.gz)

```bash
grace_collect \
    -wd /path/to/vasp_runs \
    --output-dataset-filename collected.pkl.gz \
    --free-atom-energy "Li:auto Ni:auto Mn:auto O:auto" \
    --selection all
```

Or use the Python wrapper:

```bash
python scripts/collect_vasp_dataset.py \
    --working-dir /path/to/vasp_runs \
    --output      collected.pkl.gz \
    --elements    Li Ni Mn O \
    --selection   all
```

### Step 3 — Train a GRACE model

```bash
python scripts/generate_gracemaker_input.py \
    --train-file  collected.pkl.gz \
    --model-type  GRACE-2L \
    --complexity  medium \
    --elements    Li Ni Mn O \
    --cutoff      6.0 \
    --output      input.yaml

gracemaker input.yaml
```

### Step 4a — Run ASE MD / relaxation with GRACE

```bash
python scripts/grace_ase_calculator.py \
    --structure   POSCAR \
    --model       seed/1/saved_model \
    --task        relax \
    --fmax        0.01 \
    --output      relaxed.extxyz
```

### Step 4b — Run LAMMPS MD with GRACE

```bash
python scripts/grace_lammps_input.py \
    --structure   POSCAR \
    --model       seed/1/saved_model \
    --elements    Li Ni Mn O \
    --temperature 300 \
    --steps       100000 \
    --output-dir  lammps_run/
```

### Step 5 — Validate the model

```bash
python scripts/validate_grace_model.py \
    --model      seed/1/saved_model \
    --test-data  test_set.extxyz \
    --output-dir validation/
```

---

## GRACE model family overview

| Model | Architecture | Speed | Accuracy | Best for |
|---|---|---|---|---|
| **GRACE/FS** | Local ACE + FS | ★★★★★ | ★★★ | Millions of atoms, CPU/MPI |
| **GRACE-1L** | 1-layer message passing | ★★★★ | ★★★★ | GPU, 10k–100k atoms |
| **GRACE-2L** | 2-layer message passing | ★★★ | ★★★★★ | GPU, highest accuracy |

Foundation models (download with `grace_models download <name>`):
- `GRACE-2L-OMAT-medium` — trained on OMat24
- `GRACE-2L-SMAX-medium` — trained on OMat24 + MPtrj + SALEX
- `GRACE-2L-MPtrj-medium` — trained on MPtrj

---

## Citing

If you use this workflow, please cite:

```bibtex
@article{bochkarev2024grace,
  title   = {Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant Message Passing},
  author  = {Bochkarev, Anton and Lysogorskiy, Yury and Drautz, Ralf},
  journal = {Phys. Rev. X},
  volume  = {14},
  pages   = {021036},
  year    = {2024}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
