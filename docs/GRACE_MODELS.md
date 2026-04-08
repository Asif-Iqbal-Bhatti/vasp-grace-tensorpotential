# GRACE Model Family — Detailed Guide

This document explains the three GRACE model variants, their trade-offs,
and when to choose each one for VASP-based MLIP workflows.

---

## 1. GRACE/FS — Fast, Local, C++ Native

**Architecture:** Local ACE with a finnis-sinclair-type non-linear readout.  
**Backend:** Native C++ in LAMMPS (`grace/fs` pair style). No TensorFlow at inference time.  
**Speed:** Up to millions of atoms on CPU with MPI domain decomposition.

### When to use GRACE/FS
- Production MD with very large supercells (> 10,000 atoms)
- CPU-only clusters without GPU availability
- Workflows requiring maximum throughput (e.g., phonon calculations with many finite displacements)
- Systems where you need the `extrapolation` monitoring feature to detect out-of-distribution structures

### Training
```bash
python scripts/generate_gracemaker_input.py \
    --train-file collected.pkl.gz \
    --model-type GRACE/FS \
    --complexity medium \
    --elements Li Ni Mn O \
    --output input_fs.yaml

gracemaker input_fs.yaml
```

After training, export the YAML model for C++:
```bash
gracemaker input_fs.yaml -r -s -sf   # -sf flag exports FS_model.yaml
```

### LAMMPS usage
```lammps
pair_style   grace/fs
pair_coeff   * * seed/1/FS_model.yaml Li Ni Mn O

# With extrapolation grade monitoring:
pair_style   grace/fs extrapolation
pair_coeff   * * seed/1/FS_model.yaml seed/1/FS_model.asi Li Ni Mn O
fix          grace_gamma all pair 100 grace/fs gamma 1
compute      max_gamma all reduce max f_grace_gamma
```

### MPI parallelisation
Only GRACE/FS (1-layer) supports domain decomposition MPI:
```bash
mpirun -np 16 lmp -in in.lammps
```

---

## 2. GRACE-1L — GPU-Accelerated Local Model

**Architecture:** One message-passing layer; local interactions only.  
**Backend:** TensorFlow SavedModel in LAMMPS (`grace` pair style); GPU required for efficiency.  
**Speed:** ~10–100k atoms on a single A100 at useful throughput.

### When to use GRACE-1L
- Moderate-sized systems (1,000–50,000 atoms) on a GPU cluster
- Good accuracy with lower training cost than GRACE-2L
- When a foundation model fine-tune is needed but GRACE-2L is too slow

### LAMMPS usage
```lammps
pair_style   grace pad_verbose
pair_coeff   * * seed/1/saved_model Li Ni Mn O
```

For large systems (OOM risk):
```lammps
pair_style   grace/1layer/chunk chunksize 2048
pair_coeff   * * seed/1/saved_model Li Ni Mn O
```

---

## 3. GRACE-2L — Highest Accuracy, Semi-Local

**Architecture:** Two message-passing layers; captures longer-range interactions.  
**Backend:** TensorFlow SavedModel in LAMMPS (`grace` pair style); GPU required.  
**Speed:** Tens of thousands of atoms on a single A100 at useful throughput.

### When to use GRACE-2L
- Highest accuracy is required (e.g., HE-DRX cathode screening, ionic conductors)
- Fine-tuning from a foundation model (GRACE-2L-OMAT-medium, GRACE-2L-SMAX-medium)
- Systems with complex many-body interactions (grain boundaries, interfaces, defects)

### Foundation model fine-tuning
```bash
# Download the foundation model first
grace_models download GRACE-2L-OMAT-medium

# Fine-tune on your VASP data
python scripts/generate_gracemaker_input.py \
    --train-file collected.pkl.gz \
    --finetune   GRACE-2L-OMAT-medium \
    --elements   Li Ni Mn O \
    --output     finetune_input.yaml

gracemaker finetune_input.yaml
```

### LAMMPS usage
```lammps
pair_style   grace pad_verbose
pair_coeff   * * seed/1/saved_model Li Ni Mn O

# Chunked for large systems:
pair_style   grace/2layer/chunk chunksize 4096
pair_coeff   * * seed/1/saved_model Li Ni Mn O
```

### GPU parallelisation (multi-GPU)
```bash
# Each MPI rank uses one GPU
TF_CPP_MIN_LOG_LEVEL=1 mpirun -np 4 --bind-to none \
    bash -c 'CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_RANK % 4)) lmp -in in.lammps'
```

---

## Model Selection Decision Tree

```
Large system (> 50k atoms)?
    YES → GRACE/FS  (CPU/MPI)
    NO  → GPU available?
              NO  → GRACE/FS
              YES → Need maximum accuracy?
                        YES → GRACE-2L
                        NO  → GRACE-1L  (faster training)
```

---

## Foundation Models Available

| Model name | Training data | Elements | Best for |
|---|---|---|---|
| `GRACE-2L-OMAT-medium` | OMat24 | Full periodic table | General materials |
| `GRACE-2L-SMAX-medium` | OMat24 + MPtrj + SALEX | Full periodic table | Battery materials, solid electrolytes |
| `GRACE-2L-MPtrj-medium` | MPtrj | ~90 elements | Materials Project compositions |
| `GRACE-1L-OMAT-medium` | OMat24 | Full periodic table | Faster GPU inference |

Download with:
```bash
grace_models list                         # list available models
grace_models download GRACE-2L-SMAX-medium   # download specific model
```

---

## Units Reference

| Quantity | VASP | ASE | LAMMPS (metal) |
|---|---|---|---|
| Energy | eV | eV | eV |
| Length | Å | Å | Å |
| Force | eV/Å | eV/Å | eV/Å |
| Stress | kBar | eV/Å³ (Voigt) | bar |
| Time | fs | fs | ps |
| Temperature | K | K | K |

> **Important:** gracemaker uses `REF_virial` (3×3 matrix, eV, not stress/volume)
> as the default virial key. The `vasp_to_extxyz.py` script converts VASP kBar
> stress → virial in eV automatically.
