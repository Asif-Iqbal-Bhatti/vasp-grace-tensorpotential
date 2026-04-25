# Quick Reference: vasp-grace Package Guide

## Installation

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Standard install
pip install .
```

## Command-Line Usage

```bash
# Basic usage (uses CONTCAR and INCAR)
vasp-grace

# With explicit paths
vasp-grace --poscar POSCAR --incar INCAR

# Get help
vasp-grace --help
```

## Module Organization

```
vasp_grace/
├── __init__.py          → Package entry point
├── cli.py               → Command-line interface & main dispatcher
├── parser.py            → INCAR file parsing with type inference
├── calculator.py        → GRACE model loading and utilities
├── calculations.py      → High-level calculation workflows
├── outputs.py           → VASP-compatible output file writing
└── utils.py             → Helper functions
```

## Common Programming Tasks

### Task: Parse an INCAR File

```python
from vasp_grace import parse_incar

incar = parse_incar("INCAR")
print(incar["NSW"])      # Number of ionic steps
print(incar["IBRION"])   # Ionic movement algorithm
```

### Task: Load a GRACE Calculator

```python
from vasp_grace.calculator import get_calculator
from ase.io import read

atoms = read("POSCAR", format="vasp")
calc = get_calculator("GRACE-2L-OAM")
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### Task: Run a Geometry Optimization

```python
from vasp_grace import parse_incar
from vasp_grace.calculations import run_geometry_optimization
from ase.io import read

atoms = read("POSCAR", format="vasp")
incar = parse_incar("INCAR")

run_geometry_optimization(atoms, incar)
```

### Task: Run Single-Point Calculation

```python
from vasp_grace.calculations import run_single_point

run_single_point(atoms, incar)
```

### Task: Run Molecular Dynamics

```python
from vasp_grace.calculations import run_molecular_dynamics

run_molecular_dynamics(atoms, incar)
```

### Task: Run Phonon Calculation

```python
from vasp_grace.calculations import run_ase_phonons

run_ase_phonons(atoms, incar)
```

## Adding New INCAR Properties

### Step 1: Add to Parser

Edit `vasp_grace/parser.py`:

```python
params = {
    # ... existing params ...
    "MY_NEW_PARAM": 1.5,      # Float parameter
    "MY_NEW_BOOL": False,     # Boolean parameter
    "MY_NEW_STRING": "default",  # String parameter
}
```

### Step 2: Use in Calculations

In `vasp_grace/calculations.py` or a new calculation function:

```python
def my_calculation(atoms, incar):
    my_val = incar["MY_NEW_PARAM"]
    if incar["MY_NEW_BOOL"]:
        # Do something with my_val
        pass
```

### Step 3: Add Trigger Logic

In `vasp_grace/cli.py`:

```python
if incar["MY_TRIGGER"] == True:
    my_calculation(atoms, incar)
```

## Supported INCAR Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `IBRION` | int | -1 | Ionic movement algorithm |
| `NSW` | int | 0 | Number of ionic steps |
| `ISIF` | int | 3 | Cell shape/volume adjustment |
| `EDIFFG` | float | -0.01 | Force convergence criterion |
| `GRACE_MODEL` | str | "GRACE-2L-OAM" | GRACE model identifier |
| `POTIM` | float | None | Time step (MD/phonons) |
| `TEBEG` | float | 300.0 | Initial temperature |
| `MDALGO` | int | 0 | MD ensemble type |
| `NFREE` | int | 2 | Phonon displacement points |
| `LPHON_DISPERSION` | bool | False | Calculate phonon band structure |
| `PHON_DOS` | int | 0 | Calculate phonon DOS |
| `PHON_NEDOS` | int | 1000 | DOS points |
| `PHON_SIGMA` | float | 0.001 | DOS smearing |

## Calculation Types (by IBRION value)

| IBRION | Description | NSW | Key Settings |
|--------|-------------|-----|--------------|
| -1 | Single point | 0 | Just energy/forces/stress |
| 0 | Molecular dynamics | >0 | TEBEG, MDALGO, POTIM |
| 1 | LBFGS optimization | >0 | ISIF controls cell relaxation |
| 2 | Conjugate gradient | >0 | (not used, defaults to FIRE2) |
| 5,6 | Phonons (finite-disp.) | 1 | NFREE, POTIM (displacement) |

## Molecular Dynamics Ensembles (by MDALGO)

| MDALGO | Ensemble | Description |
|--------|----------|-------------|
| 0 | NVE | Microcanonical (velocity Verlet) |
| 1 | NVT | Andersen thermostat |
| 2 | NVT | Berendsen thermostat |
| 3 | NVT | Langevin thermostat |
| (ISIF≥3) | NPT | Nose-Hoover chain barostat |

## Output Files Generated

- **`OUTCAR`** - Main output (energies, forces, stresses)
- **`OSZICAR`** - SCF/ionic convergence summary
- **`CONTCAR`** - Final structure
- **`XDATCAR`** - Trajectory (MD/opt)
- **`force_constants.npy`** - Phonon force constants (phonon runs)
- **`phonon_gamma.dat`** - Gamma-point frequencies
- **`grace_opt.traj`** - ASE trajectory file

## Extending the Package

### Adding a New Calculation Module

1. Create `vasp_grace/my_feature.py`:
```python
def run_my_feature(atoms, incar):
    """Description of my feature."""
    # Implementation
    pass
```

2. Import in `cli.py`:
```python
from .my_feature import run_my_feature
```

3. Add dispatcher logic:
```python
if should_run_my_feature(incar):
    run_my_feature(atoms, incar)
```

### Package Distribution

Build distributions for PyPI:

```bash
# Install build tools
pip install build

# Create distributions
python -m build

# Upload to PyPI
twine upload dist/*
```

## Troubleshooting

### Import Error: No module named 'ase'

Dependencies aren't installed. Run:
```bash
pip install -e .
```

### TypeError: unsupported operand type(s)

Check that parameters in INCAR match their expected types in `parser.py`.

### GPU not being used

Ensure TensorFlow GPU support is installed:
```bash
pip install tensorflow[and-cuda]
```

## Full Documentation

- **Installation details**: [INSTALL.md](INSTALL.md)
- **Refactoring overview**: [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- **Original README**: [README.md](README.md)
