# vasp-grace

`vasp-grace` is a lightweight Python package that allows you to use the **GRACE Machine Learning Potentials** as a drop-in replacement for VASP. By mimicking the VASP executable, it seamlessly integrates GRACE with higher-level materials science workflows (e.g., Phonopy, USPEX, CALYPSO) that normally parse VASP inputs and outputs.

## Features
- **Direct VASP Compatibility:** Reads standard `POSCAR` and `INCAR` files.
- **Modular Architecture:** Organized into logical modules for easy extension and maintenance.
- **Geometry Optimization:** Supports VASP's `IBRION` and `ISIF` tags for structural optimization natively via ASE.
- **Molecular Dynamics:** Supports multiple ensemble types (NVE, NVT, NPT) via ASE.
- **Phonon Calculations:** Finite-displacement phonons with band structure and DOS plotting.
- **NEB:** Nudged Elastic Band calculations using ASE.
- **Output Generation:** Writes standard `OUTCAR`, `OSZICAR`, and `CONTCAR` files mimicking VASP output structures.
- **Backend:** Native Python integration using the GRACE `tensorpotential` ASE calculator (GPU accelerated via TensorFlow).
- **pip-installable:** Install as a standard Python package with `pip install .`

## Installation

### Quick Start

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Use from command line:
   ```bash
   vasp-grace --poscar POSCAR --incar INCAR
   ```

### Detailed Instructions

See [INSTALL.md](INSTALL.md) for comprehensive installation and usage guide.
