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

## Citation

If you use `vasp-grace` in your research, please cite it as:

```bibtex
@software{bhatti_vasp_grace_2026,
  author       = {Bhatti, Asif Iqbal},
  title        = {{vasp-grace}: GRACE Machine Learning Potentials as a drop-in VASP replacement},
  year         = {2026},
  url          = {https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential},
  note         = {Version 2.0.0}
}
```

You should also cite the underlying **GRACE / tensorpotential** model and **ASE**:

- Bochkarev, A. *et al.* "Efficient parametrization of the atomic cluster expansion." *Physical Review Materials* **6**, 013804 (2022). https://doi.org/10.1103/PhysRevMaterials.6.013804
- Larsen, A. H. *et al.* "The atomic simulation environment — a Python library for working with atoms." *J. Phys.: Condens. Matter* **29**, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e

> **Note:** If you have published a paper describing this software, please update the BibTeX entry above with the full journal reference and DOI.
