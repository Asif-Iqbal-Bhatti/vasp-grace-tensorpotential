# vasp-grace

`vasp-grace` is a lightweight Python package that allows you to use the **GRACE Machine Learning Potentials** as a drop-in replacement for VASP. By mimicking the VASP executable, it seamlessly integrates GRACE with higher-level materials science workflows (e.g., Phonopy,) that normally parse VASP inputs and outputs.

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
- **Active Learning / UQ:** Committee-model uncertainty quantification to identify structures for DFT labelling (`active_learning.py`).
- **Monte Carlo sampling:** Metropolis MC with GRACE energies, swap moves for cation disorder, and live UQ flagging (`montecarlo.py`).

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

## Active Learning & Uncertainty Quantification

`active_learning.py` uses a committee of N GRACE models to measure disagreement in predicted forces as a proxy for model uncertainty. High-uncertainty structures are saved as POSCARs ready for DFT labelling — closing the finetuning loop for difficult geometries like grain boundaries.

```bash
# Screen a single structure
python active_learning.py --poscar POSCAR --models m1.pb m2.pb m3.pb

# Screen a full MD trajectory (every 10th frame)
python active_learning.py --xdatcar XDATCAR --models m1.pb m2.pb m3.pb --stride 10

# Screen a directory of POSCARs + write per-atom uncertainty map
python active_learning.py --poscar_dir ./structures --models m1.pb m2.pb --per_atom
```

Outputs: `uncertainty_log.dat`, `flagged/POSCAR_uncertain_NNNN`, optionally `per_atom_uncertainty.dat`.

## Monte Carlo Sampling

`montecarlo.py` runs Metropolis MC using GRACE as the energy engine. Supports:
- **Displacement moves** — random atomic perturbations
- **Swap moves** — exchange two atom species (e.g. Cl ↔ S for antisite-disorder at a grain boundary interface)
- **Live UQ** — when a model committee is provided, flags high-uncertainty MC snapshots for DFT

```bash
# Basic MC at 800 K
python montecarlo.py --poscar POSCAR --model GRACE-2L-OAM --temperature 800 --steps 50000

# With Cl ↔ S swap moves (20% of steps)
python montecarlo.py --poscar POSCAR --model m1.pb --swap Li S --temperature 1000 --steps 100000

# Committee MC: energy from model[0], UQ from full committee
python montecarlo.py --poscar POSCAR --models m1.pb m2.pb m3.pb --temperature 800 --steps 50000
```

Outputs: `MC_energies.dat`, `XDATCAR_MC`, `CONTCAR_MC`, optionally `flagged/POSCAR_MC_NNNNN`.

## Citation

If you use `vasp-grace-tensorpotential` in your research, please cite it as:

```bibtex
@software{bhatti_vasp_grace_2026,
  author       = {Bhatti, Asif Iqbal},
  title        = {{vasp-grace-tensorpotential}: GRACE Machine Learning Potentials as a drop-in VASP replacement},
  year         = {2026},
  url          = {https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential},
  note         = {Version 2.0.0}
}
```


- Bochkarev, A. *et al.* "Efficient parametrization of the atomic cluster expansion." *Physical Review Materials* **6**, 013804 (2022). https://doi.org/10.1103/PhysRevMaterials.6.013804
- Larsen, A. H. *et al.* "The atomic simulation environment — a Python library for working with atoms." *J. Phys.: Condens. Matter* **29**, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e

> **Note:** If you have published a paper describing this software, please update the BibTeX entry above with the full journal reference and DOI.
