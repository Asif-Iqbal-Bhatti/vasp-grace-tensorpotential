# vasp-grace-tensorpotential

> **GRACE Machine Learning Potentials as a drop-in VASP replacement.**  
> Reads standard `POSCAR`/`INCAR` files and routes calculations through the GRACE `tensorpotential` ASE calculator (TensorFlow / GPU accelerated).

---

## Workflow

```
POSCAR / INCAR
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        main.py  (VASP drop-in)                  │
│  IBRION=1/2 → geometry opt │ IBRION=0 → MD │ IBRION=5/6 → phonon│
│  ISIF=3 → cell relax       │ NEB           │ elastic constants   │
└────────────┬────────────────────────────────────────────────────┘
             │ OUTCAR · CONTCAR · OSZICAR · phonon.*.json
             │
    ┌────────┴────────────────────────────────────────────┐
    │               Analysis & Advanced Modules            │
    └──────────────────────────────┬──────────────────────┘
                                   │
    ┌──────────────────────────────┼──────────────────────────────┐
    │                              │                              │
    ▼                              ▼                              ▼
┌───────────────┐        ┌─────────────────┐         ┌───────────────────┐
│ Structure /   │        │  Ion dynamics   │         │  Condensed matter │
│ Uncertainty   │        │  & transport    │         │  / Topology       │
├───────────────┤        ├─────────────────┤         ├───────────────────┤
│active_learning│        │  lihopping      │         │topological_phonons│
│montecarlo     │        │  thermal_cond.  │         │phonon_berry       │
│dislocation    │        │                 │         │topology (TDA)     │
│               │        │                 │         │moire              │
└───────────────┘        └─────────────────┘         └───────────────────┘
```

---

## Modules at a Glance

| Module | What it does | Key outputs |
|--------|-------------|-------------|
| `main.py` | VASP drop-in: opt · MD · phonon · NEB · elastic | `OUTCAR`, `CONTCAR`, band/DOS plots |
| `active_learning.py` | Committee-model UQ — flags uncertain structures for DFT | `uncertainty_log.dat`, `flagged/POSCAR_*` |
| `montecarlo.py` | Metropolis MC with displacement + swap moves, live UQ | `MC_energies.dat`, `XDATCAR_MC` |
| `dislocation.py` | Edge/screw dislocations via Volterra or Stroh formalism | `POSCAR_dislocation`, `dislocation_info.txt` |
| `lihopping.py` | Sojourn-filtered ion hop detection, Arrhenius Ea | `hop_events.dat`, `arrhenius.png` |
| `thermal_conductivity.py` | Müller-Plathe rNEMD — lattice κ(T), no stress tensor needed | `kappa_summary.txt`, `temperature_profile.png` |
| `topology.py` | Persistent homology (β₀ β₁ β₂) — symmetry-free fingerprinting | `betti_curves.dat`, `betti_curves.png` |
| `topological_phonons.py` | Weyl/Dirac phonon points via band inversion + Berry phase | `phonon_crossings.dat`, `phonon_topo_bands.png` |
| `phonon_berry.py` | Zak phases + Berry curvature Ω(k) + Chern numbers | `phonon_zak.dat`, `phonon_chern.png` |
| `moire.py` | Commensurate twisted-bilayer supercells (CSL method) | `POSCAR_moire`, `stacking_map.png` |

---

## Project Structure

```
vasp-grace-tensorpotential/
├── src/
│   └── vasp_grace/          # all modules (importable package)
│       ├── main.py
│       ├── active_learning.py
│       ├── montecarlo.py
│       ├── dislocation.py
│       ├── lihopping.py
│       ├── thermal_conductivity.py
│       ├── topology.py
│       ├── topological_phonons.py
│       ├── phonon_berry.py
│       └── moire.py
├── tests/
│   ├── test_modules.py      # integration test suite
│   └── (POSCAR, INCAR_*, OUTCAR, ...)
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential
cd vasp-grace-tensorpotential
pip install -e .
```

**Dependencies:** `ase`, `numpy`, `scipy`, `matplotlib`, `tensorflow`, `tensorpotential`  
Optional: `gudhi` (`pip install gudhi`) for full β₂ persistent homology in `topology.py`.

---

## Quick Usage

After `pip install -e .`, all modules are available as CLI commands:

```bash
# VASP drop-in
vasp-grace --poscar POSCAR --incar INCAR

# Active learning: flag uncertain structures
grace-active-learning --poscar_dir ./structs --models m1.pb m2.pb m3.pb

# Metropolis MC + swap moves (Cl ↔ S disorder)
grace-mc --poscar POSCAR --model GRACE-2L-OAM \
         --swap Cl S --temperature 800 --steps 50000

# Dislocation (isotropic edge)
grace-dislocation --poscar POSCAR --type edge --burgers 2.8 --poisson 0.28

# Li-ion hop detection + Arrhenius Ea
grace-lihopping --traj XDATCAR --ref POSCAR --species Li --timestep 2.0
grace-lihopping --arrhenius --temps 600 800 1000 \
                --trajs traj_600K traj_800K traj_1000K --ref POSCAR --species Li

# Lattice thermal conductivity (rNEMD)
grace-kappa --poscar POSCAR --model GRACE-2L-OAM --temperature 300 --steps 200000

# Topological Data Analysis (crystal vs grain boundary)
grace-topology --poscar POSCAR_bulk --compare POSCAR_gb --rmax 8.0

# Weyl/Dirac phonon points + Berry phase
grace-topo-phonons --poscar POSCAR --model GRACE-2L-OAM --berry

# Phonon Zak phases + Berry curvature
grace-phonon-berry --poscar POSCAR --model GRACE-2L-OAM --curvature

# Moiré superlattice (scan commensurate angles, then build)
grace-moire --poscar POSCAR --scan --m_max 10
grace-moire --poscar POSCAR --m 5 --n 6 --relax --model GRACE-2L-OAM
```

### Running tests

```bash
python tests/test_modules.py
```

---

## Dislocation physics

`dislocation.py` supports two formulations:

| Method | Input | Displacement field |
|--------|-------|--------------------|
| Isotropic Volterra | Poisson ratio ν | Screw: `u_z = b/2π·arctan2(y,x)` · Edge: Hirth & Lothe |
| Anisotropic Stroh | `ELASTIC_Cij_GPa.dat` | 6×6 Stroh eigenvalue problem |

Both support dipole configuration (±b at ±Lx/4) for full 3D periodicity.

Inspired by: [atomsk](https://atomsk.univ-lille.fr) · [atomman](https://github.com/usnistgov/atomman) · [E. Clouet's tools](https://emmanuel.clouet.free.fr)

---

## Citation

```bibtex
@software{bhatti_vasp_grace_2026,
  author = {Bhatti, Asif Iqbal},
  title  = {{vasp-grace-tensorpotential}: A wrapper to use GRACE MLIP as a substitute for VASP/DFT backend for solid-state systems and advanced topological material analysis},
  year   = {2026},
  url    = {https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential},
  note   = {Version 2.0.0}
}
```

- Bochkarev, A. *et al.* "Efficient parametrization of the atomic cluster expansion." *Phys. Rev. Mater.* **6**, 013804 (2022). https://doi.org/10.1103/PhysRevMaterials.6.013804
- Larsen, A. H. *et al.* "The atomic simulation environment." *J. Phys.: Condens. Matter* **29**, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e

> If this work is useful to you, a star on the repo and a citation mean a lot. Thank you.
