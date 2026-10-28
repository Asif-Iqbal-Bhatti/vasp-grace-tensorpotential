# vasp-grace-tensorpotential

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-blue)](pyproject.toml)
[![Website](https://img.shields.io/badge/docs-website-0b7285)](https://asif-iqbal-bhatti.github.io/vasp-grace-tensorpotential/)

> **GRACE machine-learning potentials as a drop-in VASP replacement, plus a toolkit for defects, ion transport and topological phonons.**
> Reads standard `POSCAR`/`INCAR` files, routes the calculation through the GRACE `tensorpotential` ASE calculator (TensorFlow, GPU accelerated), and writes `OUTCAR`, `OSZICAR`, `CONTCAR` and `XDATCAR` the way VASP does.

**Website:** <https://asif-iqbal-bhatti.github.io/vasp-grace-tensorpotential/>

---

## Why vasp-grace

- **Keep your VASP workflow.** Same input files, same tags, same output files. Existing scripts that parse `OUTCAR` or `XDATCAR` keep working.
- **DFT-quality forces at MD cost.** GRACE foundation models (e.g. `GRACE-2L-OAM`) cover most of the periodic table; you can also load your own fine-tuned model.
- **Physics on top, not just a wrapper.** Ten command-line tools cover uncertainty-driven active learning, Monte Carlo, dislocations, Li-ion hopping, thermal conductivity, persistent homology, phonon topology and moiré bilayers.

---

## Workflow

```
POSCAR / INCAR
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     vasp-grace  (VASP drop-in)                  │
│  IBRION=1/2 → geometry opt │ IBRION=0 → MD │ IBRION=5/6 → phonon│
│  ISIF=3 → cell relax       │ ISIF≥3 + IBRION=5/6 → elastic Cij   │
└────────────┬────────────────────────────────────────────────────┘
             │ OUTCAR · CONTCAR · OSZICAR · XDATCAR · phonon.*.json
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

## Modules at a glance

| Command | Module | What it does | Key outputs |
|---------|--------|--------------|-------------|
| `vasp-grace` | `main.py` | VASP drop-in: single point · relaxation · MD · phonons · elastic tensor | `OUTCAR`, `CONTCAR`, `XDATCAR`, `phonon_band.png`, `ELASTIC_Cij_GPa.dat` |
| `grace-active-learning` | `active_learning.py` | Committee-model uncertainty; flags structures that need DFT | `uncertainty_log.dat`, `flagged/POSCAR_*` |
| `grace-mc` | `montecarlo.py` | Metropolis MC with displacement and swap moves, live uncertainty | `MC_energies.dat`, `XDATCAR_MC` |
| `grace-dislocation` | `dislocation.py` | Edge, screw and mixed dislocations (isotropic Volterra or anisotropic Stroh) | `POSCAR_dislocation`, `dislocation_info.txt` |
| `grace-lihopping` | `lihopping.py` | Sojourn-filtered ion hop detection, hop networks, Arrhenius Eₐ | `hop_events.dat`, `arrhenius.png` |
| `grace-kappa` | `thermal_conductivity.py` | Müller-Plathe reverse NEMD lattice thermal conductivity κ | `kappa_summary.txt`, `temperature_profile.png` |
| `grace-topology` | `topology.py` | Persistent homology (β₀ β₁ β₂): symmetry-free structural fingerprint | `betti_curves.dat`, `betti_curves.png` |
| `grace-topo-phonons` | `topological_phonons.py` | Weyl/Dirac phonon crossings with local Berry phase | `phonon_crossings.dat`, `phonon_topo_bands.png` |
| `grace-phonon-berry` | `phonon_berry.py` | Phonon Zak phases, Berry curvature Ω(k), Chern numbers | `phonon_zak.dat`, `phonon_chern.png` |
| `grace-moire` | `moire.py` | Commensurate twisted-bilayer supercells (CSL), stacking maps | `POSCAR_moire`, `stacking_map.png` |

Every command accepts `--model` as either a GRACE foundation-model name (default `GRACE-2L-OAM`) or a path to your own fitted model. In `vasp-grace` the same value goes in the `INCAR` tag `GRACE_MODEL`.

---

## Installation

```bash
git clone https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential
cd vasp-grace-tensorpotential
pip install -e .
```

**Dependencies:** `ase`, `numpy`, `scipy`, `matplotlib`, `tensorflow`, `tensorpotential`, [`elastic`](https://github.com/jochym/Elastic) (elastic constants in `vasp-grace`).

**Optional extras**

| Extra | Install | Adds |
|-------|---------|------|
| `tda` | `pip install -e ".[tda]"` | `gudhi`, for full β₁/β₂ persistent homology in `grace-topology` |
| `examples` | `pip install -e ".[examples]"` | `marimo`, for the interactive example notebook |
| `dev` | `pip install -e ".[dev]"` | `pytest`, `black`, `pylint` |

The first call to a foundation model downloads it to `~/.cache/grace/`.

---

## Quick usage

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

Run any command with `--help` for the full option list.

---

## `vasp-grace` INCAR reference

`vasp-grace` reads the tags below and ignores everything else, so an existing VASP `INCAR` can be used unchanged.

| Tag | Default | Effect in vasp-grace |
|-----|---------|----------------------|
| `GRACE_MODEL` | `GRACE-2L-OAM` | Foundation-model name or path to a fitted model |
| `IBRION` | `-1` | `-1` single point · `0` MD · `1` LBFGS · `2` FIRE · `5`/`6` finite-displacement phonons · `7`/`8` not supported (DFPT) |
| `NSW` | `0` | Ionic / MD steps; `0` gives a single point |
| `ISIF` | `3` | Relaxation: `3` positions + cell, `4` positions + cell shape at fixed volume, otherwise positions only. MD: `≥3` switches to NPT. Phonons: `≥3` also computes the elastic tensor |
| `EDIFFG` | `-0.01` | Negative value is the force criterion (eV/Å); positive values fall back to 0.05 eV/Å |
| `POTIM` | `1.0` fs | MD time step |
| `TEBEG` | `300` | MD temperature (K) |
| `MDALGO` | `0` | `0` NVE (Velocity Verlet) · `1` Andersen · `2` Berendsen · `3` Langevin. NPT (`ISIF≥3`) uses the MTK barostat at 1 bar |
| `NFREE` | `2` | Central differences (other values fall back to ±δ) |
| `LPHON_DISPERSION` | `.FALSE.` | Write the phonon band structure (`phonon_band.png`) |
| `PHON_DOS` | `0` | `>0` writes the phonon DOS (`phonon_dos.dat`, `phonon_dos.png`); mesh from `QPOINTS`/`KPOINTS` if present |
| `PHON_NEDOS`, `PHON_SIGMA` | `1000`, `0.001` | DOS grid points and smearing |

Phonons use the `POSCAR` you supply as the displaced cell, so give a supercell `POSCAR` for a supercell phonon calculation. Elastic constants are fitted with [jochym/Elastic](https://github.com/jochym/Elastic) and written to `ELASTIC_Cij_GPa.dat`, `ELASTIC_Bij_GPa.dat` and `ELASTIC_fit_info.txt`.

---

## Interactive example

```bash
pip install -e ".[examples]"
marimo edit examples/quickstart.py
```

The notebook loads a GRACE model, runs a single point, fits an equation of state, relaxes the cell, compares topological fingerprints of an ordered and a rattled crystal, and scans commensurate moiré angles, all with live controls. For fcc Cu with `GRACE-2L-OAM` it gives V₀ = 11.84 Å³ and B₀ ≈ 142 GPa.

---

## Dislocation physics

`dislocation.py` supports two formulations:

| Method | Input | Displacement field |
|--------|-------|--------------------|
| Isotropic Volterra | Poisson ratio ν | Screw: `u_z = b/2π·arctan2(y,x)` · Edge: Hirth & Lothe |
| Anisotropic Stroh | `ELASTIC_Cij_GPa.dat` | 6×6 Stroh eigenvalue problem |

Dipoles (±b at ±Lx/4, the default) keep full 3D periodicity; use `--single` for an isolated dislocation. The anisotropic route reads the elastic tensor that `vasp-grace` writes, so `IBRION=6, ISIF=3` followed by `grace-dislocation --method anisotropic --elastic ELASTIC_Cij_GPa.dat` is a complete GRACE-only pipeline.

Inspired by: [atomsk](https://atomsk.univ-lille.fr) · [atomman](https://github.com/usnistgov/atomman) · [E. Clouet's tools](https://emmanuel.clouet.free.fr)

---

## Scope: what a machine-learning potential can and cannot give you

GRACE predicts the Born–Oppenheimer energy surface: energies, forces and stresses. Everything that follows from the lattice is in reach (structures, phonons, elastic constants, diffusion, thermal transport, lattice topology). Quantities that need electrons are not: band structures, electronic Berry phases or Z₂/Chern invariants, electron–phonon coupling, superconducting gaps and magnetism. For those, use vasp-grace to prepare and pre-relax structures, then hand them to DFT, Wannier90 or EPW.

Foundation models are accurate to a few meV/atom on typical structures. For subtle energy differences (charge-density-wave distortions, competing polymorphs) check against DFT and fine-tune with structures flagged by `grace-active-learning`.

---

## Roadmap

Planned additions, focused on condensed-matter use cases:

- **Anharmonic phonons at finite temperature** (TDEP-style fit from GRACE MD), for dynamically stabilised phases such as superconducting hydrides.
- **Soft-mode / CDW finder**: follow imaginary phonon modes into the distorted ground state and report the energy gain and space group.
- **Phonon surface states** from surface Green's functions, the surface signature of Weyl and nodal-line phonons.
- **Chiral phonons**: phonon angular momentum and circular polarisation at valley points of 2D materials.
- **Allen–Dynes Tc helper** that combines DFT α²F(ω) or λ with GRACE phonon frequencies.
- **phonopy backend** for symmetry-reduced displacements, irreducible representations and LO–TO splitting.

Suggestions and pull requests are welcome via the [issue tracker](https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential/issues).

---

## Project structure

```
vasp-grace-tensorpotential/
├── src/
│   └── vasp_grace/          # all modules (importable package)
│       ├── calculator.py    # shared GRACE model loader
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
├── examples/
│   └── quickstart.py        # interactive marimo notebook
├── docs/
│   └── index.html           # project website (GitHub Pages)
├── tests/
│   ├── test_modules.py      # integration test suite
│   └── (POSCAR, INCAR_*, OUTCAR, ...)
├── README.md
├── pyproject.toml
└── LICENSE
```

### Running tests

```bash
python tests/test_modules.py
```

The suite covers syntax, CLI help, structure-only modules (topology, dislocation, moiré, Li hopping) and, when `tensorpotential` is installed, end-to-end GRACE runs including phonon topology.

---

## Citation

```bibtex
@software{bhatti_vasp_grace_2026,
  author = {Bhatti, Asif Iqbal},
  title  = {{vasp-grace-tensorpotential}: A Python toolkit interfacing GRACE machine learning interatomic potentials with VASP workflows for solid-state dynamics, defect analysis, and topological phonon characterisation},
  year   = {2026},
  url    = {https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential},
  note   = {Version 2.0.0}
}
```

Please also cite the methods you use:

- Bochkarev, A., Lysogorskiy, Y. & Drautz, R. "Graph atomic cluster expansion for semilocal interactions beyond equivariant message passing." *Phys. Rev. X* **14**, 021036 (2024). https://doi.org/10.1103/PhysRevX.14.021036
- Bochkarev, A. *et al.* "Efficient parametrization of the atomic cluster expansion." *Phys. Rev. Mater.* **6**, 013804 (2022). https://doi.org/10.1103/PhysRevMaterials.6.013804
- Larsen, A. H. *et al.* "The atomic simulation environment." *J. Phys.: Condens. Matter* **29**, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e
- Müller-Plathe, F. "A simple nonequilibrium molecular dynamics method for calculating the thermal conductivity." *J. Chem. Phys.* **106**, 6082 (1997). https://doi.org/10.1063/1.473271 (for `grace-kappa`)

> If this work is useful to you, a star on the repo and a citation mean a lot. Thank you.

---

## License

MIT. See [LICENSE](LICENSE).
