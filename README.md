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
- **Active Learning / UQ:** Committee-model uncertainty quantification to identify structures for DFT labelling (`active_learning.py`).
- **Monte Carlo sampling:** Metropolis MC with GRACE energies, swap moves for cation disorder, and live UQ flagging (`montecarlo.py`).
- **Dislocation builder:** Edge, screw, and mixed dislocation structures via isotropic Volterra or anisotropic Stroh formalism, with dipole configuration and GRACE relaxation (`dislocation.py`).
- **Li-ion hop detection:** Sojourn-filtered hop event detection from MD/MC trajectories, per-site hop rates, hop network, and Arrhenius activation energy (`lihopping.py`).
- **Thermal conductivity:** Lattice κ(T) via Müller-Plathe reverse NEMD — no per-atom stress needed, works with any GRACE model (`thermal_conductivity.py`).
- **Topological Data Analysis:** Persistent homology (β₀, β₁, β₂ Betti curves) of crystal and grain boundary structures — symmetry-free structural fingerprinting (`topology.py`).
- **Topological phonons:** Weyl/Dirac phonon point detection via band inversion and Berry phase Wilson loop on each detected crossing (`topological_phonons.py`).
- **Phonon Berry phase / Zak phase:** Band-resolved Zak phases along closed BZ paths and Berry curvature Ω(k) on a 2D k-mesh — classifies phonon bands as topological or trivial (`phonon_berry.py`).
- **Moiré superlattice builder:** Commensurate twisted-bilayer supercells via Coincidence Site Lattice (CSL) method, stacking analysis (AA/AB/SP regions), and optional GRACE relaxation (`moire.py`).

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

## Li-ion Hop Detection

`lihopping.py` tracks mobile ions (Li, Na, etc.) across reference lattice sites during MD or MC trajectories, detects hop events with a sojourn-time filter (eliminates false positives from oscillating atoms), and computes per-site hop rates, mean residence times, and the hop network.

```bash
# Detect Li hops from an MD trajectory
python lihopping.py --traj XDATCAR --ref POSCAR --species Li --timestep 2.0

# With sojourn filter: only count hops where atom stays ≥3 frames
python lihopping.py --traj XDATCAR --ref POSCAR --species Li --min_sojourn 3

# Arrhenius activation energy from multiple temperatures
python lihopping.py --arrhenius --temps 600 800 1000 \
                    --trajs XDATCAR_600 XDATCAR_800 XDATCAR_1000 \
                    --ref POSCAR --species Li --timestep 2.0
```

Outputs: `hop_events.dat`, `hop_statistics.dat`, `hop_network.dat`, `arrhenius.dat`, `arrhenius.png`.

## Thermal Conductivity

`thermal_conductivity.py` computes lattice thermal conductivity κ via the Müller-Plathe reverse NEMD method. Divides the cell into slabs, periodically swaps z-velocities between hot and cold regions, and measures the resulting temperature gradient. No per-atom stress tensor required — works directly with any GRACE model.

```bash
# Basic run at 300 K
python thermal_conductivity.py --poscar POSCAR --model GRACE-2L-OAM \
                               --temperature 300 --steps 200000

# Custom slab count and swap interval
python thermal_conductivity.py --poscar POSCAR --model my_model.pb \
                               --temperature 500 --steps 300000 --nslabs 30
```

Outputs: `temperature_profile.dat`, `kappa_convergence.dat`, `kappa_summary.txt`, `temperature_profile.png`.

## Topological Data Analysis

`topology.py` fingerprints crystal and grain boundary structures using persistent homology — no crystal symmetry assumed. Computes Betti numbers β₀ (connected components), β₁ (loops/channels), β₂ (voids) as a function of distance threshold. Identifies structurally distinct regions (bulk vs GB) without needing order parameters.

Uses `scipy` by default (β₀ exact via union-find, β₁ from graph cycle rank). Install `gudhi` for full β₂ and persistence diagrams: `pip install gudhi`.

```bash
# Single structure
python topology.py --poscar POSCAR --rmax 8.0

# Compare bulk vs grain boundary
python topology.py --poscar POSCAR_bulk --compare POSCAR_gb

# Li sublattice only, from trajectory
python topology.py --xdatcar XDATCAR --species Li --stride 10 --rmax 6.0
```

Outputs: `betti_curves.dat`, `persistence_diagram.dat`, `tda_summary.txt`, `betti_curves.png`.

## Topological Phonons

`topological_phonons.py` detects Weyl and Dirac phonon points in the phonon band structure computed with GRACE. Builds on cached force-constant data from the phonon module (`phonon.*.json`).

**Physics:**
- Band crossing detection via near-degeneracy and band inversion across adjacent k-points
- Berry phase γ = -Im[ln ∏ ⟨uₙ(kⱼ)|uₙ(kⱼ₊₁)⟩] on a small closed loop around each crossing
- γ ≈ π indicates a Weyl phonon or topological nodal line

```bash
# Detect crossings along a k-path (uses cached phonon.*.json)
python topological_phonons.py --poscar POSCAR --model GRACE-2L-OAM

# Custom path and threshold
python topological_phonons.py --poscar POSCAR --model GRACE-2L-OAM \
                              --path GXMGZ --nkpts 300 --tol 0.5

# Compute Berry phase at each detected crossing
python topological_phonons.py --poscar POSCAR --model GRACE-2L-OAM --berry
```

Outputs: `phonon_topo_bands.png`, `phonon_crossings.dat`, `phonon_topo_summary.txt`.

## Phonon Berry Phase & Zak Phase

`phonon_berry.py` computes the Zak phase (Berry phase along a closed BZ path) for each phonon band and optionally maps Berry curvature Ω(k) across a 2D k-mesh.

**Physics:**
- Zak phase γₙ = 0 → trivial band; γₙ = π → topological (phononic analogue of topological insulator)
- Berry curvature Ωₙ(k) = Im[∂kx uₙ† ∂ky uₙ] computed via finite differences
- Chern number C = (1/2π) ∫∫ Ωₙ(k) dkx dky integrates the curvature over the BZ

```bash
# Zak phases for all bands (GXG = closed BZ path)
python phonon_berry.py --poscar POSCAR --model GRACE-2L-OAM

# Berry curvature on a 2D k-mesh + Chern numbers
python phonon_berry.py --poscar POSCAR --model GRACE-2L-OAM \
                       --curvature --nkx 30 --nky 30
```

Outputs: `phonon_zak.dat`, `phonon_zak.png`, optionally `phonon_berry_curv.dat`, `phonon_chern.dat`, `phonon_chern.png`, `phonon_berry_summary.txt`.

## Moiré Superlattice Builder

`moire.py` constructs commensurate twisted-bilayer supercells for moiré physics studies. Uses the Coincidence Site Lattice (CSL) method to find the exact twist angle and supercell matrix for integer (m, n) pairs.

**Physics:**
- Hexagonal lattice: cos θ = (m²+4mn+n²) / [2(m²+mn+n²)], N = m²+mn+n²
- Square lattice: cos θ = (m²-n²)/(m²+n²), N = m²+n²
- Stacking analysis classifies each top-layer atom as AA, AB, or SP

```bash
# Build twisted bilayer at (m,n) = (5,6) ≈ 13.2°
python moire.py --poscar POSCAR_monolayer --m 5 --n 6

# Scan all commensurate angles up to m=10
python moire.py --poscar POSCAR --scan --m_max 10

# Build and relax with GRACE
python moire.py --poscar POSCAR --m 5 --n 6 --gap 3.35 \
                --relax --model GRACE-2L-OAM
```

Outputs: `POSCAR_moire`, `moire_info.txt`, `stacking_map.dat`, `stacking_map.png`.

## Dislocation Builder

`dislocation.py` creates edge, screw, and mixed dislocation structures from a perfect crystal POSCAR — similar to atomsk's `--dislocation` mode and atomman's dislocation module.

**Physics implemented:**
- **Isotropic Volterra** — closed-form displacement field (screw: `u_z = b/2π arctan2(y,x)`; edge: full Hirth & Lothe formula including Poisson ratio)
- **Anisotropic Stroh formalism** — solves the 6×6 Stroh eigenvalue problem using elastic constants from `ELASTIC_Cij_GPa.dat` (produced by the elastic module in `main.py`)
- **Dipole configuration** — two dislocations of opposite sign at ±Lx/4 for full 3D periodicity

```bash
# Screw dislocation dipole (isotropic)
python dislocation.py --poscar POSCAR --type screw --burgers 2.8

# Edge dislocation with Poisson ratio
python dislocation.py --poscar POSCAR --type edge --burgers 2.8 --poisson 0.28

# Anisotropic Stroh using elastic constants from elastic module
python dislocation.py --poscar POSCAR --type edge --burgers 2.8 \
                      --method anisotropic --elastic ELASTIC_Cij_GPa.dat

# Miller index input for slip plane and line direction
python dislocation.py --poscar POSCAR --type screw --burgers 2.8 \
                      --hkl 1 1 0 --uvw 1 -1 0

# Build then relax the core structure with GRACE
python dislocation.py --poscar POSCAR --type edge --burgers 2.8 \
                      --relax --model my_grace_model.pb --fmax 0.05
```

Outputs: `POSCAR_dislocation`, `dislocation_info.txt`, optionally `dislocation_relax.traj`.

The dislocation builder draws inspiration from the following tools — please also consider citing them if you use this module:

- **atomsk** — Hirel, P. *Atomsk: A tool for manipulating and converting atomic data files.* Computer Physics Communications **197**, 212–219 (2015). https://doi.org/10.1016/j.cpc.2015.07.012 | https://atomsk.univ-lille.fr
- **atomman** — Hale, L. M. *et al.* NIST Atomistic Simulation Environment (atomman). https://github.com/usnistgov/atomman
- **E. Clouet's dislocation tools** — Clouet, E. *Dislocation modeling in solids.* https://emmanuel.clouet.free.fr — reference implementation of the Stroh anisotropic formalism and Volterra displacement field used as theoretical basis here.

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

> **Note:** I don't have a paper and i dont know i will but it was an inspiration form my work and I hope if it is useful please cite and like this repo it means a lot to me and my career.
