<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" alt="vasp-grace logo: a 3×3 lattice with one bonded path" width="96">
  </picture>
</p>

# vasp-grace-tensorpotential

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-blue)](pyproject.toml)
[![Website](https://img.shields.io/badge/docs-website-0b7285)](https://asif-iqbal-bhatti.github.io/vasp-grace-tensorpotential/)

**Run your VASP inputs on a GRACE machine-learning potential, then study defects, ion transport and topological phonons with the same model.**

vasp-grace reads `POSCAR` and `INCAR`, computes energies, forces and stresses with a [GRACE](https://doi.org/10.1103/PhysRevX.14.021036) model (TensorFlow, GPU accelerated), and writes `OUTCAR`, `OSZICAR`, `CONTCAR` and `XDATCAR` the way VASP does. Eleven command-line tools build on it.

**Website:** <https://asif-iqbal-bhatti.github.io/vasp-grace-tensorpotential/>

---

## What do you want to do?

| I want to… | Command | More |
|---|---|---|
| Run my VASP inputs (relax, MD, phonons, elastic constants) on a potential | `vasp-grace` | [VASP drop-in](#vasp-drop-in) |
| Find structures the model has not seen, to label with DFT | `grace-active-learning` | [Uncertainty](#uncertainty-when-to-trust-the-model) |
| Sample site disorder or swaps with Monte Carlo | `grace-mc` | [Uncertainty](#uncertainty-when-to-trust-the-model) |
| Build an edge, screw or mixed dislocation | `grace-dislocation` | [Dislocations](#dislocations) |
| Find ion hops and an activation energy from MD | `grace-lihopping` | `--help` |
| Compute lattice thermal conductivity | `grace-kappa` | `--help` |
| Know whether a crystal can be a band insulator, and whether that survives soft phonons and temperature | `grace-symmetry` | [Symmetry and chirality](#band-insulator-verdict-and-chirality) |
| Find phonon Weyl points and their topological charges | `grace-topo-phonons` | [Phonon Weyl points](#phonon-weyl-points) |
| Compute phonon Zak phases and Berry curvature | `grace-phonon-berry` | [Phonon Weyl points](#phonon-weyl-points) |
| Follow phonon nodes and their charges with temperature | `grace-topo-phonons --temperature` | [Phonon nodes versus temperature](#phonon-nodes-versus-temperature) |
| Fingerprint a structure without using symmetry (grain boundaries, amorphous regions) | `grace-topology` | `--help` |
| Build a twisted bilayer at a commensurate angle | `grace-moire` | `--help` |

Every command takes `--model` (any GRACE model, see [Choosing a model](#choosing-a-grace-model)) and `--help`.

---

## Install

```bash
git clone https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential
cd vasp-grace-tensorpotential
pip install -e .
```

Optional extras, installed the same way (`pip install -e ".[condmat]"`):

| Extra | Adds | Needed for |
|---|---|---|
| `condmat` | [symlint](https://github.com/Asif-Iqbal-Bhatti/symlint), [phonopy](https://phonopy.github.io/phonopy/) | band-insulator verdicts in `grace-symmetry` (Python ≥ 3.10); symmetrized force constants in the phonon topology tools |
| `tda` | gudhi | full β₁/β₂ persistent homology in `grace-topology` |
| `examples` | marimo | the interactive notebook |
| `dev` | pytest, black, pylint | development |

The first use of a GRACE foundation model downloads it to `~/.cache/grace/`.

---

## Quick start

**1. Run a VASP-style calculation.** Use your usual `POSCAR` and an `INCAR` with one extra tag:

```
GRACE_MODEL = GRACE-2L-OAM
IBRION = 2        # relax with FIRE
ISIF   = 3        # positions and cell
NSW    = 200
EDIFFG = -0.01
```

```bash
vasp-grace --poscar POSCAR --incar INCAR
```

You get `OUTCAR`, `OSZICAR` and `CONTCAR` as from VASP, so existing scripts keep working.

**2. Analyse the result with a tool:**

```bash
grace-symmetry     --poscar CONTCAR                 # band-insulator verdict, soft modes, chirality
grace-topo-phonons --poscar CONTCAR --node 0 0 0    # charges of the phonon nodes at Γ
```

**3. Or explore interactively:**

```bash
pip install -e ".[examples]"
marimo edit examples/quickstart.py
```

The notebook relaxes a crystal, fits its equation of state and tries the analysis tools with live controls. For fcc Cu with `GRACE-2L-OAM` it gives V₀ = 11.84 Å³ and B₀ ≈ 142 GPa.

---

## How it fits together

```
POSCAR + INCAR ──► vasp-grace ──► OUTCAR · CONTCAR · XDATCAR · ELASTIC_Cij_GPa.dat
                       │
      ┌────────────────┼──────────────────────────┐
      ▼                ▼                          ▼
 Structure and     Ion dynamics            Condensed matter (condmat/)
 uncertainty       and transport           ─────────────────────────────
 ───────────────   ──────────────          grace-symmetry      verdict, chirality
 active-learning   lihopping               grace-topo-phonons  Weyl points, charges
 mc                kappa                   grace-phonon-berry  Zak phase, curvature
 dislocation                               grace-topology      persistent homology
                                           grace-moire         twisted bilayers
```

Outputs feed each other. For example, `vasp-grace` with `IBRION = 6, ISIF = 3` writes the elastic tensor that `grace-dislocation --method anisotropic` reads.

---

## Choosing a GRACE model

`--model` (or `GRACE_MODEL` in the `INCAR`) accepts any GRACE foundation model or a path to your own fine-tuned model. The default is `GRACE-2L-OAM`. Run `grace_models list` to see all models and which are downloaded.

| Model | When to use it |
|---|---|
| `GRACE-2L-OAM` | Default. Energies line up with Materials Project–style PBE data |
| `GRACE-2L-OMAT` | Fitted on OMat24 alone |
| `GRACE-2L-OMAT-medium-*`, `-large-*`, `GRACE-2L-SMAX-*` | Newer models with built-in per-atom uncertainty (`--calm`) |
| `GRACE-1L-*`, `GRACE-FS-*` | Faster, for long MD on large cells |
| any `…-fp64` | Double precision, for phonons and elastic constants where finite differences amplify noise |

<details>
<summary>How GRACE model names are built</summary>

Names follow `GRACE-<architecture>-<training data>[-<size>][-<variant>][-fp64]`:

| Part | Options | Meaning |
|---|---|---|
| Architecture | `FS`, `1L`, `2L`, `3L` | `FS` (Finnis–Sinclair-like) and `1L` are local single-layer models. `2L` adds a second message-passing layer (semi-local). `3L` is the three-layer generation. |
| Training data | `OAM`, `OMAT`, `SMAX`, `SMAX-OMAT` | `OAM`: pre-fitted on OMat24, fine-tuned on sAlex + MPtraj. `OMAT`: OMat24 only. `SMAX`, `SMAX-OMAT`: newer training sets (`grace_models info <name>`). |
| Size | `medium`, `large` | Larger is more accurate and slower. |
| Variant | `base`, `ft-E`, `ft-AM` | The base model and its fine-tuned variants. |
| Precision | `-fp64` | Full double precision; the bare name is fp32. |

The foundation models are released under the Academic Software License.

</details>

---

## Uncertainty: when to trust the model

`grace-active-learning` and `grace-mc` flag structures the model has not seen, so you can label them with DFT and fine-tune.

| | CALM (`--calm MODEL`) | Committee (`--models m1 m2 …`) |
|---|---|---|
| Needs | One model with built-in uncertainty | Two or more models |
| Signal | Per-atom extrapolation grade γ; γ > 1 means outside the training data | Spread of predicted forces (eV/Å) |
| Default threshold | γ > 1 | σ_F > 0.1 eV/Å |
| Cost | One evaluation | One per model |

```bash
grace-active-learning --xdatcar XDATCAR --calm GRACE-2L-OMAT-medium-ft-AM --stride 10
grace-mc --poscar POSCAR --model GRACE-2L-OMAT-medium-ft-AM --calm --swap Cl S --temperature 800
```

Both write `uncertainty_log.dat` and `flagged/POSCAR_*`. Example: on the 108-atom argyrodite cell in `tests/POSCAR` with `GRACE-2L-OMAT-medium-base`, max γ is 0.85 for the relaxed cell and 1.50 and 3.51 after random displacements of 0.3 and 0.6 Å, so only the distorted cells are flagged. In Python, `load_calculator(name, uq=True)` from `vasp_grace.calculator` gives per-atom γ in `calc.results["gamma"]`.

---

## VASP drop-in

`vasp-grace` reads the tags below and ignores the rest, so an existing `INCAR` works unchanged.

| To run | Set |
|---|---|
| Single point | `NSW = 0` (or `IBRION = -1`) |
| Relaxation | `IBRION = 1` (LBFGS) or `2` (FIRE); `ISIF = 3` relaxes the cell too |
| Molecular dynamics | `IBRION = 0`, `MDALGO`, `TEBEG`, `POTIM`; `ISIF ≥ 3` gives NPT |
| Phonons | `IBRION = 5` or `6`, `LPHON_DISPERSION = .TRUE.`, `PHON_DOS = 1` |
| Elastic constants | `IBRION = 6` with `ISIF = 3` |

<details>
<summary>Full INCAR reference</summary>

| Tag | Default | Effect |
|---|---|---|
| `GRACE_MODEL` | `GRACE-2L-OAM` | Model name or path |
| `IBRION` | `-1` | `-1` single point · `0` MD · `1` LBFGS · `2` FIRE · `5`/`6` finite-displacement phonons · `7`/`8` not supported (DFPT) |
| `NSW` | `0` | Ionic or MD steps; `0` gives a single point |
| `ISIF` | `3` | Relaxation: `3` positions + cell, `4` cell shape at fixed volume, otherwise positions only. MD: `≥3` switches to NPT. Phonons: `≥3` also computes the elastic tensor |
| `EDIFFG` | `-0.01` | Negative: force criterion in eV/Å; positive values fall back to 0.05 eV/Å |
| `POTIM` | `1.0` fs | MD time step |
| `TEBEG` | `300` | MD temperature (K) |
| `MDALGO` | `0` | `0` NVE · `1` Andersen · `2` Berendsen · `3` Langevin. NPT uses the MTK barostat at 1 bar |
| `NFREE` | `2` | Central differences |
| `LPHON_DISPERSION` | `.FALSE.` | Write the phonon band structure (`phonon_band.png`) |
| `PHON_DOS` | `0` | `> 0` writes the phonon DOS; mesh from `QPOINTS`/`KPOINTS` if present |
| `PHON_NEDOS`, `PHON_SIGMA` | `1000`, `0.001` | DOS grid points and smearing |

Phonons displace the `POSCAR` you give, so provide a supercell for a supercell phonon calculation. Elastic constants are fitted with [jochym/Elastic](https://github.com/jochym/Elastic) and written to `ELASTIC_Cij_GPa.dat`, `ELASTIC_Bij_GPa.dat` and `ELASTIC_fit_info.txt`.

</details>

---

## Dislocations

```bash
grace-dislocation --poscar POSCAR --type screw --burgers 2.8 --relax                     # isotropic
grace-dislocation --poscar POSCAR --type edge  --burgers 2.8 --method anisotropic \
                  --elastic ELASTIC_Cij_GPa.dat                                          # Stroh
```

<details>
<summary>Method</summary>

| Method | Input | Displacement field |
|---|---|---|
| Isotropic Volterra | Poisson ratio ν | Screw: `u_z = b/2π·arctan2(y,x)` · Edge: Hirth & Lothe |
| Anisotropic Stroh | `ELASTIC_Cij_GPa.dat` (from `vasp-grace`, any crystal class) | 6×6 Stroh eigenvalue problem |

Dipoles (±b at ±Lx/4, the default) keep 3D periodicity; `--single` builds an isolated dislocation. Inspired by [atomsk](https://atomsk.univ-lille.fr), [atomman](https://github.com/usnistgov/atomman) and [E. Clouet's tools](https://emmanuel.clouet.free.fr).

</details>

---

## Condensed matter

The tools in `src/vasp_grace/condmat/` study what the lattice alone can tell you about electronic and phononic topology.

### Band-insulator verdict and chirality

```bash
grace-symmetry --poscar POSCAR                    # relax, phonons, verdict, chirality
grace-symmetry --poscar POSCAR --md 300 --handoff # add 300 K MD and VASP + SOC inputs
```

A nonmagnetic crystal can be a band insulator only if its electron count per primitive cell fits its space group (Watanabe, Po, Zaletel and Vishwanath, PRL 117, 096404 (2016)); [symlint](https://github.com/Asif-Iqbal-Bhatti/symlint) applies that rule. The verdict is only as good as the space group, and `grace-symmetry` tests the space group with GRACE: after relaxation, along every soft phonon mode, and in the time-averaged structure at temperature. It also reports structural chirality and writes VASP + SOC inputs with an IrRep script for the DFT check.

<details>
<summary>Method and checks</summary>

1. **Relaxation** with the space group held fixed, then the verdict for the relaxed primitive cell.
2. **Soft phonons** at Γ and at the zone-boundary points of an n×n×n supercell. Each imaginary mode is followed into its distorted supercell (true phonon eigen-displacements, checked against F = −mω²u) and the rule is applied again. For degenerate modes the highest-symmetry direction is used. Zone-boundary modes multiply the cell and change the electron count: the Peierls / CDW route to opening a gap.
3. **Temperature** (`--md T`): the space group of the time-averaged structure, with a tolerance above its statistical noise. No space group is given when the average is too noisy or atoms diffuse.
4. **Chirality**: Sohncke-group test and the enantiomer's space group.
5. **CALM γ** on every structure for models with built-in uncertainty.
6. **DFT hand-off** (`--handoff`): the IrRep script reads `NELECT` from the SCF `OUTCAR` and extends the band window past the Fermi level.

| Structure (GRACE-2L-OAM) | Result |
|---|---|
| Si, Fd-3m | ν = 28 allowed; still Fd-3m on average at 300 K |
| Te, P3₁21 | ν = 156 allowed; chiral, enantiomer P3₂21 |
| SrIrO₃, Pnma | ν = 556 symmetry-enforced gapless; no soft modes at the checked q-points |
| SrTiO₃, cubic | ferroelectric Γ mode (−3.6 THz → Cm) and octahedral-rotation R mode (−2.1 THz → R-3c, cell doubles) |
| Na, simple cubic (ν = 11) | soft X and M modes double the cell to P2₁/m and P4/nmm, where ν = 22 is still forbidden |

Limits: for degenerate soft modes the highest-symmetry subgroup can be missed (cubic SrTiO₃'s R mode gives R-3c rather than I4/mcm), and only q-points commensurate with the chosen supercell are checked.

</details>

### Phonon Weyl points

```bash
grace-topo-phonons --poscar POSCAR --node 0 0 0 --node 0.5 0.5 0.5   # charges at Γ and R
grace-topo-phonons --poscar POSCAR --search 10 --bands 3 8           # search the whole zone
grace-phonon-berry --poscar POSCAR --curvature                       # Berry curvature on a plane
```

`grace-topo-phonons` gives each phonon band crossing its topological charge, the Chern number on a small sphere around it: ±1 for a Weyl point, ±2 for a double-Weyl or spin-1 node, 0 for a crossing that is not a monopole. Results go to `phonon_nodes.dat`. For CoSi it finds the spin-1 nodes at Γ (|C| = 2) and the charge-2 fourfold nodes at R, the double-Weyl phonons reported by Zhang et al., PRL 120, 016401 (2018) and observed by Miao et al., PRL 121, 035302 (2018).

<details>
<summary>Method, checks and limits</summary>

- **Force constants** come from [phonopy](https://phonopy.github.io/phonopy/) when installed (`--fc phonopy`, the default then): symmetry-reduced displacements, projected onto the space group, and every dynamical matrix from phonopy's `get_dynamical_matrix_at_q`. Symmetry-related nodes are then exactly equivalent (for CoSi the gap at all 24 images of a Weyl node is 10⁻¹⁰ THz, against 0.03 THz with ASE finite differences). `--fc ase` uses ASE instead.
- **Chern numbers** use the gauge-invariant lattice method of Fukui, Hatsugai and Suzuki, J. Phys. Soc. Jpn. 74, 1674 (2005). A charge counts only when it is the same integer on spheres of radius r and r/2; otherwise the crossing is reported as unresolved (a nodal line or surface, or two nodes very close together), never as a Weyl point. For nodes where bands stay paired along lines (screw axes), the charge is read from the gaps between the pairs.
- **Search** (`--search N`): an N³ scan of the zone, refined with Nelder–Mead, plus the symmetry copies of every node (space-group rotations and time reversal). The charges in each gap must sum to zero over the zone (Nielsen–Ninomiya); the summary reports this per gap.
- **Checks:** analytic models give exactly ±1 (Weyl), ±2 (double Weyl), (+2, 0, −2) (spin-1), and a two-node lattice model is found complete with charges summing to zero. CoSi (a = 4.436 Å with GRACE-2L-OAM, 4.443 Å measured) gives |C| = 2 at Γ and charge 2 at R; generic Weyl points (±1) appear in symmetry-related sets with charges stable from r = 0.002 to 0.01 Å⁻¹. `grace-phonon-berry` gives exactly 0 for all 24 CoSi bands on the k_z = 0 plane, as time reversal requires.
- **Zone-wide balance:** with phonopy force constants, the symmetry copies of a node coincide with nodes found independently, so the balance is a real completeness check. For CoSi, gap 3|4 sums to zero with a 16³ search; a 10³ search misses one 12-member orbit and reports +12, and capping the refined gap minima at 60 per gap (`--max_seeds 60`, the old default) reports −12. If a gap does not close, search on a denser mesh or raise `--max_seeds`. In space groups with nodal surfaces (P2₁3 pairs bands across the zone faces) the gaps inside those pairs need not balance from point nodes alone. With `--fc ase` the copies sit 0.005–0.01 Å⁻¹ apart and the balance is unreliable.
- **Sign convention:** Bloch waves exp(+iq·R) (phonopy's), Berry connection A = i⟨u|∇u⟩, outward normal; the lower band of H = k·σ has C = +1. Both `--fc` backends use it.

</details>

### Phonon nodes versus temperature

```bash
grace-topo-phonons --poscar POSCAR --node 0 0 0 --node 0.5 0.5 0.5 \
    --temperature 100 300 600                            # follow the Γ and R nodes
grace-topo-phonons --poscar POSCAR --search 16 --bands 3 3 \
    --temperature 100 300 600                            # follow every Weyl point found
```

`--temperature` replaces the harmonic force constants by effective ones fitted from GRACE molecular dynamics at each T, the idea of the temperature-dependent effective potential (TDEP) method of Hellman et al., Phys. Rev. B 87, 104111 (2013); the fit is done by phonopy with [symfc](https://github.com/symfc/symfc) (A. Seko and A. Togo, Phys. Rev. B 110, 214302 (2024)). Every node is then followed from 0 K, its charge recomputed, and annihilation, merging and new nodes are reported. Results go to `weyl_trajectories.dat`, `phonon_topo_vs_T.txt`, `phonon_nodes_T<T>.dat` and `weyl_nodes_vs_T.png`. symfc is not installed with vasp-grace: `pip install symfc`.

<details>
<summary>Method, checks and limits</summary>

- **Effective force constants:** NPT MD (zero pressure) gives the average cell at T; NVT MD in phonopy's supercell of that cell gives `--md_snapshots` displacement–force pairs (default 150, 200 fs apart). The time-averaged structure, symmetrized back onto the space group, is the reference, so Wyckoff parameters can move with T. Displacements and forces are centred over the snapshots before the fit, which removes a static offset between the reference and the true mean (`static_offset` in the output; it should sit at the noise level, about 1/√N for N snapshots). Two more fits, from each half of the snapshots, give the error bars.
- **Tracking:** nodes pinned by symmetry (`--node`) are matched one-to-one at each T by eigenvector overlap, since bands reorder as frequencies shift. Other nodes are re-converged from their previous position. A node with another node inside its charge sphere is reported as annihilated (charge 0 next to an opposite charge), merged (it landed on another node) or unresolved; a node with charge 0 and nothing inside the sphere has vanished. With `--search`, the zone is searched again at the highest T for new nodes.
- **Checks (CoSi, 3×3×3, GRACE-2L-OAM):** at 100 K the effective frequencies at Γ, R, X, M and a generic q are within 0.05–0.10 THz of harmonic, a uniform softening that matches the thermal expansion. All Γ and R nodes soften monotonically to 600 K and keep charge 2, with three MD seeds and also with GRACE-2L-OMAT-large-ft-AM-fp64. The half-run error bars match the scatter between MD seeds. Runs and scripts are in [`validation/`](validation/README.md).
- **Limits:** the MD is classical, so there is no zero-point motion and low-T results approach the harmonic force constants, not the quantum ground state. Effective harmonic force constants describe the average lattice, not phonon lifetimes. Generic Weyl points that lie within a few hundredths of 1/Å of each other are not resolved once they move by as much at higher T: in CoSi the gap-3|4 Weyl points reorganize between 300 and 600 K in a way that changes with the MD seed. Two GRACE models also differ by as much as the whole 0–600 K softening (2–4 %), so compare temperatures within one model, and treat absolute frequencies with the accuracy of the model. TDEP itself is a separate code (F. Knoop et al., J. Open Source Softw. 9, 6150 (2024)); vasp-grace does not use it.

</details>

---

## What a potential can and cannot tell you

GRACE models the energy surface of the nuclei: energies, forces and stresses.

| In reach | Needs DFT |
|---|---|
| Structures, relaxations, equations of state | Electronic band structures and their topology (Z₂, Chern) |
| Phonons, elastic constants, lattice instabilities | Electron–phonon coupling and superconducting T<sub>c</sub> |
| Diffusion, ion hopping, thermal conductivity | Magnetism |
| Defects, dislocations, grain boundaries, twisted bilayers | Energy differences below a few meV/atom without fine-tuning |
| Phonon topology and structural chirality | |

Use vasp-grace to screen and pre-relax structures, then hand them to VASP, Wannier90 or EPW (`grace-symmetry --handoff` writes the inputs).

---

## Further reading

New to topology in condensed matter? These cover the ideas behind the condensed-matter tools: Berry phases, Chern numbers, Weyl points and symmetry-protected band crossings.

- **[Topology in condensed matter: tying quantum knots](https://topocondmat.org/)**: a free online course with lecture notes, videos and interactive code.
- **B. A. Bernevig with T. L. Hughes, *Topological Insulators and Topological Superconductors*** (Princeton University Press, 2013): a graduate textbook.
- **S. M. Girvin and K. Yang, *[Modern Condensed Matter Physics](https://www.cambridge.org/highereducation/books/modern-condensed-matter-physics/F0A27AC5DEA8A40EA6EA5D727ED8B14E)*** (Cambridge University Press, 2019): a graduate textbook covering lattice dynamics and phonons, band theory, Berry phases and topological phases.

---

## Roadmap

- **Uncertainty on topological charges**: node charges across a committee of GRACE models.
- **Topological phase diagrams under strain and pressure**, with automatic tracking of node creation and annihilation.
- **Phonon surface states** from surface Green's functions.
- **Chiral phonons**: phonon angular momentum and circular polarisation.
- **Allen–Dynes T<sub>c</sub> helper** combining DFT α²F(ω) or λ with GRACE phonons.

Suggestions and pull requests are welcome in the [issue tracker](https://github.com/Asif-Iqbal-Bhatti/vasp-grace-tensorpotential/issues).

---

## Project layout and tests

<details>
<summary>Repository layout</summary>

```
vasp-grace-tensorpotential/
├── src/vasp_grace/
│   ├── calculator.py           shared GRACE model loader, CALM uncertainty
│   ├── main.py                 vasp-grace (VASP drop-in)
│   ├── active_learning.py      grace-active-learning
│   ├── montecarlo.py           grace-mc
│   ├── dislocation.py          grace-dislocation
│   ├── lihopping.py            grace-lihopping
│   ├── thermal_conductivity.py grace-kappa
│   └── condmat/
│       ├── symmetry_stability.py   grace-symmetry
│       ├── chirality.py
│       ├── topological_phonons.py  grace-topo-phonons
│       ├── phonon_berry.py         grace-phonon-berry
│       ├── phonon_tools.py         eigen-solver, Chern numbers, node search, phonopy backend
│       ├── topology.py             grace-topology
│       └── moire.py                grace-moire
├── examples/quickstart.py      interactive marimo notebook
├── docs/                       website (GitHub Pages) and logo
└── tests/test_modules.py       test suite
```

</details>

```bash
python tests/test_modules.py
```

The suite checks every module's syntax and `--help`, runs the structure-only tools, and with `tensorpotential` installed runs GRACE end to end, including analytic Chern-number models and CoSi phonon topology.

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

<details>
<summary>Please also cite the methods you use</summary>

- Bochkarev, A., Lysogorskiy, Y. & Drautz, R. Graph atomic cluster expansion for semilocal interactions beyond equivariant message passing. *Phys. Rev. X* **14**, 021036 (2024). https://doi.org/10.1103/PhysRevX.14.021036
- Bochkarev, A. *et al.* Efficient parametrization of the atomic cluster expansion. *Phys. Rev. Mater.* **6**, 013804 (2022). https://doi.org/10.1103/PhysRevMaterials.6.013804
- Larsen, A. H. *et al.* The atomic simulation environment. *J. Phys.: Condens. Matter* **29**, 273002 (2017). https://doi.org/10.1088/1361-648X/aa680e
- Togo, A., Chaput, L., Tadano, T. & Tanaka, I. Implementation strategies in phonopy and phono3py. *J. Phys.: Condens. Matter* **35**, 353001 (2023). https://doi.org/10.1088/1361-648X/acd831 (phonon topology tools)
- Fukui, T., Hatsugai, Y. & Suzuki, H. Chern numbers in discretized Brillouin zone. *J. Phys. Soc. Jpn.* **74**, 1674 (2005). https://doi.org/10.1143/JPSJ.74.1674 (Chern numbers)
- Watanabe, H., Po, H. C., Zaletel, M. P. & Vishwanath, A. Filling-enforced gaplessness in band structures of the 230 space groups. *Phys. Rev. Lett.* **117**, 096404 (2016). https://doi.org/10.1103/PhysRevLett.117.096404 (`grace-symmetry`)
- Müller-Plathe, F. A simple nonequilibrium molecular dynamics method for calculating the thermal conductivity. *J. Chem. Phys.* **106**, 6082 (1997). https://doi.org/10.1063/1.473271 (`grace-kappa`)

</details>

If this work is useful to you, a star and a citation are much appreciated.

---

## License

MIT. See [LICENSE](LICENSE).
