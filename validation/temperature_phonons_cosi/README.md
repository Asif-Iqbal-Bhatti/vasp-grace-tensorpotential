# Validation: phonon topology versus temperature in CoSi

Recorded runs and scripts behind the checks of `grace-topo-phonons --temperature`
(`phonon_tools.effective_phonons`, `topological_phonons.track_nodes`), so that anyone can verify
them. Model GRACE-2L-OAM, 3×3×3 supercell, CoSi (P2₁3) relaxed with GRACE (`POSCAR`,
V = 87.32 Å³). Runs made 2026-10-28/29 on an RTX 2000 Ada Laptop GPU. `TODO.md` (items 1–3) gives
the context and conclusions.

## Layout

| Path | Contents |
|---|---|
| `POSCAR` | CoSi structure used everywhere |
| `runs/gamma_R_vs_T_biased/` | Γ/R nodes at 0/100/300/600 K, 150 snapshots, **before** the fix (commit `82c2089`). Kept to show the bias. |
| `runs/gamma_R_vs_T_fixed/` | The same run **after** the fixes (`e8dc57f` + `c5f712f`). The reference result. |
| `runs/gamma_diagnosis/` | Logs of the Γ-anomaly diagnosis and the 100 K MD snapshots before and after the fix (`.npz`) |
| `runs/weyl_zone_search_16/` | Harmonic 16³ zone search of gap 3\|4: 95 distinct Weyl nodes (101 with symmetry copies), charges balanced |
| `scripts/control.py` | Finite differences versus symfc fits of random displacements (no MD) |
| `scripts/md_dump.py` | Runs `effective_phonons` at T and saves what the fit sees, for `refit.py` |
| `scripts/refit.py` | CPU-only refits and diagnostics of saved snapshots |
| `scripts/smoke_track_nodes.py` | CPU-only check of the one-to-one node matching |

The run directories hold the standard outputs of `grace-topo-phonons` (`run.log`,
`weyl_trajectories.dat`, `phonon_topo_vs_T.txt`, `phonon_nodes*.dat`, plots) and the harmonic
`phonopy_params.yaml`. They were made with

```bash
grace-topo-phonons --poscar POSCAR --path GXMGRX --nkpts 40 \
    --node 0 0 0 --node 0.5 0.5 0.5 --temperature 100 300 600 --md_snapshots 150   # gamma_R_vs_T_*
grace-topo-phonons --poscar POSCAR --path GXMGRX --nkpts 40 --search 16 --bands 3 3  # weyl_zone_search_16
```

## What each run shows

**1. The fit itself is sound** (`runs/gamma_diagnosis/control.log`, from `scripts/control.py`).
Finite differences at 0.005/0.015/0.03 Å give the top Γ triplet at 12.256/12.258/12.266 THz. symfc
fits of 40 random-displacement supercells from the 0 K structure give 12.259 THz at 0.01 Å. Removing
the mean translation or subtracting the forces of the perfect supercell changes nothing. Larger
amplitudes stiffen Γ (12.285 at 0.03 Å, 12.366 at 0.06 Å), so anharmonicity cannot explain a
softening. Forces on the perfect supercell are ≤ 0.001 eV/Å.

**2. Before the fix, Γ was biased low** (`runs/gamma_R_vs_T_biased/weyl_trajectories.dat`). Top Γ
triplet: 12.258 (0 K), 11.809 (100 K), 12.072 (300 K), 11.915 (600 K) THz. The drop at 100 K followed
by a rise is not physical. Half-run spreads were as small as 0.005 THz, because both halves share the
bias. The matching bug is also visible: node 1 shares bands 9–11 with node 2 at all T, and at 600 K
node 7 takes bands 4–7 from node 8 ("charges changed").

**3. Diagnosis of the cause** (`runs/gamma_diagnosis/md_100K_before_fix.npz`, `scripts/refit.py`).
Displacements and forces centred over the snapshots give the top Γ triplet at 12.217 THz instead of
11.809. The mean displacement per atom is 0.0083 Å (static offset 0.158 of the rms, 0.294 along the
top triplet). It is linear in the position in the supercell, with shear strains up to 1.4×10⁻³: the
reference cell, which `refine_symmetry` replaced with a symmetric (slightly rotated) cell, differed
from the raw NPT-average cell that the MD ran in. Using the 0 K Wyckoff positions as the reference
changes nothing, so the Wyckoff parameters are not the cause.

**4. After the fix** (`runs/gamma_diagnosis/md100_fixed.log`, `md_100K_after_fix.npz`).
`static_offset` 0.079, the 1/√150 noise level. Top Γ triplet 12.217 THz. Against the harmonic values,
max |Δν| = 0.09 (Γ), 0.05 (R), 0.10 (X), 0.07 (M) and 0.07 THz (q = (0.2, 0.1, 0)). Before the fix:
1.03, 0.09, 0.20, 0.06 and 0.44 THz. The remaining uniform softening of about 0.05 THz matches the
0.3 % thermal expansion at 100 K.

**5. Γ/R nodes versus T with the fixed code** (`runs/gamma_R_vs_T_fixed/`). All 11 nodes soften
monotonically from 0 to 600 K, stay on their own bands (no double claims, no mixed modes) and keep
charge 2. `static_offset` is 0.079–0.080 at every T. The table is in `TODO.md` item 1.

## How to check

Needs the repository on `PYTHONPATH` (or `pip install -e .`) and phonopy with symfc. From this
directory:

```bash
export PYTHONPATH=$(git rev-parse --show-toplevel)/src
# CPU only, seconds: reproduce points 3 and 4 from the saved snapshots
python scripts/refit.py runs/gamma_diagnosis/md_100K_before_fix.npz runs/gamma_R_vs_T_fixed/phonopy_params.yaml
python scripts/refit.py runs/gamma_diagnosis/md_100K_after_fix.npz  runs/gamma_R_vs_T_fixed/phonopy_params.yaml
# CPU only, a few minutes: node matching (prints PASS)
python scripts/smoke_track_nodes.py POSCAR runs/gamma_R_vs_T_fixed/phonopy_params.yaml
# GPU, minutes: point 1
python scripts/control.py
# GPU, about 20 min per run on a laptop GPU: new 100 K snapshots (md_100K.npz) with the current code
python scripts/md_dump.py 100 npt
# GPU, about 1 hour: point 5, compare with runs/gamma_R_vs_T_fixed/weyl_trajectories.dat
grace-topo-phonons --poscar POSCAR --path GXMGRX --nkpts 40 \
    --node 0 0 0 --node 0.5 0.5 0.5 --temperature 100 300 600 --md_snapshots 150
```

In `refit.py` the per-band Γ projections inside a degenerate triplet depend on the eigenvector
basis that the diagonalization returns. Compare the `triplet` lines, which do not. The MD uses
`seed=0`, but GPU force round-off makes new trajectories differ slightly, so expect agreement within
the half-run spreads, not to the last digit.

Software used: Python 3.11.8, tensorpotential 0.6.2.1, tensorflow 2.19.1 (TF32 disabled),
phonopy 3.1.0, symfc 1.5.4, spglib 2.7.0, ase 3.29.0, numpy 1.26.4, scipy 1.17.1.
