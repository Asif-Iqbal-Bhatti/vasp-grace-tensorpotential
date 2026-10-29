# TODO

Open work, in the order to pick it up.

## 1. Phonon topology at finite temperature (`grace-topo-phonons --temperature`)

**Status: implemented, experimental.** Not yet documented in the README beyond the roadmap.

### Done

- `phonon_tools.effective_phonons()`: NPT MD for the average cell at T, NVT snapshots in phonopy's supercell, symmetrized time-averaged reference structure (Wyckoff parameters can move), second-order force constants fitted by phonopy + symfc, plus two half-run fits for error bars.
- `topological_phonons.track_nodes()`: follows every node from the harmonic (0 K) result. Weyl points are re-converged from their previous position; nodes pinned by symmetry (from `--node`) are re-identified at each T by a one-to-one assignment on eigenvector overlap (item 3), because bands reorder as frequencies shift. Charges are recomputed, vanished nodes and pair annihilation are flagged, and with `--search` a full zone search runs at the highest T for new nodes.
- Outputs: `weyl_trajectories.dat`, `phonon_topo_vs_T.txt`, `phonon_nodes_T<T>.dat`, `weyl_nodes_vs_T.png`.

### Validated (CoSi, GRACE-2L-OAM, 3×3×3 supercell)

- **Comparison with harmonic at 100 K (after the item 2 fix, 150 snapshots):** effective frequencies at Γ, R, X, M and a generic q-point are within 0.05–0.10 THz of the harmonic phonopy values, a uniform softening of about 0.05 THz from thermal expansion.
- **Structure:** the space group stays P2₁3 after symmetrizing the average structure at 100, 300 and 600 K. Thermal expansion from NPT: V = 87.32 Å³ (0 K), 88.33 Å³ (300 K), 89.19 Å³ (600 K).
- **R-point charges (before the item 2 fix; to re-check):** all six fourfold nodes keep charge 2 between their band pairs up to 600 K, and the gaps inside each pair stay exactly closed. They are pinned by symmetry, so the bias should not change this.

### Open

1. **Re-run needed: every temperature result before commit "Fix static strain bias" is biased** (item 2). This includes the 150-snapshot table that was recorded here (Γ triplets 0.4–1.0 THz too low at 100 K, generic q up to 0.44 THz). The half-run spreads with 150 snapshots (mostly 0.001–0.08 THz) still hold as a noise estimate, but not as an accuracy estimate. Re-run the Γ and R check below with the fixed code; it also exercises the new node matching (item 3) on real MD.
2. **Fixed: the Γ anomaly was a static strain in the displacements.** `refine_symmetry` replaces the cell as well as the positions, so the reference structure had the symmetrized cell while the NVT MD ran in the raw NPT-average cell (shear up to 1.4×10⁻³). The displacements u therefore carried a static offset linear in position, 16 % of the thermal rms at 100 K. A fit through the origin attenuates ω² by σ²/(σ² + c²), most for coherent high-frequency Γ modes and at low T, and both half-runs share the offset, so their spread (0.005 THz) could not see it.
   - Ruled out on the way: symfc and the rigid-translation removal (random displacements of 0.01 Å from the 0 K structure reproduce finite differences at Γ to 0.001 THz), fp32 noise (|F| ≤ 0.001 eV/Å on the perfect supercell), anharmonicity (larger static amplitudes stiffen Γ: 12.26 → 12.37 THz at 0.06 Å), and the Wyckoff parameters of the reference. The mean residual forces (0.037/0.065/0.094 eV/Å at 100/300/600 K) scale as √T: finite-sample noise.
   - Fix: the NPT-average cell is symmetrized before the NVT MD, so the MD and the reference share one cell (checked to 10⁻⁶ Å), and u and F are centred over the snapshots of each fit (exact for a harmonic crystal). `static_offset` in the output reports the offset before centring; it is 0.079 at 100 K with 150 snapshots, the 1/√150 noise level.
   - Result at 100 K, 150 snapshots: max |ν − ν_harmonic| = 0.09 (Γ), 0.05 (R), 0.10 (X), 0.07 (M), 0.07 THz (generic q), against 1.03, 0.09, 0.20, 0.06 and 0.44 THz before. The remaining uniform −0.05 THz matches the 0.3 % thermal expansion. Top Γ triplet: 12.217 THz (harmonic 12.258).
3. **Fixed: node matching is now one-to-one.** Before, pinned nodes were matched to the nearest multiplet of the same size by frequency alone, so two nodes could claim one multiplet: at 100 K node 1 (Γ, bands 6–8) took bands 9–11 from node 2, and at 600 K node 7 (R, bands 0–3) took bands 4–7 from node 8 and was reported as "charges changed" although no charge changed. Now the pinned nodes at each q are assigned with the Hungarian algorithm on eigenvector-subspace overlap with the previous temperature, |V_old† V_new|²/m, frequency breaking ties. An overlap below 0.5 is reported as "modes mixed", and below 0.1, or with no free multiplet of the node's size left, the node is reported as lost. Checked on the CoSi harmonic spectra: the identity assignment holds, overlap wins over a misleading remembered frequency, and a split multiplet loses its node. Not yet re-run with MD.
4. **Weyl-point tracking has not been run yet.** Plan: a zone search of CoSi gap 3|4 at 16³, tracking at 100/300/600 K with 150 snapshots, then the new-node search at 600 K. Expect several hours. Run it alone on the GPU, after the item 1 re-run.
5. **Before calling it done:**
   - README section for `--temperature`, with method, limits (classical MD, no zero-point motion) and TDEP/symfc citations taken from their own documentation
   - a test for `effective_phonons` (short MD, check space group kept and frequencies close to harmonic)
   - remove "experimental" from the help text

Command for the Γ and R check (about 1 hour alone on the GPU):

```bash
grace-topo-phonons --poscar POSCAR --path GXMGRX --nkpts 40 \
    --node 0 0 0 --node 0.5 0.5 0.5 --temperature 100 300 600 --md_snapshots 150
```

## 2. Uncertainty on topological charges

Node charges for every member of a GRACE committee (fine-tuned or seed-varied models): how often each node and charge appears, and the spread of its position and frequency. It can reuse `node_chern` and `follow_node`.

## 3. Topological phase diagrams under strain or pressure

Relax with GRACE along a strain or pressure path and follow nodes with `follow_node`, as the temperature tracking does, flagging creation and annihilation.

## Practical notes

- **One GPU:** run GRACE jobs one after another, not in parallel. Parallel runs compete for the single GPU and the CPU, and every job is slower.
- `TF_FORCE_GPU_ALLOW_GROWTH` is set in `calculator.py`, so a job no longer reserves all GPU memory. CPU-bound parts (Chern numbers, node search) still use all cores through numpy.
