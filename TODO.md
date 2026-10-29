# TODO

Open work, in the order to pick it up.

## 1. Phonon topology at finite temperature (`grace-topo-phonons --temperature`)

**Status: implemented and documented** (README, "Phonon nodes versus temperature"). Validated for symmetry-pinned nodes and the overall softening; generic Weyl points are not resolved at 300–600 K (item 4).

### Done

- `phonon_tools.effective_phonons()`: NPT MD for the average cell at T, NVT snapshots in phonopy's supercell, symmetrized time-averaged reference structure (Wyckoff parameters can move), second-order force constants fitted by phonopy + symfc, plus two half-run fits for error bars.
- `topological_phonons.track_nodes()`: follows every node from the harmonic (0 K) result. Weyl points are re-converged from their previous position; nodes pinned by symmetry (from `--node`) are re-identified at each T by a one-to-one assignment on eigenvector overlap (item 3), because bands reorder as frequencies shift. Charges are recomputed, vanished nodes and pair annihilation are flagged, and with `--search` a full zone search runs at the highest T for new nodes.
- Outputs: `weyl_trajectories.dat`, `phonon_topo_vs_T.txt`, `phonon_nodes_T<T>.dat`, `weyl_nodes_vs_T.png`.

### Validated (CoSi, GRACE-2L-OAM, 3×3×3 supercell)

- **Comparison with harmonic at 100 K (after the item 2 fix, 150 snapshots):** effective frequencies at Γ, R, X, M and a generic q-point are within 0.05–0.10 THz of the harmonic phonopy values, a uniform softening of about 0.05 THz from thermal expansion.
- **Structure:** the space group stays P2₁3 after symmetrizing the average structure at 100, 300 and 600 K. Thermal expansion from NPT: V = 87.32 Å³ (0 K), 88.33 Å³ (300 K), 89.19 Å³ (600 K).
- **Γ and R charges:** all five threefold Γ nodes and all six fourfold R nodes keep charge 2 up to 600 K, with the fixed code (item 1).

### Open

1. **Γ and R nodes re-checked with the fixed code (items 2 and 3): done.** CoSi, 150 snapshots per T, command below (about 1 hour alone on the GPU). Results from before commit "Fix static strain bias" (e8dc57f) are biased and replaced by these.
   - Every node softens monotonically with T (THz, ± half-run spread):

     | Node | 0 K | 100 K | 300 K | 600 K |
     |---|---|---|---|---|
     | Γ, bands 6–8 | 6.425 | 6.376 ± 0.032 | 6.258 ± 0.037 | 6.093 ± 0.015 |
     | Γ, bands 9–11 | 6.926 | 6.876 ± 0.003 | 6.764 ± 0.014 | 6.634 ± 0.040 |
     | Γ, bands 12–14 | 8.791 | 8.736 ± 0.060 | 8.583 ± 0.121 | 8.391 ± 0.219 |
     | Γ, bands 17–19 | 9.909 | 9.860 ± 0.017 | 9.743 ± 0.024 | 9.601 ± 0.043 |
     | Γ, bands 21–23 | 12.258 | 12.217 ± 0.050 | 12.075 ± 0.087 | 11.911 ± 0.097 |
     | R, bands 0–3 | 5.101 | 5.053 ± 0.007 | 4.944 ± 0.017 | 4.807 ± 0.014 |
     | R, bands 4–7 | 5.263 | 5.222 ± 0.005 | 5.133 ± 0.019 | 5.028 ± 0.043 |
     | R, bands 8–11 | 7.249 | 7.204 ± 0.018 | 7.094 ± 0.031 | 6.950 ± 0.029 |
     | R, bands 12–15 | 7.738 | 7.720 ± 0.001 | 7.615 ± 0.022 | 7.481 ± 0.047 |
     | R, bands 16–19 | 9.970 | 9.959 ± 0.083 | 9.860 ± 0.114 | 9.733 ± 0.092 |
     | R, bands 20–23 | 12.513 | 12.474 ± 0.079 | 12.331 ± 0.104 | 12.140 ± 0.097 |

   - All 11 nodes stay on their own bands (no double claims, no "modes mixed"), and every charge is unchanged up to 600 K, as symmetry requires. Static offset before centring: 0.079–0.080 of the rms at all T, the noise level.
   - Remaining noise: spreads up to 0.22 THz (Γ, bands 12–14, 600 K), so the 100 K shifts (about 0.05 THz) are often within error. 150 snapshots stays the default; more snapshots would be needed to resolve small shifts.
2. **Fixed: the Γ anomaly was a static strain in the displacements.** `refine_symmetry` replaces the cell as well as the positions, so the reference structure had the symmetrized cell while the NVT MD ran in the raw NPT-average cell (shear up to 1.4×10⁻³). The displacements u therefore carried a static offset linear in position, 16 % of the thermal rms at 100 K. A fit through the origin attenuates ω² by σ²/(σ² + c²), most for coherent high-frequency Γ modes and at low T, and both half-runs share the offset, so their spread (0.005 THz) could not see it.
   - Ruled out on the way: symfc and the rigid-translation removal (random displacements of 0.01 Å from the 0 K structure reproduce finite differences at Γ to 0.001 THz), fp32 noise (|F| ≤ 0.001 eV/Å on the perfect supercell), anharmonicity (larger static amplitudes stiffen Γ: 12.26 → 12.37 THz at 0.06 Å), and the Wyckoff parameters of the reference. The mean residual forces (0.037/0.065/0.094 eV/Å at 100/300/600 K) scale as √T: finite-sample noise.
   - Fix: the NPT-average cell is symmetrized before the NVT MD, so the MD and the reference share one cell (checked to 10⁻⁶ Å), and u and F are centred over the snapshots of each fit (exact for a harmonic crystal). `static_offset` in the output reports the offset before centring; it is 0.079 at 100 K with 150 snapshots, the 1/√150 noise level.
   - Result at 100 K, 150 snapshots: max |ν − ν_harmonic| = 0.09 (Γ), 0.05 (R), 0.10 (X), 0.07 (M), 0.07 THz (generic q), against 1.03, 0.09, 0.20, 0.06 and 0.44 THz before. The remaining uniform −0.05 THz matches the 0.3 % thermal expansion. Top Γ triplet: 12.217 THz (harmonic 12.258).
3. **Fixed: node matching is now one-to-one.** Before, pinned nodes were matched to the nearest multiplet of the same size by frequency alone, so two nodes could claim one multiplet: at 100 K node 1 (Γ, bands 6–8) took bands 9–11 from node 2, and at 600 K node 7 (R, bands 0–3) took bands 4–7 from node 8 and was reported as "charges changed" although no charge changed. Now the pinned nodes at each q are assigned with the Hungarian algorithm on eigenvector-subspace overlap with the previous temperature, |V_old† V_new|²/m, frequency breaking ties. An overlap below 0.5 is reported as "modes mixed", and below 0.1, or with no free multiplet of the node's size left, the node is reported as lost. Checked on the CoSi harmonic spectra: the identity assignment holds, overlap wins over a misleading remembered frequency, and a split multiplet loses its node. Confirmed on MD in the item 1 re-run.
4. **Weyl points in gap 3|4 tracked (H200, Sofia HPC): softening is robust, the 600 K topology is not.** Runs and scripts in `validation/` (see `validation/README.md`).
   - **Zone search needs `--max_seeds 300`.** With the default of 60 the 16³ search of gap 3|4 found 37 distinct nodes and the charges summed to −12; with 300 it finds 82 distinct nodes (88 with symmetry copies) and the balance is zero. (The laptop search reported 95/101 from the same 82 crossing points; the difference is in the merging of crossings into distinct nodes and is not resolved.)
   - **0–300 K (3 MD seeds, 150 snapshots):** at 100 K all 82 nodes keep their charge in every seed; the static offset is 0.079–0.089 (noise level) at every T. Mean shift from 0 K: −0.03 THz (100 K), −0.13 THz (300 K), −0.23 THz (600 K). The half-run error bars match the scatter between seeds (median 0.013 vs 0.008 THz at 100 K, 0.024 vs 0.029 THz at 300 K), so they are a fair noise estimate.
   - **300–600 K: the gap-3|4 nodes reorganize, and which nodes do what depends on the MD seed.** Several orbits that start 0.046 Å⁻¹ apart at 0 K close in (0.010 Å⁻¹ at 300 K). Seed 0: 12 +/− pairs annihilate at 600 K, 9 nodes re-converge onto other nodes, 12 have a neighbour inside the sphere. Seed 1: 29 nodes annihilate already at 300 K, and 21 new nodes appear at 600 K. Seed 2: nothing at 300 K, and 45 nodes are left at charge-0 gap minima at 600 K. Only 8/82 nodes (on the (111) diagonals, 5.17 THz at 0 K) are intact at 600 K in all three seeds. Statement that holds: the gap-3|4 Weyl nodes near the k_x = 0 planes annihilate and re-form between 300 and 600 K; the details are below the resolution (positions ±0.01–0.03 Å⁻¹, frequencies ±0.1 THz at 600 K).
   - **Model dependence is as large as the temperature effect.** GRACE-2L-OMAT-large-ft-AM-fp64 is 2–4 % stiffer than GRACE-2L-OAM (R nodes 5.29/5.52 vs 5.10/5.26 THz, top Γ triplet 12.52 vs 12.26 THz), as much as the whole 0→600 K softening. With OMAT-large (300 snapshots, static offset 0.053–0.054 at every T, V = 87.03/87.31/88.25/88.91 Å³ at 0/100/300/600 K) the qualitative picture is the same: all 11 Γ/R nodes soften monotonically and keep charge 2 up to 600 K (Γ bands 12–14 unresolved at 600 K: a neighbouring gap closes on the sphere), somewhat more than with OAM (top Γ triplet 12.52 → 12.07 THz against 12.26 → 11.91; lowest R node 5.29 → 4.93 against 5.10 → 4.81). Gap 3|4 (16³ search, 71 Weyl nodes, balance +7 with 5 unresolved crossings already at 0 K): all intact at 100 K (mean −0.04 THz), 52/71 intact at 300 and 600 K (mean −0.16 and −0.26 THz); 7 nodes merge into others and 12 are left at charge-0 minima at 300 K. Run and outputs: `validation/weyl_vs_T_omat_large_fp64/` (made before the "vanished" check and the charge-0 filter of the new-node search, so charge-0 points appear there as "charges changed" and as 24 "new nodes").
   - **Fixed in `track_nodes`: crowded and empty nodes were reported as "tracked" or "charges changed".** The old summary said "every node survived" at 600 K while the charges summed to −18. New `crowded_nodes()`: a node with a neighbour (symmetry copies included) inside the charge sphere is "annihilated" (charge 0 next to an opposite charge), "merged" (re-converged onto another node; dropped) or "unresolved" (charges not reliable). A node with charge 0 and no neighbour inside the sphere has "vanished" (a gap minimum, not a node). All of these, and every charge change, now go into the events. Checked against the 600 K force constants of seeds 0 and 2 (`validation/scripts_h200/replay_track_600K.py`) and by a unit test.
   - Conclusion for the method: with an MLIP and classical MD, follow symmetry-pinned nodes (Γ, R; item 1) and the overall softening with T. Generic Weyl points that sit within ~0.05 Å⁻¹ of each other are not resolved at 300–600 K, and their positions depend on the model by more than on T.
5. **Before calling it done:**
   - ~~README section for `--temperature`~~ done, with method, checks, limits (classical MD, no zero-point motion, generic Weyl points, model dependence) and the citations given by symfc (package metadata) and TDEP (its documentation page)
   - ~~a test for `effective_phonons`~~ done: `tests/test_modules.py`, CoSi 2×2×2 at 100 K, 40 snapshots (33 s on an H200): space group kept, static offset 0.109, optical frequencies within 0.23 THz of harmonic
   - ~~remove "experimental" from the help text~~ done; the help now points to the README for what the method resolves. Defaults raised to the validated settings: `--md_snapshots 150` (was 60, while item 1 already assumed 150) and `--max_seeds 300` (was 60, which misses CoSi gap-3|4 nodes)

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

- **One GPU (laptop):** run GRACE jobs one after another, not in parallel. Parallel runs compete for the single GPU and the CPU, and every job is slower. On the HPC, submit independent runs as separate jobs (one GPU each) at the same time.
- **Sofia HPC (H200):** GRACE-2L-OAM takes 14.9 ms per force call for 216 atoms (laptop: 47 ms), GRACE-2L-OMAT-large-ft-AM-fp64 19.8 ms. A 3-temperature Weyl run with 150 snapshots takes 18 min. Job scripts in `validation/` show the partition rules (24 CPUs per GPU; conda's `bin` on PATH, since batch jobs do not activate it). `symfc` is needed for `--temperature` and is not in `pyproject.toml`.
- `TF_FORCE_GPU_ALLOW_GROWTH` is set in `calculator.py`, so a job no longer reserves all GPU memory. CPU-bound parts (Chern numbers, node search) still use all cores through numpy.
