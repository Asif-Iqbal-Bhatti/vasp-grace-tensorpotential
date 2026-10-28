# TODO

Open work, in the order to pick it up.

## 1. Phonon topology at finite temperature (`grace-topo-phonons --temperature`)

**Status: implemented, experimental.** Not yet documented in the README beyond the roadmap.

### Done

- `phonon_tools.effective_phonons()`: NPT MD for the average cell at T, NVT snapshots in phonopy's supercell, symmetrized time-averaged reference structure (Wyckoff parameters can move), second-order force constants fitted by phonopy + symfc, plus two half-run fits for error bars.
- `topological_phonons.track_nodes()`: follows every node from the harmonic (0 K) result. Weyl points are re-converged from their previous position; nodes pinned by symmetry (from `--node`) are re-identified at each T by multiplicity and nearest frequency, because bands reorder as frequencies shift. Charges are recomputed, vanished nodes and pair annihilation are flagged, and with `--search` a full zone search runs at the highest T for new nodes.
- Outputs: `weyl_trajectories.dat`, `phonon_topo_vs_T.txt`, `phonon_nodes_T<T>.dat`, `weyl_nodes_vs_T.png`.

### Validated (CoSi, GRACE-2L-OAM, 3×3×3 supercell)

- **Comparison with harmonic at 100 K:** effective frequencies at X, M, R and two generic q-points are within 0.06–0.10 THz (0.4–0.6 %) of the harmonic phonopy values.
- **Structure:** the space group stays P2₁3 after symmetrizing the average structure at 100, 300 and 600 K. Thermal expansion from NPT: V = 87.32 Å³ (0 K), 88.33 Å³ (300 K), 89.19 Å³ (600 K).
- **R-point charges:** all six fourfold nodes keep charge 2 between their band pairs up to 600 K, and the gaps inside each pair stay exactly closed.

### Open

1. **Statistical noise: mostly solved with 150 snapshots.**
   - With 60 snapshots, half-run spreads were 0.03–0.24 THz, as large as the temperature shifts.
   - With 150 snapshots per T (200 fs apart) they are mostly 0.001–0.08 THz, and several nodes now soften with temperature well beyond their error bars, e.g. (THz, 0 / 100 / 300 / 600 K):

     | Node | 0 K | 100 K | 300 K | 600 K |
     |---|---|---|---|---|
     | Γ triplet, bands 9–11 | 6.926 | 6.837 ± 0.021 | 6.764 ± 0.007 | 6.632 ± 0.040 |
     | R, bands 8–11 | 7.249 | 7.159 ± 0.019 | 7.091 ± 0.036 | 6.944 ± 0.036 |
     | R, bands 12–15 | 7.738 | 7.709 ± 0.008 | 7.611 ± 0.012 | 7.472 ± 0.027 |

   - Some spreads remain larger: up to 0.21 THz for the Γ triplet at bands 12–14 at 600 K. Use 150+ snapshots as the default.
   - All Γ and R charges are unchanged up to 600 K, as their symmetry requires. Full output: `weyl_trajectories.dat` from the run below (about 1 hour alone on the GPU).
2. **Γ anomaly: confirmed, not noise.** The top Γ triplet (bands 21–23) is at 12.258 THz harmonic and 11.809 ± 0.005 THz at 100 K, then rises to 12.072 at 300 K and falls to 11.915 at 600 K. A 0.45 THz drop at 100 K with a 0.005 THz error bar is a systematic Γ-specific bias, not physics: classical MD near 0 K should approach the harmonic force constants. It also likely explains the lower Γ triplet (bands 6–8, item 3). Candidates to check, cheapest first:
   - the rigid-translation removal (`U -= U.mean(...)`) and the acoustic sum rule in the symfc fit (Γ optical modes are where an imperfect sum rule shows up)
   - the reference structure (mean residual force 0.037 eV/Å at 100 K)
   - fp32 force noise (try a `-fp64` model)
   - a phonopy random-displacement fit at a small amplitude, as a control
3. **Bug: node matching is not one-to-one.** Pinned nodes are matched to the nearest multiplet of the same size by frequency alone, so two nodes can claim the same multiplet:
   - At 100 K, node 1 (Γ triplet, bands 6–8, 6.42 THz) was matched to bands 9–11, which already belonged to node 2.
   - At 600 K, node 7 (R, bands 0–3) was matched to bands 4–7, which belonged to node 8, and was reported as "charges changed". No charge actually changed.
   - Fix: a one-to-one assignment (Hungarian algorithm on frequency difference), preferably using eigenvector overlap with the previous temperature's multiplet, and flag a node as lost when no unclaimed multiplet is close.
4. **Weyl-point tracking has not been run yet.** Plan: a zone search of CoSi gap 3|4 at 16³, tracking at 100/300/600 K with 150 snapshots, then the new-node search at 600 K. Expect several hours. Run it alone on the GPU, after items 2 and 3.
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
