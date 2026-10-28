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
- **Structure:** the space group stays P2₁3 after symmetrizing the average structure at 100, 300 and 600 K. Thermal expansion from NPT: V = 87.27 Å³ (0 K), 88.33 Å³ (300 K), 89.19 Å³ (600 K).
- **R-point charges:** all six fourfold nodes keep charge 2 between their band pairs up to 600 K, and the gaps inside each pair stay exactly closed.

### Open

1. **Statistical noise.** With 60 snapshots, half-run spreads are 0.03–0.24 THz, about as large as the temperature shifts (≈0.3 THz at R by 600 K).
   - A rerun with 150 snapshots per T was running when this was written: scratch directory `tdep_GR2`, command below. Record whether the spreads drop clearly below the shifts.
   - If they don't: more snapshots, a check of snapshot decorrelation (velocity autocorrelation of the slowest optical mode), several independent seeds, and symfc cutoff options.
2. **Γ high-frequency modes at 100 K sit up to 0.4 THz below harmonic.** Classical MD that close to 0 K should nearly reproduce the harmonic force constants. Candidates to check:
   - fp32 force noise (try a `-fp64` model)
   - the reference structure (mean residual force was 0.06 eV/Å at 100 K)
   - the fit residual (12 % at 100 K)
   - a phonopy random-displacement fit at a small amplitude, as a control
3. **Re-validate pinned-node tracking.** The multiplicity/nearest-frequency matching was added after the first run, which falsely reported the Γ triplet (bands 6–8) as vanished at 100 K because of band reordering. The 150-snapshot rerun is the first test of the fix.
4. **Weyl-point tracking has not been run yet.** Plan: a zone search of CoSi gap 3|4 at 16³, tracking at 100/300/600 K, then the new-node search at 600 K. Expect a few hours. Run it alone on the GPU.
5. **Before calling it done:**
   - README section for `--temperature`, with method, limits (classical MD, no zero-point motion) and TDEP/symfc citations taken from their own documentation
   - a test for `effective_phonons` (short MD, check space group kept and frequencies close to harmonic)
   - remove "experimental" from the help text

Command for the running check (Γ and R nodes only, about 1 hour):

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
