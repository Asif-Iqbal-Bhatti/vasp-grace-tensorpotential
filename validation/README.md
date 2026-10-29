# Validation runs

Recorded runs and scripts behind the checks in `TODO.md`, so that anyone can verify them.
Laptop runs (RTX 2000 Ada, 2026-10-28/29) are in `2026-10-28_session/`, `2026-10-29_session/` and
`temperature_phonons_cosi/` (Γ/R nodes versus T, with their own README). The folders below were
run on the Sofia HPC (NVIDIA H200, partition `zen4_h200`) on 2026-10-29; each holds its Slurm job
script, the log and the standard `grace-topo-phonons` outputs. CoSi (P2₁3), 3×3×3 supercell.

| Folder | What it shows |
|---|---|
| `h200_env_check/` | Same 0 K frequencies at Γ and R as the laptop (to 0.001 THz) with phonopy 4.8.3 / symfc 1.7.3 / numpy 2.1.3; force calls 3× faster (216 atoms: 14.9 ms) |
| `weyl_zone_search_16_h200/` | Harmonic 16³ search of gap 3\|4 with `--max_seeds 300`: 82 distinct nodes, 88 with symmetry copies, charges balanced |
| `weyl_vs_T_h200_maxseeds60_aborted/` | The same search with the default `--max_seeds 60`: 37 nodes, balance −12. Kept to show why the flag is needed |
| `weyl_vs_T_h200/` | Weyl tracking at 100/300/600 K, 150 snapshots, code before the crowding fix: the summary says "every node survived" at 600 K although the charges sum to −18 |
| `weyl_600K_diagnosis/` | 600 K effective force constants of that run (full fit and both half-fits, `.yaml.xz`, read by `phonopy.load`), the per-node diagnosis (`weyl_600K_diag.py`) and the replay of the fixed `track_nodes` on them |
| `weyl_vs_T_h200_fixed/` | The same run with the fixed code (MD seed 0): 12 +/− pairs annihilated, 9 merged, 12 unresolved at 600 K |
| `weyl_vs_T_h200_fixed_seed1/`, `_seed2/` | MD seeds 1 and 2: the 300–600 K events differ from seed 0 (see `TODO.md` item 4) |
| `weyl_600K_diagnosis_seed2/` | 600 K force constants of seed 2 and the replay of the final code on them |
| `weyl_vs_T_omat_large_fp64/` | GRACE-2L-OMAT-large-ft-AM-fp64: relaxation, zone search, Γ/R and Weyl nodes at 100/300/600 K with 300 snapshots |
| `committee_cosi/` | `--committee`: the GRACE-2L-OAM nodes recomputed with 6 fp64 models (each relaxed), Γ/R and the gap-3\|4 Weyl points, and a zone search per model; `v1_before_partly_resolved/` is the first run, before partly resolved charges were told apart from changed ones |
| `pressure_cosi/`, `strain_z_tension_cosi/`, `strain_z_compression_cosi/` | `--pressure 5 10 20 40` and `--strain ±0.002 … ±0.02` in a few large steps: nodes jump onto neighbours ("merged", 30 of 82 at 0 → 5 GPa). Kept to show why the steps must be small. `strain_z_tension_cosi/` also holds `strain_R_check.py`'s log: the R charge on spheres of four radii and the Weyl points inside the R multiplet at 0.5 % and 1 % strain |
| `pressure_fine_cosi/`, `strain_z_tension_fine_cosi/`, `strain_z_compression_fine_cosi/` | The same paths in steps of 1–5 GPa and 0.1 % strain: no merging (3 nodes at −1.8 % only); see `TODO.md` item 3 |
| `tests_h200/` | Test suite (55 passed, 2 skipped: gudhi, symlint), the `effective_phonons` test alone, the Γ/R node-matching smoke test |
| `scripts_h200/` | `fc_dump.py` (save effective force constants at T), `weyl_600K_diag.py`, `replay_track_600K.py`, `relax_cosi.py`, `strain_R_check.py` |

Command of the Weyl runs (`--seed N` for the other seeds; the OMAT-large run adds `--model`,
`--node 0 0 0 --node 0.5 0.5 0.5` and `--md_snapshots 300`):

```bash
grace-topo-phonons --poscar POSCAR --path GXMGRX --nkpts 40 \
    --search 16 --bands 3 3 --temperature 100 300 600 --md_snapshots 150 --max_seeds 300
```
