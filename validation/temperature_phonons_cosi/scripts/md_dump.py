"""
Run effective_phonons at T and save the snapshots the fit sees (displacements U and
forces F, after the code's own preprocessing), the 0 K cell and the reference cell,
for offline refits with refit.py. Run from the directory holding POSCAR:
    python md_dump.py 100 npt          -> md_100K.npz  (about 20 min on a laptop GPU)
The recorded runs/gamma_diagnosis/md_100K_before_fix.npz was made with
effective_phonons as of c5f712f (before the fix in e8dc57f); md_100K_after_fix.npz
with e8dc57f, where U and F are already centred over the snapshots.
"""
import sys, json, numpy as np
from ase.io import read
from vasp_grace.calculator import load_calculator
from vasp_grace.condmat import phonon_tools as pt

T = float(sys.argv[1]); npt = sys.argv[2] == "npt"
prim = read("POSCAR"); calc = load_calculator("GRACE-2L-OAM")
captured = []
orig = pt._fit_force_constants
def spy(prim_ref, supercell, U, F):
    captured.append((prim_ref.copy(), U.copy(), F.copy())); return orig(prim_ref, supercell, U, F)
pt._fit_force_constants = spy
eff = pt.effective_phonons(prim, calc, T, (3, 3, 3), n_snapshots=150, sample_every=100, npt=npt, seed=0)
g = pt.eigensystem(eff["ph"], [0, 0, 0])[0]
print("info", eff["info"]); print("Γ", np.round(g[[6, 9, 12, 17, 21]], 3), flush=True)
prim_ref, U, F = captured[0]
np.savez_compressed(f"md_{T:g}K.npz", U=U, F=F, prim_cell=prim.cell.array,
                    prim_scaled=prim.get_scaled_positions(), numbers=prim.numbers,
                    ref_cell=prim_ref.cell.array, ref_scaled=prim_ref.get_scaled_positions(),
                    info=json.dumps(eff["info"]))
