"""
Γ anomaly control: harmonic finite differences versus symfc fits of random
displacements from the 0 K structure (no MD). Run from the directory holding POSCAR:
    python control.py [model]          -> runs/gamma_diagnosis/control.log
"""
import sys, numpy as np
from ase import Atoms
from ase.io import read
from vasp_grace.calculator import load_calculator
from vasp_grace.condmat.phonon_tools import (phonopy_phonons, _phonopy_for, _ase_supercell,
                                             _fit_force_constants, eigensystem, PhonopyPhonons)
prim = read("POSCAR")
calc = load_calculator(sys.argv[1] if len(sys.argv) > 1 else "GRACE-2L-OAM")
SC = (3, 3, 3)
def show(label, ph, extra=""):
    g = eigensystem(ph, [0, 0, 0])[0]; r = eigensystem(ph, [0.5, 0.5, 0.5])[0]
    print(f"{label:34s} Γ[6,9,12,17,21]={np.round(g[[6,9,12,17,21]],3)}  R[0,8,20]={np.round(r[[0,8,20]],3)} {extra}", flush=True)

for d in (0.005, 0.015, 0.03):
    show(f"FD delta={d}", phonopy_phonons(prim, calc, SC, d, save=None))

ph0 = _phonopy_for(prim, SC)
sc0 = _ase_supercell(ph0); sc0.calc = calc
f0 = sc0.get_forces()
print("max |F| on perfect supercell:", np.abs(f0).max(), flush=True)
for amp in (0.01, 0.03, 0.06):
    ph = _phonopy_for(prim, SC)
    ph.generate_displacements(distance=amp, number_of_snapshots=40, random_seed=1)
    U = np.array(ph.displacements); F = []
    for s in ph.supercells_with_displacements:
        a = Atoms(s.symbols, cell=s.cell, scaled_positions=s.scaled_positions, pbc=True); a.calc = calc
        F.append(a.get_forces())
    F = np.array(F)
    p, res = _fit_force_constants(prim, SC, U, F)
    show(f"symfc random amp={amp}", p, f"residual {res:.4f}")
    p, res = _fit_force_constants(prim, SC, U - U.mean(axis=1, keepdims=True), F)
    show(f"  same, U-=mean", p, f"residual {res:.4f}")
    p, res = _fit_force_constants(prim, SC, U, F - f0)
    show(f"  same, F-=F(perfect)", p, f"residual {res:.4f}")
