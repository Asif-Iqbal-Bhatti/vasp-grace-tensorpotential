"""
Relax a structure (cell and positions, space group kept) with a GRACE model and
time its force calls, before a grace-topo-phonons run with that model:
    python relax_cosi.py POSCAR_start MODEL  -> POSCAR (relaxed), relax.log
"""
import sys, time, numpy as np, spglib
from ase.io import read, write
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from vasp_grace.calculator import load_calculator

atoms = read(sys.argv[1]); model = sys.argv[2]
calc = load_calculator(model, verbose=False)
atoms.calc = calc
atoms.set_constraint(FixSymmetry(atoms, symprec=1e-3))
BFGS(FrechetCellFilter(atoms), logfile="relax.log").run(fmax=1e-4, steps=500)
atoms.set_constraint()
sg = spglib.get_spacegroup((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=1e-3)
print(f"{model}: a = {atoms.cell.lengths()[0]:.5f} Å, V = {atoms.get_volume()/len(atoms)*8:.3f} Å³ per cell, "
      f"{sg}, max |F| = {np.abs(atoms.get_forces()).max():.1e} eV/Å, "
      f"max |stress| = {np.abs(atoms.get_stress()).max():.1e} eV/Å³, "
      f"x(Co) = {atoms.get_scaled_positions()[0, 0]:.5f}, x(Si) = {atoms.get_scaled_positions()[4, 0]:.5f}")
write("POSCAR", atoms, format="vasp", direct=True)
a = atoms.repeat(3); a.calc = calc; rng = np.random.default_rng(0)
for _ in range(3):
    a.positions += 0.01 * rng.standard_normal(a.positions.shape); a.get_forces()
t = time.perf_counter()
for _ in range(10):
    a.positions += 0.01 * rng.standard_normal(a.positions.shape); a.get_forces()
print(f"{len(a)} atoms: {(time.perf_counter() - t) / 10 * 1e3:.1f} ms per force call", flush=True)
